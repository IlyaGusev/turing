import pandas as pd
import numpy as np
import os

from sklearn.feature_extraction.text import CountVectorizer
import string
from nltk.tokenize import TreebankWordTokenizer

import itertools
from nltk.corpus import stopwords
import nltk

from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import pickle

np.random.seed(1337)

nltk.download("stopwords")
stop_words = stopwords.words("english")
punct = str.maketrans({p: None for p in string.punctuation})

with open("words.txt") as wordfile:
    words = set(x.strip().lower() for x in wordfile.readlines())


def process_data(path, exclude=[], train=True, vectorizers=None):
    def process_file(path):
        user = {'Alice': "A", 'Bob': "B"}

        df = pd.read_json(path).set_index("dialogId")
        df['speaker'] = df["thread"].apply(lambda x: [user[msg['userId']] for msg in x])
        df['thread'] = df["thread"].apply(lambda x: [msg['text'] for msg in x], convert_dtype=False)
        df['thread_raw'] = df["thread"].apply(lambda x: " ".join(x))
        if train:
            df["qualA"] = df["evaluation"].apply(lambda x: sorted(x, key=lambda x: x['userId'])[0]['quality'])
            df["qualB"] = df["evaluation"].apply(lambda x: sorted(x, key=lambda x: x['userId'])[1]['quality'])
            df["botA"] = df["users"].apply(lambda x: sorted(x, key=lambda x: x['id'])[0]['userType'] == 'Bot')
            df["botB"] = df["users"].apply(lambda x: sorted(x, key=lambda x: x['id'])[1]['userType'] == 'Bot')
        df.drop(['users'], axis=1, inplace=True)
        if train:
            df.drop(['evaluation'], axis=1, inplace=True)

        return df

    def add_features(data, vectorizers):
        def preprocess(text, lower, punctuation, stops):
            if lower:
                text = text.lower()
            if punctuation == "exclude":
                text = text.translate(str.maketrans({p: None for p in string.punctuation}))
            if stops:
                text = " ".join(word for word in text.split(" ") if word.lower().translate(punct) not in stop_words)

            return text

        punct_modes = ["exclude", "leave"]
        lower_modes = [True, False]
        stops_modes = [True, False]

        new_vectorizers = {}

        for preproc_mode in itertools.product(lower_modes, punct_modes, stops_modes):
            preproc = lambda text: preprocess(text, *preproc_mode)

            if vectorizers:
                count_thread, count_context, count_char_thread = vectorizers["{}_{}_{}".format(*preproc_mode)]
            else:
                count_thread = CountVectorizer(analyzer='word', ngram_range=(1, 2), max_features=4000,
                                               tokenizer=TreebankWordTokenizer().tokenize)
                count_thread.fit(data["thread_raw"])
                count_context = CountVectorizer(analyzer='word', ngram_range=(1, 2), max_features=4000,
                                                tokenizer=TreebankWordTokenizer().tokenize)
                count_context.fit(data["context"])
                count_char_thread = CountVectorizer(analyzer='char', ngram_range=(1, 4), max_features=4000)
                count_char_thread.fit(data["thread_raw"])

                new_vectorizers["{}_{}_{}".format(*preproc_mode)] = (count_thread, count_context, count_char_thread)

            def get_speaker(speaker):
                return lambda row: [preproc(x) for x in np.array(row['thread'])[np.array(row['speaker']) == speaker]]

            data["thread_split_A_{}_{}_{}".format(*preproc_mode)] = data.apply(get_speaker("A"), axis=1)
            data['thread_split_B_{}_{}_{}'.format(*preproc_mode)] = data.apply(get_speaker("B"), axis=1)

            def join_speaker(speaker):
                return lambda row: " ".join(row["thread_split_{}_{}_{}_{}".format(speaker, *preproc_mode)])

            data["thread_joined_A_{}_{}_{}".format(*preproc_mode)] = data.apply(join_speaker("A"), axis=1)
            data["thread_joined_B_{}_{}_{}".format(*preproc_mode)] = data.apply(join_speaker("B"), axis=1)

            def get_first(speaker):
                return lambda row: " ".join(row["thread_split_{}_{}_{}_{}".format(speaker, *preproc_mode)])

            data["start_A_{}_{}_{}".format(*preproc_mode)] = data.apply(get_first("A"), axis=1)
            data["start_B_{}_{}_{}".format(*preproc_mode)] = data.apply(get_first("B"), axis=1)

            data["counts_all_{}_{}_{}".format(*preproc_mode)] = count_thread.transform(
                data["thread_raw"]).toarray().tolist()
            data["counts_A_{}_{}_{}".format(*preproc_mode)] = count_thread.transform(
                data["thread_joined_A_{}_{}_{}".format(*preproc_mode)]).toarray().tolist()
            data["counts_B_{}_{}_{}".format(*preproc_mode)] = count_thread.transform(
                data["thread_joined_B_{}_{}_{}".format(*preproc_mode)]).toarray().tolist()
            data["counts_char_A_{}_{}_{}".format(*preproc_mode)] = count_char_thread.transform(
                data["thread_joined_A_{}_{}_{}".format(*preproc_mode)]).toarray().tolist()
            data["counts_char_B_{}_{}_{}".format(*preproc_mode)] = count_char_thread.transform(
                data["thread_joined_B_{}_{}_{}".format(*preproc_mode)]).toarray().tolist()

            data["counts_context_{}_{}_{}".format(*preproc_mode)] = count_context.transform(
                data["context"]).toarray().tolist()

            def run_len(target, func):
                return lambda row: [
                    func((len(list(g)) for person, g in itertools.groupby(row["speaker"]) if person == target),
                         default=0)]

            data["f_max_run_A_{}_{}_{}".format(*preproc_mode)] = data.apply(run_len("A", max), axis=1)
            data["f_max_run_B_{}_{}_{}".format(*preproc_mode)] = data.apply(run_len("B", max), axis=1)
            data["f_min_run_A_{}_{}_{}".format(*preproc_mode)] = data.apply(run_len("A", min), axis=1)
            data["f_min_run_B_{}_{}_{}".format(*preproc_mode)] = data.apply(run_len("B", min), axis=1)

            def typo_count(target):
                return lambda row: [sum(
                    1 for word in preproc(row["thread_joined_{}_{}_{}_{}".format(target, *preproc_mode)]).split() if
                    word not in words)]

            data["f_typos_A_{}_{}_{}".format(*preproc_mode)] = data.apply(typo_count("A"), axis=1)
            data["f_typos_B_{}_{}_{}".format(*preproc_mode)] = data.apply(typo_count("B"), axis=1)
            data["f_typos_frac_A_{}_{}_{}".format(*preproc_mode)] = data.apply(lambda row: [
                row["f_typos_A_{}_{}_{}".format(*preproc_mode)][0] / (
                    1 + len(preproc(row["thread_joined_A_{}_{}_{}".format(*preproc_mode)]).split()))], axis=1)
            data["f_typos_frac_B_{}_{}_{}".format(*preproc_mode)] = data.apply(lambda row: [
                row["f_typos_B_{}_{}_{}".format(*preproc_mode)][0] / (
                    1 + len(preproc(row["thread_joined_B_{}_{}_{}".format(*preproc_mode)]).split()))], axis=1)

            def relevant_words(target):
                return lambda row: [sum(
                    1 for word in preproc(row["thread_joined_{}_{}_{}_{}".format(target, *preproc_mode)]).split() if
                    word in (preproc(x) for x in row['context'].split()))]

            data["f_relevant_A_{}_{}_{}".format(*preproc_mode)] = data.apply(relevant_words("A"), axis=1)
            data["f_relevant_B_{}_{}_{}".format(*preproc_mode)] = data.apply(relevant_words("B"), axis=1)
            data["f_relevant_frac_A_{}_{}_{}".format(*preproc_mode)] = data.apply(lambda row: [
                row["f_relevant_A_{}_{}_{}".format(*preproc_mode)][0] / (
                    1 + len(preproc(row["thread_joined_A_{}_{}_{}".format(*preproc_mode)]).split()))], axis=1)
            data["f_relevant_frac_B_{}_{}_{}".format(*preproc_mode)] = data.apply(lambda row: [
                row["f_relevant_B_{}_{}_{}".format(*preproc_mode)][0] / (
                    1 + len(preproc(row["thread_joined_B_{}_{}_{}".format(*preproc_mode)]).split()))], axis=1)

            def unanswered_messages(target):
                return lambda row: [
                    sum(len(list(g)) != 1 for person, g in itertools.groupby(row["speaker"]) if person == target)]

            data["f_unanswered_A_{}_{}_{}".format(*preproc_mode)] = data.apply(unanswered_messages("B"), axis=1)
            data["f_unanswered_B_{}_{}_{}".format(*preproc_mode)] = data.apply(unanswered_messages("A"), axis=1)

            def count_stop_words(target):
                return lambda row: [sum(
                    1 for word in preproc(row["thread_joined_A_{}_{}_{}".format(*preproc_mode)]).split() if
                    word in stop_words)]

            data["f_stops_A_{}_{}_{}".format(*preproc_mode)] = data.apply(count_stop_words("A"), axis=1)
            data["f_stops_B_{}_{}_{}".format(*preproc_mode)] = data.apply(count_stop_words("B"), axis=1)

        if not vectorizers:
            vectorizers = new_vectorizers

        return data, vectorizers

    if os.path.isdir(path):
        data = pd.concat(
            [
                process_file(os.path.join(path, file))
                for file in os.listdir(path)
                if (
                       file.startswith("train") and train or
                       file.startswith("test") and not train
                   ) and file not in exclude
            ]
        )
    else:
        data = process_file(path)

    data, vectorizers = add_features(data, vectorizers)

    return data, vectorizers


with open("models_mryab/vectorizers_4d.pkl", "rb") as f:
    vectorizers = pickle.load(f)

feat_templates = [
    # 'counts_all_{}_{}_{}',
    'counts_A_{}_{}_{}',
    'counts_B_{}_{}_{}',
    # 'counts_context_{}_{}_{}',
    'f_max_run_A_{}_{}_{}',
    'f_max_run_B_{}_{}_{}',
    'f_min_run_A_{}_{}_{}',
    'f_min_run_B_{}_{}_{}',
    'f_typos_A_{}_{}_{}',
    'f_typos_B_{}_{}_{}',
    'f_typos_frac_A_{}_{}_{}',
    'f_typos_frac_B_{}_{}_{}',
    'f_relevant_A_{}_{}_{}',
    'f_relevant_B_{}_{}_{}',
    'f_relevant_frac_A_{}_{}_{}',
    'f_relevant_frac_B_{}_{}_{}',
    'f_unanswered_A_{}_{}_{}',
    'f_unanswered_B_{}_{}_{}',
    'f_stops_A_{}_{}_{}',
    'f_stops_B_{}_{}_{}',
]

features = []

punct_modes = ["exclude", "leave"]
lower_modes = [True, False]
stops_modes = [False, True]

for template, *preproc_mode in itertools.product(feat_templates, lower_modes, punct_modes, stops_modes):
    features.append(template.format(*preproc_mode))

test, *_ = process_data("data/test/", train=False, vectorizers=vectorizers)
T = ((test[features]).values)
T = np.stack([np.concatenate(T[i]) for i in range(T.shape[0])])

with open("models_mryab/mryab_stregr_a_4.pkl", "rb") as file:
    stregr_a = pickle.load(file)

with open("models_mryab/mryab_stregr_a_4.pkl", "rb") as file:
    stregr_b = pickle.load(file)

pred_a = stregr_a.predict(T)
pred_b = stregr_b.predict(T)

subm = pd.DataFrame(np.vstack([pred_a, pred_b]).T, index=test.index, columns=["Alice", "Bob"])
subm.to_csv("data/answer-mryab-4days.csv")
