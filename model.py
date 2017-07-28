from scipy.stats import spearmanr

import pandas as pd
import numpy as np
import os

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.model_selection import cross_val_score, KFold

from lightgbm.sklearn import LGBMRegressor
import string
from nltk.tokenize import TreebankWordTokenizer
from xgboost import XGBRegressor
from difflib import SequenceMatcher

import pickle

import itertools
import stop_words

np.random.seed(1337)

with open("/usr/share/dict/words") as wordfile:
    words = set(x.strip().lower() for x in wordfile.readlines())


def spearmancorr(est, X, y):
    rho, pval = spearmanr(np.reshape(y, (-1, 1)), np.reshape(est.predict(X), (-1, 1)), axis=0)
    return rho


stopwords = set(stop_words.get_stop_words("english"))
punct = str.maketrans({p: None for p in string.punctuation})


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
                text = " ".join(word for word in text.split(" ") if word.lower().translate(punct) not in stopwords)

            return text

        punct_modes = ["exclude", "leave"]
        lower_modes = [True, False]
        stops_modes = [True, False]

        new_vectorizers = {}

        for preproc_mode in itertools.product(lower_modes, punct_modes, stops_modes):
            def preproc(text):
                return preprocess(text, *preproc_mode)

            if vectorizers:
                tfidf_thread, tfidf_context, count_thread, count_context, tfidf_char_thread, count_char_thread = vectorizers["{}_{}_{}".format(*preproc_mode)]
            else:
                tfidf_thread = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=4000, tokenizer=TreebankWordTokenizer().tokenize)
                tfidf_thread.fit(data["thread_raw"])
                tfidf_context = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=4000, tokenizer=TreebankWordTokenizer().tokenize)
                tfidf_context.fit(data["context"])
                count_thread = CountVectorizer(analyzer='word', ngram_range=(1, 2), max_features=4000, tokenizer=TreebankWordTokenizer().tokenize)
                count_thread.fit(data["thread_raw"])
                count_context = CountVectorizer(analyzer='word', ngram_range=(1, 2), max_features=4000, tokenizer=TreebankWordTokenizer().tokenize)
                count_context.fit(data["context"])
                tfidf_char_thread = TfidfVectorizer(analyzer='char', ngram_range=(1, 4), max_features=4000)
                tfidf_char_thread.fit(data["thread_raw"])
                count_char_thread = CountVectorizer(analyzer='char', ngram_range=(1, 4), max_features=4000)
                count_char_thread.fit(data["thread_raw"])

                new_vectorizers["{}_{}_{}".format(*preproc_mode)] = (tfidf_thread, tfidf_context, count_thread, count_context, tfidf_char_thread, count_char_thread)

            def get_speakers_text(speaker):
                return lambda row: [preproc(x) for x in np.array(row['thread'])[np.array(row['speaker']) == speaker]]

            data["thread_split_A_{}_{}_{}".format(*preproc_mode)] = data.apply(get_speakers_text("A"), axis=1)
            data['thread_split_B_{}_{}_{}'.format(*preproc_mode)] = data.apply(get_speakers_text("B"), axis=1)

            def join_speaker(speaker):
                return lambda row: " ".join(row["thread_split_{}_{}_{}_{}".format(speaker, *preproc_mode)])

            data["thread_joined_A_{}_{}_{}".format(*preproc_mode)] = data.apply(join_speaker("A"), axis=1)
            data["thread_joined_B_{}_{}_{}".format(*preproc_mode)] = data.apply(join_speaker("B"), axis=1)

            def get_first(speaker):
                return lambda row: " ".join(row["thread_split_{}_{}_{}_{}".format(speaker, *preproc_mode)])

            data["start_A_{}_{}_{}".format(*preproc_mode)] = data.apply(get_first("A"), axis=1)
            data["start_B_{}_{}_{}".format(*preproc_mode)] = data.apply(get_first("B"), axis=1)

            data["tfidf_all_{}_{}_{}".format(*preproc_mode)] = tfidf_thread.transform(data["thread_raw"]).toarray().tolist()
            data["tfidf_A_{}_{}_{}".format(*preproc_mode)] = tfidf_thread.transform(data["thread_joined_A_{}_{}_{}".format(*preproc_mode)]).toarray().tolist()
            data["tfidf_B_{}_{}_{}".format(*preproc_mode)] = tfidf_thread.transform(data["thread_joined_B_{}_{}_{}".format(*preproc_mode)]).toarray().tolist()
            data["tfidf_char_A_{}_{}_{}".format(*preproc_mode)] = tfidf_char_thread.transform(data["thread_joined_A_{}_{}_{}".format(*preproc_mode)]).toarray().tolist()
            data["tfidf_char_B_{}_{}_{}".format(*preproc_mode)] = tfidf_char_thread.transform(data["thread_joined_B_{}_{}_{}".format(*preproc_mode)]).toarray().tolist()

            data["tfidf_context_{}_{}_{}".format(*preproc_mode)] = tfidf_context.transform(data["context"]).toarray().tolist()

            data["counts_all_{}_{}_{}".format(*preproc_mode)] = count_thread.transform(data["thread_raw"]).toarray().tolist()
            data["counts_A_{}_{}_{}".format(*preproc_mode)] = count_thread.transform(data["thread_joined_A_{}_{}_{}".format(*preproc_mode)]).toarray().tolist()
            data["counts_B_{}_{}_{}".format(*preproc_mode)] = count_thread.transform(data["thread_joined_B_{}_{}_{}".format(*preproc_mode)]).toarray().tolist()
            data["counts_char_A_{}_{}_{}".format(*preproc_mode)] = count_char_thread.transform(data["thread_joined_A_{}_{}_{}".format(*preproc_mode)]).toarray().tolist()
            data["counts_char_B_{}_{}_{}".format(*preproc_mode)] = count_char_thread.transform(data["thread_joined_B_{}_{}_{}".format(*preproc_mode)]).toarray().tolist()

            data["counts_context_{}_{}_{}".format(*preproc_mode)] = count_context.transform(data["context"]).toarray().tolist()

            def run_len(target, func):
                return lambda row: [func((len(list(g)) for person, g in itertools.groupby(row["speaker"]) if person == target), default=0)]

            data["f_max_run_A_{}_{}_{}".format(*preproc_mode)] = data.apply(run_len("A", max), axis=1)
            data["f_max_run_B_{}_{}_{}".format(*preproc_mode)] = data.apply(run_len("B", max), axis=1)
            data["f_min_run_A_{}_{}_{}".format(*preproc_mode)] = data.apply(run_len("A", min), axis=1)
            data["f_min_run_B_{}_{}_{}".format(*preproc_mode)] = data.apply(run_len("B", min), axis=1)

            def typo_count(target):
                return lambda row: [sum(1 for word in preproc(row["thread_joined_{}_{}_{}_{}".format(target, *preproc_mode)]).split() if word not in words)]

            data["f_typos_A_{}_{}_{}".format(*preproc_mode)] = data.apply(typo_count("A"), axis=1)
            data["f_typos_B_{}_{}_{}".format(*preproc_mode)] = data.apply(typo_count("B"), axis=1)
            data["f_typos_frac_A_{}_{}_{}".format(*preproc_mode)] = data.apply(lambda row: [row["f_typos_A_{}_{}_{}".format(*preproc_mode)][0] / (1 + len(row["thread_joined_A_{}_{}_{}".format(*preproc_mode)].split()))], axis=1)
            data["f_typos_frac_B_{}_{}_{}".format(*preproc_mode)] = data.apply(lambda row: [row["f_typos_B_{}_{}_{}".format(*preproc_mode)][0] / (1 + len(row["thread_joined_B_{}_{}_{}".format(*preproc_mode)].split()))], axis=1)

            def relevant_words(target):
                return lambda row: [sum(1 for word in preproc(row["thread_joined_{}_{}_{}_{}".format(target, *preproc_mode)]).split() if word in (preproc(x) for x in row['context'].split()))]

            data["f_relevant_A_{}_{}_{}".format(*preproc_mode)] = data.apply(relevant_words("A"), axis=1)
            data["f_relevant_B_{}_{}_{}".format(*preproc_mode)] = data.apply(relevant_words("B"), axis=1)
            data["f_relevant_frac_A_{}_{}_{}".format(*preproc_mode)] = data.apply(lambda row: [row["f_relevant_A_{}_{}_{}".format(*preproc_mode)][0] / (1 + len(row["thread_joined_A_{}_{}_{}".format(*preproc_mode)].split()))], axis=1)
            data["f_relevant_frac_B_{}_{}_{}".format(*preproc_mode)] = data.apply(lambda row: [row["f_relevant_B_{}_{}_{}".format(*preproc_mode)][0] / (1 + len(row["thread_joined_B_{}_{}_{}".format(*preproc_mode)].split()))], axis=1)

            def unanswered_messages(target):
                return lambda row: [sum(len(list(g)) != 1 for person, g in itertools.groupby(row["speaker"]) if person == target)]

            data["f_unanswered_A_{}_{}_{}".format(*preproc_mode)] = data.apply(unanswered_messages("B"), axis=1)
            data["f_unanswered_B_{}_{}_{}".format(*preproc_mode)] = data.apply(unanswered_messages("A"), axis=1)

            def count_stop_words(target):
                return lambda row: [sum(1 for word in preproc(row["thread_joined_A_{}_{}_{}".format(*preproc_mode)]).split() if word in stopwords)]

            data["f_stops_A_{}_{}_{}".format(*preproc_mode)] = data.apply(count_stop_words("A"), axis=1)
            data["f_stops_B_{}_{}_{}".format(*preproc_mode)] = data.apply(count_stop_words("B"), axis=1)

            def longest_common_substring(target):
                return lambda row: [SequenceMatcher(None, row["thread_joined_A_{}_{}_{}".format(*preproc_mode)], preproc(row["context"])).find_longest_match(0, len(row["thread_joined_A_{}_{}_{}".format(*preproc_mode)]), 0, len(preproc(row["context"]))).size]

            data["f_lcs_A_{}_{}_{}".format(*preproc_mode)] = data.apply(longest_common_substring("A"), axis=1)
            data["f_lcs_B_{}_{}_{}".format(*preproc_mode)] = data.apply(longest_common_substring("B"), axis=1)

            data["f_scared_A_{}_{}_{}".format(*preproc_mode)] = data.apply(lambda row: [len(preproc(row['context'])) > 1000 and len(row["thread_joined_A_{}_{}_{}".format(*preproc_mode)]) == 0], axis=1)
            data["f_scared_B_{}_{}_{}".format(*preproc_mode)] = data.apply(lambda row: [len(preproc(row['context'])) > 1000 and len(row["thread_joined_B_{}_{}_{}".format(*preproc_mode)]) == 0], axis=1)

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


def build_feature_list(templates):
    features = []

    punct_modes = ["exclude", "leave"]
    lower_modes = [True, False]
    stops_modes = [False, True]

    for template, *preproc_mode in itertools.product(templates, lower_modes, punct_modes, stops_modes):
        features.append(template.format(*preproc_mode))

    return features


feature_templates = [
    'counts_A_{}_{}_{}',
    'counts_B_{}_{}_{}',
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
    'f_lcs_A_{}_{}_{}',
    'f_lcs_B_{}_{}_{}'
]

features = build_feature_list(feature_templates)


data, vectorizers = process_data("../train/", train=True)

X = data[features].values
X = np.stack([np.concatenate(X[i]) for i in range(X.shape[0])])

y_A = data["qualA"].values
y_B = data["qualB"].values

clf_A = LGBMRegressor(n_estimators=100, num_leaves=1000)
clf_A.fit(X, y_A)
clf_B = LGBMRegressor(n_estimators=100, num_leaves=1000)
clf_B.fit(X, y_B)

valid, _ = process_data("../data/train_20170727.json", train=True, vectorizers=vectorizers)

my = pd.read_csv("my.csv")
svm_lasso = pd.read_csv("svm_lasso.csv")
xgb_lgbm = pd.read_csv("xgb_lgbm.csv")
subm = pd.read_csv("subm.csv", header=None)

lol = pd.concat([pd.DataFrame(X), my[["Alice", "Bob"]], svm_lasso[["Alice", "Bob"]], xgb_lgbm[["Alice", "Bob"]], subm[[1, 2]]], axis=1).values

clf_2_A = XGBRegressor()
clf_2_A.fit(lol, y_A)
clf_2_B = XGBRegressor()
clf_2_B.fit(lol, y_B)


X_27 = np.stack([np.concatenate(X_27[i]) for i in range(X_27.shape[0])])

y_A_27 = data["qualA"].values
y_B_27 = data["qualB"].values

test, *_ = process_data("../data/test_20170727.json", train=False, vectorizers=vectorizers)

T = test[features].values
T = np.stack([np.concatenate(T[i]) for i in range(T.shape[0])])

pred_A = clf_A.predict(T)
pred_B = clf_B.predict(T)

pd.DataFrame(np.stack([pred_A, pred_B]).T, index=test.index, columns=["Alice", "Bob"]).to_csv("pred.csv")

y_V = valid["qualA"].values

cv = cross_val_score(XGBRegressor(), lol, y_V, scoring=spearmancorr, cv=KFold(10, True, 123), verbose=3)
cv.mean(), cv.std()

pred

spearmanr(np.reshape(pred[["Alice", "Bob"]].values, (-1, 1)), np.reshape(valid[["qualA", "qualB"]].values, (-1, 1)))
