import copy
import re
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from pymorphy2 import MorphAnalyzer
from nltk.stem.snowball import SnowballStemmer
from nltk import pos_tag
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from scipy import stats
from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier

from parse_data import parse_dir
morph_ru = MorphAnalyzer()
morph_en = SnowballStemmer("english")


def spearman(a, b):
    return stats.spearmanr(a, b)[0]


def text_to_wordlist(sentence):
    regexp = "[^а-яА-Яёa-zA-Z.,;!?:()><_]"
    sentence = re.sub(regexp, " ", sentence)
    result = sentence.lower().split()
    return result


def text_to_charlist(sentence):
    return list(sentence)


def stem_sentence(sentence, language):
    words = text_to_wordlist(sentence)
    for j in range(len(words)):
        if language == 'ru':
            words[j] = morph_ru.parse(words[j])[0].normal_form
        if language == 'en':
            words[j] = morph_en.stem(words[j])
    return " ".join(words)


def get_sentence_tags(sentence, language):
    words = text_to_wordlist(sentence)
    tags = []
    if language == 'en':
        if len(words) != 0:
            tags = [i[1] for i in pos_tag(words)]
    if language == 'ru':
        for j in range(len(words)):
            pos = morph_ru.parse(words[j])[0].tag.POS
            if pos is not None:
                tags.append(pos)
    return " ".join(tags)


def bow(train_texts, test_texts, language='en', stem=False, tokenizer=text_to_wordlist, preprocessor=None,
        use_tfidf=False, max_features=None, bow_ngrams=(1,2), analyzer='word'):
    train = copy.deepcopy(train_texts)
    test = copy.deepcopy(test_texts)
    if stem:
        for i in range(len(train)):
            train[i] = stem_sentence(train[i], language)
        for i in range(len(test)):
            test[i] = stem_sentence(test[i], language)

    if use_tfidf:
        vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=bow_ngrams, tokenizer=tokenizer,
                                     preprocessor=preprocessor, max_features=max_features)
    else:
        vectorizer = CountVectorizer(analyzer=analyzer, ngram_range=bow_ngrams, tokenizer=tokenizer,
                                     preprocessor=preprocessor, max_features=max_features)
    data = train+test
    data = vectorizer.fit_transform(data)
    train_data = data[:len(train)]
    test_data = data[len(train):]
    return train_data.todense(), test_data.todense()


def count_in_a_row(mask):
    count = 0
    max_count = 0
    for i in mask:
        if i == 1:
            count += 1
            max_count = max(count, max_count)
        else:
            count = 0
    return max_count


def collect_all_features():
    df = parse_dir()
    data = pd.DataFrame()
    data["userMessages"] = df["AliceMessages"].tolist() + df["BobMessages"].tolist()
    data["userOpponentMessages"] = df["BobMessages"].tolist() + df["AliceMessages"].tolist()
    data["userMessageMask"] = df["AliceMessageMask"].tolist() + df["BobMessageMask"].tolist()
    data["userConcatenatedMessages"] = data["userMessages"].apply(lambda x: " ".join(x))
    data["userOpponentConcatenatedMessages"] = data["userMessages"].apply(lambda x: " ".join(x))
    data["userIsBot"] = df["AliceIsBot"].tolist() + df["BobIsBot"].tolist()
    data["userScores"] = df["AliceScore"].tolist() + df["BobScore"].tolist()

    data["messageNum"] = data["userMessages"].apply(lambda x: len(x))
    data["numChars"] = data["userMessages"].apply(lambda x: sum([len(msg) for msg in x]))
    data["numWords"] = data["userMessages"].apply(lambda x: sum([len(msg.split()) for msg in x]))
    data["avgChars"] = data["userMessages"].apply(lambda x: np.mean([0] + [len(msg) for msg in x]))
    data["avgWords"] = data["userMessages"].apply(lambda x: np.mean([0] + [len(msg.split()) for msg in x]))
    data["msgInARow"] = data["userMessageMask"].apply(lambda x: count_in_a_row(x))

    # data["RmessageNum"] = data["userOpponentMessages"].apply(lambda x: len(x))
    # data["RnumChars"] = data["userOpponentMessages"].apply(lambda x: sum([len(msg) for msg in x]))
    # data["RnumWords"] = data["userOpponentMessages"].apply(lambda x: sum([len(msg.split()) for msg in x]))
    # data["RavgChars"] = data["userOpponentMessages"].apply(lambda x: np.mean([0] + [len(msg) for msg in x]))
    # data["RavgWords"] = data["userOpponentMessages"].apply(lambda x: np.mean([0] + [len(msg.split()) for msg in x]))

    print("BoW step...")
    bow_train_data, _ = bow(data["userConcatenatedMessages"].tolist(), [], language='en', stem=False,
                            tokenizer=text_to_wordlist, use_tfidf=False, bow_ngrams=(1, 1))
    data = pd.concat([data, pd.DataFrame(bow_train_data)], axis=1)

    print("RBoW step...")
    bow_train_data, _ = bow(data["userOpponentConcatenatedMessages"].tolist(), [], language='en', stem=False,
                            tokenizer=text_to_wordlist, use_tfidf=False, bow_ngrams=(1, 1))
    data = pd.concat([data, pd.DataFrame(bow_train_data)], axis=1)

    print("BoC step...")
    bow_train_data, _ = bow(data["userConcatenatedMessages"].tolist(), [], language='en', stem=False,
                            tokenizer=text_to_charlist, use_tfidf=False, bow_ngrams=(1, 1))
    data = pd.concat([data, pd.DataFrame(bow_train_data)], axis=1)

    print("RBoC step...")
    bow_train_data, _ = bow(data["userOpponentConcatenatedMessages"].tolist(), [], language='en', stem=False,
                            tokenizer=text_to_charlist, use_tfidf=False, bow_ngrams=(1, 1))
    data = pd.concat([data, pd.DataFrame(bow_train_data)], axis=1)

    # print("POS step...")
    # pos_train_data = []
    # for text in data["userConcatenatedMessages"].tolist():
    #     pos_train_data.append(get_sentence_tags(text, "en"))
    # pos_train_data, _ = bow(pos_train_data, [], language='en', stem=False, tokenizer=text_to_wordlist,
    #                         use_tfidf=False, bow_ngrams=(1, 1))
    # data = pd.concat([data, pd.DataFrame(pos_train_data)], axis=1)
    return data


def answer_bot(data, train_indices, test_indices):
    answers = np.array([int(i) for i in data["userIsBot"].tolist()])
    data = data.drop(["userMessages", "userMessageMask", "userConcatenatedMessages", "userIsBot",
                      "userScores", "userOpponentMessages", "userOpponentConcatenatedMessages"], axis=1)

    l = data.shape[0]//2
    pairs = {i: l+i if i < l else i-l for i in range(2*l)}
    train_indices += [pairs[i] for i in train_indices]
    test_indices += [pairs[i] for i in test_indices]
    assert len(set(train_indices).intersection(set(test_indices))) == 0

    clf = LinearSVC(tol=0.1)
    clf.fit(data.loc[train_indices, :], answers[train_indices])
    return clf.predict(data.loc[test_indices, :])


def predict_regression(data, train_indices, test_indices):
    bot_answers = answer_bot(data, train_indices, test_indices)
    answers = np.array([int(i) for i in data["userScores"].tolist()])
    l = data.shape[0] // 2
    pairs = {i: l + i if i < l else i - l for i in range(2 * l)}
    bot_scores_indices = [pairs[i] for i in data[data['userIsBot'] == True].index.tolist()]
    data = data.drop(["userMessages", "userMessageMask", "userConcatenatedMessages",
                      "userIsBot", "userScores", "userOpponentMessages", "userOpponentConcatenatedMessages"], axis=1)

    train_indices += [pairs[i] for i in train_indices]
    test_indices += [pairs[i] for i in test_indices]
    assert len(set(train_indices).intersection(set(test_indices))) == 0
    train_indices = list(set(train_indices).difference(set(bot_scores_indices)))

    clf = Lasso()
    clf.fit(data.loc[train_indices, :], answers[train_indices])
    preds = clf.predict(data.loc[test_indices])
    for i, is_bot in enumerate(bot_answers):
        if is_bot == 1.0:
            preds[test_indices.index(pairs[test_indices[i]])] = 0
    del data
    return spearman(answers[test_indices], preds)

scores = []
features = collect_all_features()
for i in range(20):
    train_indices, test_indices, _, __ = train_test_split(range(features.shape[0]//2),
                                                          range(features.shape[0]//2), test_size=0.1, random_state=i)
    score = predict_regression(features, train_indices, test_indices)
    print(score)
    scores.append(score)
scores = np.array(scores)
print("CV: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std()))