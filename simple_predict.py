import copy
import re
import pandas as pd
import numpy as np
import os
import editdistance

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from pymorphy2 import MorphAnalyzer
from nltk.stem.snowball import SnowballStemmer
from nltk import pos_tag
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from scipy import stats
from nltk.tokenize import TweetTokenizer
from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import ShuffleSplit, cross_val_score

from parse_data import parse_dir, parse
morph_ru = MorphAnalyzer()
morph_en = SnowballStemmer("english")


def spearman(a, b):
    return stats.spearmanr(a, b)[0]


# def collect_vocab(messages):
#     words = set()
#     for message in messages:
#         for word in text_to_wordlist(message):
#             words.add(word)
#     a = np.zeros((len(words), len(words)))
#     words = list(words)
#     for i in range(len(words)):
#         for j in range(len(words)):
#             a[i][j] = editdistance.eval(words[i], words[j])
#             if a[i][j] == 1 and len(words[i]) > 4 and len(words[j]) > 4 and i != j:
#                 print(words[i], words[j])


def text_to_wordlist(sentence):
    tokenizer = TweetTokenizer()
    return tokenizer.tokenize(sentence)


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
        use_tfidf=False, max_features=None, bow_ngrams=(1, 1), analyzer='word'):
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


def collect_all_features(filenames):
    df = parse(filenames)
    data = pd.DataFrame()
    data["dialogId"] = df["dialogId"].tolist() + df["dialogId"].tolist()
    data["userMessages"] = df["AliceMessages"].tolist() + df["BobMessages"].tolist()
    data["userOpponentMessages"] = df["BobMessages"].tolist() + df["AliceMessages"].tolist()
    data["userMessageMask"] = df["AliceMessageMask"].tolist() + df["BobMessageMask"].tolist()
    separator = "      "
    data["userConcatenatedMessages"] = data["userMessages"].apply(lambda x: separator.join(x))
    data["userOpponentConcatenatedMessages"] = data["userOpponentMessages"].apply(lambda x: separator.join(x))
    data["userIsBot"] = df["AliceIsBot"].tolist() + df["BobIsBot"].tolist()
    data["userScores"] = df["AliceScore"].tolist() + df["BobScore"].tolist()
    # collect_vocab(data["userConcatenatedMessages"].tolist())

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
    bow_train_data, _ = bow(data["userConcatenatedMessages"].tolist(), [], tokenizer=text_to_wordlist, bow_ngrams=(1, 1))
    data = pd.concat([data, pd.DataFrame(bow_train_data)], axis=1)

    # print("RBoW step...")
    # bow_train_data, _ = bow(data["userOpponentConcatenatedMessages"].tolist(), [], tokenizer=text_to_wordlist)
    # data = pd.concat([data, pd.DataFrame(bow_train_data)], axis=1)

    print("BoC step...")
    bow_train_data, _ = bow(data["userConcatenatedMessages"].tolist(), [], tokenizer=text_to_charlist, bow_ngrams=(1, 3))
    data = pd.concat([data, pd.DataFrame(bow_train_data)], axis=1)

    # print("RBoC step...")
    # bow_train_data, _ = bow(data["userOpponentConcatenatedMessages"].tolist(), [], tokenizer=text_to_charlist)
    # data = pd.concat([data, pd.DataFrame(bow_train_data)], axis=1)

    # print("POS step...")
    # pos_train_data = []
    # for text in data["userConcatenatedMessages"].tolist():
    #     pos_train_data.append(get_sentence_tags(text, "en"))
    # pos_train_data, _ = bow(pos_train_data, [], language='en', stem=False, tokenizer=text_to_wordlist,
    #                         use_tfidf=False, bow_ngrams=(1, 1))
    # data = pd.concat([data, pd.DataFrame(pos_train_data)], axis=1)
    return data


def answer_bot(data, train_indices, test_indices):
    answers = np.array([int(i) if not np.isnan(i) else np.NaN for i in data["userIsBot"].tolist()])
    data = data.drop(["userMessages", "userMessageMask", "userConcatenatedMessages", "userIsBot", "dialogId",
                      "userScores", "userOpponentMessages", "userOpponentConcatenatedMessages"], axis=1)
    clf = LinearSVC(tol=0.1)
    # cv = ShuffleSplit(10, test_size=0.1, random_state=42)
    # cv_scores = cross_val_score(clf, data.loc[train_indices, :], answers[train_indices], cv=cv,
    #                             scoring=make_scorer(roc_auc_score))
    # print("CV is_bot: %0.3f (+/- %0.3f)" % (cv_scores.mean(), cv_scores.std() * 2))

    clf.fit(data.loc[train_indices, :], answers[train_indices])
    return clf.predict(data.loc[test_indices, :])


def predict_regression(data, train_indices, test_indices):
    bot_answers = answer_bot(data, train_indices, test_indices)
    answers = np.array([int(i) if not np.isnan(i) else np.NaN for i in data["userScores"].tolist()])

    l = data.shape[0] // 2
    pairs = {i: l + i if i < l else i - l for i in range(2 * l)}
    bot_scores_indices = [pairs[i] for i in data[data['userIsBot'] == True].index.tolist()]
    train_indices = copy.deepcopy(list(set(train_indices).difference(set(bot_scores_indices))))

    data = data.drop(["userMessages", "userMessageMask", "userConcatenatedMessages", "dialogId",
                      "userIsBot", "userScores", "userOpponentMessages", "userOpponentConcatenatedMessages"], axis=1)
    clf = Lasso()
    # cv = ShuffleSplit(10, test_size=0.1, random_state=42)
    # cv_scores = cross_val_score(clf, data.loc[train_indices, :], answers[train_indices], cv=cv, scoring=make_scorer(spearman))
    # print("CV regr: %0.3f (+/- %0.3f)" % (cv_scores.mean(), cv_scores.std() * 2))

    clf.fit(data.loc[train_indices, :], answers[train_indices])
    preds = clf.predict(data.loc[test_indices])
    for i, is_bot in enumerate(bot_answers):
        if is_bot == 1.0:
            preds[test_indices.index(pairs[test_indices[i]])] = 0
    del data
    return preds


def predict(train_filenames, test_filenames):
    if len(test_filenames) == 0:
        scores = []
        features = collect_all_features(train_filenames)
        for i in range(20):
            train_indices, test_indices, _, __ = train_test_split(range(features.shape[0] // 2),
                                                                  range(features.shape[0] // 2), test_size=0.1,
                                                                  random_state=i)
            l = features.shape[0] // 2
            pairs = {i: l + i if i < l else i - l for i in range(2 * l)}
            train_indices += [pairs[i] for i in train_indices]
            test_indices += [pairs[i] for i in test_indices]
            assert len(set(train_indices).intersection(set(test_indices))) == 0

            answers = np.array([int(i) for i in features["userScores"].tolist()])[test_indices]
            preds = predict_regression(features, train_indices, test_indices)

            score = spearman(preds, answers)
            print(score)
            scores.append(score)
        scores = np.array(scores)
        print("CV: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std()))
    else:
        features = collect_all_features(train_filenames+test_filenames)
        train_indices = features["userScores"].index[features["userScores"].apply(lambda x: not np.isnan(x))].tolist()
        test_indices = features["userScores"].index[features["userScores"].apply(np.isnan)].tolist()
        preds = predict_regression(features, train_indices, test_indices).tolist()
        alice_preds = preds[:len(preds)//2]
        bob_preds = preds[len(preds)//2:]
        submission = pd.DataFrame({'dialogId': features["dialogId"][test_indices[:len(test_indices)//2]],
                                   'Alice': alice_preds,
                                   'Bob': bob_preds})
        submission = submission[["dialogId", "Alice", "Bob"]]
        submission.to_csv(os.path.join(os.getcwd(), 'submitions', 'answerRegr.csv'), index=False)


def local_scorer(train_filename, submition):
    df = parse([train_filename])
    subm = pd.read_csv(submition, index_col="dialogId")
    preds = np.array(subm.Alice.tolist() + subm.Bob.tolist())
    answer = np.array(df["AliceScore"].tolist() + df["BobScore"].tolist())
    print(spearman(answer, preds))

predict(["data/train_20170724.json", "data/train_20170725.json", "data/train_20170726.json"], [])
# local_scorer("data/train_20170726.json", "submitions/answerRegr.csv")