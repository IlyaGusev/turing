import os
import gc
import itertools
import copy

import pandas as pd
import numpy as np
import xgboost as xgb
from scipy import stats

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_auc_score, make_scorer
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

import stop_words

from util import bow, text_to_wordlist, text_to_charlist
from parse_data import parse


def spearman(a, b):
    return stats.spearmanr(a, b)[0]


def collect_all_features(filenames):
    df = parse(filenames)
    data = pd.DataFrame()
    data["dialogId"] = df["dialogId"].tolist() + df["dialogId"].tolist()
    data["context"] = df["context"].tolist() + df["context"].tolist()
    data["userMessages"] = df["AliceMessages"].tolist() + df["BobMessages"].tolist()
    data["userOpponentMessages"] = df["BobMessages"].tolist() + df["AliceMessages"].tolist()
    data["userMessageMask"] = df["AliceMessageMask"].tolist() + df["BobMessageMask"].tolist()
    separator = "      "
    data["userConcatenatedMessages"] = data["userMessages"].apply(lambda x: separator.join(x))
    data["userOpponentConcatenatedMessages"] = data["userOpponentMessages"].apply(lambda x: separator.join(x))
    data["userIsBot"] = df["AliceIsBot"].tolist() + df["BobIsBot"].tolist()
    data["userScores"] = df["AliceScore"].tolist() + df["BobScore"].tolist()

    hand_crafted_enable = True
    custom_enable = True
    bow_enable = True
    boc_enable = True
    rhand_crafted_enable = False
    rbow_enable = False
    rboc_enable = False
    if hand_crafted_enable:
        data["isEmpty"] = data["userMessages"].apply(lambda x: len(x) == 0)
        data["isEmptyDialog"] = (data["userOpponentMessages"].apply(lambda x: len(x) == 0)) & \
                                (data["userMessages"].apply(lambda x: len(x) == 0))
        data["messageNum"] = data["userMessages"].apply(lambda x: len(x))
        data["numChars"] = data["userMessages"].apply(lambda x: sum([len(msg) for msg in x]))
        data["numWords"] = data["userMessages"].apply(lambda x: sum([len(msg.split()) for msg in x]))
        data["avgChars"] = data["userMessages"].apply(lambda x: np.mean([0] + [len(msg) for msg in x]))
        data["avgWords"] = data["userMessages"].apply(lambda x: np.mean([0] + [len(msg.split()) for msg in x]))
        if custom_enable:
            with open("words.txt") as wordfile:
                system_words = set(x.strip().lower() for x in wordfile.readlines())
            masks = data["userMessageMask"].tolist()
            data["msgInARow"] = [max([0] + [len(list(x)) for x in (g for k, g in itertools.groupby(mask) if k == 1)])
                                 for mask in masks]
            not_dict_word_count = [sum([1 for word in text_to_wordlist(msg) if word not in system_words])
                                   for msg in data["userConcatenatedMessages"].tolist()]
            len_msg = [len(text_to_wordlist(msg, remove_stopwords=False)) for msg in data["userConcatenatedMessages"].tolist()]
            data["typoCount"] = not_dict_word_count
            data["typoCountPart"] = [float(count) / (1 + len_msg[i]) for i, count in enumerate(not_dict_word_count)]
            context_word_count = [sum([1 for word in text_to_wordlist(text) if word in data["context"].tolist()[i]])
                                  for i, text in enumerate(data["userConcatenatedMessages"].tolist())]
            data["relevantWords"] = context_word_count
            data["relevantWordsPart"] = [float(count) / (1 + len_msg[i]) for i, count in enumerate(context_word_count)]
            data["groupOf1"] = [sum([len(list(x)) == 1 for x in (g for k, g in itertools.groupby(mask) if k == 1)])
                                for mask in masks]
            data["groupOfNot1"] = [sum([len(list(x)) != 1 for x in (g for k, g in itertools.groupby(mask) if k == 1)])
                                   for mask in masks]
            stopwords = set(stop_words.get_stop_words("english"))
            data["stopWordsCount"] = [sum([1 for word in text_to_wordlist(msg, remove_stopwords=False)
                                           if word in stopwords]) for msg in data["userConcatenatedMessages"].tolist()]
            data["notStopWordsCount"] = [sum([1 for word in text_to_wordlist(msg, remove_stopwords=False)
                                              if word not in stopwords]) for msg in data["userConcatenatedMessages"].tolist()]

    if rhand_crafted_enable:
        data["RmessageNum"] = data["userOpponentMessages"].apply(lambda x: len(x))
        data["RnumChars"] = data["userOpponentMessages"].apply(lambda x: sum([len(msg) for msg in x]))
        data["RnumWords"] = data["userOpponentMessages"].apply(lambda x: sum([len(msg.split()) for msg in x]))
        data["RavgChars"] = data["userOpponentMessages"].apply(lambda x: np.mean([0] + [len(msg) for msg in x]))
        data["RavgWords"] = data["userOpponentMessages"].apply(lambda x: np.mean([0] + [len(msg.split()) for msg in x]))

    if bow_enable:
        print("BoW step...")
        bow_train_data, _ = bow(data["userConcatenatedMessages"].tolist(), [], tokenizer=text_to_wordlist, bow_ngrams=(1, 2))
        data = pd.concat([data, pd.DataFrame(bow_train_data)], axis=1)

    if rbow_enable:
        print("RBoW step...")
        bow_train_data, _ = bow(data["userOpponentConcatenatedMessages"].tolist(), [], tokenizer=text_to_wordlist)
        data = pd.concat([data, pd.DataFrame(bow_train_data)], axis=1)

    if boc_enable:
        print("BoC step...")
        bow_train_data, _ = bow(data["userConcatenatedMessages"].tolist(), [], tokenizer=text_to_charlist, bow_ngrams=(1, 3))
        data = pd.concat([data, pd.DataFrame(bow_train_data)], axis=1)

    if rboc_enable:
        print("RBoC step...")
        bow_train_data, _ = bow(data["userOpponentConcatenatedMessages"].tolist(), [], tokenizer=text_to_charlist)
        data = pd.concat([data, pd.DataFrame(bow_train_data)], axis=1)
    return data


def answer_bot(data, train_indices, test_indices, clf_name="lgbm"):
    answers = np.array([int(i) if not np.isnan(i) else np.NaN for i in data["userIsBot"].tolist()])
    data = data.drop(["userMessages", "userMessageMask", "userConcatenatedMessages", "userIsBot", "dialogId", "context",
                      "userScores", "userOpponentMessages", "userOpponentConcatenatedMessages"], axis=1)
    if clf_name == "lgbm":
        clf = LGBMClassifier(n_estimators=500, num_leaves=1000, learning_rate=0.01, subsample=0.9, seed=42)
        clf.fit(data.loc[train_indices, :], answers[train_indices], eval_metric="accuracy")
    if clf_name == "svm":
        clf = LinearSVC(tol=0.1)
        clf.fit(data.loc[train_indices, :], answers[train_indices])
    if clf_name == "xgb":
        data.columns = [i for i in range(len(data.columns))]
        clf = XGBClassifier(n_estimators=500, max_depth=7, learning_rate=0.02, subsample=0.6, base_score=0.35, seed=42)
        X, y = data.loc[train_indices, :], answers[train_indices]
        clf.fit(X, y, eval_metric="error", verbose=True)

    preds = clf.predict(data.loc[test_indices, :])

    run_cv = False
    if run_cv:
        cv = ShuffleSplit(10, test_size=0.1, random_state=42)
        cv_scores = cross_val_score(clf, data.loc[train_indices, :], answers[train_indices], cv=cv,
                                    scoring=make_scorer(roc_auc_score))
        print("CV is_bot: %0.3f (+/- %0.3f)" % (cv_scores.mean(), cv_scores.std() * 2))

    del data
    gc.collect()
    return preds


def predict_regression(data, train_indices, test_indices, clf_name="xgb", reg_name="lgbm"):
    bot_answers = answer_bot(data, train_indices, test_indices, clf_name)
    answers = np.array([int(i) if not np.isnan(i) else np.NaN for i in data["userScores"].tolist()])

    l = data.shape[0] // 2
    pairs = {i: l + i if i < l else i - l for i in range(2 * l)}
    bot_scores_indices = [pairs[i] for i in data[data['userIsBot'] == True].index.tolist()]
    train_indices = copy.deepcopy(list(set(train_indices).difference(set(bot_scores_indices))))

    data = data.drop(["userMessages", "userMessageMask", "userConcatenatedMessages", "dialogId", "context",
                      "userIsBot", "userScores", "userOpponentMessages", "userOpponentConcatenatedMessages"], axis=1)

    if reg_name == "lasso":
        clf = Lasso()
        clf.fit(data.loc[train_indices, :], answers[train_indices])
    if reg_name == "lgbm":
        clf = LGBMRegressor(n_estimators=100)
        clf.fit(data.loc[train_indices, :], answers[train_indices], eval_metric="accuracy")
    if reg_name == "xgb":
        data.columns = [i for i in range(len(data.columns))]
        clf = XGBRegressor(n_estimators=500, max_depth=7, learning_rate=0.02, subsample=0.6, seed=42)
        X, y = data.loc[train_indices, :], answers[train_indices]
        clf.fit(X, y)
    run_cv = False
    if run_cv:
        cv = ShuffleSplit(10, test_size=0.1, random_state=42)
        cv_scores = cross_val_score(clf, data.loc[train_indices, :], answers[train_indices], cv=cv,
                                    scoring=make_scorer(spearman))
        print("CV regr: %0.3f (+/- %0.3f)" % (cv_scores.mean(), cv_scores.std() * 2))

    preds = clf.predict(data.loc[test_indices])
    for i, is_bot in enumerate(bot_answers):
        if is_bot == 1.0:
            preds[test_indices.index(pairs[test_indices[i]])] = 0
    del data
    gc.collect()
    return preds


def predict(train_filenames, test_filenames, clf_name="xgb", reg_name="lgbm"):
    if len(test_filenames) == 0:
        print("Validation")
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
            preds = predict_regression(features, train_indices, test_indices, clf_name, reg_name)

            # suspicios_indices = [pairs[test_indices[i]] for i, answer in enumerate(answers) if answer != 0.0 and preds[i] == 0.0]
            # print(len(suspicios_indices))
            # print(features.loc[suspicios_indices, :]["userConcatenatedMessages"].tolist())
            score = spearman(preds, answers)
            scores.append(score)
            print("Split: ", i, "Score: ", score, "Mean: ", np.mean(scores), "Median: ",
                  np.median(scores), "Std: ", np.std(scores))
        scores = np.array(scores)
        print("CV: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std()))
    else:
        print("Prediction")
        features = collect_all_features(train_filenames+test_filenames)
        train_indices = features["userScores"].index[features["userScores"].apply(lambda x: not np.isnan(x))].tolist()
        test_indices = features["userScores"].index[features["userScores"].apply(np.isnan)].tolist()
        preds = predict_regression(features, train_indices, test_indices, clf_name, reg_name).tolist()
        alice_preds = preds[:len(preds)//2]
        bob_preds = preds[len(preds)//2:]
        submission = pd.DataFrame({'dialogId': features["dialogId"][test_indices[:len(test_indices)//2]],
                                   'Alice': alice_preds,
                                   'Bob': bob_preds})
        submission = submission[["dialogId", "Alice", "Bob"]]
        submission.to_csv(os.path.join(os.getcwd(), 'data', 'answer.csv'), index=False)


def local_scorer(train_filename, submition):
    df = parse([train_filename])
    subm = pd.read_csv(submition, index_col="dialogId")
    preds = np.array(subm.Alice.tolist() + subm.Bob.tolist())
    answer = np.array(df["AliceScore"].tolist() + df["BobScore"].tolist())
    print(spearman(answer, preds))


def avg_blending(filenames):
    dfs = []
    for filename in filenames:
        dfs.append(pd.read_csv(filename, header=0))
    df = pd.DataFrame()
    df["dialogId"] = dfs[0]["dialogId"]
    users = ["Alice", "Bob"]
    for user in users:
        df[user] = np.zeros((dfs[0].shape[0]))
        for data in dfs:
            df[user] = df[user] + data[user]
        df[user] = df[user].apply(lambda x: x/len(dfs))
    df = df[["dialogId", "Alice", "Bob"]]
    df.to_csv(os.path.join('data', 'finalAnswer.csv'), index=False)

# if __name__ == "__main__":
    # train_dir = "data/train"
    # test_dir = "data/test"
    #
    # def get_all_files_in_dir(dir_name):
    #     return [os.path.join(dir_name, filename) for filename in os.listdir(dir_name)]
    # predict(get_all_files_in_dir(train_dir), get_all_files_in_dir(test_dir), clf_name="svm", reg_name="lasso")
    # local_scorer("data/train/train_20170727.json", "data/avg.csv")
    # avg_blending(["data/answer-0.62-xgb-lgbm.csv", "data/answer-0.66-svm-lasso.csv"])