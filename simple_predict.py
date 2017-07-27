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

from lightgbm import LGBMClassifier, LGBMRegressor

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
    bow_enable = True
    boc_enable = True
    rhand_crafted_enable = False
    rbow_enable = False
    rboc_enable = False
    custom_enable = True
    if hand_crafted_enable:
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
            len_msg = [len(text_to_wordlist(msg)) for msg in data["userConcatenatedMessages"].tolist()]
            data["notDictWordCount"] = not_dict_word_count
            data["notDictWordCountPart"] = [float(count) / (1 + len_msg[i]) for i, count in enumerate(not_dict_word_count)]
            context_word_count = [sum([1 for word in text_to_wordlist(text) if word in data["context"].tolist()[i]])
                                  for i, text in enumerate(data["userConcatenatedMessages"].tolist())]
            data["wordInContext"] = context_word_count
            data["wordInContextPart"] = [float(count) / (1 + len_msg[i]) for i, count in enumerate(context_word_count)]

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


def answer_bot(data, train_indices, test_indices):
    answers = np.array([int(i) if not np.isnan(i) else np.NaN for i in data["userIsBot"].tolist()])
    data = data.drop(["userMessages", "userMessageMask", "userConcatenatedMessages", "userIsBot", "dialogId", "context",
                      "userScores", "userOpponentMessages", "userOpponentConcatenatedMessages"], axis=1)
    data.columns = [i for i in range(len(data.columns))]

    xgb_enable = True
    svm_enable = False
    lgbm_enable = False
    if xgb_enable:
        params = {}
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'auc'
        params['eta'] = 0.02
        params['max_depth'] = 7
        params['subsample'] = 0.6
        params['base_score'] = 0.2
        params['silent'] = 1
        rounds = 500
        d_train = xgb.DMatrix(data.loc[train_indices, :], label=answers[train_indices])
        d_test = xgb.DMatrix(data.loc[test_indices, :])
        watchlist = [(d_train, 'train'),]
        clf = xgb.train(params, d_train, rounds, watchlist, early_stopping_rounds=50, verbose_eval=50)
        preds = [0 if i < 0.5 else 1 for i in clf.predict(d_test)]

    if lgbm_enable or svm_enable:
        if lgbm_enable:
            clf = LGBMClassifier(n_estimators=100, num_leaves=1000)
        if svm_enable:
            clf = LinearSVC(tol=0.1)
        clf.fit(data.loc[train_indices, :], answers[train_indices])
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


def predict_regression(data, train_indices, test_indices):
    bot_answers = answer_bot(data, train_indices, test_indices)
    answers = np.array([int(i) if not np.isnan(i) else np.NaN for i in data["userScores"].tolist()])

    l = data.shape[0] // 2
    pairs = {i: l + i if i < l else i - l for i in range(2 * l)}
    bot_scores_indices = [pairs[i] for i in data[data['userIsBot'] == True].index.tolist()]
    train_indices = copy.deepcopy(list(set(train_indices).difference(set(bot_scores_indices))))

    data = data.drop(["userMessages", "userMessageMask", "userConcatenatedMessages", "dialogId", "context",
                      "userIsBot", "userScores", "userOpponentMessages", "userOpponentConcatenatedMessages"], axis=1)
    lasso_enable = False
    lgbm_enable = True
    if lasso_enable:
        clf = Lasso()
    if lgbm_enable:
        clf = LGBMRegressor()
    run_cv = False
    if run_cv:
        cv = ShuffleSplit(10, test_size=0.1, random_state=42)
        cv_scores = cross_val_score(clf, data.loc[train_indices, :], answers[train_indices], cv=cv,
                                    scoring=make_scorer(spearman))
        print("CV regr: %0.3f (+/- %0.3f)" % (cv_scores.mean(), cv_scores.std() * 2))

    clf.fit(data.loc[train_indices, :], answers[train_indices])
    preds = clf.predict(data.loc[test_indices])
    for i, is_bot in enumerate(bot_answers):
        if is_bot == 1.0:
            preds[test_indices.index(pairs[test_indices[i]])] = 0
    del data
    gc.collect()
    return preds


def predict(train_filenames, test_filenames):
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
            preds = predict_regression(features, train_indices, test_indices)

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
        preds = predict_regression(features, train_indices, test_indices).tolist()
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
    df.to_csv(os.path.join('submitions', 'avg.csv'), index=False)

if __name__ == "__main__":
    train_dir = "data/train"
    test_dir = "data/test"

    def get_all_files_in_dir(dir_name):
        return [os.path.join(dir_name, filename) for filename in os.listdir(dir_name)]
    predict(get_all_files_in_dir(train_dir), get_all_files_in_dir(test_dir))
    # local_scorer("data/train_20170726.json", "submitions/answerRegr.csv")
    # avg_blending(["submitions/answer27-0.6.csv", "submitions/Max-answer27-0.5.csv", "submitions/Nik-answer27-0.33.csv"])