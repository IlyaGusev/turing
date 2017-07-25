import copy
import re
import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from pymorphy2 import MorphAnalyzer
from nltk.stem.snowball import SnowballStemmer
from nltk import pos_tag
from sklearn.svm import LinearSVC
from scipy.sparse import hstack
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import ShuffleSplit, cross_val_score

from parse_data import parse
morph_ru = MorphAnalyzer()
morph_en = SnowballStemmer("english")


def text_to_wordlist(sentence, cyrillic=False):
    regexp = "[^а-яА-Яёa-zA-Z.,;!?:()><_]"
    if cyrillic:
        regexp = "[^а-яА-Яё]"
    sentence = re.sub(regexp, " ", sentence)
    result = sentence.lower().split()
    return result


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
    return train_data, test_data


def predict(filename):
    dialogs = parse(filename, get_df=False)
    l = len(dialogs)
    answers = []
    messages = []
    additional_features = [[], [], [], []]
    for user_id in ["Bob", "Alice"]:
        answer = [int(bool(dialog.first_user_is_bot) and dialog.first_user_id == user_id or
                  bool(dialog.second_user_is_bot) and dialog.second_user_id == user_id) for dialog in dialogs]
        answers += answer
        texts = [[message.text for message in dialog.messages if message.user_id == user_id] for dialog in dialogs]
        concatenated_messages = []
        for text in texts:
            concatenated_messages.append("")
            for message in text:
                concatenated_messages[-1] += message + ". "
        messages += concatenated_messages
        additional_features[0] += [len(text) for text in texts]
        additional_features[1] += [np.mean([0] + [len(message.split()) for message in text]) for text in texts]
        additional_features[2] += [np.mean([0] + [len(message) for message in text]) for text in texts]
        # Количество подряд идущих сообщений.
        for dialog in dialogs:
            i = 0
            max_i = 0
            for message in dialog.messages:
                if message.user_id == user_id:
                    i += 1
                    if max_i < i:
                        max_i = i
                else:
                    i = 0
            additional_features[3].append(max_i)

    print("Bow step...")
    bow_train_data, _ = bow(messages, [], language='en', stem=True,
                            tokenizer=text_to_wordlist, use_tfidf=False, max_features=None,
                            bow_ngrams=(1, 1))

    # print("POS step...")
    # pos_train_data = []
    # for text in messages:
    #     pos_train_data.append(get_sentence_tags(text, "en"))
    # pos_train_data, _ = bow(pos_train_data, [], language='en', stem=False, tokenizer=text_to_wordlist,
    #                         use_tfidf=False, max_features=None, bow_ngrams=(1, 1))
    data = hstack([bow_train_data, np.transpose(additional_features)])
    clf = LinearSVC(tol=0.1)
    cv = ShuffleSplit(10, test_size=0.1, random_state=42)
    cv_scores = cross_val_score(clf, data, answers, cv=cv, scoring=make_scorer(roc_auc_score))
    print("CV: %0.3f (+/- %0.3f)" % (cv_scores.mean(), cv_scores.std() * 2))
    clf.fit(data, answers)
    pred = clf.predict(data)
    ids = [dialog.dialog_id for dialog in dialogs]
    submission = pd.DataFrame({'dialogId': ids,
                               'Alice': pred[l:],
                               'Bob': pred[:l]})
    submission.to_csv(os.path.join(os.getcwd(), 'answerSVM.csv'), index=False)