import copy
import re

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk import pos_tag
from nltk.tokenize import TweetTokenizer


def text_to_wordlist(sentence):
    tokenizer = TweetTokenizer()
    return [word.lower() for word in tokenizer.tokenize(sentence)]


def text_to_wordlist_word_only(sentence):
    regexp = "[^а-яА-Яёa-zA-Z.,;!?:()><_]"
    sentence = re.sub(regexp, " ", sentence)
    result = sentence.lower().split()
    return result


def text_to_charlist(sentence):
    return list(sentence)


def bow(train_texts, test_texts, tokenizer=text_to_wordlist, preprocessor=None,
        use_tfidf=False, max_features=None, bow_ngrams=(1, 1), analyzer='word'):
    train = copy.deepcopy(train_texts)
    test = copy.deepcopy(test_texts)

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