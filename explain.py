import numpy as np
import pylab as pl
import pandas as pd
from nltk.tokenize import TweetTokenizer
from matplotlib.pyplot import cm
from collections import Counter
from nltk.collocations import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import TruncatedSVD

from parse_data import parse
from util import bow, text_to_wordlist_word_only
from simple_predict import collect_all_features


filenames = ["data/train_20170724.json", "data/train_20170725.json", "data/train_20170726.json"]
df = parse(filenames)
print(df.describe())


def count_ngrams_freq(tokenized_texts, n):
    ngrams = Counter()
    for text in tokenized_texts:
        for i in range(len(text)-n+1):
            ngram = " ".join(text[i:i+n])
            ngrams[ngram] += 1
    return ngrams


def count_bigrams_mi(tokenized_texts):
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words([token for text in tokenized_texts for token in text+['EOF']])
    finder.apply_freq_filter(3)
    return finder.nbest(bigram_measures.pmi, 10)

tokenizer = TweetTokenizer()
data = pd.DataFrame()
data["dialogId"] = df["dialogId"].tolist() + df["dialogId"].tolist()
data["userMessages"] = df["AliceMessages"].tolist() + df["BobMessages"].tolist()
separator = "      "
data["userConcatenatedMessages"] = data["userMessages"].apply(lambda x: text_to_wordlist_word_only(separator.join(x)))
data["userIsBot"] = df["AliceIsBot"].tolist() + df["BobIsBot"].tolist()
data["userScores"] = df["AliceScore"].tolist() + df["BobScore"].tolist()
tokenized_texts = data["userConcatenatedMessages"].tolist()
X = collect_all_features(filenames)
X = X.drop(["userMessages", "userMessageMask", "userConcatenatedMessages", "dialogId",
            "userIsBot", "userScores", "userOpponentMessages", "userOpponentConcatenatedMessages"], axis=1)
X = X.drop(["messageNum", "numChars", "numWords", "avgChars",  "avgWords",  "msgInARow"], axis=1)

print(X.head(5))

print(count_ngrams_freq(tokenized_texts, 2).most_common(10))
print(count_bigrams_mi(tokenized_texts))


def clustering(X, n_clusters):
    km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, verbose=False, random_state=42)
    km.fit(X)
    y = km.labels_.tolist()
    return y

n_clusters = 8
y = clustering(X, n_clusters)
print(y[:100])


def reduce_dimensionality(X):
    svd = TruncatedSVD(3)
    return svd.fit_transform(X)


def plot2D(X, y, i, j, n_clusters):
    x_min, x_max = X[:, i].min() + 0.8, X[:, i].max() - 1.1
    y_min, y_max = X[:, j].min() + 0.8, X[:, j].max() - 1.1
    pl.figure(1)
    pl.clf()

    color = iter(cm.rainbow(np.linspace(0, 1, n_clusters)))
    df = pd.DataFrame(dict(x=X[:, i], y=X[:, j], label=y))
    groups = df.groupby('label')
    for label, group in groups:
        pl.plot(group.x, group.y, 'k.', markersize=2, color=next(color))

    pl.title('K-means clustering')
    pl.xlim(x_min, x_max)
    pl.ylim(y_min, y_max)
    pl.xticks(())
    pl.yticks(())
    pl.show()

reduced = reduce_dimensionality(X)
plot2D(reduced, y, 0, 1, n_clusters)
plot2D(reduced, y, 1, 2, n_clusters)
plot2D(reduced, y, 0, 2, n_clusters)

clusters = [[] for i in range(n_clusters)]
for i in range(len(y)):
    clusters[y[i]].append(tokenized_texts[i])


def describe_cluster(cluster):
    size = len(cluster)
    common_words = count_ngrams_freq(cluster, 1).most_common(10)
    common_bigrams = count_ngrams_freq(cluster, 2).most_common(10)
    mi = count_bigrams_mi(cluster)
    mean_len = sum([len(text) for text in cluster]) / len(cluster)
    description = "Cluster of " + str(size) + " texts, with " + str(mean_len) + " average word length\n"
    common_words = ", ".join([pair[0] for pair in common_words])
    description += "Common words: " + str(common_words) + "\n"
    common_bigrams = ", ".join([pair[0] for pair in common_bigrams])
    description += "Common bigrams: " + str(common_bigrams) + "\n"
    mi = ", ".join([" ".join(pair) for pair in mi])
    description += "MI: " + str(mi)
    print(description)


for i in range(n_clusters):
    print("-----------------------------" + str(i) + " CLUSTER" + "-----------------------------")
    describe_cluster(clusters[i])