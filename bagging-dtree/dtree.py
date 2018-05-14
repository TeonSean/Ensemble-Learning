import tfidf
from sklearn.decomposition import TruncatedSVD
from sklearn import tree
import pickle
import math


def iteration():
    return 3 * (len(tfidf.word) / int(math.log(len(tfidf.word))))


def train():
    print('reducing dimension...')
    svd = TruncatedSVD(n_components=int(math.log(len(tfidf.word))))
    svd.fit(tfidf.scores)
    print('done')
    print('generating random samples...')
    s_score, s_label = tfidf.rand_sample()
    print('training dtree classifier...')
    clf = tree.DecisionTreeClassifier()
    clf.fit(svd.transform(s_score), s_label)
    print('done')
    return clf, svd


def predict(clf, svd):
    return clf.predict(svd.transform(tfidf.vscores))


def save(id, clf, svd):
    with open('model/' + str(id) + '.clf', 'wb') as f:
        pickle.dump(clf, f)
        pickle.dump(svd, f)


def load(id):
    with open('model/' + str(id) + '.clf', 'rb') as f:
        clf = pickle.load(f)
        svd = pickle.load(f)
    return clf, svd
