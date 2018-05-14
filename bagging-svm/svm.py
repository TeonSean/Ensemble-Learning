import tfidf
from sklearn import svm
import pickle


def train():
    clf = svm.NuSVC(verbose=True)
    print('training svm classifier...')
    scores, labels = tfidf.rand_sample()
    clf.fit(scores, labels)
    print('done')
    return clf


def predict(clf):
    return clf.predict(tfidf.vscores)


def save(id, clf):
    with open('model/' + str(id) + '.clf', 'wb') as f:
        pickle.dump(clf, f)


def load(id):
    with open('model/' + str(id) + '.clf', 'rb') as f:
        clf = pickle.load(f)
    return clf
