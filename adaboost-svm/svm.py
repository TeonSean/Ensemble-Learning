import tfidf
from sklearn import svm
import pickle


def train(sample_weight):
    clf = svm.NuSVC(verbose=True)
    print('training svm classifier...')
    clf.fit(tfidf.tscores, tfidf.labels, sample_weight)
    print('done')
    return clf


def validate(clf):
    labels = clf.predict(tfidf.tscores)
    matches = []
    for idx in range(len(labels)):
        matches.append(labels[idx] == tfidf.labels[idx])
    return matches


def predict(clf):
    return clf.predict(tfidf.vscores)


def save(id, clf, weight):
    with open('model/svm-' + str(id) + '.clf', 'wb') as f:
        pickle.dump(clf, f)
        pickle.dump(weight, f)


def load(id):
    with open('model/svm-' + str(id) + '.clf', 'rb') as f:
        clf = pickle.load(f)
        weight = pickle.load(f)
    return clf, weight


classifier_name = 'svm'
