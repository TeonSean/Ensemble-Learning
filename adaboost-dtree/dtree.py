import tfidf
from sklearn import tree
import pickle


def train(sample_weight):
    print('training dtree classifier...')
    clf = tree.DecisionTreeClassifier()
    clf.fit(tfidf.tscores, tfidf.labels, sample_weight=sample_weight)
    print('done')
    print('validating...')
    labels = clf.predict(tfidf.tscores)
    matches = []
    for idx in range(len(labels)):
        matches.append(labels[idx] == tfidf.labels[idx])
    print('done')
    return clf, matches


def predict(clf):
    return clf.predict(tfidf.vscores)


def save(id, clf, weight):
    with open('model/' + str(id) + '.clf', 'wb') as f:
        pickle.dump(clf, f)
        pickle.dump(weight, f)


def load(id):
    with open('model/' + str(id) + '.clf', 'rb') as f:
        clf = pickle.load(f)
        weight = pickle.load(f)
    return clf, weight
