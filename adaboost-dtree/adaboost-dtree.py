from dtree import *
from adaboost import *


def learn(id, iteration):
    sample_weight = None
    clfs = []
    weights = []
    for i in range(iteration):
        print('iteration ' + str(i))
        clf, matches = train(sample_weight)
        early_term, clf_weight, sample_weight = iterate(matches, sample_weight)
        if early_term:
            break
        clfs.append(clf)
        weights.append(clf_weight)
    save(id, clfs, weights)
    return clfs, weights


def reuse(id):
    return load(id)


def classify(id, clfs, weights):
    prediction = []
    for idx, clf in enumerate(clfs):
        print('predicting with classifier ' + str(idx) + '...')
        prediction.append(predict(clf))
        print('done')
    merge(id, prediction, weights)


test_id = 'test'
clfs, weights = learn(test_id, 50)
#clfs, svds, weights = reuse(id)
classify(test_id, clfs, weights)
