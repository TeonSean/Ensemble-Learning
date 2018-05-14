from svm import *
from adaboost import *


def learn(id, iteration):
    sample_weight = None
    clfs = []
    weights = []
    for i in range(iteration):
        print('iteration ' + str(i))
        clf = train(sample_weight)
        matches = validate(clf)
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
    for clf in clfs:
        prediction.append(predict(clf))
    merge(id + classifier_name, prediction, weights)


test_id = 'test'
clfs, weights = learn(test_id, 10)
#clfs, weights = reuse(id)
classify(test_id, clfs, weights)
