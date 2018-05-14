from svm import *
from bagging import *


def learn(id, iteration):
    clfs = []
    for i in range(iteration):
        print('iteration ' + str(i))
        clf = train()
        clfs.append(clf)
    save(id, clfs)
    return clfs


def reuse(id):
    return load(id)


def classify(id, clfs):
    prediction = []
    for idx, clf in enumerate(clfs):
        print('predicting with classifier ' + str(idx) + '...')
        prediction.append(predict(clf))
        print('done')
    merge(id, prediction)


test_id = 'test'
clfs = learn(test_id, 10)
#clfs = reuse(id)
classify(test_id, clfs)
