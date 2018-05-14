from dtree import *
from bagging import *


def learn(id):
    clfs = []
    svds = []
    iter = 500
    print(str(iter) + ' iterations in total')
    for i in range(iter):
        print('iteration ' + str(i))
        clf, svd = train()
        clfs.append(clf)
        svds.append(svd)
    save(id, clfs, svds)
    return clfs, svds


def reuse(id):
    return load(id)


def classify(id, clfs, svds):
    prediction = []
    for idx, clf in enumerate(clfs):
        print('predicting with classifier ' + str(idx) + '...')
        prediction.append(predict(clf, svds[idx]))
        print('done')
    merge(id, prediction)


test_id = 'test'
clfs, svds = learn(test_id)
#clfs, svds = reuse(id)
classify(test_id, clfs, svds)
