from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import csv

vectorizer = CountVectorizer()
transformer = TfidfTransformer()
reviews = []
labels = []
mock_labels = []
nr_train = 0
nr_val = 0
with open('train.csv', 'r') as f:
    reader = csv.reader(f)
    lines = []
    for line in reader:
        lines.append(line)
    for line in lines[1:]:
        reviews.append(line[1].strip())
        labels.append(int(line[0]))
        nr_train += 1
with open('validation.csv', 'r') as f:
    reader = csv.reader(f)
    lines = []
    for line in reader:
        lines.append(line)
    for line in lines[1:]:
        reviews.append(line[1].strip())
        mock_labels.append(1)
        nr_val += 1
tfidf = transformer.fit_transform(vectorizer.fit_transform(reviews))
word = vectorizer.get_feature_names()

def train_review_score(id, re):
    assert(re == reviews[id])
    ls = tfidf.getrow(id).toarray()[0]
    re = {}
    for idx, val in enumerate(ls):
        if not val == 0:
            re[idx] = val
    return re

def validation_review_score(id, re):
    assert(re == reviews[id + nr_train])
    ls = tfidf.getrow(id + nr_train).toarray()[0]
    re = {}
    for idx, val in enumerate(ls):
        if not val == 0:
            re[idx] = val
    return re

