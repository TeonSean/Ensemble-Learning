from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import csv
import random

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
print('generating tf-idf scores...')
scores = transformer.fit_transform(vectorizer.fit_transform(reviews))
vscores = transformer.transform(vectorizer.transform(reviews[nr_train:]))
word = vectorizer.get_feature_names()
print('done')

def rand_sample():
    selected = set()
    for i in range(nr_train):
        selected.add(random.randint(0, nr_train - 1))
    selected_sample = [reviews[i] for i in range(nr_train) if i in selected]
    selected_sample = transformer.transform(vectorizer.transform(selected_sample))
    selected_label = [labels[i] for i in range(nr_train) if i in selected]
    return selected_sample, selected_label