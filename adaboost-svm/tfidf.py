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
print('generating tf-idf scores...')
transformer.fit(vectorizer.fit_transform(reviews))
tscores = transformer.transform(vectorizer.transform(reviews[:nr_train]))
vscores = transformer.transform(vectorizer.transform(reviews[nr_train:]))
word = vectorizer.get_feature_names()
print('done')