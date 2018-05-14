import math
import csv
import random


def merge(id, candid):
    results = []
    print('merging...')
    for i in range(len(candid[0])):
        results.append({-1: 0, 1: 0, 0: 0})
    for i in range(len(candid)):
        for j in range(len(results)):
            results[j][candid[i][j]] += 1
    print('done')
    result = []
    for d in results:
        if d[-1] > d[0]:
            if d[-1] > d[1]:
                result.append(d[-1])
            else:
                result.append(d[1])
        else:
            if d[0] > d[1]:
                result.append(d[0])
            else:
                result.append(d[1])
    with open('result/' + str(id) + '-weak.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        i = random.randint(0, len(candid) - 1)
        for idx, label in enumerate(candid[i]):
            writer.writerow([idx + 1, 1 if label > 0 else -1 if label < 0 else 0])
    with open('result/' + str(id) + '-standard.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        for idx, label in enumerate(result):
            writer.writerow([idx + 1, 1 if label > 0 else -1 if label < 0 else 0])
    with open('result/' + str(id) + '-strict.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        for idx, label in enumerate(result):
            writer.writerow([idx + 1, 1 if label > 0.05 else -1 if label < -0.05 else 0])
    with open('result/' + str(id) + '-loose.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        for idx, label in enumerate(result):
            writer.writerow([idx + 1, 1 if label > 0.25 else -1 if label < -0.25 else 0])