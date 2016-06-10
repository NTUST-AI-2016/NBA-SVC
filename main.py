import numpy as np
import ipdb
import csv

from sklearn.svm import SVC

# Name,Pos,G,GS,MP,FG,FGA,FG%,3P,3PA,3P%,2P,2PA,2P%,eFG%,FT,FTA,FT%,ORB,DRB,TRB,AST,STL,BLK,TOV,PF,PTS

position_dict = {
    'C'  : 0,
    'PF' : 1,
    'SF' : 2,
    'PG' : 3,
    'SG' : 4
}

rows    = []
classes = []

with open('data.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for idx, row in enumerate(spamreader):
        if idx == 0:
            continue

        rows.append([float(x) for x in row[2:-1]])
        classes.append(position_dict[row[1]])

X = np.array(rows)
y = np.array(classes)

# SVC usage from:
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

clf = SVC()
clf.fit(X, y)

print(clf.score(X, y)) # => 1.0

# predict_results = clf.predict(rows)
# correct = [idx for idx, result in enumerate(predict_results) if result == classes[idx]]
# ipdb.set_trace()
