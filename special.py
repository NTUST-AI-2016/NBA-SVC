import numpy as np
import ipdb
import csv

from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn import metrics

# Name,Pos,G,GS,MP,FG,FGA,FG%,3P,3PA,3P%,2P,2PA,2P%,eFG%,FT,FTA,FT%,ORB,DRB,TRB,AST,STL,BLK,TOV,PF,PTS

position_dict = {
    'PG-SG'  : 0,
    'SG-SF' : 1,
    'PG-SF' : 2,
    'SF-PF' : 3,
    'PF-C' : 4
}


rows    = []
classes = []

with open('special_pos.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for idx, row in enumerate(spamreader):
        if idx == 0:
            continue

        rows.append([float(x) for x in row[3:-1]])
        classes.append(position_dict[row[1]])

X = np.array(rows)
y = np.array(classes)

# SVC usage from:
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
clf = SVC(C=0.01, gamma= 1e-08)
clf.fit(X, y)

predicted = cross_validation.cross_val_predict(clf, X, y, cv=10)
scores = cross_validation.cross_val_score(clf, X, y, cv=10)

print(clf.score(X, y)) # => 1.0

# predict_results = clf.predict(rows)
# correct = [idx for idx, result in enumerate(predict_results) if result == classes[idx]]
ipdb.set_trace()
