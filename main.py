import numpy as np
import ipdb
import csv

from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn import metrics

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

        rows.append([float(x) for x in row[3:-1]])
        classes.append(position_dict[row[1]])

X = np.array(rows)
y = np.array(classes)

# SVC usage from:
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# C=13894954.94373136, gamma= 1.8420699693267165e-07
# C=3.3598182862837742, gamma= 0.026366508987303555
# C=2.6366508987303581, gamma= 0.026366508987303555
# C=10000000, gamma= 1e-08
# C= 1.0, gamma= 0.086596432336006529
# C=1.5998587196060574, gamma= 0.095409547634999634
# clf = SVC(C=10000000, gamma= 1e-08)
# clf = SVC(C= 1.0, gamma= 0.086596432336006529)
# clf = SVC(C=17.575106248547929, gamma= 0.0039069399370546126) # 0.98
# clf = SVC(C=13894954.94373136, gamma= 1.8420699693267165e-07) # => 0.48999999999999994
clf = SVC(C=1778279.4100389229, gamma= 5.6234132519034904e-07)
# clf = SVC(C=1.5998587196060574, gamma= 0.095409547634999634) #
clf.fit(X, y)


predicted = cross_validation.cross_val_predict(clf, X, y, cv=10)
scores = cross_validation.cross_val_score(clf, X, y, cv=5)

print(clf.score(X, y)) # => 1.0

# predict_results = clf.predict(rows)
# correct = [idx for idx, result in enumerate(predict_results) if result == classes[idx]]
ipdb.set_trace()
