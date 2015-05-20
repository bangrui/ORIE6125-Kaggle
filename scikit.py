import numpy as np
from sklearn import svm,tree
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA

# Read in the training data
train = np.genfromtxt('new_train.csv',delimiter = ',')
X = train[:,1:94]
Y = (train[:,94]).ravel()

# Use 1000 sample data to test our code
sample = random.sample(range(len(Y)),1000)
X = X[sample,]
Y = Y[sample]

# Read in the test data
test = np.genfromtxt('new_test.csv',delimiter = ',')
test_X = test[:,1:94]

# use 1000 sample data to test our code
sample = random.sample(range(len(test_X)),1000)
test_X = test_X[sample,]

# Benchmark: Randomforest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X,Y)

BenchMark = rf.predict(test_X)

# SVM training
clf1 = svm.SVC()
clf1.fit(X,Y)

# SVM prediction
SVM_predict = clf1.predict(test_X)
print "Accuracy of SVM is ", 1.0 * np.sum(BenchMark==SVM_predict) / 1000

# Decision Tree training
clf2 = tree.DecisionTreeClassifier()
clf2 = clf2.fit(X,Y)

# Decision Tree prediction
DT_predict = clf2.predict(test_X)
print "Accuracy of DT is ", 1.0 * np.sum(BenchMark==DT_predict) / 1000

# Multinomial Naive Bayes
mnb = MultinomialNB()
NB_predict = mnb.fit(X,Y).predict(test_X)
print "Accuracy of MNB is ", 1.0 * np.sum(BenchMark==NB_predict) / 1000


# PCA
pca = PCA(n_components=50)
X_PCA = pca.fit_transform(X)
print X_PCA
test_X_PCA = pca.fit_transform(test_X)

# SVM on PCA data 
clf1.fit(X_PCA,Y)
SVM_predict = clf1.predict(test_X_PCA)
print "Accuracy of SVM after PCA is ", 1.0 * np.sum(BenchMark==SVM_predict) / 1000
