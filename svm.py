from sklearn import svm,tree
import random
from numpy import genfromtxt
from sklearn.ensemble import RandomForestClassifier

# Read in the training data
train = genfromtxt('new_train.csv',delimiter = ',')
X = train[:,1:94]
Y = (train[:,94]).ravel()
print Y
print len(Y)

# Use 1000 sample data to test our code
sample = random.sample(range(len(Y)),1000)
X = X[sample,]
Y = Y[sample]


# Read in the test data
test = genfromtxt('new_test.csv',delimiter = ',')
test_X = test[:,1:94]

# use 1000 sample data to test our code
sample = random.sample(range(len(test_X)),1000)
test_X = test_X[sample,]

# SVM training
clf1 = svm.SVC()
clf1.fit(X,Y)

# SVM prediction
predict_Y = clf1.predict(test_X)
print predict_Y

# Decision Tree training
clf2 = tree.DecisionTreeClassifier()
clf2 = clf2.fit(X,Y)


# Decision Tree prediction
predict_Y = clf2.predict(test_X)
print predict_Y

# plot the decision tree


# Benchmark: Randomforest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X,Y)

predict_Y = rf.predict(test_X)
print predict_Y
