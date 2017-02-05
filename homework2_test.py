
import numpy as np
import matplotlib.pyplot as plt

import urllib
import scipy.optimize
import random
from math import exp
from math import log
from operator import itemgetter, attrgetter
from sklearn.decomposition import PCA
from collections import defaultdict
random.seed(0)


np.random.seed(0)
def parseData(fname):
    for l in urllib.urlopen(fname):
        yield eval(l)


print "Reading data..."
dataFile = open("winequality-white.csv")
header = dataFile.readline()
fields = ["constant"] + header.strip().replace('"', '').split(';')
featureNames = fields[:-1]
labelName = fields[-1]
lines = [[1.0] + [float(x) for x in l.split(';')] for l in dataFile]
random.shuffle(lines)
X = [l[:-1] for l in lines]
y = [l[-1] > 5 for l in lines]
sy = [l[-1] for l in lines]
print "done"


def inner(x, y):
    return sum([x[i] * y[i] for i in range(len(x))])


def sigmoid(x):
    return 1.0 / (1 + exp(-x))


##################################################
# Logistic regression by gradient ascent         #
##################################################

# NEGATIVE Log-likelihood
def f(theta, X, y, lam):
    loglikelihood = 0
    for i in range(len(X)):
        logit = inner(X[i], theta)
        loglikelihood -= log(1 + exp(-logit))
        if not y[i]:
            loglikelihood -= logit
    for k in range(len(theta)):
        loglikelihood -= lam * theta[k] * theta[k]
    # for debugging
    # print "ll =", loglikelihood
    return -loglikelihood


# NEGATIVE Derivative of log-likelihood
def fprime(theta, X, y, lam):
    dl = [0] * len(theta)
    for i in range(len(X)):
        logit = inner(X[i], theta)
        for k in range(len(theta)):
            dl[k] += X[i][k] * (1 - sigmoid(logit))
            if not y[i]:
                dl[k] -= X[i][k]
    for k in range(len(theta)):
        dl[k] -= lam * 2 * theta[k]
    return np.array([-x for x in dl])


X_train = X[:int(len(X) / 3)]
y_train = y[:int(len(y) / 3)]
X_validate = X[int(len(X) / 3):int(2 * len(X) / 3)]
y_validate = y[int(len(y) / 3):int(2 * len(y) / 3)]
X_test = X[int(2 * len(X) / 3):]
y_test = y[int(2 * len(X) / 3):]


##################################################
# Train                                          #
##################################################

def train(lam):
    theta, _, _ = scipy.optimize.fmin_l_bfgs_b(f, [0] * len(X[0]), fprime, pgtol=10, args=(X_train, y_train, lam))
    return theta


##################################################
# Predict                                        #
##################################################

def performance(theta):
    scores_train = [inner(theta, x) for x in X_train]
    scores_validate = [inner(theta, x) for x in X_validate]
    scores_test = [inner(theta, x) for x in X_test]

    predictions_train = [s > 0 for s in scores_train]
    predictions_validate = [s > 0 for s in scores_validate]
    predictions_test = [s > 0 for s in scores_test]

    correct_train = [(a == b) for (a, b) in zip(predictions_train, y_train)]
    correct_validate = [(a == b) for (a, b) in zip(predictions_validate, y_validate)]
    correct_test = [(a == b) for (a, b) in zip(predictions_test, y_test)]

    acc_train = sum(correct_train) * 1.0 / len(correct_train)
    acc_validate = sum(correct_validate) * 1.0 / len(correct_validate)
    acc_test = sum(correct_test) * 1.0 / len(correct_test)
    return acc_train, acc_validate, acc_test
lam = 0.01
theta = train(lam)
acc_train, acc_validate, acc_test = performance(theta)
print("lambda = " + str(lam) + ";\ttrain=" + str(acc_train) + "; validate=" + str(acc_validate) + "; test=" + str(acc_test))

##################################################
# true positives, true negatives, false positives#
# , false negatives, and the Balanced Error Rate.#
##################################################
def four_type(theta, _X, _Y):
  scores_test = [inner(theta, x) for x in _X]
  t_n = 0
  t_p = 0
  f_p = 0
  f_n = 0
  p = 0
  n = 0
  for y in _Y:
    if y:
      p = p + 1
    else:
      n = n + 1
  ll = len(_Y)
  for i in range(0, ll, 1):
    if _Y[i]:
      if scores_test[i] > 0:
        t_p = t_p + 1
      else:
        f_p = f_p + 1
    else:
      if scores_test[i] > 0:
        f_n = f_n + 1
      else :
        t_n = t_n + 1
  print "true postive: " + str(t_p)
  print "false postive: " + str(f_p)
  print "true negative: " + str(t_n)
  print "false negative: " + str(f_n)
  print "Balanced Error Rate: " + str(1 - 0.5 * (t_p * 1.0 / p + t_n * 1.0 / n))

four_type(theta, X_test, y_test)
##################################################
# precision and  recall                          #
##################################################
class rankdata:
  def __init__(self, conf, label):
    self.conf = conf
    self.label = label
  def __repr__(self):
    return ((self.conf, self.label))
  def get(self):
    return self.conf, self.label
  def __getitem__(self, item):
    if item == 0:
      return self.conf
    else:
      return self.label
def construct(_X, _Y):
  T = []
  for k in range(len(_X)):
    T.append(rankdata(_X[k], _Y[k]))

  T = sorted(T, key=itemgetter(0), reverse=True)

  allrelevant = 0
  for yy in _Y:
    if yy:
      allrelevant = allrelevant + 1

  retrive = 10;
  relevant = 0
  for i in range(0,retrive,1):
    a, b = T[i].get()
    if a > 0 and b == True:
      relevant = relevant + 1
  print "Top 10: precision: " + str(relevant * 1.0 / retrive) + " recall: " + str(relevant * 1.0 / allrelevant)

  retrive = 500;
  relevant = 0
  for i in range(0,retrive,1):
    a, b = T[i].get()
    if a > 0 and b == True:
      relevant = relevant + 1
  print "Top 500: " + "precision: " + str(relevant * 1.0 / retrive) + " recall: " + str(relevant * 1.0 / allrelevant)

  retrive = 1000;
  relevant = 0
  for i in range(0, retrive, 1):
    a, b = T[i].get()
    if a > 0 and b == True:
      relevant = relevant + 1
  print "Top 1000: " + "precision: " + str(relevant * 1.0 / retrive) + " recall: " + str(relevant * 1.0 / allrelevant)

def p_and_r(theta):
  scores_test = [inner(theta, x) for x in X_test]

  print "Test data:"
  construct(scores_test, y_test)
p_and_r(theta)
##################################################
# Draw plot                                      #
##################################################
def drawpic(theta,_X,_Y):
  T = []
  for k in range(len(_X)):
    T.append(rankdata(_X[k], _Y[k]))
  T = sorted(T, key=itemgetter(0), reverse=True)


  allrelevant = 0
  for yy in _Y:
    if yy:
      allrelevant = allrelevant + 1

  ll = len(_Y)
  l_x = []
  l_y = []
  relevant = 0
  print ll
  for i in range(0, ll, 1):
    a, b = T[i].get()
    if a > 0 and b == True:
      relevant = relevant + 1
    xx = relevant * 1.0 / (i + 1)
    l_x.append(xx)
    yy = relevant * 1.0 / allrelevant
    l_y.append(yy)
  plt.plot(l_y,l_x)
  plt.xlabel("recall")
  plt.ylabel("precision")
  plt.show()

sc = [inner(theta, x) for x in X_test]
drawpic(theta, sc, y_test)

##################################################
# PCA_1                                          #
##################################################

def PCA_Mean(_X):

  s = []
  for i in range(0, 11, 1):
    s.append(0)
  for t in _X:
    for i in range(0, 11, 1):
      s[i] = s[i] + t[i + 1]
  for i in range(0, 11, 1):
    s[i] = s[i] * 1.0 / len(_X)
  su = 0
  for t in _X:
    for i in range(0, 11, 1):
      su = su + (t[i + 1] - s[i]) * (t[i + 1] - s[i])
  print "reconstruction error: ", su

PCA_Mean(X_train)

##################################################
# PCA_2                                          #
##################################################
_X = []
for t in X_train:
  _X.append(t[1:])
pca = PCA(n_components=5)
pca.fit(_X)
print "5 parameters: "
print pca.components_

##################################################
# PCA_3                                          #
##################################################
_X = []
for t in X_train:
  _X.append(t[1:])

n = 4
pca = PCA(n_components=4)
pca.fit(_X)
trans0 = pca.components_
trans0 = np.matrix(trans0)

_x = np.matrix(_X).T
_y = trans0 * _x
print _y.size
_yy = trans0.T * _y
print _yy.size
tmp = np.subtract(_yy.A1, _x.A1)
tmp = tmp ** 2



print "reconstruction error for 4 dimensions: " + str(np.sum(tmp))

##################################################
# PCA_4                                          #
##################################################
y = sy[:int(len(sy) / 3)]
X=[]
_y = np.matrix(y)
for i in range(0,len(X_train), 1):
  X.append([1])
for i in range(1, 12, 1):
  pca = PCA(n_components=i)
  pca.fit(X_train)
  trans0 = pca.components_
  trans0 = np.matrix(trans0)
  XX = trans0 * (np.matrix(X_train).T)
  XX = np.column_stack((X, XX.T))

  thetas = np.linalg.inv(XX.T * XX) * XX.T * _y.T
  predict = (XX * thetas).T
#0.492429542035
  sub = np.subtract(predict.A1, _y.A1)
  sub = sub ** 2
  print "MSE with " + str(i) +" dimensions: "+ str( sum(sub)/len(sub))


