import pandas as pd
import pickle
import sklearn
from sklearn import preprocessing
from sklearn.metrics import precision_recall_curve
import re
import json
import os
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from scipy import stats

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LogisticRegression

import statsmodels.api as sm
from scipy.stats import ttest_ind, ttest_ind_from_stats,chi2_contingency, pearsonr

def train(trainx, trainy, testx, save_path, maxiteration):
  statsmodel_log = sm.Logit(trainy, trainx).fit_regularized(method='l1', trim_mode='size', maxiter=maxiteration)
  prediction = statsmodel_log(testx)
  np.save(save_path, prediction)

if __name__ == "__main__":
  trainx, trainy, testx, save_path, maxiteration
  train(trainx, trainy, testx, save_path)

