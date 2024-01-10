#
import cv2
from functools import partial
import math
import numpy as np
import pandas as pd

import smote_variants as sv
import logging

from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest, SelectFromModel

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from ITMO_FS.filters.univariate import anova

from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, AllKNN, TomekLinks
from imblearn.over_sampling import RandomOverSampler, SVMSMOTE, ADASYN, SMOTE, KMeansSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

from helpers import *



def mrmre_score (X, y, nFeatures):
    Xp = pd.DataFrame(X, columns = range(X.shape[1]))
    yp = pd.DataFrame(y, columns=['Target'])

    # we need to pre-specify the max solution length...
    solutions = mrmr.mrmr_ensemble(features = Xp, targets = yp, solution_length=nFeatures, solution_count=5)
    scores = [0]*Xp.shape[1]
    for k in solutions.iloc[0]:
        for j, z in enumerate(k):
            scores[z] = scores[z] + Xp.shape[1] - j
    scores = np.asarray(scores, dtype = np.float32)
    scores = scores/np.sum(scores)
    return scores


def bhattacharyya_score_fct (X, y):
    yn = y/np.sum(y)
    yn = np.asarray(yn, dtype = np.float32)
    scores = [0]*X.shape[1]
    for j in range(X.shape[1]):
        xn = (X[:,j] - np.min(X[:,j]))/(np.max(X[:,j] - np.min(X[:,j])))
        xn = xn/np.sum(xn)
        xn = np.asarray(xn, dtype = np.float32)
        scores[j] = cv2.compareHist(xn, yn, cv2.HISTCMP_BHATTACHARYYA)

    scores = np.asarray(scores, dtype = np.float32)
    return -scores


def train_model(paramSet, X_train, y_train, X_train_resampled = None, y_train_resampled = None):
    p_Res, p_FSel, p_Clf = paramSet
    resampler = createResampler (p_Res)
    fselector = createFSel (p_FSel)
    classifier = createClf (p_Clf)

    # apply both
    with np.errstate(divide='ignore',invalid='ignore'):
        # apply at all?
        if X_train_resampled is None or y_train_resampled is None:
            X_train_resampled, y_train_resampled = resampler.fit_resample (X_train, y_train)
        fselector.fit (X_train_resampled, y_train_resampled)
        X_fs_train = fselector.transform (X_train_resampled)
        y_fs_train = y_train_resampled

        classifier.fit (X_fs_train, y_fs_train)

        # extract selected feats as well
        selected_feature_idx = fselector.get_support()

    return [resampler, fselector, classifier, selected_feature_idx]



def test_model(trained_model, X_valid, y_valid):
    resampler, fselector, classifier, _ = trained_model # ignore resampler

    # apply model
    try:
        X_fs_valid = fselector.transform (X_valid)
        y_fs_valid = y_valid

        #print (X_fs_valid)
        y_pred = classifier.predict_proba (X_fs_valid)[:,1]
        t = np.array(y_valid)
        p = np.array(y_pred)
    except Exception as e:
        #print (classifier.predict_proba (X_fs_valid))
        print (classifier.predict_proba (X_fs_valid).shape)
        raise Exception(e)
    return p, t



class MWMOTE:
    def __init__(self, random_state=None, **kwargs):
        self.oversampler = sv.MWMOTE(**kwargs, random_state=random_state)

    def fit_resample(self, X, y):
        X_samp, y_samp = self.oversampler.sample(X.values, y.values)
        X_samp_df = pd.DataFrame(X_samp, columns=X.columns)
        y_samp_df = pd.Series(y_samp, name=y.name)
        return X_samp_df, y_samp_df


class Polynom_fit_SMOTE:
    def __init__(self, random_state=None, topology = None, **kwargs):
        logger = logging.getLogger('smote_variants')
        logger.setLevel(logging.ERROR)
        # thank you, need to revert the latest change
        if topology == "star":
            self.oversampler = sv.polynom_fit_SMOTE_star(**kwargs, random_state=random_state)
        if topology == "bus":
            self.oversampler = sv.polynom_fit_SMOTE_bus(**kwargs, random_state=random_state)
        if topology == "poly":
            self.oversampler = sv.polynom_fit_SMOTE_poly(**kwargs, order = 2, random_state=random_state)

    def fit_resample(self, X, y):
        X_samp, y_samp = self.oversampler.sample(X.values, y.values)
        X_samp_df = pd.DataFrame(X_samp, columns=X.columns)
        y_samp_df = pd.Series(y_samp, name=y.name)
        return X_samp_df, y_samp_df



def createResampler (fExp, ignoreApply = False):
    method = fExp[0][0]

    if method == "RUS":
        resampler = RandomUnderSampler(sampling_strategy = 'majority', random_state = 42)

    if method == "MWMOTE":
        resampler = MWMOTE(random_state=42)

    if method == "PolySMOTE":
        topology = fExp[0][1]["topology"]
        resampler = Polynom_fit_SMOTE(topology = topology, random_state=42)

    if method == "TomekLinks":
        resampler = TomekLinks()

    if method == "ROS":
        resampler = RandomOverSampler(sampling_strategy = 'minority', random_state = 42)

    if method == "SMOTE":
        k = fExp[0][1]["k"]
        resampler = SMOTE(k_neighbors = k, sampling_strategy = 'minority', random_state = 42)

    if method == "SMOTETomek":
        k = fExp[0][1]["k"]
        smote = SMOTE(k_neighbors = k, sampling_strategy = 'minority', random_state = 42)
        resampler = SMOTETomek(smote = smote, random_state = 42)

    if method == "None":
        # since we have a binary problem this will do nothing
        resampler = RandomOverSampler(sampling_strategy = 'not minority', random_state = 42)

    return resampler



def createFSel (fExp):
    method = fExp[0][0]
    nFeatures = fExp[0][1]["nFeatures"]

    if method == "LASSO":
        C = fExp[0][1]["C"]
        clf = LogisticRegression(penalty='l1', max_iter = 100, solver='liblinear', C = C, random_state = 42)
        pipe = SelectFromModel(clf, prefit=False, max_features=nFeatures, threshold=-np.inf)

    if method == "Anova":
        pipe = SelectKBest(anova, k = nFeatures)

    if method == "Bhattacharyya":
        pipe = SelectKBest(bhattacharyya_score_fct, k = nFeatures)

    if method == "ET":
        clf = ExtraTreesClassifier(random_state = 42)
        pipe = SelectFromModel(clf, prefit=False, max_features=nFeatures, threshold=-np.inf)

    return pipe



def createClf (cExp):
    method = cExp[0][0]

    if method == "kNN":
        n = cExp[0][1]["k"]
        model = KNeighborsClassifier(n)

    if method == "LogisticRegression":
        C = cExp[0][1]["C"]
        model = LogisticRegression(max_iter=500, solver='liblinear', C = C, random_state = 42)

    if method == "NaiveBayes":
        model = GaussianNB()

    if method == "RandomForest":
        n_estimators = cExp[0][1]["n_estimators"]
        model = RandomForestClassifier(n_estimators = n_estimators)

    if method == "RBFSVM":
        C = cExp[0][1]["C"]
        g = cExp[0][1]["gamma"]
        model = SVC(kernel = "rbf", C = C, gamma = g, probability = True)

    if method == "NeuralNetwork":
        N1 = cExp[0][1]["layer_1"]
        N2 = cExp[0][1]["layer_2"]
        N3 = cExp[0][1]["layer_3"]
        model = MLPClassifier (hidden_layer_sizes=(N1,N2,N3,), random_state=42, max_iter = 1000)

    return model

#
