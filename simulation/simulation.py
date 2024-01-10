import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from imblearn.over_sampling import SMOTE
from matplotlib.colors import ListedColormap
import matplotlib.lines as mlines
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from joblib import parallel_backend, Parallel, delayed, load, dump



def generateData (Np, Nn):
    x1 = np.random.uniform(-1 ,1, Nn + Np)
    x2 = np.random.uniform(-1 ,1, Nn + Np)
    y = np.array([0]*Nn + [1]*Np)
    X = np.vstack((x1, x2)).T
    return X, y


def wrongPipeline (X, y):
    sm = SMOTE(random_state=42)
    X_smote_all, y_smote_all = sm.fit_resample(X, y)
    X_smote_all_train, X_smote_all_test, y_smote_all_train, y_smote_all_test = train_test_split(X_smote_all, y_smote_all, test_size=0.2, random_state=42)
    clf = SVC(gamma=50, C=500, random_state=42, probability = True)
    clf.fit(X_smote_all_train, y_smote_all_train)
    y_pred_proba = clf.predict_proba(X_smote_all_test)[:, 1]
    iAUC = roc_auc_score(y_smote_all_test, y_pred_proba)
    return iAUC

def correctPipeline (X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sm = SMOTE(random_state=42)
    X_smote, y_smote = sm.fit_resample(X_train, y_train)
    clf = SVC(gamma=50, C=500, random_state=42, probability = True)
    clf.fit(X_smote, y_smote)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    cAUC = roc_auc_score(y_test, y_pred_proba)
    return cAUC


def getDifference(b, Np):
    X, y  = generateData (Np, int(Np*b/100))
    try:
        cAUC = correctPipeline (X, y)
        iAUC = wrongPipeline (X,y)
    except:
        return None
    return (b, Np, iAUC, cAUC)


if __name__ == '__main__':
    np.random.seed(42)

    ncpus = 30
    reps = 100
    with parallel_backend("loky", inner_max_num_threads=1):
        cres = Parallel (n_jobs = ncpus)(delayed(getDifference)(b, N) for b in range(5,101,5) for r in range(reps) for N in [100,250,500,1000])
    dump (cres, "./results.dump")
    #cres = load("./results.dump")
    cres = [i for i in cres if i is not None]

    res = pd.DataFrame(cres)
    res.columns = ["Balance", "N", "Incorrect", "Correct"]

    if 1 == 1:
        fig = plt.figure(figsize = (9,6.6))
        cpl = ["red", "orange", "green", "blue"]
        sns.set_style("whitegrid", {'patch.facecolor': (0, 0, 0, 1.0)})  # RGBA colors
        sns.lineplot(x='Balance', y='Correct', hue='N', data=res, linestyle = "--", linewidth=1.5, palette=cpl, alpha=1, err_kws={"alpha": .1234})
        sns.lineplot(x='Balance', y='Incorrect', hue='N', data=res, linestyle = "-", linewidth=1.5, palette=cpl, alpha=1, err_kws={"alpha": .1234})
        plt.ylabel("AUC")
        plt.xlabel("Balance [%]")
        handles, labels = plt.gca().get_legend_handles_labels()
        handles = handles[:-4]
        labels = labels[:-4]

        title_line = mlines.Line2D([], [], color='none', label='')
        handles.append(title_line)
        labels.append('')

        title_line = mlines.Line2D([], [], color='none', label='Resampling')
        handles.append(title_line)
        labels.append('Resampling')

        title_line = mlines.Line2D([], [], color='none', label='Size of negative class')
        handles.insert(0,title_line)
        labels.insert(0, 'Size of negative class')

        correct_line = mlines.Line2D([], [], color='black', linestyle='--', label='Correct')
        incorrect_line = mlines.Line2D([], [], color='black', label='Incorrect')
        handles.extend([correct_line, incorrect_line])
        labels.extend(['Correct', 'Incorrect'])
        plt.legend(handles, labels)
        fig.savefig("./simulation.png", facecolor='w', bbox_inches='tight', dpi = 500)



    #

#
