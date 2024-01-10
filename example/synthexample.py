import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

np.random.seed(42)

sf = 2.0
fsize = (11,11)
psize = 300

def generateData ():
    theta = np.random.uniform(0, 2*np.pi, 100) # Random angles
    r = np.random.normal(0.4, 0.2, 100) # Random radii
    r = np.clip(r, 0., 1) # Clip the radii to the ring range
    x1 = r * np.cos(theta) # x-coordinates
    y1 = r * np.sin(theta) # y-coordinates
    label1 = np.hstack((-np.ones(10), np.ones(90)))


    theta = np.random.uniform(0, 2*np.pi, 10) # Random angles
    r = np.random.normal(0, 0.2, 10) # Random radii
    x2 = r * np.cos(theta) # x-coordinates
    y2 = r * np.sin(theta) # y-coordinates
    label2 = -np.ones(10) # Labels

    # Combine the points and labels
    P = np.vstack ((x1, y1)).T
    N = np.vstack ((x2, y2)).T
    X = np.vstack((P,N)) # Features
    y = np.hstack((label1, label2)) # Labels

    X_test = X[:20]
    y_test = y[:20]
    return X, y, X_test, y_test


def doPlot(X, y, fname):
    fig, axs = plt.subplots(figsize=fsize)
    colormap = ListedColormap(['orange', 'darkcyan'])
    scatter = axs.scatter(X[:, 0], X[:, 1], c=y, cmap=colormap, s=psize)
    legend = axs.legend(handles=scatter.legend_elements()[0], title='Label', labels=['Positive', 'Negative'], loc='lower left', fontsize=30, title_fontsize=30, markerscale=4,  handlelength=1)
    legend.get_frame().set_boxstyle('round')
    legend.get_frame().set_edgecolor('black')
    axs.tick_params(axis='both', labelsize=20)
    axs.set_xticks([])  # Remove x-axis ticks
    axs.set_yticks([])  # Remove y-axis ticks
    axs.set_xticklabels([])  # Remove x-axis tick labels
    axs.set_yticklabels([])  # Remove y-axis tick labels
    plt.tight_layout()
    axs.set_xlim([-0.7, 0.7])
    axs.set_ylim([-0.7, 0.7])
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['bottom'].set_visible(False)
    axs.spines['left'].set_visible(False)
    fig.savefig(fname, facecolor='w', bbox_inches='tight')
    pass


def doDecisionPlot(X, y, X_test, y_test, clf, fname):
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-0.08, X[:, 0].max(), 200),
                         np.linspace(X[:, 1].min()-0.08, X[:, 1].max(), 200))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, axs = plt.subplots(figsize=fsize)
    colormap = ListedColormap(['orange', 'darkcyan'])
    axs.contourf(xx, yy, Z, cmap=ListedColormap(['orange', 'darkcyan']), alpha=0.3)
    scatter = axs.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=ListedColormap(['orange', 'darkcyan']), s=psize)
    legend = axs.legend(handles=scatter.legend_elements()[0], title='Label', labels=['Positive', 'Negative'], loc='lower left', fontsize=30, title_fontsize=30, markerscale=4,  handlelength=1)
    legend.get_frame().set_boxstyle('round')
    legend.get_frame().set_edgecolor('black')
    axs.tick_params(axis='both', labelsize=20)
    axs.set_xticks([])  # Remove x-axis ticks
    axs.set_yticks([])  # Remove y-axis ticks
    axs.set_xticklabels([])  # Remove x-axis tick labels
    axs.set_yticklabels([])  # Remove y-axis tick labels
    plt.tight_layout()
    axs.set_xlim([-0.7, 0.7])
    axs.set_ylim([-0.7, 0.7])
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['bottom'].set_visible(False)
    axs.spines['left'].set_visible(False)

    # Compute AUC on the test set
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    axs.text(0.97, 0.03, f'AUC: {auc:.2f}', verticalalignment='bottom', horizontalalignment='right', transform=axs.transAxes, color='black', fontsize=36, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    fig.savefig(fname, facecolor='w', bbox_inches='tight')
    pass



def wrongPipeline (X, y, X_test, y_test):
    sm = SMOTE(random_state=42)
    X_smote_all, y_smote_all = sm.fit_resample(X, y)
    doPlot (X_smote_all, y_smote_all, "./wrong_after_smote.png")

    # then remove test set from the data
    X_smote_all_train = X_smote_all[20:]
    y_smote_all_train = y_smote_all[20:]
    doPlot (X_smote_all_train, y_smote_all_train, "./wrong_train.png")

    # apply classifier
    clf = SVC(gamma=50, C=500, random_state=42, probability = True)
    clf.fit(X_smote_all_train, y_smote_all_train)

    doDecisionPlot (X_smote_all_train, y_smote_all_train, X_test, y_test, clf, "./wrong_decision.png")



def correctPipeline (X, y, X_test, y_test):
    #  remove test set from the data
    X = X[20:]
    y = y[20:]

    sm = SMOTE(random_state=42)
    X_smote, y_smote = sm.fit_resample(X, y)
    doPlot (X_smote, y_smote, "./correct_after_smote.png")
    doPlot (X_smote, y_smote, "./correct_train.png") # the same

    # apply classifier
    clf = SVC(gamma=50, C=500, random_state=42, probability = True)
    clf.fit(X_smote, y_smote)

    doDecisionPlot (X_smote, y_smote, X_test, y_test, clf, "./correct_decision.png")



if __name__ == '__main__':
    X, y, X_test, y_test  = generateData()
    doPlot (X, y, "./original_data.png")
    doPlot (X[20:], y[20:], "./train_data_only.png")
    doPlot (X_test, y_test, "./test_data_only.png")

    wrongPipeline(X, y, X_test, y_test)
    correctPipeline(X, y, X_test, y_test)


#
