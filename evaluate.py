import itertools
from joblib import parallel_backend, Parallel, delayed, load, dump
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import os
import pandas as pd
from scipy.stats import ttest_rel, linregress
from scipy import stats
import seaborn as sns
import time
from pprint import pprint
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from scipy.stats import pearsonr, friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
from joblib import parallel_backend, Parallel, delayed, load, dump


from parameters import *
from helpers import *
from loadData import *



def extractDF (resultsA):
    df = []
    for r in range(len(resultsA)):
        res = {"AUC":resultsA[r]["AUC"], "Repeat": resultsA[r]["Repeat"]}
        res["Resampling"] = str(resultsA[r]["Params"][0]) # same for all
        res["Clf_method"] = str(resultsA[r]["Params"][2][0][0])
        res["FSel"] = str(resultsA[r]["Params"][1])
        res["FSel"] = str(resultsA[r]["Params"][1])
        res["FSel_method"] = str(resultsA[r]["Params"][1][0][0])
        for f in range(kFold):
            res[f"Fold_{f}_GT"] = resultsA[r]["Preds"][f][1]
            res[f"Fold_{f}_Preds"] = resultsA[r]["Preds"][f][0]
            res[f"FSel_{f}_names"] = str(resultsA[r]["Preds"][f][2])
        df.append(res)
    df = pd.DataFrame(df)
    return df



# https://towardsdatascience.com/expected-calibration-error-ece-a-step-by-step-visual-explanation-with-python-code-c3e9aa12937d#:~:text=Definition,into%20M%20equally%20spaced%20bins.
def expected_calibration_error(samples, true_labels, M=5):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

   # keep confidences / predicted "probabilities" as they are
    confidences = samples
    # get binary class predictions from confidences
    predicted_label = (samples>0.5).astype(float)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prop_in_bin = in_bin.astype(float).mean()

        if prop_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].astype(float).mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece



def getCalibrationTable (dfA, d):
    dfA = dfA.copy()
    for k in dfA.index:
        row = dfA.loc[k]
        all_gt = np.concatenate([row[f"Fold_{j}_GT"] for j in range(5)])
        all_preds = np.concatenate([row[f"Fold_{j}_Preds"] for j in range(5)])

        brier_score = brier_score_loss(all_gt, all_preds)
        prob_true, prob_pred = calibration_curve(all_gt, all_preds, n_bins=5, strategy='quantile')
        #ECE = float(expected_calibration_error(prob_pred, prob_true, 5))
        dfA.at[k, "Brier"] = brier_score
        #dfA.at[k, "ECE"] = ECE

    Bmean = pd.DataFrame(dfA).groupby(["Resampling"])["Brier"].mean().round(3)
    Bmean = Bmean.rename({s:getName(s) for s in Bmean.keys()})
    # Emean = pd.DataFrame(dfA).groupby(["Resampling"])["ECE"].mean().round(3)
    # Emean = Emean.rename({s:getName(s) for s in Emean.keys()})

    tableB = []
    ctable = pd.DataFrame(Bmean)
    ctable[d] = [s[0] for s in list(zip(*[Bmean.values]))]
    ctable = ctable.drop(["Brier"], axis = 1)
    tableB.append (ctable)
    return tableB


def getAUCTable (dfA, d):
    Amean = pd.DataFrame(dfA).groupby(["Resampling"])["AUC"].mean().round(3)
    Amean = Amean.rename({s:getName(s) for s in Amean.keys()})
    Astd = pd.DataFrame(dfA).groupby(["Resampling"])["AUC"].std().round(3)
    Astd = Astd.rename({s:getName(s) for s in Astd.keys()})

    tableSD = []
    ctable = pd.DataFrame(Amean)
    ctable[d] = [str(s[0]) + " +/- " + str(s[1]) for s in list(zip(*[Amean.values, Astd.values]))]
    ctable = ctable.drop(["AUC"], axis = 1)
    tableSD.append (ctable)

    tableM = []
    ctable = pd.DataFrame(Amean)
    ctable[d] = [s[0] for s in list(zip(*[Amean.values, Astd.values]))]
    ctable = ctable.drop(["AUC"], axis = 1)
    tableM.append (ctable)

    return tableM, tableSD


def drawArray (table3, cmap = None, clipRound = True, fsize = (9,7), aspect = None, DPI = 400, fontsize = None, fName = None, paper = False):

    def colorticks(event=None):
        locs, labels = plt.xticks()
        for k in range(len(labels)):
            labels[k].set_color(getColor(labels[k]._text))

        locs, labels = plt.yticks()
        for k in range(len(labels)):
            labels[k].set_color(getColor(labels[k]._text))


    #table3 = tO.copy()
    table3 = table3.copy()
    if clipRound == True:
        for k in table3.index:
            for l in table3.columns:
                if str(table3.loc[k,l])[-2:] == ".0":
                    table3.loc[k,l] = str(int(table3.loc[k,l]))
    # display graphically
    scMat = table3.copy()
    strMat = table3.copy()
    strMat = strMat.astype( dtype = "str")
    # replace nans in strMat
    strMat = strMat.replace("nan", "")

    if 1 == 1:
        plt.rc('text', usetex=True)
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"]})
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'''
            \usepackage{mathtools}
            \usepackage{helvet}
            \renewcommand{\familydefault}{\sfdefault}        '''

        fig, ax = plt.subplots(figsize = fsize, dpi = DPI)
        sns.set(style='white')
        #ax = sns.heatmap(scMat, annot = cMat, cmap = "Blues", fmt = '', annot_kws={"fontsize":21}, linewidth = 2.0, linecolor = "black")
        dx = np.asarray(scMat, dtype = np.float64)

        def getPal (cmap):
            if cmap == "g":
                #np.array([0.31084112, 0.51697441, 0.22130127, 1.        ])*255
                pal = sns.light_palette("#4f8338", reverse=False, as_cmap=True)
            elif cmap == "o":
                pal = sns.light_palette("#ef0000", reverse=False, as_cmap=True)
            elif cmap == "+":
                pal  = sns.diverging_palette(20, 120, as_cmap=True)
            elif cmap == "-":
                pal  = sns.diverging_palette(120, 20, as_cmap=True)
            else:
                pal = sns.light_palette("#ffffff", reverse=False, as_cmap=True)
            return pal


        if len(cmap) > 1:
            for j, (cm, vmin, vcenter, vmax) in enumerate(cmap):
                pal = getPal(cm)
                m = np.ones_like(dx)
                m[:,j] = 0
                Adx = np.ma.masked_array(dx, m)
                tnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                ax.imshow(Adx, cmap=pal, norm = tnorm, interpolation='nearest', aspect = aspect)
                #cba = plt.colorbar(pa,shrink=0.25)
        else:
            if cmap[0][0] == "*":
                for j in range(scMat.shape[1]):
                    pal = getPal("o")
                    m = np.ones_like(dx)
                    m[:,j] = 0
                    Adx = np.ma.masked_array(dx, m)
                    vmin = np.min(scMat.values[:,j])
                    vmax = np.max(scMat.values[:,j])
                    vcenter = (vmin + vmax)/2
                    tnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                    ax.imshow(Adx, cmap=pal, norm = tnorm, interpolation='nearest', aspect = aspect)
                    #cba = plt.colorbar(pa,shrink=0.25)
            else:
                cm, vmin, vcenter, vmax = cmap[0]
                pal = getPal(cm)
                tnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                ax.imshow(dx, cmap=pal, norm = tnorm, interpolation='nearest', aspect = aspect)

        # Major ticks
        mh, mw = scMat.shape
        ax.set_xticks(np.arange(0, mw, 1))
        ax.set_yticks(np.arange(0, mh, 1))

        # Minor ticks
        ax.set_xticks(np.arange(-.5, mw, 1), minor=True)
        ax.set_yticks(np.arange(-.5, mh, 1), minor=True)

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

        for i, c in enumerate(scMat.index):
            for j, f in enumerate(scMat.keys()):
                ax.text(j, i, strMat.at[c, f],    ha="center", va="center", color="k", fontsize = fontsize)
        plt.tight_layout()
        ax.xaxis.set_ticks_position('top') # the rest is the same
        ax.set_xticklabels(scMat.keys(), rotation = 45, ha = "left", fontsize = fontsize)
        ax.set_yticklabels(scMat.index, rotation = 0, ha = "right", fontsize = fontsize)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_tick_params ( labelsize= fontsize)
        colorticks()

    if fName is not None:
        if paper == True:
            fig.savefig(f"./paper/{fName}.png", facecolor = 'w', bbox_inches='tight')
        fig.savefig(f"./results/{fName}.png", facecolor = 'w', bbox_inches='tight')




def createRankingTable (table1):
    tableAUC = table1.copy()
    tR = tableAUC.rank(axis = 0, ascending = False).mean(axis = 1)

    tR = pd.DataFrame(tR).round(1)
    tR.columns = ["Mean rank"]
    tR = tR.sort_values(["Mean rank"])
    rTable = tR.copy()
    tM = tableAUC.mean(axis= 1)
    tM = tM.round(3)
    rTable["Mean bias in AUC"] = tM

    drawArray(rTable, aspect = 0.6, fsize = (10,7), cmap = [("-", 3.5, (3.5+6.5)/2, 6.5), ("+", -0.015, 0.0, 0.015), ("+", -0.06, 0.0, 0.06)], fName = "Figure2", paper = True)
    rTable.to_csv("./results/ranking.csv")

    return tR, tableA


def computeTable (d):
    from loadData import Arita2018, Carvalho2018, Hosny2018A, Hosny2018B, Hosny2018C
    from loadData import Ramella2018, Saha2018, Lu2019, Sasaki2019, Toivonen2019, Keek2020
    from loadData import Li2020, Park2020, Song2020, Veeraraghavan2020

    data = eval (d+"().getData('./data/')")
    data,_ = imputeData(data, None)
    try:
        resultsA = load(f"results/resampling_{d}.dump")
    except:
        print ("Does not exist:", d)
        return None

    # extract infos
    dfA = extractDF (resultsA)

    tableBrier = getCalibrationTable(dfA, d)
    tableBrier = pd.DataFrame(tableBrier[0])
    # table2 = pd.DataFrame(tableSD[0])

    tableM, tableSD = getAUCTable(dfA, d)
    table1 = pd.DataFrame(tableM[0])
    table2 = pd.DataFrame(tableSD[0])
    return tableM, table1, tableSD, table2, tableBrier

#

def generateBiasPlots (pMat,  result_df, ylim_low = 0, ylim_high = 0.4, ppos = 0.95, figsize = (20,23), fname = "Figure4"):
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['text.usetex'] = False
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=figsize)
    fig.subplots_adjust(hspace=0.5, wspace=0.4)  # Adjust both vertical and horizontal space
    axes = axes.flatten()
    sns.set_style("white")
    for i, column in enumerate(pMat.index):
        ax = axes[i]
        sns.scatterplot(x='Balance', y=column, data=result_df, color='blue', label=column, ax=ax)
        x_vals = np.linspace(result_df['Balance'].min(), result_df['Balance'].max(), 100)
        slope, intercept, r_value, p_value, std_err = linregress(result_df['Balance'], result_df[column])
        sns.regplot(x='Balance', y=column, data=result_df, scatter=False, color='gray', ci=None, ax=ax)
        sns.regplot(x='Balance', y=column, data=result_df, scatter=False, color='gray', ci=95, label=None, line_kws={'linewidth': 0}, ax=ax)
        if p_value < 0.001:
            ax.text(0.05, ppos, f'P<0.001', transform=ax.transAxes, fontsize=18, verticalalignment='top')
        else:
            ax.text(0.05, ppos, f'P={p_value:.3f}', transform=ax.transAxes, fontsize=18, verticalalignment='top')
        ax.set_ylim(ylim_low, ylim_high)
        ax.legend().set_visible(False)
        ax.set_title(column, fontsize=20)  # Increased fontsize to 2x
        ax.set_ylabel('Bias in AUC', fontsize=18)  # Increased fontsize to 2x
        ax.set_xlabel('Balance', fontsize=18)  # Increased fontsize to 2x
        ax.tick_params(axis='both', which='major', labelsize=18)  # Increased tick label fontsize to 2x
    for j in range(13,15):
        fig.delaxes(axes[j])

    fig.savefig(f"./paper/{fname}.png", facecolor = 'w', bbox_inches='tight')



def getBrierPlots (dfB):
    df_before = dfB.loc[dfB.index.str.endswith('_Before')]
    df_during = dfB.loc[dfB.index.str.endswith('_During')]

    fMat = (df_during).round(3)
    pMat = pd.DataFrame(fMat)
    pMat.index = [k.replace("_During", '') for k in df.index if "During" in str(k)]
    drawArray(pMat, cmap = [("-", 0.0, 0.25, 0.5)], fsize = (11,7),  aspect = 0.7, fontsize = 15, fName = f"SuppTable_2", paper = True)

    df_before = df_before.drop(["No resampling_Before"], axis = 0).copy()
    df_during = df_during.drop(["No resampling_During"], axis = 0).copy()

    df_diff = pd.DataFrame(df_before.values - df_during.values, columns=df_before.columns, index=df_before.index)
    df_diff.index = df_diff.index.str.replace('_Before', '')
    fMat = df_diff.copy()
    fMat = (fMat).round(3)
    pMat = pd.DataFrame(fMat)
    drawArray(pMat, cmap = [("+", -0.1, 0.0, 0.1)], fsize = (11,7),  aspect = 0.7, fontsize = 15, fName = f"Figure4", paper = True)

    table1 = pd.read_excel("./paper/Table_1.xlsx")
    tableZ = pMat.T.reset_index(drop = False)
    result_df = pd.merge(tableZ, table1, left_on="index", right_on='Dataset')
    generateBiasPlots (pMat,  result_df, ylim_low = -0.2, ylim_high = 0.03, ppos = 0.10, figsize = (18,26), fname = "Figure5")

    median = np.median(pMat.values)
    p25 = np.percentile(pMat, 25)
    p75 = np.percentile(pMat, 75)
    print ("Brier -- Median", median, "IQR", p25, "-", p75)



def getAUCPlots (dfA):
    df_before = df.loc[df.index.str.endswith('_Before')]
    df_during = df.loc[df.index.str.endswith('_During')]

    fMat = (df_during).round(3)
    pMat = pd.DataFrame(fMat)
    pMat.index = [k.replace("_During", '') for k in df.index if "During" in str(k)]
    drawArray(pMat, cmap = [("o", 0.5, 0.75, 1.0)], fsize = (11,7),  aspect = 0.7, fontsize = 15, fName = f"SuppTable_1", paper = True)

    # now remove None because we hide it


    # arita2018: predich IDH wildtype, paper: 0.83 on validation set
    # carvalho2018: predicts survival, not comparable, but C-index around 0.60, so it fits
    # Hosny2018A: trianing only, havartRT   AUC = 0.66 (95% CI 0.58–0.75, p < 0.001)
    # Hosny2018B: training only, Maastro   AUC = 0.66 (95% CI 0.58–0.75, p < 0.001)
    # Hosny2018C: training only mofitt surgery and AUC = 0.58 (95% CI 0.44–0.75, p = 0.275)
    # Ramella2018: 0.82 (0.73-0.91)
    # FIX RAMELLA is PET-CT
    # Saha2018: 0.697 (95% CI: 0.647–0.746, p < .0001)
    # Lu2019: validation: C-index from 0.659 to 0.679, could fit to 0.736
    # Sasaki2019: AUC not specified, but sens/spec 0.67, so fits  pMGMT methylation
    # Toivonen2019: 0.88 (95% CI of 0.82–0.95), fits
    # Keek2020: we:3-year mortality they survival, for all sites C-index 0.60-0.65, so fits
    # Li2020: high vs low grade, AUC:0.88 fits perfect
    # Park2020: 0.621 (95% CI: 0.560–0.682) in validation, fits perfect
    # Song2020: 0.814, no CI. but unclear, what subset they selected, they say 185 cases were selected
    # from PROSTATEX. in the figure 5 one can see 0.916 for validation, no CI
    # Veeraraghavan2020: low from high: train: 0.74a (0.64, 0.84) but unreasonable high 0.87 (0.73, 0.95) on test.
    # they included clinical parameters, so unclear how to compare.but no methodological errors.
    np.max(df_during,axis = 0)

    df_before = df_before.drop(["No resampling_Before"], axis = 0).copy()
    df_during = df_during.drop(["No resampling_During"], axis = 0).copy()

    df_diff = pd.DataFrame(df_before.values - df_during.values, columns=df_before.columns, index=df_before.index)
    df_diff.index = df_diff.index.str.replace('_Before', '')
    fMat = df_diff.copy()
    df_diff.mean(axis=1)
    df_diff.std(axis=1)

    fMat = (fMat).round(3)
    pMat = pd.DataFrame(fMat)
    drawArray(pMat, cmap = [("o", 0, 0.05, 0.5)], fsize = (11,7),  aspect = 0.7, fontsize = 15, fName = f"Figure2", paper = True)

    table1 = pd.read_excel("./paper/Table_1.xlsx")
    tableZ = pMat.T.reset_index(drop = False)
    result_df = pd.merge(tableZ, table1, left_on="index", right_on='Dataset')
    generateBiasPlots (pMat,  result_df, ylim_low = 0.0, ylim_high = 0.4, ppos = 0.95, figsize = (18,26), fname = "Figure3")

    median = np.median(pMat.values)
    p25 = np.percentile(pMat, 25)
    p75 = np.percentile(pMat, 75)
    print ("AUC -- Median", median, "IQR", p25, "-", p75)


if __name__ == '__main__':
    # gather data
    with parallel_backend("loky", inner_max_num_threads=1):
        cres = Parallel (n_jobs = ncpus)(delayed(computeTable)(d) for d in dList)

    table1 = []
    tableB = []
    for j, _ in enumerate(dList):
        tM, t1, tSD, t2, tB = cres[j]
        table1.append(t1)
        tableB.append(tB)
    df = pd.concat(table1, axis = 1)
    dfB = pd.concat(tableB, axis = 1)

    pMat = getAUCPlots (df)
    bMat = getBrierPlots (dfB)

#
