#
from math import sqrt
import os
import shutil


def getColor (name):
    color = "black"
    if name == "Random undersampling":
        color = "red"
    if "Polynomial-fit SMOTE" in name:
        color = "blue"
    if "Majority Weighted Minority Oversampling" in name:
        color = "blue"
    if "MWMOTE" in name:
        color = "blue"
    if "Tomek links" in name:
        color = "red"
    if "Random oversampling" in name:
        color = "blue"
    if "SMOTE [k=" in name:
        color = "blue"
    if "SMOTE+Tomek links [k=" in name:
        color = "#004800"
    if name == "None":
        color = "black"
    return color


def getName (s, detox = False):
    v = eval(s)[0]
    pos = v[1]["Position"]
    if "RUS" == v[0]:
        name = "Random Undersampling"
    if "TomekLinks" == v[0]:
        name = f"Tomek links"
    if "ROS" == v[0]:
        name = "Random Oversampling"
    if "SMOTE" == v[0]:
        name = f"SMOTE [k={v[1]['k']}]"
    if "PolySMOTE" == v[0]:
        name = f"Polynomial-fit SMOTE [{v[1]['topology']}]"
    if "MWMOTE" == v[0]:
        name = f"MWMOTE"
    if "SMOTETomek" == v[0]:
        name = f"SMOTE+Tomek links [k={v[1]['k']}]"
    if "None" == v[0]:
        name = f"No resampling"
    name = name + "_" + str(pos)

    if detox == True:
        name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "").replace(",", "_")
    return name



def recreatePath (path, create = True):
    print ("Recreating path ", path)
    try:
        shutil.rmtree (path)
    except:
        pass

    if create == True:
        try:
            os.makedirs (path)
        except:
            pass
    print ("Done.")



def findOptimalCutoff (fpr, tpr, threshold, verbose = False):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    fpr, tpr, threshold

    Returns
    -------
    list type, with optimal cutoff value

    """

    # own way
    minDistance = 2
    bestPoint = (2,-1)
    for i in range(len(fpr)):
        p = (fpr[i], tpr[i])
        d = sqrt ( (p[0] - 0)**2 + (p[1] - 1)**2 )
        if verbose == True:
            print (p, d)
        if d < minDistance:
            minDistance = d
            bestPoint = p

    if verbose == True:
        print ("BEST")
        print (minDistance)
        print (bestPoint)
    sensitivity = bestPoint[1]
    specificity = 1 - bestPoint[0]
    return sensitivity, specificity


#
