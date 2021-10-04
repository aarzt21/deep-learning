#Majority voting of the three models 

import os
import pandas as pd
import numpy as np

os.chdir("C:/Users/valan/Desktop/Preds")


preds1 = pd.read_csv("VGGC16_PREDS.txt", header=None) #best model
preds2 = pd.read_csv("resnet224_PREDS.txt.txt", header=None)   # 
preds3 = pd.read_csv("resnet50_250_PREDS.txt", header=None)


merge = pd.concat([preds1,preds2,preds3], axis=1)

merge["Sum"] = merge.sum(axis=1)

def f(row):
    if row['Sum'] >= 2:
        val = 1

    else:
        val = 0
    return val

merge['VOTE_RES'] = merge.apply(f, axis=1)


np.savetxt("majority_vgg_resNet.txt", merge["VOTE_RES"], fmt = "%d")