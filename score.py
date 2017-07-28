#!/usr/bin/env python3

# usage: python3 score.py data.json labels.csv

from scipy.stats import spearmanr
import sys
import numpy as np
import pandas as pd
df=pd.read_json(sys.argv[1]).set_index("dialogId")
df["qualA"]=df.evaluation.apply(lambda x: sorted(x,key=lambda x:x['userId'])[0]['quality'])
df["qualB"]=df.evaluation.apply(lambda x: sorted(x,key=lambda x:x['userId'])[1]['quality'])
subm=pd.read_csv(sys.argv[2],index_col="dialogId")
print(spearmanr(np.hstack([df.qualA,df.qualB]),np.hstack([subm.Alice,subm.Bob]))[0])