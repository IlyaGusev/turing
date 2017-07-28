from parse_data import  parse
import pandas as pd
import numpy as np
from scipy import stats

def spearman(a, b):
    return stats.spearmanr(a, b)[0]


def local_scorer(train_filename, submition):
    df = parse([train_filename])
    subm = pd.read_csv(submition, index_col="dialogId")
    preds = np.array(subm.Alice.tolist() + subm.Bob.tolist())
    answer = np.array(df["AliceScore"].tolist() + df["BobScore"].tolist())
    print(spearman(answer, preds))

local_scorer("data/train/train_20170727.json", "data/final_answer.csv")