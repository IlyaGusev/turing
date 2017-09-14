import numpy as np
import pandas as pd
from scipy import stats

from code.parse_data import parse


def spearman(a, b):
    return stats.spearmanr(a, b)[0]


def local_scorer(filename, submition):
    df = parse([filename])
    subm = pd.read_csv(submition, index_col="dialogId")
    preds = np.array(subm.Alice.tolist() + subm.Bob.tolist())
    answer = np.array(df["AliceScore"].tolist() + df["BobScore"].tolist())
    print(spearman(answer, preds))

if __name__ == "__main__":
    local_scorer("../data/train/train_final.json", "../submitions/final_answer.csv")