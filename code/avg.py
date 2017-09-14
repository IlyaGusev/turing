import pandas as pd
import os
import numpy as np


def avg_blending(filenames, final_answer_path):
    dfs = []
    for filename in filenames:
        dfs.append(pd.read_csv(filename, header=0))
    df = pd.DataFrame()
    df["dialogId"] = dfs[0]["dialogId"]
    users = ["Alice", "Bob"]
    for user in users:
        df[user] = np.zeros((dfs[0].shape[0]))
        for data in dfs:
            df[user] = df[user] + data[user]
        df[user] = df[user].apply(lambda x: x/len(dfs))
    df = df[["dialogId", "Alice", "Bob"]]
    df.to_csv(final_answer_path, index=False)

if __name__ == "__main__":
    answers_path = "../submitions"
    answers = [os.path.join(answers_path, filename) for filename in os.listdir(answers_path) if ".csv" in filename]
    avg_blending(answers, final_answer_path=os.path.join(answers_path, "final_answer.csv"))