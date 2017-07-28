import pandas as pd
import numpy as np
import os

from sklearn.linear_model import Lasso

np.random.seed(1337)


def process_file(path):
    user = {'Alice': "A", 'Bob': "B"}

    df = pd.read_json(path).set_index("dialogId")
    df['speaker'] = df["thread"].apply(lambda x: [user[msg['userId']] for msg in x])
    df['thread'] = df["thread"].apply(lambda x: [msg['text'] for msg in x], convert_dtype=False)
    df['thread_raw'] = df["thread"].apply(lambda x: " ".join(x))
    df["qualA"] = df["evaluation"].apply(lambda x: sorted(x, key=lambda x: x['userId'])[0]['quality'])
    df["qualB"] = df["evaluation"].apply(lambda x: sorted(x, key=lambda x: x['userId'])[1]['quality'])
    df["botA"] = df["users"].apply(lambda x: sorted(x, key=lambda x: x['id'])[0]['userType'] == 'Bot')
    df["botB"] = df["users"].apply(lambda x: sorted(x, key=lambda x: x['id'])[1]['userType'] == 'Bot')
    df.drop(['users'], axis=1, inplace=True)
    df.drop(['evaluation'], axis=1, inplace=True)

    return df[["qualA", "qualB"]]


def f(path):
    data = pd.concat(
        [
            process_file(os.path.join(path, file))
            for file in os.listdir(path)
            if file.startswith("train")
        ], axis=1
    )

    return data

preds = pd.DataFrame()
for file in os.listdir("data"):
    if os.path.isfile(os.path.join("data", file)) and file.endswith(".csv"):
        filename = os.path.join("data", file)
        preds[file] = pd.read_csv(filename)["Alice"].tolist()+pd.read_csv(filename)["Bob"].tolist()

y = f("data/train")["qualA"].tolist() + f("data/train")["qualB"].tolist()

clf = Lasso()
clf.fit(preds, y)
print([file for file in os.listdir("data") if os.path.isfile(os.path.join("data", file)) and file.endswith(".csv")])
print(clf.coef_)