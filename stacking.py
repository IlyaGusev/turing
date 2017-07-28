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
        ]
    )

    return data


preds = [pd.read_csv(file)[["Alice", "Bob"]] for file in os.listdir("data") if os.path.isfile(file) and file.endswith(".csv")]

X = pd.concat(preds, axis=1).values
y = f("data/train")

clf = Lasso()
clf.fit(X, y)
