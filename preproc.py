import pandas as pd
from keras.preprocessing.text import Tokenizer


def process_data(filename,train=True):
    user={'Alice':1,'Bob':2}
    df=pd.read_json(filename).set_index("dialogId")
    df['speaker']=df.thread.apply(lambda x:[user[msg['userId']] for msg in x])
    df['thread']=df.thread.apply(lambda x: [msg['text'] for msg in x],convert_dtype=False)
    df['thread_raw']=df.thread.apply(lambda x:" ".join(x))
    if train:
        df["qualA"]=df.evaluation.apply(lambda x: sorted(x,key=lambda x:x['userId'])[0]['quality'])
        df["qualB"]=df.evaluation.apply(lambda x: sorted(x,key=lambda x:x['userId'])[1]['quality'])
    df["botA"]=df.users.apply(lambda x: sorted(x,key=lambda x:x['id'])[0]['userType']=='Bot')
    df["botB"]=df.users.apply(lambda x: sorted(x,key=lambda x:x['id'])[1]['userType']=='Bot')
    if train:
        df.drop(['users','evaluation'],axis=1,inplace=True)
    else:
        df.drop(['evaluation'],axis=1,inplace=True)
    return df