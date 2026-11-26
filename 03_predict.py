"""Refit selected configuration on all labelled data and predict test labels."""
from pathlib import Path
import json
import pandas as pd
from pipeline import features,model
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    d=ROOT/'data'/'processed'; choice=json.loads((ROOT/'results'/'selection.json').read_text())
    learn=pd.concat([pd.read_pickle(d/'train.pkl'),pd.read_pickle(d/'valid.pkl')]);test=pd.read_pickle(d/'test.pkl')
    x=features(learn);m=model(choice['model'],x,choice['depth']);m.fit(x,learn.target.eq('G').astype(int))
    pred=m.predict(features(test)).reshape(-1);out=pd.DataFrame({'uid':test.uid,'target':['G' if a==1 else 'B' for a in pred]})
    assert len(out)==len(test) and out.uid.is_unique
    out.to_csv(d/'predictions.csv',index=False)
