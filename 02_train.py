"""Select hyperparameters on training folds; evaluate on held-out data."""
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.metrics import accuracy_score,roc_auc_score,balanced_accuracy_score,f1_score,confusion_matrix
from pipeline import features,model,SEED
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    data=ROOT/'data'/'processed'; out=ROOT/'results';out.mkdir(exist_ok=True)
    train=pd.read_pickle(data/'train.pkl'); valid=pd.read_pickle(data/'valid.pkl')
    x,y=features(train),train.target.eq('G').astype(int); v,z=features(valid),valid.target.eq('G').astype(int)
    cv=StratifiedKFold(3,shuffle=True,random_state=SEED)
    rows=[]
    for name,depth in [('Dummy',None),('Random Forest',12),('Random Forest',20),('CatBoost',4),('CatBoost',6)]:
        scores=[]
        for a,b in cv.split(x,y):
            estimator=model(name,x,depth)
            estimator.fit(x.iloc[a],y.iloc[a])
            scores.append(roc_auc_score(y.iloc[b],estimator.predict_proba(x.iloc[b])[:,1]))
        import numpy as np
        scores=np.asarray(scores)
        rows.append(dict(model=name,depth=depth,cv_auc=float(scores.mean()),cv_sd=float(scores.std())))
        print(rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(out/'cv_scores.csv',index=False)
    best=max(rows,key=lambda r:r['cv_auc']); results=[]
    for name in ['Dummy','Random Forest','CatBoost']:
        choice=max((r for r in rows if r['model']==name),key=lambda r:r['cv_auc'])
        m=model(name,x,choice['depth']);m.fit(x,y);p=m.predict_proba(v)[:,1];pred=m.predict(v).reshape(-1).astype(int)
        results.append(dict(model=name,accuracy=accuracy_score(z,pred),balanced_accuracy=balanced_accuracy_score(z,pred),f1=f1_score(z,pred),roc_auc=roc_auc_score(z,p),confusion_matrix=confusion_matrix(z,pred).tolist()))
        print(results[-1],flush=True)
    (out/'validation.json').write_text(json.dumps(results,indent=2));(out/'selection.json').write_text(json.dumps(best,indent=2))
