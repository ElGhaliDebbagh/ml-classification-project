from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from catboost import CatBoostClassifier
SEED=42
def features(df):
    x=df.drop(columns=['uid','target','Nom de la commune','Nom du département','Nom de la région','X','Y','insee_code'],errors='ignore').copy()
    for c in x.select_dtypes(exclude='number'):
        x[c]=x[c].astype('string').fillna('__MISSING__').astype(str)
    return x

def model(name,x,depth):
    cats=list(x.select_dtypes(exclude='number').columns)
    nums=list(x.select_dtypes(include='number').columns)
    if name=='CatBoost':
        return CatBoostClassifier(iterations=150,depth=depth,learning_rate=.1,cat_features=cats,random_seed=SEED,thread_count=4,verbose=False,allow_writing_files=False)
    if name=='Dummy': return DummyClassifier(strategy='most_frequent')
    prep=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore',min_frequency=10),cats),('num',SimpleImputer(strategy='median',add_indicator=True),nums)])
    return Pipeline([('prep',prep),('model',RandomForestClassifier(n_estimators=40,max_depth=depth,min_samples_leaf=5,n_jobs=4,random_state=SEED))])
