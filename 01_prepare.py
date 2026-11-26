"""Shared deterministic preparation for labelled and unlabelled data."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = DATA / 'processed'
SEED = 42

def read(name):
    return pd.read_csv(DATA / (name + '.csv'), dtype={k:'string' for k in ['insee_code','DEP','Reg','job_dep','PREVIOUS_DEP','work_desc','N3','N2','N1']})

def prepare(prefix):
    df = read(prefix + '_dataset')
    for suffix in ['Emp_tenure','job','retired_former','retired_jobs','retired_pension','sport']:
        other = read(prefix + '_dataset_' + suffix)
        df = df.merge(other, on='uid', how='left', validate='one_to_one', suffixes=('', '_ret' if suffix=='retired_jobs' else '_'+suffix))
    for name in ['city_adm','city_pop','city_loc']:
        df = df.merge(read(name), on='insee_code', how='left', validate='many_to_one')
    df = df.merge(read('departments'), on='DEP', how='left', validate='many_to_one')
    df = df.merge(read('regions'), on='Reg', how='left', validate='many_to_one')
    mapping = read('code_work_desc_map').set_index('N3')
    for col in ['work_desc','work_desc_ret']:
        if col in df:
            for level in ['N1','N2']:
                df[col+'_'+level] = df[col].map(mapping[level])
    df['has_sport_record'] = df['CLUB'].notna().astype(int)
    df['has_tenure_record'] = df['Emp_tenure'].notna().astype(int)
    return df

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    learn, test = prepare('learn'), prepare('test')
    assert learn.uid.is_unique and test.uid.is_unique
    assert not set(learn.uid) & set(test.uid)
    assert learn.target.notna().all()
    assert set(learn.columns)-{'target'} == set(test.columns)
    train, valid = train_test_split(learn, test_size=.2, random_state=SEED, stratify=learn.target)
    for name, frame in [('train',train),('valid',valid),('test',test)]:
        frame.to_pickle(OUT / (name+'.pkl'))
        print(name, frame.shape)
