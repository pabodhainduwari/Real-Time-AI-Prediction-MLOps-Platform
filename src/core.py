import json, os, threading
from pathlib import Path
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(os.getenv('DATA_DIR','data')); ROOT.mkdir(parents=True,exist_ok=True)
MODEL=ROOT/'model.joblib'; HISTORY=ROOT/'predictions.jsonl'; METADATA=ROOT/'metadata.json'; TRAIN=ROOT/'training.csv'
FEATURES=['age','income','tenure','usage','support_tickets']
LOCK=threading.RLock()

def seed_data(n=1500,seed=42):
 rng=np.random.default_rng(seed)
 d=pd.DataFrame({'age':rng.integers(18,75,n),'income':rng.uniform(18000,130000,n),'tenure':rng.integers(0,20,n),'usage':rng.uniform(0,100,n),'support_tickets':rng.poisson(3,n)})
 logits=-1.5+0.3*(d.age-40)/15-0.65*(d.income-65000)/30000-0.7*(d.tenure-8)/6-0.8*(d.usage-50)/30+0.65*(d.support_tickets-3)/3
 p=1/(1+np.exp(-logits));d['churn']=rng.binomial(1,p)
 return d

def train(data=None,source='synthetic'):
 with LOCK:
  if data is None: data=pd.read_csv(TRAIN) if TRAIN.exists() else seed_data()
  if not set(FEATURES+['churn']).issubset(data.columns):raise ValueError('Training data requires feature columns and churn')
  data=data[FEATURES+['churn']].dropna().copy()
  if len(data)<50 or data.churn.nunique()<2:raise ValueError('At least 50 labeled rows and both classes required')
  Xtr,Xte,ytr,yte=train_test_split(data[FEATURES],data.churn,test_size=.2,random_state=42,stratify=data.churn)
  pipe=Pipeline([('scale',StandardScaler()),('model',RandomForestClassifier(n_estimators=120,max_depth=9,random_state=42,n_jobs=-1))]);pipe.fit(Xtr,ytr)
  proba=pipe.predict_proba(Xte)[:,1];metrics={'accuracy':round(float(accuracy_score(yte,pipe.predict(Xte))),4),'roc_auc':round(float(roc_auc_score(yte,proba)),4),'rows':len(data)}
  baseline={c:{'mean':float(data[c].mean()),'std':max(float(data[c].std()),1e-6)} for c in FEATURES}
  version=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
  tmp=MODEL.with_suffix('.tmp');joblib.dump({'pipeline':pipe,'baseline':baseline,'version':version},tmp);os.replace(tmp,MODEL)
  metadata={'version':version,'trained_at':datetime.now(timezone.utc).isoformat(),'source':source,**metrics};METADATA.write_text(json.dumps(metadata,indent=2))
  try:
   import mlflow
   mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI','file:./mlruns'))
   mlflow.set_experiment('churn-prediction')
   with mlflow.start_run(run_name=version):
    mlflow.log_params({'estimator':'RandomForestClassifier','n_estimators':120,'source':source})
    mlflow.log_metrics({'accuracy':metrics['accuracy'],'roc_auc':metrics['roc_auc']})
  except Exception as e: print('MLflow logging unavailable:',e)
  return metadata

def load():
 with LOCK:
  if not MODEL.exists():train()
  return joblib.load(MODEL)

def predict(features):
 with LOCK:
  bundle=load();row=pd.DataFrame([features],columns=FEATURES)
  prob=float(bundle['pipeline'].predict_proba(row)[0,1]);out={'prediction':int(prob>=.5),'churn_probability':round(prob,5),'model_version':bundle['version']}
  with HISTORY.open('a') as f:f.write(json.dumps({'timestamp':datetime.now(timezone.utc).isoformat(),**features,**out})+'\n')
  return out

def drift(window=100,threshold=2.0):
 with LOCK:
  base=load()['baseline'];lines=HISTORY.read_text().splitlines()[-window:] if HISTORY.exists() else []
  if len(lines)<20:return {'drift_detected':False,'scores':{},'sample_size':len(lines),'message':'At least 20 predictions required'}
  d=pd.DataFrame([json.loads(x) for x in lines]);scores={c:round(abs(float(d[c].mean())-base[c]['mean'])/base[c]['std'],4) for c in FEATURES}
  return {'drift_detected':any(v>threshold for v in scores.values()),'scores':scores,'sample_size':len(d),'threshold':threshold,'method':'absolute standardized mean shift'}
