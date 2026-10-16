from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from src.core import load, predict, drift, train, METADATA
import json, os, time
from fastapi import Header

REQUESTS=Counter('prediction_requests_total','Prediction requests')
LATENCY=Histogram('prediction_latency_seconds','Prediction latency')
@asynccontextmanager
async def lifespan(app):
 load();yield
app=FastAPI(title='Real-Time MLOps Prediction Platform',version='1.0.0',lifespan=lifespan)
class Customer(BaseModel):
 age:int=Field(ge=18,le=120)
 income:float=Field(ge=0)
 tenure:float=Field(ge=0,le=100)
 usage:float=Field(ge=0)
 support_tickets:int=Field(ge=0)
@app.get('/health')
def health():return {'status':'healthy'}
@app.post('/predict')
def make_prediction(customer:Customer):
 REQUESTS.inc()
 with LATENCY.time():return predict(customer.model_dump())
@app.get('/monitor/drift')
def monitor_drift(window:int=100):return drift(max(20,min(window,10000)))
@app.get('/model/info')
def model_info():return json.loads(METADATA.read_text())
@app.post('/model/retrain')
def retrain(x_admin_token:str|None=Header(default=None)):
 token=os.getenv('ADMIN_TOKEN')
 if not token or x_admin_token!=token:raise HTTPException(403,'Valid admin token required')
 return train(source='manual_api')
@app.get('/metrics')
def metrics():return Response(generate_latest(),media_type=CONTENT_TYPE_LATEST)
