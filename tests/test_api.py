from fastapi.testclient import TestClient
from app.main import app

def test_endpoints():
 with TestClient(app) as client:
  assert client.get('/health').status_code==200
  r=client.post('/predict',json={'age':32,'income':55000,'tenure':4,'usage':55,'support_tickets':2})
  assert r.status_code==200 and 0<=r.json()['churn_probability']<=1
  assert client.get('/monitor/drift').status_code==200
  assert client.get('/metrics').status_code==200
  assert client.post('/model/retrain').status_code==403

def test_invalid_input():
 with TestClient(app) as client:assert client.post('/predict',json={'age':-1}).status_code==422
