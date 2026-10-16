# Real-Time AI Prediction Platform with Automated MLOps

**Individual two-week project | October 2–16, 2026**

Customer churn prediction reference implementation: FastAPI real-time inference, sklearn model pipeline, MLflow experiment logs, rolling drift monitoring, Streamlit dashboard, Docker Compose, GitHub Actions CI, and optional SSH deployment workflow.

## Quick start

```bash
cp .env.example .env
# set ADMIN_TOKEN to a strong value
docker compose up --build -d
```

- API docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- Health: http://localhost:8000/health
- Drift: http://localhost:8000/monitor/drift
- Prometheus metrics: http://localhost:8000/metrics

## Example inference

```bash
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{"age":35,"income":60000,"tenure":5,"usage":45,"support_tickets":2}'
```

## Local development

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
streamlit run dashboard/app.py
pytest -q
```

## Model training and updates

- First launch creates a reproducible **synthetic** labeled churn dataset in memory and trains a Random Forest.
- To train on **real labeled data**, supply `data/training.csv` with `age,income,tenure,usage,support_tickets,churn`, then run `python -c "from src.core import train; print(train(source='labeled_csv'))"` in the API container. Do not train on unlabeled prediction logs.
- `python -m scripts.retrain` evaluates drift; set `AUTO_RETRAIN=true` to enable drift-triggered retraining on available labeled data. Run this script **inside the long-lived API container or an attached persistent volume** via an external scheduler.
- Drift method: absolute standardized mean shift compared to training baseline, default threshold 2, minimum 20 samples. It is a lightweight heuristic, not statistical proof.
- Experiment tracking defaults to local `mlruns/`; configure `MLFLOW_TRACKING_URI` for persistent shared tracking.
- POST `/model/retrain` requires `X-Admin-Token` matching `ADMIN_TOKEN`.

## CI/CD and cloud deployment

CI workflow tests and builds Docker image on push/PR. Optional deploy workflow runs only when GitHub repository variable `ENABLE_SSH_DEPLOY=true` and secrets `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY` are configured; VM must already have repo cloned at `/srv/mlops-platform` and Docker Compose installed. Protect main branch and configure TLS/reverse proxy, firewall, persistent backups, and proper secrets before exposing to the internet.

**Important:** The included GitHub Actions scheduled workflow is a *template*; it executes in an ephemeral runner, so it does not see production inference history and is configured not to retrain. Set up a scheduled command in the deployed container/VM for actual ongoing drift-triggered retraining. Cloud hosting and public URL require your own credentials and are **not provisioned by this ZIP**. A real continuous learning loop requires newly collected, verified labels and evaluation gates before model promotion.

## Deliverables

| Deliverable | Location / status |
|---|---|
| GitHub-ready source | Repository tree (ready to push, not published) |
| ML training and inference | `src/core.py`, `app/main.py` |
| Dockerized application | `Dockerfile`, `docker-compose.yml` |
| Live deployed system | Deployment workflow included; actual deployment pending credentials |
| CI/CD | `.github/workflows/` |
| Monitoring dashboard | `dashboard/app.py` |
| Documentation | `README.md`, `docs/` |
| Architecture diagram | `docs/architecture.mmd`, `docs/architecture.png` |

## Limitations

Demo data are synthetic; performance metrics do not establish business accuracy. Drift alerts do not automatically imply safe retraining. This is a deployable reference project, not a verified production service. Add auth to prediction endpoints, rate limiting, observability persistence, secure MLflow hosting, integration tests and human/model validation before enterprise use.
