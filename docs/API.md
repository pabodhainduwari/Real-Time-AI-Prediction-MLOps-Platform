# API reference

- `GET /health` service status
- `POST /predict` JSON customer fields: age, income, tenure, usage, support_tickets
- `GET /model/info` model version, accuracy, ROC AUC and train metadata
- `GET /monitor/drift?window=100` recent prediction distribution shift
- `GET /metrics` Prometheus metrics
- `POST /model/retrain` header `X-Admin-Token: <ADMIN_TOKEN>`; retrains on labeled CSV if available, otherwise demo synthetic data

Swagger/OpenAPI interactive reference: `/docs`.
