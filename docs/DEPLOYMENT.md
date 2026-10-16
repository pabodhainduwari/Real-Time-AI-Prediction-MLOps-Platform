# Deployment checklist

1. Provision a Linux VM with Docker Engine and Compose plugin; restrict SSH and firewall access.
2. Clone repository into `/srv/mlops-platform` and create `.env` with a strong `ADMIN_TOKEN`.
3. Start `docker compose up -d --build`; test `/health`, `/docs`, and dashboard.
4. Configure a domain, HTTPS reverse proxy and authentication for public services.
5. Configure GitHub repository variable `ENABLE_SSH_DEPLOY=true` and secrets `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY`.
6. Use a deployed-host cron job or orchestrator to run `docker compose exec -T api env AUTO_RETRAIN=true python -m scripts.retrain` every six hours, only after a reliable labeled CSV refresh and model validation process is in place.
7. Persist and back up modeldata volume; separately persist MLflow artifacts and monitoring history as needed.
8. Verify rollback and alerts before enabling automatic promotion in a real system.

No cloud account, DNS, public deployment or GitHub repository is automatically created by this package.
