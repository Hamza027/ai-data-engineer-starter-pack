# AI Data Engineer Starter Pack

A small, production-style starter repo that demonstrates:
- a FastAPI service
- ETL job execution
- Postgres run logging
- containerized stack with Docker Compose
- small PyTorch training scripts

## Run locally with uv

### Install deps
```bash
uv sync

## Quickstart (Docker)

```bash
docker compose up -d --build

# Then run the smoke test:

.\smoke_test.ps1

# Commit it:

```powershell
git add README.md
git commit -m "Add quickstart instructions"
git log --oneline --max-count=5