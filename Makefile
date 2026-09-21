.PHONY: setup infrastructure datasets seed migrate backend frontend test evaluate
setup:
	python -m pip install -r backend/requirements.txt
	cd frontend && npm install
infrastructure:
	docker compose up -d postgres
datasets:
	python scripts/download_datasets.py
seed:
	python scripts/load_datasets.py
migrate:
	python scripts/migrate_recovery.py
backend:
	cd backend && uvicorn app.main:app --reload --port 8000
frontend:
	cd frontend && npm run dev
test:
	cd backend && pytest -q
	cd frontend && npm run build
evaluate:
	cd backend && python -m app.services.evaluation
