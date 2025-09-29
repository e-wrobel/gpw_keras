# Makefile for training and serving the LSTM FastAPI app
# Usage examples:
#   make venv            # create virtualenv (.venv)
#   make install         # install dependencies from requirements.txt
#   make train           # train models for all tickers (writes to models/)
#   make serve           # run FastAPI server on http://127.0.0.1:8000
#   make dev             # run FastAPI with --reload
#   make health          # curl /healthz (pretty-printed if jq installed)
#   make predict TICKER=AAPL STEPS=5   # sample forecast call

SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

PY311 := $(shell command -v python3.11 2>/dev/null || true)
PYTHON := $(if $(PY311),$(PY311),python3)
VENV := .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
UVICORN := $(VENV)/bin/uvicorn

# FastAPI serve params
HOST ?= 0.0.0.0
PORT ?= 8000
MODELS_DIR ?= models

# Predict params
TICKER ?= AAPL
STEPS ?= 5

.PHONY: help venv install freeze train serve dev health predict clean

help:
	@echo "Targets:"
	@echo "  venv     - create virtualenv ($(VENV))"
	@echo "  install  - install deps from requirements.txt"
	@echo "  freeze   - write current deps to requirements.txt"
	@echo "  train    - run main.py to train & export models into $(MODELS_DIR)/"
	@echo "  serve    - run FastAPI with uvicorn (host=$(HOST) port=$(PORT))"
	@echo "  dev      - serve with --reload"
	@echo "  health   - curl /healthz"
	@echo "  predict  - curl /predict with TICKER=$(TICKER) STEPS=$(STEPS)"
	@echo "  clean    - remove caches/__pycache__"

$(VENV):
	$(PYTHON) -V
	$(PYTHON) -m venv $(VENV)
	@echo "\n[OK] Virtualenv created at $(VENV)"

venv: $(VENV)

install: venv
	$(PIP) install --upgrade pip
	@if [ -f requirements.txt ]; then \
		$(PIP) install -r requirements.txt; \
	else \
		echo "requirements.txt not found. Installing minimal deps..."; \
		$(PIP) install fastapi uvicorn pydantic yfinance numpy matplotlib tensorflow-macos==2.16.1 tensorflow-metal==1.1.0; \
	fi
	@echo "\n[OK] Dependencies installed"

freeze: venv
	$(PIP) freeze > requirements.txt
	@echo "[OK] requirements.txt updated"

train: venv
	$(PY) main.py
	@echo "\n[OK] Training finished. Artifacts under $(MODELS_DIR)/"

serve: venv
	MODELS_DIR=$(MODELS_DIR) $(UVICORN) app:app --host $(HOST) --port $(PORT)

# Developer mode with code autoreload
dev: venv
	MODELS_DIR=$(MODELS_DIR) $(UVICORN) app:app --host $(HOST) --port $(PORT) --reload

health:
	curl -s http://127.0.0.1:$(PORT)/healthz | jq . || curl -s http://127.0.0.1:$(PORT)/healthz

predict:
	curl -s -X POST "http://127.0.0.1:$(PORT)/predict" \
	 -H "Content-Type: application/json" \
	 -d '{"ticker":"$(TICKER)","steps":$(STEPS)}' | jq . || true

clean:
	rm -rf __pycache__ **/__pycache__ .pytest_cache .mypy_cache
	find . -name '*.pyc' -delete
	@echo "[OK] Cleaned"