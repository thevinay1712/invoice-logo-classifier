.PHONY: help setup test lint train evaluate

help:
	@echo "Available targets:"
	@echo "  setup     - Install dependencies"
	@echo "  test      - Run pytest suite"
	@echo "  lint      - Run code linting and formatting checks"
	@echo "  train     - Execute model training pipeline (stub)"
	@echo "  evaluate  - Run evaluation and threshold calibration (stub)"

setup:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

test:
	python -m pytest tests/ -v

lint:
	python -m ruff check invoice_logo_classifier tests

train:
	@echo "Training pipeline stub - to be implemented in training phase."

evaluate:
	@echo "Evaluation pipeline stub - to be implemented in evaluation phase."
