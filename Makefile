.PHONY: install lint typecheck test pipeline app build help

help:
	@echo "Available commands:"
	@echo "  install   : Install dependencies using Poetry"
	@echo "  lint      : Run Ruff for linting and formatting"
	@echo "  typecheck : Run Mypy for type checking"
	@echo "  test      : Run unit tests with Pytest"
	@echo "  pipeline  : Execute the ML forecasting pipeline"
	@echo "  app       : Launch the Streamlit dashboard"
	@echo "  build     : Build the Docker image for production"

install:
	poetry install

lint:
	poetry run ruff check src tests
	poetry run ruff format --check src tests

typecheck:
	poetry run mypy src

test:
	poetry run pytest tests

pipeline:
	poetry run python -m src.main

app:
	poetry run streamlit run src/app.py

build:
	docker build -t stock-forecaster:latest .
