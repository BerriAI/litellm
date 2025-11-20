# LiteLLM Makefile
# Simple Makefile for running tests and basic development tasks

.PHONY: help test test-unit test-integration test-unit-helm lint format install-dev install-proxy-dev install-test-deps install-helm-unittest check-circular-imports check-import-safety

# Default target
help:
	@echo "Available commands:"
	@echo "  make install-dev        - Install development dependencies"
	@echo "  make install-proxy-dev  - Install proxy development dependencies"
	@echo "  make install-dev-ci     - Install dev dependencies (CI-compatible, pins OpenAI)"
	@echo "  make install-proxy-dev-ci - Install proxy dev dependencies (CI-compatible)"
	@echo "  make install-test-deps  - Install test dependencies"
	@echo "  make install-helm-unittest - Install helm unittest plugin"
	@echo "  make format             - Apply Black code formatting"
	@echo "  make format-check       - Check Black code formatting (matches CI)"
	@echo "  make lint               - Run all linting (Ruff, MyPy, Black check, circular imports, import safety)"
	@echo "  make lint-ruff          - Run Ruff linting only"
	@echo "  make lint-mypy          - Run MyPy type checking only"
	@echo "  make lint-black         - Check Black formatting (matches CI)"
	@echo "  make check-circular-imports - Check for circular imports"
	@echo "  make check-import-safety - Check import safety"
	@echo "  make test               - Run all tests"
	@echo "  make test-unit          - Run unit tests (tests/test_litellm)"
	@echo "  make test-integration   - Run integration tests"
	@echo "  make test-unit-helm     - Run helm unit tests"

# Installation targets
install-dev:
	poetry install --with dev

install-proxy-dev:
	poetry install --with dev,proxy-dev --extras proxy

# CI-compatible installations (matches GitHub workflows exactly)
install-dev-ci:
	pip install openai==2.8.0
	poetry install --with dev
	pip install openai==2.8.0

install-proxy-dev-ci:
	poetry install --with dev,proxy-dev --extras proxy
	pip install openai==2.8.0

install-test-deps: install-proxy-dev
	poetry run pip install "pytest-retry==1.6.3"
	poetry run pip install pytest-xdist
	cd enterprise && poetry run pip install -e . && cd ..

install-helm-unittest:
	helm plugin install https://github.com/helm-unittest/helm-unittest --version v0.4.4 || echo "ignore error if plugin exists"

# Formatting
format: install-dev
	cd litellm && poetry run black . && cd ..

format-check: install-dev
	cd litellm && poetry run black --check . && cd ..

# Linting targets
lint-ruff: install-dev
	cd litellm && poetry run ruff check . && cd ..

lint-mypy: install-dev
	poetry run pip install types-requests types-setuptools types-redis types-PyYAML
	cd litellm && poetry run mypy . --ignore-missing-imports && cd ..

lint-black: format-check

check-circular-imports: install-dev
	cd litellm && poetry run python ../tests/documentation_tests/test_circular_imports.py && cd ..

check-import-safety: install-dev
	poetry run python -c "from litellm import *" || (echo '🚨 import failed, this means you introduced unprotected imports! 🚨'; exit 1)

# Combined linting (matches test-linting.yml workflow)
lint: format-check lint-ruff lint-mypy check-circular-imports check-import-safety

# Testing targets
test:
	poetry run pytest tests/

test-unit: install-test-deps
	poetry run pytest tests/test_litellm -x -vv -n 4

test-integration:
	poetry run pytest tests/ -k "not test_litellm"

test-unit-helm: install-helm-unittest
	helm unittest -f 'tests/*.yaml' deploy/charts/litellm-helm

# LLM Translation testing targets
test-llm-translation: install-test-deps
	@echo "Running LLM translation tests..."
	@python .github/workflows/run_llm_translation_tests.py

test-llm-translation-single: install-test-deps
	@echo "Running single LLM translation test file..."
	@if [ -z "$(FILE)" ]; then echo "Usage: make test-llm-translation-single FILE=test_filename.py"; exit 1; fi
	@mkdir -p test-results
	poetry run pytest tests/llm_translation/$(FILE) \
		--junitxml=test-results/junit.xml \
		-v --tb=short --maxfail=100 --timeout=300