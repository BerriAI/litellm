# LiteLLM Makefile
# Simple Makefile for running tests and basic development tasks

.PHONY: help test test-unit test-integration lint format

# Default target
help:
	@echo "Available commands:"
	@echo "  make test               - Run all tests"
	@echo "  make test-unit          - Run unit tests"
	@echo "  make test-integration   - Run integration tests"
	@echo "  make test-unit-helm     - Run helm unit tests"

install-dev:
	poetry install --with dev

install-proxy-dev:
	poetry install --with dev,proxy-dev

lint: install-dev
	poetry run pip install types-requests types-setuptools types-redis types-PyYAML
	cd litellm && poetry run mypy . --ignore-missing-imports

# Testing
test:
	poetry run pytest tests/

test-unit:
	poetry run pytest tests/litellm/

test-integration:
	poetry run pytest tests/ -k "not litellm"

test-unit-helm:
	helm unittest -f 'tests/*.yaml' deploy/charts/litellm-helm