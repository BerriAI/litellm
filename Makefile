# LiteLLM Makefile
# Simple Makefile for running tests and basic development tasks

.PHONY: help test test-unit test-integration

# Default target
help:
	@echo "Available commands:"
	@echo "  make test               - Run all tests"
	@echo "  make test-unit          - Run unit tests"
	@echo "  make test-integration   - Run integration tests"

# Testing
test:
	poetry run pytest tests/

test-unit:
	poetry run pytest tests/litellm/

test-integration:
	poetry run pytest tests/ -k "not litellm" 