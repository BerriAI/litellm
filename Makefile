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
	@echo "  make review-bundle      - Create standard code review bundle (Markdown)"
	@echo "  make review-bundle-custom - Create custom ==== FILE style review bundle"
	@echo "  make smokes-all         - Run every smoke suite (shim-guarded)"

# Installation targets
install-dev:
	poetry install --with dev

install-proxy-dev:
	poetry install --with dev,proxy-dev --extras proxy

# CI-compatible installations (matches GitHub workflows exactly)
install-dev-ci:
	pip install openai==1.99.5
	poetry install --with dev
	pip install openai==1.99.5

install-proxy-dev-ci:
	poetry install --with dev,proxy-dev --extras proxy
	pip install openai==1.99.5

install-test-deps: install-proxy-dev
	poetry run pip install "pytest-retry==1.6.3"
	poetry run pip install pytest-xdist
	cd enterprise && python -m pip install -e . && cd ..

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

canary-run:
	@echo "Running single parity check"
	PYTHONPATH=$(PWD) python local/scripts/router_core_parity.py

canary-summarize:
	@echo "Summarizing parity JSONL"
	PYTHONPATH=$(PWD) python local/scripts/parity_summarize.py --in $${PARITY_OUT}

.PHONY: exec-rpc-up exec-rpc-restart exec-rpc-down exec-rpc-logs exec-rpc-probe

# Exec RPC service (Dockerized). Rebuilds from local tree and (re)starts on ${EXEC_RPC_PORT:-8790}.
exec-rpc-up:
	docker compose -f local/docker/compose.exec.yml up -d --build

exec-rpc-restart:
	docker compose -f local/docker/compose.exec.yml up -d --build

exec-rpc-down:
	docker compose -f local/docker/compose.exec.yml down

exec-rpc-logs:
	docker compose -f local/docker/compose.exec.yml logs -f exec-rpc

# Probe: health + python exec must include t_ms
exec-rpc-probe:
	./scripts/exec_rpc_probe.sh 127.0.0.1 $${EXEC_RPC_PORT:-8790} || true

.PHONY: review-bundle review-bundle-custom

review-bundle:
	@mkdir -p local/artifacts/review
	python local/scripts/review_bundle.py \
	  --files-from local/scripts/review/files.txt \
	  --prefix-file local/scripts/review/persona_and_rubric.md \
	  --output local/artifacts/review/review_bundle.md \
	  --single-file --token-estimator char || true

review-bundle-custom:
	@mkdir -p local/artifacts/review
	CONTEXT=local/artifacts/review/context_preface.txt OUT=local/artifacts/review/review_bundle.txt \
	python local/scripts/review/make_custom_bundle.py

.PHONY: e2e-up e2e-run e2e-down

# Bring up live services required for E2E (best-effort)
e2e-up: exec-rpc-up
	@echo "If needed, start the mini-agent app:"
	@echo "  uvicorn litellm.experimental_mcp_client.mini_agent.agent_proxy:app --host 0.0.0.0 --port 8788"

# Run the E2E nd-smokes (skips when services are missing)
e2e-run:
	pytest -q tests/ndsmoke_e2e -m ndsmoke || true

e2e-down: exec-rpc-down

# --- Dockerized mini-agent helpers -------------------------------------------
.PHONY: docker-up docker-down docker-logs ndsmoke-docker docker-ollama-up docker-ollama-down

docker-up:
	@docker network create llmnet >/dev/null 2>&1 || true
	API_PORT=$${API_PORT:-8788} API_CONTAINER_PORT=$${API_CONTAINER_PORT:-8788} \
		docker compose -f local/docker/compose.exec.yml up -d --build $${OLLAMA:+ollama} tools-stub agent-api
	@echo "Mini-agent API: http://127.0.0.1:$${API_PORT:-8788} (GET /ready)"

docker-down:
	@docker compose -f local/docker/compose.exec.yml down || true

docker-logs:
	@docker compose -f local/docker/compose.exec.yml logs -f agent-api

# Optional: bring up an Ollama daemon attached to llmnet for model inference
docker-ollama-up:
	@docker network create llmnet >/dev/null 2>&1 || true
	@docker run -d --name ollama --network llmnet -p 11434:11434 ollama/ollama || true
	@echo "Ollama base URL (host): $$([ -x scripts/resolve_ollama_base.sh ] && scripts/resolve_ollama_base.sh || echo http://127.0.0.1:11434)"

docker-ollama-down:
	@docker rm -f ollama >/dev/null 2>&1 || true

# Run only the Docker ndsmokes (skip-friendly). Defaults to codex loopback.
ndsmoke-docker:
	DOCKER_MINI_AGENT=1 \
	MINI_AGENT_API_HOST=$${MINI_AGENT_API_HOST:-127.0.0.1} \
	MINI_AGENT_API_PORT=$${MINI_AGENT_API_PORT:-8788} \
	LITELLM_ENABLE_CODEX_AGENT=1 \
	CODEX_AGENT_API_BASE=$${CODEX_AGENT_API_BASE:-http://127.0.0.1:8788} \
	LITELLM_DEFAULT_CODE_MODEL=$${LITELLM_DEFAULT_CODE_MODEL:-codex-agent/mini} \
	PYTHONPATH=$(PWD) pytest -q \
	  tests/ndsmoke/test_mini_agent_docker_ready.py \
	  tests/ndsmoke/test_codex_agent_docker_loopback_optional.py \
	  -q || true

.PHONY: ndsmoke-docker-live
ndsmoke-docker-live:
	DOCKER_MINI_AGENT=1 \
	MINI_AGENT_API_HOST=$${MINI_AGENT_API_HOST:-127.0.0.1} \
	MINI_AGENT_API_PORT=$${MINI_AGENT_API_PORT:-8788} \
	PYTHONPATH=$(PWD) pytest -q tests/ndsmoke/test_mini_agent_docker_live_optional.py -q || true

.PHONY: project-ready
project-ready:
	python scripts/mvp_check.py || true

.PHONY: project-ready-live
project-ready-live:
	READINESS_LIVE=1 STRICT_READY=1 READINESS_EXPECT=ollama,codex-agent,docker DOCKER_MINI_AGENT=1 python scripts/mvp_check.py || true
	python scripts/generate_project_ready.py || true

.PHONY: project-ready-summary
project-ready-summary:
	@if [ ! -f local/artifacts/mvp/mvp_report.json ]; then \
	  echo "No artifact found. Run \`make project-ready\` first."; \
	else \
	  jq -r '.checks[] | [.name,(.ok|tostring), (if has("skipped") then (.skipped|tostring) else "" end)] | @tsv' local/artifacts/mvp/mvp_report.json \
	    | awk 'BEGIN{FS="\t"} {em=$$2=="true"?"✅":($$3=="true"?"⏭":"❌"); printf("%-26s %s\n", $$1, em)}'; \
	fi

.PHONY: dump-readiness-env
dump-readiness-env:
	@echo "STRICT_READY=$${STRICT_READY:-0} READINESS_LIVE=$${READINESS_LIVE:-0} READINESS_EXPECT=$${READINESS_EXPECT:-} DOCKER_MINI_AGENT=$${DOCKER_MINI_AGENT:-0}"
	@echo "MINI_AGENT_API_HOST=$${MINI_AGENT_API_HOST:-127.0.0.1} MINI_AGENT_API_PORT=$${MINI_AGENT_API_PORT:-8788}"
	@echo "CODEX_AGENT_API_BASE=$${CODEX_AGENT_API_BASE:-auto} OLLAMA_API_BASE=$${OLLAMA_API_BASE:-http://127.0.0.1:11434}"

.PHONY: smokes-all
smokes-all:
	PYTHONPATH=$(PWD) python scripts/run_all_smokes.py || true

.PHONY: project-ready-all
project-ready-all:
	# Strict gate: split checks only (core + ND). Docker optional.
	READINESS_LIVE=1 STRICT_READY=1 READINESS_EXPECT=ollama,codex-agent,all_smokes_core,all_smokes_nd python scripts/mvp_check.py

.PHONY: project-ready-all-split-strict
project-ready-all-split-strict:
	READINESS_LIVE=1 STRICT_READY=1 READINESS_FAIL_ON_SKIP=1 READINESS_EXPECT=ollama,codex-agent,all_smokes_core,all_smokes_nd python scripts/mvp_check.py

.PHONY: project-ready-core-only-with-docker
project-ready-core-only-with-docker:
	READINESS_LIVE=1 STRICT_READY=1 READINESS_EXPECT=ollama,codex-agent,docker,all_smokes_core DOCKER_MINI_AGENT=1 python scripts/mvp_check.py

.PHONY: project-ready-nd-only-with-docker
project-ready-nd-only-with-docker:
	READINESS_LIVE=1 STRICT_READY=1 READINESS_EXPECT=ollama,codex-agent,docker,all_smokes_nd DOCKER_MINI_AGENT=1 python scripts/mvp_check.py

.PHONY: project-ready-core-only
project-ready-core-only:
	READINESS_LIVE=1 STRICT_READY=1 READINESS_EXPECT=ollama,codex-agent,all_smokes_core python scripts/mvp_check.py

.PHONY: project-ready-nd-only
project-ready-nd-only:
	READINESS_LIVE=1 STRICT_READY=1 READINESS_EXPECT=ollama,codex-agent,all_smokes_nd python scripts/mvp_check.py
