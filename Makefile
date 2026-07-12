# LiteLLM Makefile
# Simple Makefile for running tests and basic development tasks

.PHONY: help test test-unit test-unit-llms test-unit-proxy-guardrails test-unit-proxy-core test-unit-proxy-misc \
	test-unit-integrations test-unit-core-utils test-unit-other test-unit-root \
	test-proxy-unit-a test-proxy-unit-b test-integration test-unit-helm \
	info lint lint-dev lint-checks format \
	lint-basedpyright lint-e2e-basedpyright lint-basedpyright-budget-update lint-type-discipline lint-type-discipline-budget-update \
	lint-ruff-budget lint-ruff-budget-update lint-budget-update lint-gate \
	install-dev install-proxy-dev install-test-deps install-hooks \
	install-helm-unittest check-circular-imports check-import-safety pre-commit \
	lint-install lint-fetch-base

# Default target
help:
	@echo "Available commands:"
	@echo "  make install-dev        - Install development dependencies"
	@echo "  make install-proxy-dev  - Install proxy development dependencies"
	@echo "  make install-dev-ci     - Install dev dependencies (CI-compatible, pins OpenAI)"
	@echo "  make install-proxy-dev-ci - Install proxy dev dependencies (CI-compatible)"
	@echo "  make install-test-deps  - Install the full local test environment"
	@echo "  make install-helm-unittest - Install helm unittest plugin"
	@echo "  make install-hooks      - Install git hooks (Conventional Commits + Branches)"
	@echo "  make pre-commit         - Run CI-equivalent lint on staged files (run before committing)"
	@echo "  make format             - Apply ruff format code formatting"
	@echo "  make format-check       - Check ruff format code formatting (matches CI)"
	@echo "  make lint               - Run all linting (Ruff, basedpyright, format check, circular imports, import safety)"
	@echo "  make lint-ruff          - Run Ruff linting only"
	@echo "  make lint-basedpyright  - Run basedpyright strict, gated by per-rule error counts"
	@echo "  make lint-e2e-basedpyright - Run basedpyright over tests/e2e (zero errors allowed)"
	@echo "  make lint-basedpyright-budget-update - Ratchet basedpyright limits down by what this branch fixed"
	@echo "  make lint-format        - Check ruff format formatting (matches CI)"
	@echo "  make lint-ruff-budget - Gate the codebase total of each strict ruff rule against its limit"
	@echo "  make lint-gate        - Strict ruff gate in CI-parity mode (fetches staging, simulates the merge)"
	@echo "  make lint-ruff-budget-update - Ratchet ruff-strict-budget.json limits down by what this branch fixed"
	@echo "  make lint-budget-update - Ratchet all budgets down (ruff + type-discipline + basedpyright)"
	@echo "  make check-circular-imports - Check for circular imports"
	@echo "  make check-import-safety - Check import safety"
	@echo "  make test               - Run all tests"
	@echo "  make test-unit          - Run unit tests (tests/test_litellm)"
	@echo "  make test-unit-llms     - Run LLM provider tests (~225 files)"
	@echo "  make test-unit-proxy-guardrails - Run proxy guardrails+mgmt tests (~51 files)"
	@echo "  make test-unit-proxy-core - Run proxy auth+client+db+hooks tests (~52 files)"
	@echo "  make test-unit-proxy-misc - Run proxy misc tests (~77 files)"
	@echo "  make test-unit-integrations - Run integration tests (~60 files)"
	@echo "  make test-unit-core-utils - Run core utils tests (~32 files)"
	@echo "  make test-unit-other    - Run other tests (caching, responses, etc., ~69 files)"
	@echo "  make test-unit-root     - Run root-level tests (~34 files)"
	@echo "  make test-proxy-unit-a  - Run proxy_unit_tests (a-o, ~20 files)"
	@echo "  make test-proxy-unit-b  - Run proxy_unit_tests (p-z, ~28 files)"
	@echo "  make test-integration   - Run integration tests"
	@echo "  make test-unit-helm     - Run helm unit tests"

UV := uv
UV_RUN := $(UV) run --no-sync

LINT_DEP_INSTALL ?= install-dev
LINT_E2E_DEP_INSTALL ?= lint-install
LINT_DEP_BASE ?= lint-fetch-base
LINT_JOBS := $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
LINT_OUTPUT_SYNC := $(if $(filter output-sync,$(.FEATURES)),--output-sync=target,)

# Show info
info:
	@echo "UV: $(UV)"

# Installation targets
# --inexact: sync the locked deps without pruning anything already installed, so running
# a lint/format target doesn't tear the proxy extras (prisma, websockets, ...) out from
# under a dev's venv (CI installs its own env per job, so it is unaffected by this).
install-dev:
	$(UV) sync --inexact --frozen

install-proxy-dev:
	$(UV) sync --frozen --group proxy-dev --extra proxy

# CI-compatible installations (matches GitHub workflows exactly)
install-dev-ci:
	$(UV) sync --frozen

install-proxy-dev-ci:
	$(UV) sync --frozen --group proxy-dev --extra proxy

install-test-deps: install-proxy-dev
	$(UV) sync --frozen --all-groups --all-extras
	$(UV_RUN) prisma generate --schema litellm/proxy/schema.prisma

install-helm-unittest:
	helm plugin install https://github.com/helm-unittest/helm-unittest --version v0.4.4 || echo "ignore error if plugin exists"

# Install git hooks that enforce Conventional Commits and Conventional Branches.
# Opt-in: not chained into install-dev.
install-hooks:
	./scripts/install_git_hooks.sh

# Formatting
# Wrap width is ruff.toml's single source of truth (line-length = 120), shared by the
# formatter and the import sorter so there's no 88-vs-120 split to reconcile.
format: install-dev
	cd litellm && $(UV_RUN) ruff format --exclude '/enterprise/' . && cd ..

format-check: install-dev
	cd litellm && $(UV_RUN) ruff format --check --exclude '/enterprise/' . && cd ..

# Single fetch of the PR base so the delta-based gates below share one network round
# trip instead of each re-fetching when chained from `lint`.
lint-fetch-base:
	git fetch origin litellm_internal_staging

# Mirror test-linting.yml's lint job environment: the proxy-dev group plus a generated
# Prisma client, so basedpyright resolves the same modules CI does (without the generated
# client the DB wrappers typed against it degrade to Unknown, drifting the budget from
# CI's). --inexact tops up the venv instead of pruning the proxy extras gen:api and the
# running proxy need.
lint-install:
	$(UV) sync --inexact --frozen --group proxy-dev --group e2e-dev
	$(UV_RUN) python scripts/prisma_generate_if_needed.py

# Diff-scoped format check, identical to test-linting.yml's "Check ruff format" step:
# only the litellm Python files changed vs the base are checked, so a pre-existing
# format issue elsewhere doesn't block an unrelated commit.
lint-format-check-changed: $(LINT_DEP_INSTALL) $(LINT_DEP_BASE)
	@files=$$(git diff --name-only origin/litellm_internal_staging...HEAD -- 'litellm/**/*.py' | grep -v '^litellm/enterprise/' || true); \
	if [ -z "$$files" ]; then \
		echo "No changed litellm Python files to format-check."; \
	else \
		echo "$$files" | xargs $(UV_RUN) ruff format --check --exclude '/enterprise/'; \
	fi

# Linting targets
lint-ruff: $(LINT_DEP_INSTALL)
	cd litellm && $(UV_RUN) ruff check . && cd ..

# faster linter for developing ...
# inspiration from:
# https://github.com/astral-sh/ruff/discussions/10977
# https://github.com/astral-sh/ruff/discussions/4049
lint-format-changed: install-dev lint-fetch-base
	@git diff origin/litellm_internal_staging --unified=0 --no-color -- '*.py' | \
	perl -ne '\
		if (/^diff --git a\/(.*) b\//) { $$file = $$1; } \
		if (/^@@ .* \+(\d+)(?:,(\d+))? @@/) { \
			$$start = $$1; $$count = $$2 || 1; $$end = $$start + $$count - 1; \
			print "$$file:$$start:1-$$end:999\n"; \
		}' | \
		while read range; do \
			file="$${range%%:*}"; \
			lines="$${range#*:}"; \
			echo "Formatting $$file (lines $$lines)"; \
			$(UV_RUN) ruff format --range "$$lines" "$$file"; \
		done

lint-ruff-dev: install-dev lint-fetch-base
	@tmpfile=$$(mktemp /tmp/ruff-dev.XXXXXX) && \
	cd litellm && \
	($(UV_RUN) ruff check . --output-format=pylint || true) > "$$tmpfile" && \
	$(UV_RUN) diff-quality --violations=pylint "$$tmpfile" --compare-branch=origin/litellm_internal_staging && \
	cd .. ; \
	rm -f "$$tmpfile"

lint-ruff-FULL-dev: install-dev lint-fetch-base
	@files=$$(git diff --name-only origin/litellm_internal_staging -- '*.py'); \
	if [ -n "$$files" ]; then echo "$$files" | xargs $(UV_RUN) ruff check; \
	else echo "No changed .py files to check."; fi

lint-basedpyright: $(LINT_DEP_INSTALL) $(LINT_DEP_BASE)
	($(UV_RUN) basedpyright --outputjson || true) | $(UV_RUN) python scripts/type_check_gate.py --base origin/litellm_internal_staging

lint-e2e-basedpyright: $(LINT_E2E_DEP_INSTALL)
	$(UV_RUN) basedpyright tests/e2e

# Type-discipline budget (mutable collections / casts / type guards / kwargs /
# unexplained suppressions), the test-linting.yml step `make lint` used to omit.
lint-type-discipline: $(LINT_DEP_INSTALL) $(LINT_DEP_BASE)
	$(UV_RUN) python scripts/type_discipline_gate.py --base origin/litellm_internal_staging

# --update lowers each limit by what this branch fixed since its branch point, so
# it needs the base ref fetched to resolve the merge-base.
lint-basedpyright-budget-update: install-dev lint-fetch-base
	($(UV_RUN) basedpyright --outputjson || true) | $(UV_RUN) python scripts/type_check_gate.py --update

lint-format: format-check

lint-ruff-budget: install-dev
	$(UV_RUN) python scripts/ruff_strict_gate.py

# Strict gate, invoked the same way CI does in test-linting.yml so a local pass
# means the CI check will pass too.
lint-gate: $(LINT_DEP_INSTALL) $(LINT_DEP_BASE)
	$(UV_RUN) python scripts/ruff_strict_gate.py --base origin/litellm_internal_staging

lint-ruff-budget-update: install-dev lint-fetch-base
	$(UV_RUN) python scripts/ruff_strict_gate.py --update

lint-type-discipline-budget-update: install-dev lint-fetch-base
	$(UV_RUN) python scripts/type_discipline_gate.py --update

# Ratchet all budgets in one shot (ruff strict + type-discipline + basedpyright)
lint-budget-update: lint-ruff-budget-update lint-type-discipline-budget-update lint-basedpyright-budget-update

check-circular-imports: $(LINT_DEP_INSTALL)
	cd litellm && $(UV_RUN) python ../tests/documentation_tests/test_circular_imports.py && cd ..

check-import-safety: $(LINT_DEP_INSTALL)
	@$(UV_RUN) python -c "from litellm import *; print('[from litellm import *] OK! no issues!');" || (echo '🚨 import failed, this means you introduced unprotected imports! 🚨'; exit 1)

# Combined linting, isomorphic to test-linting.yml's lint job so a local pass means a
# green CI lint: it installs the same env (proxy-dev + generated Prisma client) and then
# runs the diff-scoped ruff format check, whole-tree ruff check, the strict-rule /
# type-discipline / basedpyright budgets as a delta vs the base, then the circular-import
# and import-safety checks. Steps that compare against the base resolve it the same way CI
# does (merge-base with origin/litellm_internal_staging). Setup (env sync, Prisma client,
# base fetch) runs once up front; the checks themselves are independent, so a sub-make
# fans them out with -j and the fast ones finish under basedpyright's shadow.
lint: lint-install lint-fetch-base
	$(MAKE) -j $(LINT_JOBS) $(LINT_OUTPUT_SYNC) LINT_DEP_INSTALL= LINT_E2E_DEP_INSTALL= LINT_DEP_BASE= lint-checks

lint-checks: lint-format-check-changed lint-ruff lint-gate lint-type-discipline lint-basedpyright lint-e2e-basedpyright check-circular-imports check-import-safety

# Faster linting for local development (only checks changed code)
lint-dev: lint-format-changed check-circular-imports check-import-safety

# Run the gating CI checks against your staged files right before committing. Mirrors
# test-linting.yml (Python), test-litellm-ui-build.yml's frontend-lint (dashboard), and
# check-ui-api-types.yml (API-type drift), skipping any whose files you didn't stage.
# Not auto-installed as a git hook so it never slows an unrelated human commit.
pre-commit:
	./scripts/pre_commit_lint.sh

# Testing targets
test: install-test-deps
	$(UV_RUN) pytest tests/

test-unit: install-test-deps
	$(UV_RUN) pytest tests/test_litellm -x -vv -n 4

# Matrix test targets (matching CI workflow groups)
test-unit-llms: install-test-deps
	$(UV_RUN) pytest tests/test_litellm/llms --tb=short -vv -n 4 --durations=20

test-unit-proxy-guardrails: install-test-deps
	$(UV_RUN) pytest tests/test_litellm/proxy/guardrails tests/test_litellm/proxy/management_endpoints tests/test_litellm/proxy/management_helpers --tb=short -vv -n 4 --durations=20

test-unit-proxy-core: install-test-deps
	$(UV_RUN) pytest tests/test_litellm/proxy/auth tests/test_litellm/proxy/client tests/test_litellm/proxy/db tests/test_litellm/proxy/hooks tests/test_litellm/proxy/policy_engine --tb=short -vv -n 4 --durations=20

test-unit-proxy-misc: install-test-deps
	$(UV_RUN) pytest tests/test_litellm/proxy/_experimental tests/test_litellm/proxy/agent_endpoints tests/test_litellm/proxy/anthropic_endpoints tests/test_litellm/proxy/common_utils tests/test_litellm/proxy/discovery_endpoints tests/test_litellm/proxy/experimental tests/test_litellm/proxy/google_endpoints tests/test_litellm/proxy/health_endpoints tests/test_litellm/proxy/image_endpoints tests/test_litellm/proxy/middleware tests/test_litellm/proxy/openai_files_endpoint tests/test_litellm/proxy/pass_through_endpoints tests/test_litellm/proxy/prompts tests/test_litellm/proxy/public_endpoints tests/test_litellm/proxy/response_api_endpoints tests/test_litellm/proxy/shutdown tests/test_litellm/proxy/spend_tracking tests/test_litellm/proxy/ui_crud_endpoints tests/test_litellm/proxy/vector_store_endpoints tests/test_litellm/proxy/test_*.py --tb=short -vv -n 4 --durations=20

test-unit-integrations: install-test-deps
	$(UV_RUN) pytest tests/test_litellm/integrations --tb=short -vv -n 4 --durations=20

test-unit-core-utils: install-test-deps
	$(UV_RUN) pytest tests/test_litellm/litellm_core_utils --tb=short -vv -n 2 --durations=20

test-unit-other: install-test-deps
	$(UV_RUN) pytest tests/test_litellm/caching tests/test_litellm/responses tests/test_litellm/secret_managers tests/test_litellm/vector_stores tests/test_litellm/a2a_protocol tests/test_litellm/anthropic_interface tests/test_litellm/completion_extras tests/test_litellm/containers tests/test_litellm/enterprise tests/test_litellm/experimental_mcp_client tests/test_litellm/google_genai tests/test_litellm/images tests/test_litellm/interactions tests/test_litellm/passthrough tests/test_litellm/router_strategy tests/test_litellm/router_utils tests/test_litellm/types --tb=short -vv -n 4 --durations=20

test-unit-root: install-test-deps
	$(UV_RUN) pytest tests/test_litellm/test_*.py --tb=short -vv -n 4 --durations=20

# Proxy unit tests (tests/proxy_unit_tests split alphabetically)
test-proxy-unit-a: install-test-deps
	$(UV_RUN) pytest tests/proxy_unit_tests/test_[a-o]*.py --tb=short -vv -n 2 --durations=20

test-proxy-unit-b: install-test-deps
	$(UV_RUN) pytest tests/proxy_unit_tests/test_[p-z]*.py --tb=short -vv -n 2 --durations=20

test-integration: install-test-deps
	$(UV_RUN) pytest tests/ -k "not test_litellm"

test-unit-helm: install-helm-unittest
	helm unittest -f 'tests/*.yaml' helm/litellm-helm

# LLM Translation testing targets
test-llm-translation: install-test-deps
	@echo "Running LLM translation tests..."
	@python .github/workflows/run_llm_translation_tests.py

test-llm-translation-single: install-test-deps
	@echo "Running single LLM translation test file..."
	@if [ -z "$(FILE)" ]; then echo "Usage: make test-llm-translation-single FILE=test_filename.py"; exit 1; fi
	@mkdir -p test-results
	$(UV_RUN) pytest tests/llm_translation/$(FILE) \
		--junitxml=test-results/junit.xml \
		-v --tb=short --maxfail=100 --timeout=300

test-llm-translation-flush-vcr-cache:
	$(UV_RUN) python tests/_flush_vcr_cache.py
