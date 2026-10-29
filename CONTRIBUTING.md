# Contributing to LiteLLM

Thank you for your interest in contributing to LiteLLM! We welcome contributions of all kinds - from bug fixes and documentation improvements to new features and integrations.

## **Checklist before submitting a PR**

Here are the core requirements for any PR submitted to LiteLLM:

- [ ] **Sign the Contributor License Agreement (CLA)** - [see details](#contributor-license-agreement-cla)
- [ ] **Keep scope isolated** - Your changes should address 1 specific problem at a time
- [ ] **Prove it works end to end** - If your change affects what the proxy or SDK does, paste the commands you ran against a live proxy and their output in the PR - [see details](#proving-your-change-works-end-to-end)

#### Proxy (Backend) PRs

- [ ] **Add testing** - Adding at least 1 test is a hard requirement - [see details](#adding-testing)
- [ ] **Ensure your PR passes all checks**:
  - [ ] [Linting / Formatting](#running-linting-and-formatting-checks) - `make lint`
  - [ ] [The tests covering your change](#running-unit-tests) pass, e.g. `uv run pytest tests/unit/<your_test_file>.py -v`. CI runs the full unit test matrix, so you don't need to run the whole suite locally

#### UI PRs

- [ ] **Ensure the UI builds successfully** - `npm run build`
- [ ] **Ensure all UI unit tests pass** - `npm run test`
- [ ] **Add tests for new components or logic** - If you are adding a new component or new logic, add corresponding tests

## **Contributor License Agreement (CLA)**

Before contributing code to LiteLLM, you must sign our [Contributor License Agreement (CLA)](https://cla-assistant.io/BerriAI/litellm). This is a legal requirement for all contributions to be merged into the main repository.

**Important:** We strongly recommend reviewing and signing the CLA before starting work on your contribution to avoid any delays in the PR process.

## Quick Start

### 1. Setup Your Local Development Environment

```bash
# Fork the repository on GitHub (click the Fork button at https://github.com/BerriAI/litellm)
# Then clone your fork locally
git clone https://github.com/YOUR_USERNAME/litellm.git
cd litellm

# Create a new branch for your feature (see "Commit and Branch Conventions" below)
git checkout -b litellm_your_feature

# Install development dependencies
make install-dev

# Install git hooks that enforce commit conventions (one-time, opt-in)
make install-hooks

# Verify your setup works
make help
```

That's it! Your local development environment is ready.

## Commit and Branch Conventions

Commits follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/). Internal contributors name branches with the `litellm_` prefix and no `/`, and fork branches can use any name. Run `make install-hooks` once per clone to enable the local hook that enforces the commit convention. See the [contributor docs](https://docs.litellm.ai/docs/extras/contributing_code#commit-and-branch-conventions) for details

### 2. Development Workflow

Here's the recommended workflow for making changes:

```bash
# Make your changes to the code
# ...

# Format your code (auto-fixes formatting issues)
make format

# Run all linting checks (matches CI exactly)
make lint

# Run the tests covering your change (CI runs the full suite)
uv run pytest tests/unit/<your_test_file>.py -v

# Commit your changes (must follow Conventional Commits — see above)
git add .
git commit -m "feat(scope): your descriptive commit message"

# Push and create a PR
git push origin litellm_your_feature
```

## Adding Testing

**Adding at least 1 test is a hard requirement for all PRs.**

### Where to Add Tests

Add your tests to the [`tests/unit/` directory](https://github.com/BerriAI/litellm/tree/main/tests/unit).

- This directory mirrors the structure of the `litellm/` directory
- **Only add mocked tests** - no real LLM API calls in this directory
- For integration tests with real APIs, use the appropriate test directories

Tests that call `litellm.register_model` or change `litellm.model_cost` should use the `local_model_cost_map` fixture from `tests/unit/conftest.py`. It provides a fresh catalog and restores the previous map, so pricing and capability overrides cannot leak into later tests

### File Naming Convention

The `tests/unit/` directory follows the same structure as `litellm/`:

- `litellm/proxy/caching_routes.py` → `tests/unit/proxy/test_caching_routes.py`
- `litellm/utils.py` → `tests/unit/test_utils.py`

### Example Test

```python
import pytest
from litellm import completion

def test_your_feature():
    """Test your feature with a descriptive docstring."""
    # Arrange
    messages = [{"role": "user", "content": "Hello"}]
    
    # Act
    # Use mocked responses, not real API calls
    
    # Assert
    assert expected_result == actual_result
```

## Running Tests and Checks

### Running Unit Tests

Run the tests covering your change:
```bash
uv run pytest tests/unit/test_your_file.py -v
```

`tests/unit` holds thousands of tests, so running all of it locally takes a long time. CI runs it as a parallel matrix (`make test-unit-llms`, `make test-unit-proxy-core`, and the other `test-unit-*` targets) on beefier boxes, so if, for whatever reason, you must run the whole suite, it's better to rely on CI to do that.

If you're running broader test suites, proxy tests, or anything that touches PostgreSQL-backed fixtures/plugins, install the full local test environment first:

```bash
make install-test-deps
```

This syncs the locked test environment used across the repo, including `psycopg` v3 plus `psycopg-binary`, `psycopg2-binary` (used by some proxy E2E tests), and a generated Prisma client for DB-backed proxy tests, so pytest startup matches CI without manual package installs.

### Running Linting and Formatting Checks

Run all linting checks (matches CI exactly):

```bash
make lint
```

Individual linting commands:
```bash
make format-check       # Check ruff format formatting
make lint-ruff          # Run Ruff linting
make lint-basedpyright  # Run basedpyright type checking
make check-circular-imports    # Check for circular imports
make check-import-safety       # Check import safety
```

Apply formatting (auto-fixes issues):
```bash
make format
```

> **Formatting is enforced in CI.** All PRs must pass the `ruff format --check` step.
>
> - **AI coding agents** (Claude Code, Copilot, Cursor, etc.): follow `AGENTS.md` and run `make format` before committing.
> - **VS Code users**: Install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) and enable format-on-save:
>   ```json
>   {
>     "[python]": {
>       "editor.defaultFormatter": "charliermarsh.ruff",
>       "editor.formatOnSave": true
>     }
>   }
>   ```

### CI Compatibility

To ensure your changes will pass CI, run the exact same checks locally:

```bash
# This runs the same checks as the GitHub workflows
make lint
make test-unit
```

For exact CI compatibility (pins OpenAI version like CI):
```bash
make install-dev-ci     # Installs exact CI dependencies
```

## Available Make Commands

Run `make help` to see all available commands:

```bash
make help                       # Show all available commands
make install-dev               # Install development dependencies
make install-proxy-dev         # Install proxy development dependencies
make install-test-deps         # Install the full local test environment
make format                    # Apply ruff format code formatting
make format-check              # Check ruff format formatting (matches CI)
make lint                      # Run all linting checks
make test-unit                 # Run unit tests
make test-integration          # Run integration tests
make test-unit-helm            # Run Helm unit tests
```

## Code Quality Standards

LiteLLM follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

Our automated quality checks include:
- **Ruff** for formatting, linting, and code quality
- **basedpyright** for static type checking
- **Circular import detection**
- **Import safety validation**

All checks must pass before your PR can be merged.

## Common Issues and Solutions

### 1. Linting Failures

If `make lint` fails:

1. **Formatting issues**: Run `make format` to auto-fix
2. **Ruff issues**: Check the output and fix manually
3. **basedpyright issues**: Add proper type hints
4. **Circular imports**: Refactor import dependencies
5. **Import safety**: Fix any unprotected imports

### 2. Test Failures

If `make test-unit` fails:

1. Check if you broke existing functionality
2. Add tests for your new code
3. Ensure tests use mocks, not real API calls
4. Check test file naming conventions

### 3. Common Development Tips

- **Use type hints**: basedpyright requires proper type annotations
- **Write descriptive commit messages**: Help reviewers understand your changes
- **Keep PRs focused**: One feature/fix per PR
- **Test edge cases**: Don't just test the happy path
- **Update documentation**: If you change APIs, update docs

## Proving Your Change Works End to End

Tests are required, but they do not prove that your change works for a user. A PR that changes what the proxy or SDK does shows a real run in its "Screenshots / Proof of Fix" section, meaning the exact commands you sent to a live proxy and the output that came back, against real provider APIs and not mocks. `pytest` output does not count as proof, because a passing test only shows that the code does what its mocks expect. A PR that changes only docs or CI shows the rendered page or the CI run instead

A good proof looks like this:

1. Install the proxy dependencies with `make install-proxy-dev`, then start the proxy from your branch with a config that has the model you changed, `uv run litellm --config your_config.yaml --detailed_debug`
2. Send the request a user would send with `curl`, and paste both the command and the response. When cost or routing is involved, include the response headers, since `x-litellm-response-cost` is what the user sees
3. For a bug fix, do this twice, once at the merge base with BerriAI/litellm's main (`git merge-base origin/main HEAD`, with your remote for BerriAI/litellm in place of `origin` in a fork clone) to show the bug and once at your branch's latest commit to show the fix. Name both commit hashes
4. If your change is reachable from more than one endpoint (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`), show each one
5. If the request can stream, show a `"stream": true` run too, because streaming builds usage and cost on a different path. A stream sends its headers before its usage exists, so show the `usage.cost` on the last chunk, which `/v1/chat/completions` sends when the request has `"stream_options": {"include_usage": true}` and other endpoints send with `include_cost_in_streaming_usage: true` under `litellm_settings`, or, when the proxy has a database, the request's row on the Admin UI's Logs page or `GET /spend/logs`
6. For UI changes, add before and after screenshots and name the page
7. If users reach your change through a coding tool such as Claude Code or Codex, drive that tool the way a user does and add before and after screenshots of it, because a `curl` or a headless run alone does not show what the user sees

```bash
curl -sD - http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "your-model", "messages": [{"role": "user", "content": "hi"}]}'
```

When the provider reports its own usage or spend numbers, compare the proxy's numbers against them in the proof

## Contributing an LLM Provider Integration

Provider work (a new provider, a new parameter, a new usage or cost field) is the most common kind of contribution, and it is where PRs get reworked most often. Follow these rules

**Find the existing machinery before you add plumbing.** LiteLLM already has generic paths that move data between the endpoints, the provider transformations, and cost tracking. A second, hand-rolled path for the same data is a common reason a provider PR gets rewritten before merge. [`ARCHITECTURE.md`](ARCHITECTURE.md) shows where the layers live, and the decision records in [`docs/adrs/`](docs/adrs/AGENTS.md) explain why they are shaped that way. For example, [the ADR on provider usage fields](docs/adrs/adr_001_provider_usage_fields_ride_as_usage_extras.md) covers how provider-specific usage fields reach cost tracking. If you cannot find a mechanism for what you need, say so in the PR or ask in [#pr-review on Slack](https://join.slack.com/t/litellmossslack/shared_invite/zt-3o7nkuyfr-p_kbNJj8taRfXGgQI1~YyA) before you build one

**Bill a new provider usage field from the usage object.** Leave the field on the usage object as an extra and read it in that provider's `cost_calculator.py`. A provider without one gets a new `cost_calculator.py` and an arm for it in `cost_per_token` (`litellm/cost_calculator.py`), which picks the calculator by provider. If the provider's transform builds `Usage` from named fields, as the Anthropic, Gemini, and Bedrock Converse ones do, pass the new field there too. Streamed requests rebuild usage in `ChunkProcessor.calculate_usage` (`litellm/litellm_core_utils/streaming_chunk_builder_utils.py`), which does not copy extras yet the way the Responses-to-chat bridge does (minus a short exclusion list), so until it does, bill the field on streams by naming it there, as Anthropic's `inference_geo` is. A provider that reports its own charge for the request, as OpenRouter does in `usage.cost`, is billed at that charge from `_hidden_params` before any calculator runs, unless the deployment sets its own prices. When the field counts a built-in tool such as web search, add the provider to `get_cost_for_web_search_request` in `litellm/llms/__init__.py`, and add a gate to `litellm/litellm_core_utils/llm_cost_calc/tool_call_cost_tracking.py` only when no existing gate fires. Then show the field's cost in your proof on every endpoint it reaches, streaming included

**Do not change what a caller sees to make internals easier.** Each endpoint keeps the schema of the API it emulates. A `/v1/responses` caller gets `input_tokens` and a `/v1/chat/completions` caller gets `prompt_tokens`, whatever the provider sent and whatever cost tracking needs. A provider usage field that shows up as an extra key is fine, since it only adds a key, but renaming, removing, or reshaping a key the endpoint defines is not. Reshaping a public response to feed an internal consumer breaks every user of that endpoint

If your change makes an architecture decision, such as a new cross-cutting mechanism or a deliberate difference from a provider's API shape, add an ADR to [`docs/adrs/`](docs/adrs/AGENTS.md) in the same PR

## Building and Running Locally

### LiteLLM Proxy Server

To run the proxy server locally:

```bash
# Install proxy dependencies
make install-proxy-dev

# Start the proxy server
uv run litellm --config your_config.yaml
```

### Docker Development

If you want to build the Docker image yourself:

```bash
# Build using the non-root Dockerfile
docker build -f docker/Dockerfile.non_root -t litellm_dev .

# Generate a master key. Requests send it as the bearer token
export LITELLM_MASTER_KEY="sk-$(openssl rand -hex 32)"

# Run with your config
docker run \
    -v $(pwd)/proxy_config.yaml:/app/config.yaml \
    -e LITELLM_MASTER_KEY \
    -p 4000:4000 \
    litellm_dev \
    --config /app/config.yaml --detailed_debug
```

## UI Development

### 1. Setup Your Local UI Development Environment

```bash
# Clone the repo (if you haven't already)
git clone https://github.com/YOUR_USERNAME/litellm.git
cd litellm

# Navigate to the UI dashboard directory
cd ui/litellm-dashboard

# Install dependencies
npm install

# Start the development server
npm run dev
```

### 2. Adding UI Tests

If you are adding a **new component** or **new logic**, you must add corresponding tests.

### 3. Running UI Unit Tests

```bash
npm run test
```

### 4. Building the UI

Ensure the UI builds successfully before submitting your PR:

```bash
npm run build
```

Local lint and budget checks follow origin's current default branch. They refresh it from the remote instead of trusting cached `origin/HEAD`. For an intentional comparison against another branch or commit, use `make check BASE_REF=<ref>` or the standalone gate's `--base <ref>` option. An explicit ref can also be used offline once it has been fetched locally. Without an override, unavailable remote metadata stops the check

## Submitting Your PR

1. **Push your branch**: `git push origin your-feature-branch`
2. **Create a PR**: Go to GitHub and open a pull request against the repository's current default branch. Run `python3 scripts/default_branch.py --branch` to check its name
3. **Fill out the PR template**: Provide clear description of changes
4. **Wait for review**: Maintainers will review and provide feedback
5. **Address feedback**: Make requested changes and push updates
6. **Merge**: Once approved, your PR will be merged!

## Getting Help

If you need help:

- 💬 [Join our Discord](https://discord.gg/wuPM9dRgDw)
- 💬 [Join our Slack](https://www.litellm.ai/support)
- 📧 Email us: ishaan@berri.ai / krrish@berri.ai
- 🐛 [Create an issue](https://github.com/BerriAI/litellm/issues/new)

## What to Contribute

Looking for ideas? Check out:

- 🐛 [Good first issues](https://github.com/BerriAI/litellm/labels/good%20first%20issue)
- 🚀 [Feature requests](https://github.com/BerriAI/litellm/labels/enhancement)
- 📚 Documentation improvements
- 🧪 Test coverage improvements
- 🔌 New LLM provider integrations

Thank you for contributing to LiteLLM! 🚀 
