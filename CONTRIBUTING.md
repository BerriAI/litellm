# Contributing to LiteLLM

Thank you for your interest in contributing to LiteLLM! We welcome contributions of all kinds - from bug fixes and documentation improvements to new features and integrations.

## **Checklist before submitting a PR**

Here are the core requirements for any PR submitted to LiteLLM:

- [ ] **Sign the Contributor License Agreement (CLA)** - [see details](#contributor-license-agreement-cla)
- [ ] **Add testing** - Adding at least 1 test is a hard requirement - [see details](#adding-testing)
- [ ] **Ensure your PR passes all checks**:
  - [ ] [Unit Tests](#running-unit-tests) - `make test-unit`
  - [ ] [Linting / Formatting](#running-linting-and-formatting-checks) - `make lint`
- [ ] **Keep scope isolated** - Your changes should address 1 specific problem at a time

## **Contributor License Agreement (CLA)**

Before contributing code to LiteLLM, you must sign our [Contributor License Agreement (CLA)](https://cla-assistant.io/BerriAI/litellm). This is a legal requirement for all contributions to be merged into the main repository.

**Important:** We strongly recommend reviewing and signing the CLA before starting work on your contribution to avoid any delays in the PR process.

## Quick Start

### 1. Setup Your Local Development Environment

```bash
# Clone the repository
git clone https://github.com/BerriAI/litellm.git
cd litellm

# Create a new branch for your feature
git checkout -b your-feature-branch

# Install development dependencies
make install-dev

# Verify your setup works
make help
```

That's it! Your local development environment is ready.

### 2. Development Workflow

Here's the recommended workflow for making changes:

```bash
# Make your changes to the code
# ...

# Format your code (auto-fixes formatting issues)
make format

# Run all linting checks (matches CI exactly)
make lint

# Run unit tests to ensure nothing is broken
make test-unit

# Commit your changes
git add .
git commit -m "Your descriptive commit message"

# Push and create a PR
git push origin your-feature-branch
```

## Adding Testing

**Adding at least 1 test is a hard requirement for all PRs.**

### Where to Add Tests

Add your tests to the [`tests/test_litellm/` directory](https://github.com/BerriAI/litellm/tree/main/tests/test_litellm).

- This directory mirrors the structure of the `litellm/` directory
- **Only add mocked tests** - no real LLM API calls in this directory
- For integration tests with real APIs, use the appropriate test directories

### File Naming Convention

The `tests/test_litellm/` directory follows the same structure as `litellm/`:

- `litellm/proxy/caching_routes.py` → `tests/test_litellm/proxy/test_caching_routes.py`
- `litellm/utils.py` → `tests/test_litellm/test_utils.py`

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

Run all unit tests (uses parallel execution for speed):

```bash
make test-unit
```

Run specific test files:
```bash
poetry run pytest tests/test_litellm/test_your_file.py -v
```

### Running Linting and Formatting Checks

Run all linting checks (matches CI exactly):

```bash
make lint
```

Individual linting commands:
```bash
make format-check       # Check Black formatting
make lint-ruff          # Run Ruff linting
make lint-mypy          # Run MyPy type checking
make check-circular-imports    # Check for circular imports
make check-import-safety       # Check import safety
```

Apply formatting (auto-fixes issues):
```bash
make format
```

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
make install-test-deps         # Install test dependencies (for running tests)
make format                    # Apply Black code formatting
make format-check              # Check Black formatting (matches CI)
make lint                      # Run all linting checks
make test-unit                 # Run unit tests
make test-integration          # Run integration tests
make test-unit-helm            # Run Helm unit tests
```

## Code Quality Standards

LiteLLM follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

Our automated quality checks include:
- **Black** for consistent code formatting
- **Ruff** for linting and code quality
- **MyPy** for static type checking
- **Circular import detection**
- **Import safety validation**

All checks must pass before your PR can be merged.

## Common Issues and Solutions

### 1. Linting Failures

If `make lint` fails:

1. **Formatting issues**: Run `make format` to auto-fix
2. **Ruff issues**: Check the output and fix manually
3. **MyPy issues**: Add proper type hints
4. **Circular imports**: Refactor import dependencies
5. **Import safety**: Fix any unprotected imports

### 2. Test Failures

If `make test-unit` fails:

1. Check if you broke existing functionality
2. Add tests for your new code
3. Ensure tests use mocks, not real API calls
4. Check test file naming conventions

### 3. Common Development Tips

- **Use type hints**: MyPy requires proper type annotations
- **Write descriptive commit messages**: Help reviewers understand your changes
- **Keep PRs focused**: One feature/fix per PR
- **Test edge cases**: Don't just test the happy path
- **Update documentation**: If you change APIs, update docs

## Building and Running Locally

### LiteLLM Proxy Server

To run the proxy server locally:

```bash
# Install proxy dependencies
make install-proxy-dev

# Start the proxy server
poetry run litellm --config your_config.yaml
```

### Docker Development

If you want to build the Docker image yourself:

```bash
# Build using the non-root Dockerfile
docker build -f docker/Dockerfile.non_root -t litellm_dev .

# Run with your config
docker run \
    -v $(pwd)/proxy_config.yaml:/app/config.yaml \
    -e LITELLM_MASTER_KEY="sk-1234" \
    -p 4000:4000 \
    litellm_dev \
    --config /app/config.yaml --detailed_debug
```

## Submitting Your PR

1. **Push your branch**: `git push origin your-feature-branch`
2. **Create a PR**: Go to GitHub and create a pull request
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