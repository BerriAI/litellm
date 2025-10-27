# VIBECODER.md - LiteLLM macOS Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying and optimizing LiteLLM on macOS, with special focus on Apple Silicon (M1, M2, M3, M4) optimization. The optimizations in this codebase automatically detect Apple Silicon and apply platform-specific performance improvements.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Pre-Installation Checklist](#pre-installation-checklist)
3. [Installation Methods](#installation-methods)
4. [VS Code Integration](#vs-code-integration)
5. [Configuration](#configuration)
6. [Testing Procedures](#testing-procedures)
7. [Performance Verification](#performance-verification)
8. [Troubleshooting](#troubleshooting)
9. [Before Merging](#before-merging)

---

## System Requirements

### Minimum Requirements
- **OS**: macOS 11.0 (Big Sur) or later
- **CPU**: Intel Core i5 or Apple Silicon (M1/M2/M3/M4)
- **RAM**: 8 GB
- **Storage**: 5 GB free space
- **Python**: 3.8 - 3.12

### Recommended for Apple Silicon
- **OS**: macOS 13.0 (Ventura) or later
- **CPU**: Apple Silicon M1 or newer
- **RAM**: 16 GB or more
- **Storage**: 10 GB free space (for caching and models)
- **Python**: 3.11+ (native ARM64 build)

### Required Tools
- Homebrew package manager
- Git
- Python 3.8+
- pip/Poetry
- (Optional) Docker Desktop for Mac
- (Optional) VS Code or Cursor IDE

---

## Pre-Installation Checklist

### 1. Verify Python Architecture (Apple Silicon)

Ensure you're running native ARM64 Python for best performance:

```bash
# Check Python architecture
python3 -c "import platform; print(f'Architecture: {platform.machine()}')"
# Expected output on Apple Silicon: arm64
# On Intel Macs: x86_64

# Check Python version
python3 --version
# Expected: Python 3.8.0 or higher
```

### 2. Install Homebrew (if not already installed)

```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Verify installation
brew --version
```

### 3. Install System Dependencies

```bash
# Install required tools
brew install python@3.11 git poetry

# Optional but recommended
brew install redis postgresql ollama

# Verify installations
python3.11 --version
git --version
poetry --version
```

---

## Installation Methods

### Method 1: Development Installation (Recommended for Contributors)

This method is recommended if you're developing or modifying LiteLLM.

```bash
# 1. Clone the repository
git clone https://github.com/BerriAI/litellm.git
cd litellm

# 2. Checkout the optimization branch (if applicable)
git checkout claude/create-pull-request-011CUWyhiQqBmQWSC76xB41n

# 3. Install development dependencies
make install-dev

# 4. Install proxy dependencies with full features
make install-proxy-dev

# 5. Verify installation
python -c "import litellm; print(litellm.__version__)"
```

### Method 2: Poetry Installation (Clean Environment)

```bash
# 1. Clone the repository
git clone https://github.com/BerriAI/litellm.git
cd litellm

# 2. Install with Poetry
poetry install --extras "proxy"

# 3. Activate the virtual environment
poetry shell

# 4. Verify installation
litellm --version
```

### Method 3: Pip Installation (Standard)

```bash
# Install latest release
pip install 'litellm[proxy]'

# Or install from source
git clone https://github.com/BerriAI/litellm.git
cd litellm
pip install -e '.[proxy]'

# Verify installation
litellm --version
```

### Method 4: Docker Installation

```bash
# Pull the official image
docker pull ghcr.io/berriai/litellm:main-latest

# Or build locally with Apple Silicon optimizations
docker build --platform linux/arm64 -t litellm:macos .

# Run the container
docker run -p 4000:4000 \
  -e OPENAI_API_KEY=your_key_here \
  litellm:macos
```

---

## VS Code Integration

### Extension Installation

1. **Install Python Extension**
   ```
   - Open VS Code
   - Press Cmd+Shift+X (Extensions)
   - Search for "Python"
   - Install "Python" by Microsoft
   ```

2. **Install Optional Extensions**
   ```
   - REST Client (for testing API endpoints)
   - YAML (for config file editing)
   - GitLens (for git integration)
   ```

### Workspace Configuration

Create `.vscode/settings.json` in your LiteLLM directory:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true
  }
}
```

### Launch Configuration

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "LiteLLM Proxy Server",
      "type": "python",
      "request": "launch",
      "module": "litellm.proxy.proxy_cli",
      "args": [
        "--config",
        "litellm/proxy/example_config_yaml/macos_optimized_config.yaml",
        "--port",
        "4000",
        "--num_workers",
        "4"
      ],
      "console": "integratedTerminal",
      "env": {
        "OPENAI_API_KEY": "${env:OPENAI_API_KEY}",
        "ANTHROPIC_API_KEY": "${env:ANTHROPIC_API_KEY}",
        "LITELLM_LOG": "INFO"
      }
    },
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal"
    }
  ]
}
```

### Tasks Configuration

Create `.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Run Tests",
      "type": "shell",
      "command": "make test-unit",
      "group": {
        "kind": "test",
        "isDefault": true
      },
      "presentation": {
        "reveal": "always",
        "panel": "new"
      }
    },
    {
      "label": "Lint Code",
      "type": "shell",
      "command": "make lint",
      "group": "build"
    },
    {
      "label": "Start Proxy",
      "type": "shell",
      "command": "litellm --config litellm/proxy/example_config_yaml/macos_optimized_config.yaml",
      "isBackground": true
    }
  ]
}
```

---

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# Proxy Configuration
LITELLM_MASTER_KEY=sk-1234  # Admin key for proxy
LITELLM_LOG=INFO             # Log level (DEBUG, INFO, WARNING, ERROR)

# Apple Silicon Optimizations (auto-detected, but can override)
DEFAULT_NUM_WORKERS_LITELLM_PROXY=4  # Number of workers
AIOHTTP_CONNECTOR_LIMIT=256          # Connection pool size
AIOHTTP_KEEPALIVE_TIMEOUT=60         # Keepalive timeout (seconds)
AIOHTTP_TTL_DNS_CACHE=600           # DNS cache TTL (seconds)

# Performance Tuning
PYTHONOPTIMIZE=2                     # Enable optimizations
PYTHONDONTWRITEBYTECODE=1           # Skip .pyc files

# Database (optional)
DATABASE_URL=sqlite:///./litellm.db  # SQLite for development
# DATABASE_URL=postgresql://user:password@localhost/litellm  # PostgreSQL for production

# Caching (optional)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
```

### Configuration File

Use the optimized config for macOS:

```bash
# Copy the example config
cp litellm/proxy/example_config_yaml/macos_optimized_config.yaml config.yaml

# Edit with your API keys and models
code config.yaml  # or use any editor

# Key sections to customize:
# - model_list: Add your API keys and models
# - litellm_settings: Configure caching and callbacks
# - general_settings: Set master key and database
```

---

## Testing Procedures

### Pre-Deployment Testing

Run these tests before deploying to production:

#### 1. Unit Tests

```bash
# Run all unit tests
make test-unit

# Run specific test file
poetry run pytest tests/test_litellm/test_completion.py -v

# Run with coverage
poetry run pytest tests/ --cov=litellm --cov-report=html
```

#### 2. Integration Tests

```bash
# Run integration tests (requires API keys)
make test-integration

# Test specific provider
poetry run pytest tests/llm_translation/test_anthropic.py -v
```

#### 3. Linting and Type Checking

```bash
# Run all linting checks
make lint

# Individual checks
make lint-ruff   # Ruff linter
make lint-mypy   # Type checking
make format      # Black formatting
```

#### 4. Proxy Server Test

```bash
# Start the proxy server
litellm --config config.yaml --port 4000

# In another terminal, test the endpoint
curl http://localhost:4000/health

# Test a completion (requires OPENAI_API_KEY in .env)
curl -X POST http://localhost:4000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

#### 5. Platform Optimization Verification

```bash
# Check that Apple Silicon optimizations are active
litellm --config config.yaml --port 4000 | head -20

# Look for output like:
# ✓ Apple Silicon detected - using uvloop with ARM64 optimizations
# Platform: darwin (arm64)
# Recommended workers: 4
# Connection pool: 256
```

### Load Testing (Optional)

```bash
# Install load testing tools
pip install locust

# Run load tests
cd tests/load_tests
locust -f locustfile.py --host=http://localhost:4000
```

---

## Performance Verification

### Benchmarking Script

Create `benchmark.py`:

```python
import time
import asyncio
from litellm import completion

async def benchmark_completion():
    """Benchmark completion performance."""
    start = time.time()

    response = await completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=50
    )

    elapsed = time.time() - start
    print(f"Completion time: {elapsed:.2f}s")
    return elapsed

async def run_benchmark(iterations=10):
    """Run multiple iterations."""
    times = []

    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}")
        elapsed = await benchmark_completion()
        times.append(elapsed)
        await asyncio.sleep(1)

    avg = sum(times) / len(times)
    print(f"\nAverage time: {avg:.2f}s")
    print(f"Min time: {min(times):.2f}s")
    print(f"Max time: {max(times):.2f}s")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
```

Run the benchmark:

```bash
python benchmark.py
```

### Monitoring

#### System Resources

```bash
# Monitor CPU and memory usage
top -pid $(pgrep -f litellm)

# Or use Activity Monitor (GUI)
open -a "Activity Monitor"
```

#### Application Logs

```bash
# Start proxy with debug logging
LITELLM_LOG=DEBUG litellm --config config.yaml

# Monitor logs in real-time
tail -f litellm_proxy.log
```

#### Metrics (Optional)

Set up Prometheus/Grafana or use built-in callbacks:

```yaml
# In config.yaml
litellm_settings:
  success_callback: ["langfuse", "posthog"]
  failure_callback: ["sentry"]
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Error: ModuleNotFoundError: No module named 'litellm'
# Solution: Ensure you're in the virtual environment
poetry shell
# Or activate manually
source .venv/bin/activate
```

#### 2. Port Already in Use

```bash
# Error: Address already in use
# Solution: Kill the process or use a different port
lsof -ti:4000 | xargs kill -9
# Or use a different port
litellm --config config.yaml --port 4001
```

#### 3. API Key Issues

```bash
# Error: Authentication failed
# Solution: Check your .env file and environment variables
echo $OPENAI_API_KEY
# Make sure to source your .env file
export $(cat .env | xargs)
```

#### 4. Performance Issues on Apple Silicon

```bash
# Check Python architecture
python -c "import platform; print(platform.machine())"
# Should output 'arm64', not 'x86_64'

# If x86_64, reinstall Python:
brew reinstall python@3.11
```

#### 5. uvloop Not Available

```bash
# Error: No module named 'uvloop'
# Solution: Install uvloop
pip install uvloop

# Verify installation
python -c "import uvloop; print('uvloop installed')"
```

### Debug Mode

Enable comprehensive debugging:

```bash
# Set environment variables
export LITELLM_LOG=DEBUG
export LITELLM_PROXY_DEBUG=true

# Start proxy with verbose output
litellm --config config.yaml --debug
```

### Getting Help

- **Documentation**: https://docs.litellm.ai
- **GitHub Issues**: https://github.com/BerriAI/litellm/issues
- **Discord**: https://discord.com/invite/wuPM9dRgDw
- **Email**: support@berri.ai

---

## Before Merging

### Pre-Merge Checklist

Before merging this branch into main, complete the following:

- [ ] **All tests pass**: `make test`
- [ ] **Linting passes**: `make lint`
- [ ] **No breaking changes** to existing functionality
- [ ] **Documentation is updated**: README.md updated with changes
- [ ] **Example configs work**: Test `macos_optimized_config.yaml`
- [ ] **Platform detection works**: Test on both Intel and Apple Silicon Macs
- [ ] **Backward compatibility**: Verify non-macOS platforms still work
- [ ] **Performance benchmarks**: Document any performance improvements
- [ ] **Code review**: Have at least one reviewer approve changes
- [ ] **Changelog updated**: Add entry to CHANGELOG.md (if exists)

### Update README.md

**IMPORTANT**: Before final merge, update the main README.md with:

1. **Summary of macOS optimizations** in the features section
2. **Installation instructions** for macOS users
3. **Configuration examples** for Apple Silicon
4. **Performance notes** about platform-aware defaults
5. **Links to VIBECODER.md** and example configs

### Deployment Validation

Test the deployment on a clean macOS system:

```bash
# On a fresh macOS installation
git clone https://github.com/BerriAI/litellm.git
cd litellm
git checkout claude/create-pull-request-011CUWyhiQqBmQWSC76xB41n

# Follow installation steps from scratch
make install-proxy-dev

# Run tests
make test-unit

# Start proxy and verify platform detection
litellm --config litellm/proxy/example_config_yaml/macos_optimized_config.yaml
```

### Final Review

1. Review all modified files
2. Ensure comments are clear and helpful
3. Check for any hardcoded values that should be configurable
4. Verify error handling is comprehensive
5. Confirm logging is appropriate (not too verbose, not too quiet)

---

## Additional Resources

### macOS-Specific Tools

- **Ollama**: Local LLM server optimized for Apple Silicon
  - Install: `brew install ollama`
  - Start: `ollama serve`
  - Pull models: `ollama pull llama2`

- **Redis**: For caching (optional)
  - Install: `brew install redis`
  - Start: `brew services start redis`

- **PostgreSQL**: For production database (optional)
  - Install: `brew install postgresql@15`
  - Start: `brew services start postgresql@15`

### Performance Tips

1. **Use SSD for cache**: Apple Silicon Macs have fast SSDs - disk cache is very effective
2. **Monitor thermal throttling**: Use `sudo powermetrics` to check CPU performance
3. **Optimize Docker**: If using Docker, enable VirtioFS for better file system performance
4. **Use native builds**: Always prefer ARM64 builds over Rosetta 2 translation

### Security Notes

- Store API keys in environment variables or secure vaults (never commit to git)
- Use HTTPS in production deployments
- Enable firewall rules for the proxy port
- Regularly update dependencies for security patches

---

## Version Information

- **LiteLLM Version**: See `litellm.__version__`
- **Optimization Branch**: `claude/create-pull-request-011CUWyhiQqBmQWSC76xB41n`
- **Documentation Date**: 2025-10-27
- **Supported macOS**: 11.0+ (Big Sur and later)
- **Supported Python**: 3.8 - 3.12

---

## Contact

For questions or issues specific to macOS optimizations:
1. Open an issue on GitHub with the `macos` label
2. Include your system information (`sw_vers` and `python --version`)
3. Provide relevant logs with `LITELLM_LOG=DEBUG`

**Happy coding on macOS!** 🚀
