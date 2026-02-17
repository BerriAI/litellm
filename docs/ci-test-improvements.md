# CI Test Improvements - Implementation Plan

## Current Issues

1. Tests pass locally but fail in CI
2. Async mock timing issues
3. Module reload causing `isinstance()` failures

## Implemented Solutions

### ✅ Already In Place

1. **`@pytest.mark.no_parallel` marker** - Line 200 in pyproject.toml
   - Use for tests with async mocks or timing dependencies
   - MCP OAuth test already uses this correctly

2. **Retry logic** - Lines 195-196 in pyproject.toml
   ```toml
   retries = 20
   retry_delay = 5
   ```

3. **Async mode** - Line 194 in pyproject.toml
   ```toml
   asyncio_mode = "auto"
   ```

### 🔧 Recommended Additions

## 1. Install pytest-rerunfailures Plugin

Add to `pyproject.toml`:

```toml
[tool.poetry.group.dev.dependencies]
pytest-rerunfailures = "^15.0"
```

Then use in tests:

```python
@pytest.mark.flaky(reruns=3, reruns_delay=1)
async def test_sometimes_flaky():
    # Test code
    pass
```

## 2. Add Test Categories

Update `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "asyncio: mark test as an asyncio test",
    "limit_leaks: mark test with memory limit for leak detection",
    "no_parallel: mark test to run sequentially (not in parallel)",
    "flaky: mark test as potentially flaky (auto-retry)",  # NEW
    "requires_credentials: mark test that needs external credentials",  # NEW
    "slow: mark test as slow running (> 10 seconds)",  # NEW
]
```

## 3. GitHub Actions Improvements

### Option A: Run Flaky Tests Separately

`.github/workflows/test.yml`:

```yaml
jobs:
  test-stable:
    name: Stable Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Stable Tests
        run: |
          poetry run pytest tests/ \
            -m "not flaky" \
            -n 4 \
            --maxfail=5

  test-flaky:
    name: Flaky Tests (with retries)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Flaky Tests
        run: |
          poetry run pytest tests/ \
            -m "flaky" \
            --reruns 3 \
            --reruns-delay 2 \
            -n 1  # Sequential for flaky tests
```

### Option B: Add Retry Step

```yaml
- name: Run Tests
  id: tests
  run: poetry run pytest tests/ -v -n 4

- name: Retry Failed Tests
  if: failure() && steps.tests.outcome == 'failure'
  run: |
    echo "Retrying failed tests..."
    poetry run pytest tests/ --lf -v --maxfail=5
```

## 4. Pre-commit Hook for Test Best Practices

`.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: check-async-test-markers
        name: Check async tests have proper markers
        entry: python scripts/check_test_markers.py
        language: python
        files: ^tests/.*test.*\.py$
```

`scripts/check_test_markers.py`:

```python
#!/usr/bin/env python3
import ast
import sys
import re

def check_test_file(filename):
    """Check that async tests with mocks have no_parallel marker"""
    with open(filename) as f:
        content = f.read()

    # Parse AST
    tree = ast.parse(content)

    issues = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name.startswith('test_'):
            # Check if it has @patch or Mock
            has_mock = 'patch' in content or 'Mock' in content or 'mock_' in node.name.lower()

            # Check if it has @pytest.mark.no_parallel
            decorators = [d for d in node.decorator_list]
            has_no_parallel = any(
                'no_parallel' in ast.unparse(d) for d in decorators
            )

            if has_mock and not has_no_parallel:
                issues.append(
                    f"{filename}:{node.lineno} - Async test '{node.name}' "
                    f"uses mocks but missing @pytest.mark.no_parallel"
                )

    return issues

if __name__ == '__main__':
    all_issues = []
    for filename in sys.argv[1:]:
        all_issues.extend(check_test_file(filename))

    if all_issues:
        print("❌ Test marker issues found:")
        for issue in all_issues:
            print(f"  {issue}")
        sys.exit(1)
    else:
        print("✅ All async tests properly marked")
        sys.exit(0)
```

## 5. Makefile Targets for Testing

`Makefile`:

```makefile
.PHONY: test test-fast test-flaky test-repeat

# Regular test run
test:
	poetry run pytest tests/ -v

# Fast: parallel execution, stop on first failure
test-fast:
	poetry run pytest tests/ -n 4 -x

# Only flaky tests with retries
test-flaky:
	poetry run pytest tests/ -m flaky --reruns 3 --reruns-delay 1

# Repeat test N times to catch flakiness
test-repeat:
	poetry run pytest tests/ --count=100 -x

# Test specific file 100 times
test-file-repeat:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make test-file-repeat FILE=tests/path/to/test.py"; \
		exit 1; \
	fi
	for i in {1..100}; do \
		echo "Run $$i/100"; \
		poetry run pytest $(FILE) || exit 1; \
	done
```

Usage:
```bash
make test-fast                              # Quick CI-like run
make test-flaky                             # Test only flaky tests
make test-file-repeat FILE=tests/test_mcp_server.py  # Stress test
```

## 6. Test Utilities Module

`tests/test_utils.py`:

```python
"""Common test utilities for handling flakiness"""
import asyncio
import functools
import os
import pytest
from typing import Callable, TypeVar

T = TypeVar('T')

def retry_on_assertion_error(max_attempts: int = 3, delay: float = 0.1):
    """Retry async test on AssertionError (for flaky CI tests)"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except AssertionError as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        print(f"  ⚠️  Attempt {attempt + 1} failed, retrying...")
                        await asyncio.sleep(delay)
            raise last_error
        return wrapper
    return decorator

def skip_in_ci(reason: str = "Flaky in CI"):
    """Skip test in CI environment"""
    is_ci = os.getenv('CI') == 'true' or os.getenv('GITHUB_ACTIONS') == 'true'
    return pytest.mark.skipif(is_ci, reason=reason)

def requires_credentials(*env_vars: str):
    """Skip test if required credentials are missing"""
    has_creds = all(os.getenv(var) for var in env_vars)
    var_names = ', '.join(env_vars)
    return pytest.mark.skipif(
        not has_creds,
        reason=f"Missing credentials: {var_names}"
    )

# Usage examples:
# @retry_on_assertion_error(max_attempts=3)
# async def test_flaky():
#     ...
#
# @skip_in_ci("Known timing issue in CI - tracked in #12345")
# async def test_problematic():
#     ...
#
# @requires_credentials("GOOGLE_APPLICATION_CREDENTIALS", "VERTEX_PROJECT")
# def test_vertex_ai():
#     ...
```

## 7. Update Test Documentation

Add to `CLAUDE.md`:

```markdown
### Test Best Practices

When writing tests:

1. **Async tests with mocks** → Add `@pytest.mark.no_parallel`
2. **Tests needing credentials** → Add `@requires_credentials("ENV_VAR")`
3. **Known flaky tests** → Add `@pytest.mark.flaky(reruns=3)`
4. **Slow tests** → Add `@pytest.mark.slow`

Example:
\`\`\`python
@pytest.mark.asyncio
@pytest.mark.no_parallel  # Has async mocks
async def test_mcp_oauth():
    with patch("module.func") as mock:
        # Test code
        mock.assert_called_once()
\`\`\`

Check test flakiness locally:
\`\`\`bash
make test-file-repeat FILE=tests/path/to/test.py
\`\`\`
```

## Priority Implementation Order

### Phase 1: Quick Wins (1 day)
1. ✅ Document test patterns (done - test-flakiness-guide.md)
2. Add `pytest-rerunfailures` to dependencies
3. Mark known flaky tests with `@pytest.mark.flaky`
4. Add test utilities module

### Phase 2: CI Improvements (2-3 days)
1. Update GitHub Actions to retry failed tests
2. Split stable vs flaky test runs
3. Add Makefile targets for common test scenarios

### Phase 3: Enforcement (1 week)
1. Add pre-commit hook for test markers
2. Update CLAUDE.md with test guidelines
3. Review and mark all async tests appropriately

## Monitoring Success

Track these metrics:
- **Flaky test rate**: Tests that fail <10% of runs
- **False failure rate**: CI failures that pass on retry
- **Test runtime**: Should decrease with better parallelization

Target goals:
- ✅ <5% flaky test rate
- ✅ <2% false failure rate
- ✅ Test suite completes in <15 minutes

## Quick Checklist

Before merging a PR with new tests:

- [ ] All async tests with mocks have `@pytest.mark.no_parallel`
- [ ] Tests requiring credentials have skip conditions
- [ ] No `isinstance()` checks on module-level imports
- [ ] Async mocks are configured before execution
- [ ] Tests pass 10 times locally: `make test-file-repeat FILE=...`
- [ ] Tests pass in parallel: `pytest tests/file.py -n 4`
