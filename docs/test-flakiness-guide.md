# Test Flakiness Solutions Guide

This guide explains how to address CI test flakiness in LiteLLM.

## Problem Summary

Tests passing locally but failing in CI due to:
- Parallel test execution (pytest-xdist)
- Async mock timing issues
- Module reload side effects
- Environment differences

## Solution 1: Better Test Isolation

### Use `@pytest.mark.no_parallel` for Sequential Tests

When tests have timing-sensitive mocks or shared state:

```python
@pytest.mark.asyncio
@pytest.mark.no_parallel  # Prevents parallel execution
async def test_oauth2_headers_passed_to_mcp_client():
    # Async mocks that need deterministic timing
    with patch("module._create_mcp_client") as mock_client:
        # Test code
        mock_client.assert_called_once()
```

**When to use:**
- ✅ Async mock assertions (`call_count`, `assert_called_once`)
- ✅ Global state manipulation
- ✅ Database schema migrations
- ✅ File system operations that aren't isolated
- ❌ Simple unit tests (adds unnecessary overhead)

### Fix Module Import Issues

For `isinstance()` checks that fail after module reload:

```python
# ❌ BAD: Module-level import becomes stale after reload
from module import MyClass

def test_something():
    obj = create_object()
    assert isinstance(obj, MyClass)  # Fails if litellm reloaded!

# ✅ GOOD: Import locally after reload
def test_something():
    from module import MyClass  # Fresh reference
    obj = create_object()
    assert isinstance(obj, MyClass)  # Always works
```

**Root cause:** `conftest.py` does `importlib.reload(litellm)` which creates new class objects.

## Solution 2: Robust Async Mock Setup

### Pattern 1: Ensure Mock Applied Before Async Code

```python
@pytest.mark.asyncio
async def test_async_function():
    # ❌ BAD: Mock might not be ready
    with patch("module.async_func") as mock:
        result = await call_async_code()
        mock.assert_called_once()  # Flaky in CI!

    # ✅ GOOD: Use AsyncMock and configure before execution
    mock_func = AsyncMock(return_value="result")
    with patch("module.async_func", mock_func):
        # Ensure mock is configured
        assert mock_func.call_count == 0

        result = await call_async_code()

        # More reliable assertion
        await asyncio.sleep(0)  # Let event loop settle
        assert mock_func.call_count == 1
```

### Pattern 2: Use `side_effect` for Async Mocks

```python
@pytest.mark.asyncio
async def test_with_side_effect():
    async def mock_implementation(*args, **kwargs):
        # Custom async logic
        return {"status": "success"}

    with patch("module.async_func", side_effect=mock_implementation):
        result = await call_code()
        assert result["status"] == "success"
```

### Pattern 3: Patch Early, Execute Late

```python
@pytest.mark.asyncio
async def test_patch_early():
    # Create and configure mock FIRST
    mock_client = AsyncMock()
    mock_client._create_mcp_client = AsyncMock(return_value={"id": "123"})

    # THEN patch
    with patch("module.ClientClass", return_value=mock_client):
        # NOW execute async code
        result = await execute_code()

        # Assertions are more reliable
        mock_client._create_mcp_client.assert_called_once()
```

## Solution 3: Retry Logic for Flaky Tests

### Method A: pytest-rerunfailures (Already Configured)

The project already has retry logic in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
retries = 20
retry_delay = 5
```

To use it for specific tests:

```python
@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_sometimes_fails():
    # Flaky test code
    pass
```

### Method B: Manual Retry Decorator

For async tests with specific retry logic:

```python
import asyncio
from functools import wraps

def retry_async(max_attempts=3, delay=0.1):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except AssertionError as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

@pytest.mark.asyncio
@retry_async(max_attempts=3)
async def test_with_retry():
    # Test that might fail due to timing
    result = await async_operation()
    assert result.call_count == 1
```

### Method C: CI-Only Retries

In GitHub Actions workflow:

```yaml
- name: Run Tests
  run: |
    poetry run pytest tests/ -v --maxfail=5 || \
    poetry run pytest tests/ --lf -v --maxfail=5 || \
    poetry run pytest tests/ --lf -v
```

This runs:
1. All tests normally
2. If fails, rerun only last failures
3. If still fails, rerun last failures again

## Solution 4: Test Fixtures for Cleanup

### Ensure Clean State Between Tests

```python
@pytest.fixture(autouse=True)
async def cleanup_mcp_state():
    """Clean up MCP state between tests"""
    yield

    # Teardown
    try:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        global_mcp_server_manager.registry.clear()
    except ImportError:
        pass
```

### Module-Level Cleanup

```python
@pytest.fixture(scope="module", autouse=True)
def reset_litellm_state():
    """Reset litellm state before module tests"""
    import litellm

    # Save state
    original_settings = {
        "disable_aiohttp_transport": litellm.disable_aiohttp_transport,
        "force_ipv4": litellm.force_ipv4,
    }

    yield

    # Restore state
    for key, value in original_settings.items():
        setattr(litellm, key, value)
```

## Solution 5: CI Environment Variables

### Skip Tests Without Credentials

```python
def _has_credentials() -> bool:
    return "GOOGLE_APPLICATION_CREDENTIALS" in os.environ

@pytest.mark.skipif(
    not _has_credentials(),
    reason="Credentials not available in CI"
)
def test_requires_credentials():
    # Test code
    pass
```

### Detect CI Environment

```python
import os

IS_CI = os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true"

@pytest.mark.skipif(IS_CI, reason="Flaky in CI, tracked in issue #XXXXX")
def test_known_flaky():
    # Test code
    pass
```

## Checklist for Test Authors

When writing tests that might be flaky:

- [ ] Does it use async mocks? → Add `@pytest.mark.no_parallel`
- [ ] Does it modify global state? → Add cleanup fixture
- [ ] Does it check `isinstance()` → Import classes locally in test
- [ ] Does it need credentials? → Add `@pytest.mark.skipif`
- [ ] Is it timing-sensitive? → Add retries or `asyncio.sleep(0)`
- [ ] Does it fail randomly in CI? → Add `@pytest.mark.flaky(reruns=3)`

## Quick Reference

| Problem | Solution | Example |
|---------|----------|---------|
| Async mock not called | Use `@pytest.mark.no_parallel` | See Pattern 3 above |
| `isinstance()` fails | Import class locally | See Fix Module Import above |
| Need credentials | Use `@pytest.mark.skipif` | See Solution 5 above |
| Random timing failures | Add `@pytest.mark.flaky` | See Method A above |
| Module state pollution | Add cleanup fixture | See Solution 4 above |

## Examples from Codebase

### Fixed: http_handler isinstance issue (PR #21388)
```python
@pytest.mark.asyncio
async def test_session_reuse_integration():
    # Import locally to get fresh class after reload
    from litellm.llms.custom_httpx.http_handler import (
        get_async_httpx_client,
        AsyncHTTPHandler as AsyncHTTPHandlerReload
    )

    client = get_async_httpx_client(...)
    assert isinstance(client, AsyncHTTPHandlerReload)  # ✅ Works!
```

### Already Correct: MCP OAuth test
```python
@pytest.mark.asyncio
@pytest.mark.no_parallel  # Prevents parallel execution issues
async def test_oauth2_headers_passed_to_mcp_client():
    # Test with async mocks
    pass
```

## Debugging Flaky Tests

### Run test 100 times locally:
```bash
for i in {1..100}; do
    poetry run pytest tests/path/to/test.py::test_name || break
done
```

### Run with pytest-repeat:
```bash
poetry run pytest tests/path/to/test.py::test_name --count=100
```

### Run in parallel to reproduce CI:
```bash
poetry run pytest tests/path/to/test.py -n 4  # 4 parallel workers
```

## Further Reading

- [pytest-xdist docs](https://pytest-xdist.readthedocs.io/)
- [pytest-asyncio docs](https://pytest-asyncio.readthedocs.io/)
- [unittest.mock docs](https://docs.python.org/3/library/unittest.mock.html)
- [pytest-rerunfailures](https://github.com/pytest-dev/pytest-rerunfailures)
