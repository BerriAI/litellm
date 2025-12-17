# Test Key Patterns Standard

This document defines the standard patterns for test/mock keys and credentials in the LiteLLM codebase.

## Standard Test Key Patterns

All test keys, mock credentials, and example tokens MUST follow one of these patterns to be automatically excluded from secret detection:

### 1. Prefix Patterns (Recommended)
- `test-*` - e.g., `test-token-123`, `test-api-key-abc`
- `mock-*` - e.g., `mock-key-xyz`, `mock-secret-456`
- `fake-*` - e.g., `fake-token-789`, `fake-credential-def`
- `example-*` - e.g., `example-key-ghi`, `example-token-jkl`

### 2. API Key Patterns
- `sk-test-*` - e.g., `sk-test-1234567890abcdef`
- `sk-mock-*` - e.g., `sk-mock-abcdef1234567890`
- `sk-fake-*` - e.g., `sk-fake-9876543210fedcba`

### 3. Environment Variable References
- `os.environ/KEY_NAME` - e.g., `os.environ/OPENAI_API_KEY`
- `os.environ['KEY_NAME']` - e.g., `os.environ['ANTHROPIC_API_KEY']`

### 4. Simple Test Values
- `sk-1234`, `sk-12345`, `sk-67890` - Simple numeric test keys (only in test files)
- `dummy-*` - e.g., `dummy-key`, `dummy-token`

### 5. Standard Test Database Credentials
- `postgres` / `postgres` - Standard test database username/password
- `user@with+special` / `pas********#$%` - Test credentials with special characters

## GitGuardian Dashboard Configuration

Add these patterns to GitGuardian dashboard under "Secrets" → "Pattern exclusions":

```
^test-.*$
^mock-.*$
^fake-.*$
^example-.*$
^sk-test-.*$
^sk-mock-.*$
^sk-fake-.*$
^sk-1234$
^sk-12345$
^sk-67890$
^dummy-.*$
^postgres$
```

**Note:** `os.environ/KEY_NAME` patterns are configuration placeholders, not test keys. If these are flagged, handle them individually by excluding specific files (e.g., `litellm/proxy/example_config_yaml/**`) or fixing the specific instances.

## Usage Guidelines

1. **Test Files**: Always use one of the standard patterns above for mock credentials
2. **Documentation**: Use `example-*` prefix for example tokens in docs
3. **CI/CD Configs**: Use standard test values like `postgres` for test databases
4. **Comments**: When showing example API keys in comments, use `sk-test-*` or `example-*` patterns

## Examples

### ✅ Good (Will be excluded)
```python
api_key = "test-token-123"
api_key = "sk-test-abcdef1234567890"
api_key = "mock-api-key-xyz"
api_key = os.environ["OPENAI_API_KEY"]
```

### ❌ Bad (Will be flagged)
```python
api_key = "sk-abcdef1234567890"  # Looks like real key
api_key = "88dc28d0f03...f1243fe6b4b"  # High entropy, no test prefix
```
