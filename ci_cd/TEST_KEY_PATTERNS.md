# Test Key Patterns Standard

Standard patterns for test/mock keys and credentials in the LiteLLM codebase. All test keys MUST follow these patterns to be automatically excluded from secret detection.

## Standard Patterns

- **Prefix patterns**: `test-*`, `mock-*`, `fake-*`, `example-*`, `dummy-*`
- **API key patterns**: `sk-test-*`, `sk-mock-*`, `sk-fake-*`
- **Simple test values**: `sk-1234`, `sk-12345`, `postgres`
- **Environment variable references**: `os.environ/KEY_NAME`, `os.environ['KEY_NAME']`

## GitGuardian Dashboard Configuration

These patterns are configured in GitGuardian dashboard under "Secrets" → "Pattern exclusions":

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
^dummy-.*$
^postgres$
^os\.environ.*
```

## Usage

- **Test files**: Use `test-*`, `mock-*`, `sk-test-*`, or simple values like `sk-1234`
- **Documentation**: Use `example-*` prefix for example tokens
- **CI/CD configs**: Use `postgres` for test database credentials
- **Configuration files**: Use `os.environ/KEY_NAME` for environment variable references

## Examples

✅ **Good** (excluded):
```python
api_key = "test-token-123"
api_key = "sk-test-abcdef1234567890"
api_key = "example-api-key-xyz"
api_key = os.environ["OPENAI_API_KEY"]
api_key = "os.environ/ANTHROPIC_API_KEY"
```

❌ **Bad** (will be flagged):
```python
api_key = "sk-abcdef1234567890"  # No test prefix
api_key = "88dc28d0f03...f1243fe6b4b"  # High entropy, no test prefix
```
