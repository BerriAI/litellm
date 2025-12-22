# Test Key Patterns Standard

Standard patterns for test/mock keys and credentials in the LiteLLM codebase to avoid triggering secret detection.

## How GitGuardian Works

GitGuardian uses **machine learning and entropy analysis**, not just pattern matching:
- **Low entropy** values (like `sk-1234`, `postgres`) are automatically ignored
- **High entropy** values (realistic-looking secrets) trigger detection
- **Context-aware** detection understands code syntax like `os.environ["KEY"]`

## Recommended Test Key Patterns

### Option 1: Low Entropy Values (Simplest)
These won't trigger GitGuardian's ML detector:

```python
api_key = "sk-1234"
api_key = "sk-12345"
database_password = "postgres"
token = "test123"
```

### Option 2: High Entropy with Test Prefixes
If you need realistic-looking test keys with high entropy, use these prefixes:

```python
api_key = "sk-test-abc123def456ghi789..."  # OpenAI-style test key
api_key = "sk-mock-1234567890abcdef1234..."  # Mock key
api_key = "sk-fake-xyz789uvw456rst123..."  # Fake key
token = "test-api-key-with-high-entropy"
```

## Configured Ignore Patterns

These patterns are in `.gitguardian.yaml` for high-entropy test keys:
- `sk-test-*` - OpenAI-style test keys
- `sk-mock-*` - Mock API keys  
- `sk-fake-*` - Fake API keys
- `test-api-key` - Generic test tokens
