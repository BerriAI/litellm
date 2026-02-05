# AWS Bedrock Provider

This directory contains the AWS Bedrock provider implementation for LiteLLM.

## Beta Headers Management

### Overview

Bedrock anthropic-beta header handling uses a centralized whitelist-based filter (`beta_headers_config.py`) across all three Bedrock APIs to ensure:
- Only supported headers reach AWS (prevents API errors)
- Consistent behavior across Invoke Chat, Invoke Messages, and Converse APIs
- Zero maintenance when new Claude models are released

### Key Features

1. **Version-Based Filtering**: Headers specify minimum version (e.g., "requires Claude 4.5+") instead of hardcoded model lists
2. **Family Restrictions**: Can limit headers to specific families (opus/sonnet/haiku)
3. **Automatic Translation**: `advanced-tool-use` → `tool-search-tool` + `tool-examples` for backward compatibility

### Adding New Beta Headers

When AWS Bedrock adds support for a new Anthropic beta header, update `beta_headers_config.py`:

```python
# 1. Add to whitelist
BEDROCK_CORE_SUPPORTED_BETAS.add("new-feature-2027-01-15")

# 2. (Optional) Add version requirement
BETA_HEADER_MINIMUM_VERSION["new-feature-2027-01-15"] = 5.0

# 3. (Optional) Add family restriction
BETA_HEADER_FAMILY_RESTRICTIONS["new-feature-2027-01-15"] = ["opus"]
```

Then add tests in `tests/test_litellm/llms/bedrock/test_beta_headers_config.py`.

### Adding New Claude Models

When Anthropic releases new models (e.g., Claude Opus 5):
- **Required code changes**: ZERO ✅
- The version-based filter automatically handles new models
- No hardcoded lists to update

### Testing

```bash
# Test beta headers filtering
poetry run pytest tests/test_litellm/llms/bedrock/test_beta_headers_config.py -v

# Test API integrations
poetry run pytest tests/test_litellm/llms/bedrock/test_anthropic_beta_support.py -v

# Test everything
poetry run pytest tests/test_litellm/llms/bedrock/ -v
```

### Debug Logging

Enable debug logging to see filtering decisions:
```bash
LITELLM_LOG=DEBUG
```

### References

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages-request-response.html)
- [Anthropic Beta Headers](https://docs.anthropic.com/claude/reference/versioning)
