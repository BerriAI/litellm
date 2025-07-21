# Enhanced /v1/models Endpoint with Fallback Support

## Summary

This implementation enhances the existing `/v1/models` endpoint to include complete fallback information for models while maintaining backward compatibility.

## Changes Made

### 1. Core Functionality (`litellm/proxy/auth/model_checks.py`)

Added `get_all_fallbacks()` function:
- Returns complete fallback list for a given model
- Supports three fallback types: `general`, `context_window`, `content_policy`
- Uses existing `get_fallback_model_group()` from router utils
- Handles errors gracefully and returns empty list when no fallbacks found

### 2. Enhanced API Endpoint (`litellm/proxy/proxy_server.py`)

Modified `model_list()` function to support:
- `include_metadata: bool = False` - Include additional metadata in response
- `fallback_type: Optional[str] = None` - Specify type of fallbacks to include

**Validation:**
- `fallback_type` must be one of: `["general", "context_window", "content_policy"]`
- Returns HTTP 400 error for invalid fallback types

### 3. Response Format

**Backward Compatible (default):**
```json
{
  "data": [
    {
      "id": "claude-4-sonnet",
      "object": "model", 
      "created": 1234567890,
      "owned_by": "openai"
    }
  ],
  "object": "list"
}
```

**With Fallback Metadata:**
```json
{
  "data": [
    {
      "id": "claude-4-sonnet",
      "object": "model",
      "created": 1234567890, 
      "owned_by": "openai",
      "metadata": {
        "fallbacks": ["bedrock-claude-sonnet-4", "google-claude-sonnet-4"]
      }
    }
  ],
  "object": "list"
}
```

## API Usage Examples

```bash
# Standard response (backward compatible)
GET /models

# Include metadata with general fallbacks (default)
GET /models?include_metadata=true

# Explicitly specify general fallbacks
GET /models?include_metadata=true&fallback_type=general

# Include context window fallbacks  
GET /models?include_metadata=true&fallback_type=context_window

# Include content policy fallbacks
GET /models?include_metadata=true&fallback_type=content_policy
```

## Fallback Logic

For configuration:
```yaml
fallbacks:
  - claude-4-sonnet:
      - bedrock-claude-sonnet-4
      - google-claude-sonnet-4
```

Expected behavior:
- `claude-4-sonnet` → `["bedrock-claude-sonnet-4", "google-claude-sonnet-4"]` (complete list)
- `bedrock-claude-sonnet-4` → `[]` (no fallbacks defined)
- `google-claude-sonnet-4` → `[]` (no fallbacks defined)

## Test Coverage

### Unit Tests (`tests/test_litellm/proxy/auth/test_model_checks_fallbacks.py`)
- 14 comprehensive test cases for `get_all_fallbacks()` function
- Tests all fallback types, error handling, edge cases
- Validates router integration and configuration parsing

### Integration Tests (`tests/proxy_unit_tests/test_models_fallback_endpoint.py`)
- 8 endpoint test cases covering full API functionality
- Tests query parameter validation, response format
- Verifies backward compatibility and new metadata features
- Tests multiple models with different fallback configurations

## Design Benefits

1. **Backward Compatibility**: Existing `/models` calls work unchanged
2. **Intuitive Default**: `include_metadata=true` automatically includes general fallbacks
3. **Generic Metadata Structure**: Extensible for future enhancements
4. **Type Safety**: Proper parameter validation and error handling
5. **Performance**: Only computes fallbacks when requested
6. **Comprehensive**: Supports all three LiteLLM fallback types

## Future Extensions

The generic metadata structure allows easy addition of new parameters:
```bash
# Future possibilities
GET /models?include_metadata=true&include_pricing=true
GET /models?include_metadata=true&include_capabilities=true
```