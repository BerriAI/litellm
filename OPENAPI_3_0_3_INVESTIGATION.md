# OpenAPI 3.0.3 Support Investigation

## Background
Customer request: Need to support OpenAPI 3.0.3 spec for routing LiteLLM endpoint behind Apigee, which requires OpenAPI 3.0.3 compatibility. Customer reports current spec is 3.1.0 and cannot be easily converted down to 3.0.3.

## Current State

### Findings from Codebase Analysis

1. **Static OpenAPI File**: `/workspace/litellm/proxy/openapi.json`
   - Currently specifies `"openapi": "3.0.0"`
   - Contains 237 lines of basic OpenAPI schema
   - Appears to be a manually maintained file

2. **Dynamic Schema Generation**: `litellm/proxy/proxy_server.py`
   - Uses FastAPI's `get_openapi()` function from `fastapi.openapi.utils`
   - Schema is generated dynamically at runtime via `get_openapi_schema()` function
   - FastAPI by default generates OpenAPI 3.1.0 schemas (in recent versions)

3. **Custom OpenAPI Handling**: `litellm/proxy/common_utils/custom_openapi_spec.py`
   - Handles Pydantic model schema generation
   - Converts Pydantic v2 `$defs` to OpenAPI `components/schemas`
   - Adds request body schemas for chat completions, embeddings, and responses API

4. **Compatibility Layer**: `litellm/proxy/common_utils/openapi_schema_compat.py`
   - Provides compatibility for FastAPI 0.120+ schema generation
   - Patches Pydantic's schema generation to handle non-serializable types
   - Fallback creates minimal schema with `"openapi": "3.0.0"`

## Key Differences: OpenAPI 3.0.3 vs 3.1.0

### Major Changes in 3.1.0 (that need to be reverted for 3.0.3 compatibility):

1. **JSON Schema Alignment**
   - **3.1.0**: Fully aligned with JSON Schema Draft 2020-12
   - **3.0.3**: Uses JSON Schema Draft 4 with modifications
   - Impact: Properties like `const`, `$dynamicRef`, `$dynamicAnchor` not supported in 3.0.3

2. **Type Arrays**
   - **3.1.0**: Supports type arrays, e.g., `"type": ["string", "null"]`
   - **3.0.3**: Only single types allowed, must use `oneOf`/`anyOf` for alternatives
   - Example transformation needed:
     ```json
     // 3.1.0
     {"type": ["string", "null"]}
     
     // 3.0.3
     {"oneOf": [{"type": "string"}, {"type": "null"}]}
     ```

3. **Nullable Keyword**
   - **3.1.0**: Uses type arrays: `"type": ["string", "null"]`
   - **3.0.3**: Uses `"nullable": true` alongside type
   - Example:
     ```json
     // 3.1.0
     {"type": ["string", "null"]}
     
     // 3.0.3
     {"type": "string", "nullable": true}
     ```

4. **Schema Combinations**
   - **3.1.0**: Uses `prefixItems` for tuple validation
   - **3.0.3**: Uses `items` as array or object

5. **Exclusions**
   - **3.1.0**: Supports `exclusiveMinimum`/`exclusiveMaximum` as numbers
   - **3.0.3**: `exclusiveMinimum`/`exclusiveMaximum` are booleans

6. **Examples**
   - **3.1.0**: Single `examples` array property
   - **3.0.3**: Single `example` property (singular)

7. **Webhooks**
   - **3.1.0**: Native `webhooks` support
   - **3.0.3**: No webhooks field

8. **Content in Parameters**
   - **3.1.0**: Allows `content` in parameter objects
   - **3.0.3**: Parameters must use `schema` property

## Required Changes for OpenAPI 3.0.3 Support

### 1. Version Control
Add configuration option to specify OpenAPI version:

```python
# In proxy_server.py or config
OPENAPI_VERSION = os.getenv("OPENAPI_VERSION", "3.1.0")  # Default to 3.1.0, allow 3.0.3
```

### 2. Schema Transformation Layer
Create a new module: `litellm/proxy/common_utils/openapi_downgrade.py`

Functions needed:
- `downgrade_openapi_schema_to_3_0_3(schema: dict) -> dict`
- `convert_type_arrays_to_nullable(schema: dict) -> dict`
- `convert_examples_to_example(schema: dict) -> dict`
- `remove_unsupported_keywords(schema: dict) -> dict`

### 3. Modify FastAPI Schema Generation

Update `get_openapi_schema_with_compat()` in `openapi_schema_compat.py`:

```python
def get_openapi_schema_with_compat(
    get_openapi_func,
    title: str,
    version: str,
    description: str,
    routes: list,
    openapi_version: str = "3.1.0",  # Add parameter
) -> Dict[str, Any]:
    # ... existing code ...
    
    # After schema generation
    if openapi_version.startswith("3.0"):
        from litellm.proxy.common_utils.openapi_downgrade import (
            downgrade_openapi_schema_to_3_0_3
        )
        openapi_schema = downgrade_openapi_schema_to_3_0_3(openapi_schema)
    
    return openapi_schema
```

### 4. FastAPI App Configuration

FastAPI doesn't directly control OpenAPI version, but we can:
- Override the `app.openapi()` method to force version
- Post-process the schema to ensure compliance

In `proxy_server.py`:
```python
def get_openapi_schema():
    # ... existing code ...
    
    # Force OpenAPI version if configured
    openapi_version = os.getenv("OPENAPI_VERSION", "3.1.0")
    openapi_schema["openapi"] = openapi_version
    
    if openapi_version.startswith("3.0"):
        from litellm.proxy.common_utils.openapi_downgrade import (
            downgrade_openapi_schema_to_3_0_3
        )
        openapi_schema = downgrade_openapi_schema_to_3_0_3(openapi_schema)
    
    return openapi_schema
```

### 5. Pydantic Schema Handling

Pydantic v2 generates JSON Schema Draft 2020-12 compliant schemas (aligned with OpenAPI 3.1.0).

Update `CustomOpenAPISpec.get_pydantic_schema()`:
```python
@staticmethod
def get_pydantic_schema(model_class, openapi_version: str = "3.1.0") -> Optional[Dict[str, Any]]:
    try:
        schema = model_class.model_json_schema()
        
        if openapi_version.startswith("3.0"):
            # Transform to 3.0.3 compatible
            from litellm.proxy.common_utils.openapi_downgrade import (
                convert_pydantic_v2_to_openapi_3_0_3
            )
            schema = convert_pydantic_v2_to_openapi_3_0_3(schema)
        
        return schema
    except AttributeError:
        # Pydantic v1 fallback
        return model_class.schema()
```

## Implementation Plan

### Phase 1: Core Transformation Functions
1. Create `openapi_downgrade.py` module
2. Implement schema transformation functions:
   - Type array to nullable conversion
   - Examples array to example conversion
   - Remove 3.1.0-specific keywords
   - Recursive schema traversal

### Phase 2: Integration
1. Add `OPENAPI_VERSION` environment variable support
2. Integrate transformation in `get_openapi_schema_with_compat()`
3. Update `CustomOpenAPISpec` methods to accept version parameter
4. Modify `get_openapi_schema()` in proxy_server.py

### Phase 3: Testing
1. Add unit tests for transformation functions
2. Test with actual Apigee validation
3. Ensure both 3.0.3 and 3.1.0 modes work correctly
4. Verify Swagger UI compatibility

### Phase 4: Documentation
1. Update README with OPENAPI_VERSION configuration
2. Add example for Apigee integration
3. Document limitations of 3.0.3 mode

## Testing Strategy

### Unit Tests
Create `tests/test_litellm/proxy/common_utils/test_openapi_downgrade.py`:

```python
def test_convert_type_arrays_to_nullable():
    """Test type array conversion"""
    schema = {"type": ["string", "null"]}
    result = convert_type_arrays_to_nullable(schema)
    assert result == {"type": "string", "nullable": true}

def test_convert_examples_to_example():
    """Test examples array to example conversion"""
    schema = {"examples": ["value1", "value2"]}
    result = convert_examples_to_example(schema)
    assert result == {"example": "value1"}

def test_full_schema_downgrade():
    """Test complete schema downgrade from 3.1.0 to 3.0.3"""
    schema_3_1 = {
        "openapi": "3.1.0",
        "components": {
            "schemas": {
                "Model": {
                    "type": "object",
                    "properties": {
                        "name": {"type": ["string", "null"]},
                        "value": {"type": "integer", "examples": [1, 2, 3]}
                    }
                }
            }
        }
    }
    result = downgrade_openapi_schema_to_3_0_3(schema_3_1)
    assert result["openapi"] == "3.0.3"
    assert result["components"]["schemas"]["Model"]["properties"]["name"]["nullable"] == True
    assert "example" in result["components"]["schemas"]["Model"]["properties"]["value"]
```

### Integration Tests
1. Start proxy server with `OPENAPI_VERSION=3.0.3`
2. Fetch schema from `/openapi.json` endpoint
3. Validate against OpenAPI 3.0.3 specification
4. Test with Apigee validator tool

## Challenges and Considerations

### 1. Pydantic v2 Incompatibility
- Pydantic v2 generates JSON Schema 2020-12 by default
- Need comprehensive transformation layer
- Some advanced Pydantic features may not translate cleanly

### 2. FastAPI Version Dependency
- Recent FastAPI versions default to OpenAPI 3.1.0
- Older versions used 3.0.2
- May need version-specific handling

### 3. Maintainability
- Two schema versions to maintain
- Potential feature limitations in 3.0.3 mode
- Need clear documentation on differences

### 4. Performance
- Schema transformation adds overhead
- Consider caching transformed schemas
- Lazy transformation only when needed

## Recommended Approach

### Option 1: Environment Variable Toggle (Recommended)
- Add `OPENAPI_VERSION` environment variable
- Default to 3.1.0 for modern tools
- Allow 3.0.3 for legacy compatibility
- Pros: Flexible, backward compatible
- Cons: More complex, two code paths to maintain

### Option 2: Separate Endpoint
- Add `/openapi-3.0.3.json` endpoint
- Keep default at 3.1.0
- Pros: Simple, no breaking changes
- Cons: Confusing to have multiple specs

### Option 3: Query Parameter
- Add `?version=3.0.3` to `/openapi.json`
- Pros: Flexible per-request
- Cons: Caching complexity

**Recommendation**: Option 1 with Option 2 as supplement
- Primary control via `OPENAPI_VERSION` env var
- Also provide `/openapi-3.0.3.json` convenience endpoint

## Next Steps

1. **Immediate**: Create proof-of-concept transformation function
2. **Short-term**: Implement full transformation layer with tests
3. **Medium-term**: Add configuration and integration
4. **Long-term**: Monitor for issues and optimize

## References

- [OpenAPI 3.0.3 Specification](https://spec.openapis.org/oas/v3.0.3)
- [OpenAPI 3.1.0 Specification](https://spec.openapis.org/oas/v3.1.0)
- [What's New in OpenAPI 3.1.0](https://www.openapis.org/blog/2021/02/16/migrating-from-openapi-3-0-to-3-1-0)
- [FastAPI OpenAPI Documentation](https://fastapi.tiangolo.com/advanced/extending-openapi/)
- [JSON Schema Migration Guide](https://json-schema.org/draft/2020-12/release-notes)

## Sample Transformation Code

```python
def downgrade_openapi_schema_to_3_0_3(schema: dict) -> dict:
    """
    Transform OpenAPI 3.1.0 schema to 3.0.3 compatible schema.
    
    Args:
        schema: OpenAPI 3.1.0 schema dictionary
        
    Returns:
        OpenAPI 3.0.3 compatible schema dictionary
    """
    import copy
    result = copy.deepcopy(schema)
    
    # Update version
    result["openapi"] = "3.0.3"
    
    # Remove webhooks if present (3.1.0 feature)
    if "webhooks" in result:
        del result["webhooks"]
    
    # Recursively process all schema objects
    result = _process_schema_object(result)
    
    return result

def _process_schema_object(obj):
    """Recursively process schema objects"""
    if isinstance(obj, dict):
        # Handle type arrays -> nullable
        if "type" in obj and isinstance(obj["type"], list):
            if "null" in obj["type"]:
                types = [t for t in obj["type"] if t != "null"]
                if len(types) == 1:
                    obj["type"] = types[0]
                    obj["nullable"] = True
                else:
                    # Multiple non-null types, use oneOf
                    obj = {
                        "oneOf": [{"type": t} for t in types],
                        "nullable": True
                    }
        
        # Handle examples -> example
        if "examples" in obj and isinstance(obj["examples"], list):
            obj["example"] = obj["examples"][0] if obj["examples"] else None
            del obj["examples"]
        
        # Remove 3.1.0 specific keywords
        for keyword in ["$dynamicRef", "$dynamicAnchor", "prefixItems", "const"]:
            if keyword in obj:
                del obj[keyword]
        
        # Recursively process nested objects
        return {k: _process_schema_object(v) for k, v in obj.items()}
    
    elif isinstance(obj, list):
        return [_process_schema_object(item) for item in obj]
    
    else:
        return obj
```

## Validation Tools

For testing 3.0.3 compliance:
1. [Swagger Editor](https://editor.swagger.io/) - validates OpenAPI specs
2. [Redoc](https://github.com/Redocly/redoc) - renders OpenAPI docs
3. [openapi-spec-validator](https://github.com/p1c2u/openapi-spec-validator) - Python validation
4. Apigee's native validation tools

## Environment Variable Configuration

Add to `.env.example`:
```bash
# OpenAPI Version - set to "3.0.3" for legacy compatibility, "3.1.0" for modern tools
OPENAPI_VERSION=3.1.0
```

Usage:
```bash
export OPENAPI_VERSION=3.0.3
litellm --config config.yaml
```
