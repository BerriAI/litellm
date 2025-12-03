# OpenAPI 3.0.3 Integration Guide

This guide explains how to integrate the OpenAPI 3.0.3 downgrade functionality into LiteLLM proxy server.

## Files Created

1. **`litellm/proxy/common_utils/openapi_downgrade.py`** - Core transformation logic
2. **`tests/test_litellm/proxy/common_utils/test_openapi_downgrade.py`** - Comprehensive unit tests

## Integration Steps

### Step 1: Update `openapi_schema_compat.py`

Modify `litellm/proxy/common_utils/openapi_schema_compat.py` to support version parameter:

```python
def get_openapi_schema_with_compat(
    get_openapi_func,
    title: str,
    version: str,
    description: str,
    routes: list,
    openapi_version: str = None,  # Add this parameter
) -> Dict[str, Any]:
    """
    Generate OpenAPI schema with compatibility handling for FastAPI 0.120+.
    
    Args:
        get_openapi_func: The FastAPI get_openapi function
        title: API title
        version: API version
        description: API description
        routes: List of routes
        openapi_version: Target OpenAPI version (e.g., "3.0.3" or "3.1.0")
        
    Returns:
        OpenAPI schema dictionary
    """
    import os
    
    # Use environment variable if not explicitly provided
    if openapi_version is None:
        openapi_version = os.getenv("OPENAPI_VERSION", "3.1.0")
    
    # ... existing code for schema generation ...
    
    try:
        openapi_schema = get_openapi_func(
            title=title,
            version=version,
            description=description,
            routes=routes,
        )
    finally:
        # ... existing cleanup code ...
    
    # NEW: Apply downgrade if 3.0.3 is requested
    if openapi_version.startswith("3.0"):
        from litellm.proxy.common_utils.openapi_downgrade import (
            get_openapi_3_0_3_compatible_version,
        )
        openapi_schema = get_openapi_3_0_3_compatible_version(openapi_schema)
        verbose_proxy_logger.info(f"Converted OpenAPI schema to version {openapi_version}")
    
    return openapi_schema
```

### Step 2: Update `proxy_server.py`

Modify the `get_openapi_schema()` function in `litellm/proxy/proxy_server.py`:

```python
def get_openapi_schema():
    if app.openapi_schema:
        return app.openapi_schema

    # Use compatibility wrapper for FastAPI 0.120+ schema generation
    from litellm.proxy.common_utils.openapi_schema_compat import (
        get_openapi_schema_with_compat,
    )
    
    # NEW: Get desired OpenAPI version from environment
    import os
    openapi_version = os.getenv("OPENAPI_VERSION", "3.1.0")

    openapi_schema = get_openapi_schema_with_compat(
        get_openapi_func=get_openapi,
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        openapi_version=openapi_version,  # NEW: Pass the version
    )

    # ... rest of existing code (WebSocket routes, LLM schemas, etc.) ...

    app.openapi_schema = openapi_schema
    return app.openapi_schema
```

### Step 3: Update `CustomOpenAPISpec` class

Modify `litellm/proxy/common_utils/custom_openapi_spec.py` to handle version-specific schemas:

```python
@staticmethod
def get_pydantic_schema(model_class, openapi_version: str = None) -> Optional[Dict[str, Any]]:
    """
    Get JSON schema from a Pydantic model, handling both v1 and v2 APIs.
    
    Args:
        model_class: Pydantic model class
        openapi_version: Target OpenAPI version (e.g., "3.0.3" or "3.1.0")
        
    Returns:
        JSON schema dict or None if failed
    """
    import os
    
    # Use environment variable if not explicitly provided
    if openapi_version is None:
        openapi_version = os.getenv("OPENAPI_VERSION", "3.1.0")
    
    try:
        # Try Pydantic v2 method first
        schema = model_class.model_json_schema()  # type: ignore
        
        # NEW: Convert to 3.0.3 if needed
        if openapi_version.startswith("3.0"):
            from litellm.proxy.common_utils.openapi_downgrade import (
                convert_pydantic_v2_to_openapi_3_0_3,
            )
            schema = convert_pydantic_v2_to_openapi_3_0_3(schema)
        
        return schema
    except AttributeError:
        try:
            # Fallback to Pydantic v1 method (already 3.0.x compatible)
            return model_class.schema()  # type: ignore
        except AttributeError:
            return None
    except Exception as e:
        verbose_proxy_logger.debug(f"Failed to generate schema for {model_class}: {e}")
        return None
```

### Step 4: Add Separate Endpoint (Optional but Recommended)

Add a dedicated endpoint for 3.0.3 spec in `proxy_server.py`:

```python
@app.get("/openapi-3.0.3.json", tags=["openapi"], include_in_schema=False)
async def get_openapi_3_0_3():
    """
    Get OpenAPI 3.0.3 compatible schema.
    
    This endpoint provides a version of the OpenAPI schema that is compatible
    with tools requiring OpenAPI 3.0.3 specification (e.g., Apigee).
    """
    from litellm.proxy.common_utils.openapi_downgrade import (
        get_openapi_3_0_3_compatible_version,
    )
    
    # Get the current schema
    current_schema = app.openapi()
    
    # Convert to 3.0.3
    schema_3_0_3 = get_openapi_3_0_3_compatible_version(current_schema)
    
    return schema_3_0_3


@app.get("/openapi.json", tags=["openapi"], include_in_schema=False)
async def get_openapi_json():
    """
    Get OpenAPI schema in JSON format.
    
    Returns OpenAPI 3.1.0 by default, or 3.0.3 if OPENAPI_VERSION=3.0.3 is set.
    """
    return app.openapi()
```

## Configuration

### Environment Variable

Add to your `.env` file or set in your environment:

```bash
# For OpenAPI 3.0.3 (required by Apigee)
export OPENAPI_VERSION=3.0.3

# For OpenAPI 3.1.0 (default, modern tools)
export OPENAPI_VERSION=3.1.0
```

### Docker Configuration

Add to your `docker-compose.yml`:

```yaml
services:
  litellm:
    image: ghcr.io/berriai/litellm:latest
    environment:
      - OPENAPI_VERSION=3.0.3
    ports:
      - "4000:4000"
```

Or in your Dockerfile:

```dockerfile
ENV OPENAPI_VERSION=3.0.3
```

## Usage Examples

### Option 1: Environment Variable (Recommended)

Set the environment variable before starting the proxy:

```bash
export OPENAPI_VERSION=3.0.3
litellm --config config.yaml --port 4000
```

All OpenAPI endpoints will return 3.0.3 compatible schemas.

### Option 2: Dedicated Endpoint

Keep default as 3.1.0, but access 3.0.3 version via dedicated endpoint:

```bash
# Start server normally (uses 3.1.0 by default)
litellm --config config.yaml --port 4000

# Access 3.0.3 schema
curl http://localhost:4000/openapi-3.0.3.json > openapi-3.0.3.json

# Use with Apigee
# Upload openapi-3.0.3.json to Apigee
```

### Option 3: Dynamic Query Parameter (Future Enhancement)

Future version could support:

```bash
curl "http://localhost:4000/openapi.json?version=3.0.3"
```

## Testing

### Run Unit Tests

```bash
# Run all OpenAPI downgrade tests
poetry run pytest tests/test_litellm/proxy/common_utils/test_openapi_downgrade.py -v

# Run specific test
poetry run pytest tests/test_litellm/proxy/common_utils/test_openapi_downgrade.py::TestTypeArrayConversion::test_string_or_null_to_nullable -v
```

### Validate Generated Schema

```bash
# Generate schema
export OPENAPI_VERSION=3.0.3
litellm --config config.yaml &
sleep 5  # Wait for startup

# Fetch schema
curl http://localhost:4000/openapi.json > generated-3.0.3.json

# Validate with swagger-cli (install: npm install -g @apidevtools/swagger-cli)
swagger-cli validate generated-3.0.3.json

# Or validate with Python
pip install openapi-spec-validator
python -c "
from openapi_spec_validator import validate_spec
from openapi_spec_validator.readers import read_from_filename

spec_dict, spec_url = read_from_filename('generated-3.0.3.json')
validate_spec(spec_dict)
print('✓ OpenAPI 3.0.3 schema is valid!')
"
```

### Test with Apigee

1. Generate the 3.0.3 schema:
```bash
export OPENAPI_VERSION=3.0.3
litellm --config config.yaml &
curl http://localhost:4000/openapi.json > litellm-openapi-3.0.3.json
```

2. Upload to Apigee:
   - Go to Apigee console
   - Navigate to Develop → Specs
   - Click "Import Spec"
   - Upload `litellm-openapi-3.0.3.json`
   - Verify it passes validation

## Known Limitations

### Features Not Supported in 3.0.3 Mode

1. **Webhooks**: 3.1.0 webhooks feature is completely removed
2. **Type Arrays**: Converted to `nullable` (may lose precision for multi-type fields)
3. **Advanced JSON Schema**: Some JSON Schema 2020-12 features removed
4. **License Identifiers**: `license.identifier` field removed
5. **Parameter Content**: Converted from `content` to `schema` (may lose some expressiveness)

### Recommendations

- **Use 3.1.0 by default**: It's more expressive and modern
- **Use 3.0.3 only when required**: For tools like Apigee that specifically require it
- **Test both versions**: Ensure your API documentation works in both modes
- **Document limitations**: Make users aware of 3.0.3 mode limitations

## Troubleshooting

### Schema Validation Fails

**Problem**: Generated 3.0.3 schema fails validation

**Solution**:
1. Check logs for transformation errors:
```bash
export LITELLM_LOG=DEBUG
litellm --config config.yaml
```

2. Compare with original 3.1.0 schema:
```bash
# Generate both versions
OPENAPI_VERSION=3.1.0 litellm --config config.yaml &
curl http://localhost:4000/openapi.json > schema-3.1.0.json
kill %1

OPENAPI_VERSION=3.0.3 litellm --config config.yaml &
curl http://localhost:4000/openapi.json > schema-3.0.3.json
kill %1

# Use diff to compare
diff schema-3.1.0.json schema-3.0.3.json
```

### Complex Types Not Converting Correctly

**Problem**: Complex Pydantic models not converting properly

**Solution**: The transformation may need enhancement for specific edge cases. Please:
1. Create a minimal reproduction case
2. Add a unit test to `test_openapi_downgrade.py`
3. Enhance the transformation logic in `openapi_downgrade.py`

### Performance Impact

**Problem**: Schema generation is slower with transformation

**Solution**:
1. Enable schema caching (already implemented via `app.openapi_schema`)
2. Pre-generate schemas at build time:
```bash
# Generate and cache schema
export OPENAPI_VERSION=3.0.3
python -c "
from litellm.proxy.proxy_server import app
schema = app.openapi()
import json
with open('cached-schema-3.0.3.json', 'w') as f:
    json.dump(schema, f, indent=2)
"
```

## Migration Path

### For Existing Deployments

**Phase 1: Add Support (No Breaking Changes)**
1. Merge the new code
2. Default remains 3.1.0
3. Users can opt-in to 3.0.3 via environment variable

**Phase 2: Test and Validate**
1. Test with Apigee and other tools
2. Gather feedback from users
3. Fix any edge cases

**Phase 3: Documentation**
1. Update main README
2. Add Apigee integration guide
3. Document configuration options

### For New Deployments

Start with the configuration you need:

```bash
# For modern tools (Swagger UI, Redoc, etc.)
export OPENAPI_VERSION=3.1.0

# For legacy tools (Apigee, older API gateways)
export OPENAPI_VERSION=3.0.3
```

## Example Use Case: Apigee Integration

Complete example for Apigee integration:

```bash
#!/bin/bash
# apigee-export.sh - Export OpenAPI 3.0.3 schema for Apigee

set -e

echo "Starting LiteLLM with OpenAPI 3.0.3..."
export OPENAPI_VERSION=3.0.3
litellm --config config.yaml --port 4000 &
LITELLM_PID=$!

echo "Waiting for server to start..."
sleep 10

echo "Fetching OpenAPI 3.0.3 schema..."
curl -s http://localhost:4000/openapi.json > litellm-apigee-spec.json

echo "Validating schema..."
npx @apidevtools/swagger-cli validate litellm-apigee-spec.json

echo "Stopping LiteLLM..."
kill $LITELLM_PID

echo "✓ Schema exported successfully to litellm-apigee-spec.json"
echo "  Upload this file to Apigee at: https://apigee.google.com/specs"
```

## Additional Resources

- [OpenAPI 3.0.3 Specification](https://spec.openapis.org/oas/v3.0.3)
- [OpenAPI 3.1.0 Specification](https://spec.openapis.org/oas/v3.1.0)
- [Apigee OpenAPI Documentation](https://cloud.google.com/apigee/docs/api-platform/develop/openapi)
- [FastAPI OpenAPI Customization](https://fastapi.tiangolo.com/advanced/extending-openapi/)
- [JSON Schema Migration Guide](https://json-schema.org/draft/2020-12/release-notes)

## Support

For issues with OpenAPI 3.0.3 conversion:
1. Check if it's a known limitation (see above)
2. Validate the original 3.1.0 schema works correctly
3. Create an issue with:
   - Original 3.1.0 schema snippet
   - Expected 3.0.3 output
   - Actual 3.0.3 output
   - Validation error (if any)
