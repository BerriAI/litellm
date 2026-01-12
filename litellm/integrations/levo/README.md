# Levo AI Integration

This integration enables sending LLM observability data to Levo AI using OpenTelemetry (OTLP) protocol.

## Overview

The Levo integration extends LiteLLM's OpenTelemetry support to automatically send traces to Levo's collector endpoint with proper authentication and routing headers.

## Features

- **Automatic OTLP Export**: Sends OpenTelemetry traces to Levo collector
- **Levo-Specific Headers**: Automatically includes `x-levo-organization-id` and `x-levo-workspace-id` for routing
- **Simple Configuration**: Just use `callbacks: ["levo"]` in your LiteLLM config
- **Environment-Based Setup**: Configure via environment variables

## Quick Start

### 1. Install Dependencies

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http opentelemetry-exporter-otlp-proto-grpc
```

### 2. Configure LiteLLM

Add to your `litellm_config.yaml`:

```yaml
litellm_settings:
  callbacks: ["levo"]
```

### 3. Set Environment Variables

```bash
export LEVOAI_API_KEY="<your-levo-api-key>"
export LEVOAI_ORG_ID="<your-levo-org-id>"
export LEVOAI_WORKSPACE_ID="<your-workspace-id>"
export LEVOAI_COLLECTOR_URL="<your-levo-collector-url>"
```

### 4. Start LiteLLM

```bash
litellm --config config.yaml
```

All LLM requests will now automatically be sent to Levo!

## Configuration

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `LEVOAI_API_KEY` | Your Levo API key for authentication |
| `LEVOAI_ORG_ID` | Your Levo organization ID for routing |
| `LEVOAI_WORKSPACE_ID` | Your Levo workspace ID for routing |
| `LEVOAI_COLLECTOR_URL` | Full collector endpoint URL from Levo support |

### Optional Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LEVOAI_ENV_NAME` | Environment name for tagging traces | `None` |

**Important**: The `LEVOAI_COLLECTOR_URL` is used exactly as provided. No path manipulation is performed.

## How It Works

1. **LevoLogger** extends LiteLLM's `OpenTelemetry` class
2. **Configuration** is read from environment variables via `get_levo_config()`
3. **OTLP Headers** are automatically set:
   - `Authorization: Bearer {LEVOAI_API_KEY}`
   - `x-levo-organization-id: {LEVOAI_ORG_ID}`
   - `x-levo-workspace-id: {LEVOAI_WORKSPACE_ID}`
4. **Traces** are sent to the collector endpoint in OTLP format

## Code Structure

```
litellm/integrations/levo/
├── __init__.py          # Exports LevoLogger
├── levo.py             # LevoLogger implementation
└── README.md           # This file
```

### Key Classes

- **LevoLogger**: Extends `OpenTelemetry`, handles Levo-specific configuration
- **LevoConfig**: Pydantic model for Levo configuration (defined in `levo.py`)

## Testing

See the test files in `tests/test_litellm/integrations/levo/`:
- `test_levo.py`: Unit tests for configuration
- `test_levo_integration.py`: Integration tests for callback registration

## Error Handling

The integration validates all required environment variables at initialization:
- Missing `LEVOAI_API_KEY`: Raises `ValueError` with clear message
- Missing `LEVOAI_ORG_ID`: Raises `ValueError` with clear message
- Missing `LEVOAI_WORKSPACE_ID`: Raises `ValueError` with clear message
- Missing `LEVOAI_COLLECTOR_URL`: Raises `ValueError` with clear message

## Integration with LiteLLM

The Levo callback is registered in:
- `litellm/litellm_core_utils/custom_logger_registry.py`: Maps `"levo"` to `LevoLogger`
- `litellm/litellm_core_utils/litellm_logging.py`: Instantiates `LevoLogger` when `callbacks: ["levo"]` is used
- `litellm/__init__.py`: Added to `_custom_logger_compatible_callbacks_literal`

## Documentation

For detailed documentation, see:
- [LiteLLM Levo Integration Docs](../../../../docs/my-website/docs/observability/levo_integration.md)
- [Levo Documentation](https://docs.levo.ai)

## Support

For issues or questions:
- LiteLLM Issues: https://github.com/BerriAI/litellm/issues
- Levo Support: support@levo.ai

