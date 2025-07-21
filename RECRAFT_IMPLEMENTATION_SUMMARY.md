# Recraft AI Image Generation Integration for LiteLLM

## Overview

This implementation adds support for Recraft AI's image generation API to LiteLLM, following the existing code style and patterns. Recraft AI offers state-of-the-art image generation with support for various styles including realistic images, digital illustrations, vector art, and icons.

## Features Implemented

### 1. Provider Registration
- Added `RECRAFT = "recraft"` to `LlmProviders` enum in `litellm/types/utils.py`
- Added "recraft" to `LITELLM_CHAT_PROVIDERS` list in `litellm/constants.py`

### 2. Configuration Structure
- Created `RecraftImageGenerationConfig` class in `litellm/llms/recraft/image_generation.py`
- Follows the same pattern as other providers (OpenAI, Azure, etc.)
- Inherits from `BaseImageGenerationConfig`

### 3. LLM HTTP Handler Integration
- Integrated with the existing HTTP handler system
- Supports both sync and async operations
- Uses the standard LiteLLM logging and error handling

### 4. Provider Configuration Manager
- Added Recraft support to `ProviderConfigManager.get_provider_image_generation_config()`
- Automatic provider detection for models prefixed with "recraft/"

### 5. Main Image Generation Routing
- Added Recraft routing logic to `litellm/images/main.py`
- Handles both `aimage_generation` (async) and `image_generation` (sync)

## API Specification Implementation

### Base URL
- Default: `https://external.api.recraft.ai/v1/images/generations`
- Configurable via `api_base` parameter

### Authentication
- Bearer token authentication
- Supports environment variables: `RECRAFT_API_KEY`, `RECRAFT_API_TOKEN`

### Supported Models
- `recraftv3` (default, latest model)
- `recraftv2` (legacy model)

### Supported Parameters

#### OpenAI-Compatible Parameters
- `n`: Number of images to generate (1-6)
- `size`: Image size (e.g., "1024x1024")
- `response_format`: "url" or "b64_json"
- `user`: User identifier

#### Recraft-Specific Parameters
- `style`: Base style ("realistic_image", "digital_illustration", "vector_illustration", "icon")
- `substyle`: Style refinement (e.g., "hand_drawn", "pixel_art")
- `negative_prompt`: What to avoid in the image
- `controls`: Advanced control parameters (colors, artistic_level, etc.)
- `text_layout`: Text positioning (Recraft V3 only)
- `style_id`: Custom style reference UUID

## Usage Examples

### Basic Usage
```python
import litellm

# Async
response = await litellm.aimage_generation(
    model="recraft/recraftv3",
    prompt="a beautiful sunset over mountains",
    api_key="your-api-key"
)

# Sync
response = litellm.image_generation(
    model="recraft/recraftv3", 
    prompt="a beautiful sunset over mountains",
    api_key="your-api-key"
)
```

### Advanced Usage with Recraft Features
```python
response = await litellm.aimage_generation(
    model="recraft/recraftv3",
    prompt="a futuristic city skyline",
    style="digital_illustration",
    substyle="hand_drawn", 
    size="1024x1024",
    n=2,
    negative_prompt="blurry, low quality",
    controls={
        "colors": [{"rgb": [0, 255, 0]}],
        "artistic_level": 3
    },
    api_key="your-api-key"
)
```

### Using Custom Styles
```python
response = await litellm.aimage_generation(
    model="recraft/recraftv3",
    prompt="a cartoon character",
    style_id="custom-style-uuid",
    api_key="your-api-key"
)
```

## Files Created/Modified

### New Files
- `litellm/llms/recraft/__init__.py`
- `litellm/llms/recraft/image_generation.py`
- `tests/image_gen_tests/test_recraft.py`

### Modified Files
- `litellm/types/utils.py` - Added RECRAFT to LlmProviders enum
- `litellm/constants.py` - Added "recraft" to provider list
- `litellm/utils.py` - Added Recraft to ProviderConfigManager
- `litellm/images/main.py` - Added Recraft routing logic

## Key Implementation Details

### Error Handling
- Validates API key presence
- Provides clear error messages for missing configuration
- Uses standard LiteLLM exception patterns

### Parameter Mapping
- Maps OpenAI-compatible parameters to Recraft equivalents
- Passes through Recraft-specific parameters
- Validates parameter support per model

### Response Transformation
- Converts Recraft API responses to OpenAI-compatible format
- Uses existing `convert_to_model_response_object` utility
- Maintains compatibility with LiteLLM response types

### Configuration Flexibility
- Supports custom API base URLs
- Environment variable configuration
- Model-specific parameter validation

## Testing

Comprehensive test suite includes:
- Provider detection and routing
- Configuration creation and validation
- Parameter mapping and transformation
- Mock API call testing
- Error handling validation

## Integration Notes

This implementation:
- ✅ Follows existing LiteLLM code patterns
- ✅ Uses the configuration structure approach
- ✅ Goes through the LLM HTTP handler
- ✅ Supports both sync and async operations
- ✅ Includes comprehensive parameter support
- ✅ Maintains OpenAI API compatibility
- ✅ Provides clear documentation and examples

The implementation is ready for use and can be extended with additional Recraft API features as needed.