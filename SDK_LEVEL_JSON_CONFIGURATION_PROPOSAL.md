# SDK-Level JSON Configuration with Transformations & Cost Tracking

## Overview

This proposal extends the JSON-based configuration system to the **LiteLLM SDK level**, enabling:
1. ✅ **SDK Integration** - Works with `litellm.image_generation()` directly
2. ✅ **Request Transformations** - Convert OpenAI format → Provider format
3. ✅ **Response Transformations** - Convert Provider format → OpenAI format
4. ✅ **Cost Tracking** - Automatic cost calculation per request
5. ✅ **Future-Proof** - Can add OpenAI and any provider via JSON

## Architecture

```
┌─────────────────────────────────┐
│  sdk_providers.json             │  ← Provider configurations
│  (API specs + transformations)  │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  SDKProviderRegistry            │  ← Load & validate configs
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  TransformationEngine           │  ← Request/Response transforms
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  CostTracker                    │  ← Calculate costs
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  litellm.image_generation()     │  ← Works automatically!
└─────────────────────────────────┘
```

## JSON Schema Design

### Complete Provider Configuration

```json
{
  "google_imagen_3": {
    "provider_name": "google_imagen",
    "provider_type": "image_generation",
    "api_base": "https://generativelanguage.googleapis.com/v1beta",
    "api_base_env": "GOOGLE_IMAGEN_API_BASE",
    
    "authentication": {
      "type": "query_param",
      "env_var": "GOOGLE_API_KEY",
      "param_name": "key"
    },
    
    "endpoints": {
      "generate": {
        "path": "/models/{model}:predict",
        "method": "POST",
        "supported_models": [
          "imagen-3.0-generate-001",
          "imagen-3.0-fast-generate-001",
          "imagen-3.0-capability-generate-001"
        ]
      }
    },
    
    "transformations": {
      "request": {
        "type": "jinja",
        "template": {
          "instances": [
            {
              "prompt": "{{ prompt }}"
            }
          ],
          "parameters": {
            "sampleCount": "{{ n | default(1) }}",
            "aspectRatio": "{{ aspect_ratio | default('1:1') }}",
            "negativePrompt": "{{ negative_prompt | default(None) }}"
          }
        }
      },
      "response": {
        "type": "jsonpath",
        "mappings": {
          "images": "$.predictions[*].bytesBase64Encoded",
          "format": "b64_json"
        }
      }
    },
    
    "cost_tracking": {
      "enabled": true,
      "cost_per_image": {
        "imagen-3.0-generate-001": 0.04,
        "imagen-3.0-fast-generate-001": 0.02,
        "imagen-3.0-capability-generate-001": 0.04
      },
      "unit": "per_image"
    },
    
    "capabilities": {
      "max_images": 4,
      "supported_sizes": ["256x256", "512x512", "1024x1024"],
      "supports_negative_prompt": true,
      "supports_aspect_ratio": true
    }
  }
}
```

### Transformation Types

#### 1. Jinja Templates (Request Transformation)
```json
{
  "transformations": {
    "request": {
      "type": "jinja",
      "template": {
        "instances": [{"prompt": "{{ prompt }}"}],
        "parameters": {
          "sampleCount": "{{ n | default(1) }}",
          "aspectRatio": "{{ size | map_size_to_ratio }}"
        }
      },
      "filters": {
        "map_size_to_ratio": {
          "1024x1024": "1:1",
          "1024x1792": "9:16",
          "1792x1024": "16:9"
        }
      }
    }
  }
}
```

#### 2. JSONPath (Response Transformation)
```json
{
  "transformations": {
    "response": {
      "type": "jsonpath",
      "mappings": {
        "images": "$.predictions[*].bytesBase64Encoded",
        "revised_prompt": "$.metadata.prompt",
        "format": "b64_json"
      }
    }
  }
}
```

#### 3. Python Function (Complex Transformations)
```json
{
  "transformations": {
    "request": {
      "type": "function",
      "module": "litellm.llms.google_imagen.transformation",
      "function": "transform_request"
    },
    "response": {
      "type": "function",
      "module": "litellm.llms.google_imagen.transformation",
      "function": "transform_response"
    }
  }
}
```

## Implementation Plan

### Phase 1: Core Infrastructure

#### File: `litellm/llms/json_providers/sdk_provider_registry.py`

```python
"""
SDK-level provider configuration registry with transformation support.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class AuthenticationConfig(BaseModel):
    """Authentication configuration"""
    type: str
    env_var: str
    param_name: Optional[str] = None
    header_name: Optional[str] = None

class EndpointConfig(BaseModel):
    """API endpoint configuration"""
    path: str
    method: str
    supported_models: List[str]

class TransformationConfig(BaseModel):
    """Transformation configuration"""
    type: str  # jinja, jsonpath, function
    template: Optional[Dict] = None
    mappings: Optional[Dict] = None
    module: Optional[str] = None
    function: Optional[str] = None
    filters: Optional[Dict] = None

class CostTrackingConfig(BaseModel):
    """Cost tracking configuration"""
    enabled: bool = True
    cost_per_image: Dict[str, float] = {}
    cost_per_token: Optional[Dict[str, Dict[str, float]]] = None
    unit: str = "per_image"

class SDKProviderConfig(BaseModel):
    """Complete SDK provider configuration"""
    provider_name: str
    provider_type: str  # image_generation, chat, embeddings, etc.
    api_base: str
    api_base_env: Optional[str] = None
    authentication: AuthenticationConfig
    endpoints: Dict[str, EndpointConfig]
    transformations: Dict[str, TransformationConfig]
    cost_tracking: CostTrackingConfig
    capabilities: Dict[str, Any] = {}

class SDKProviderRegistry:
    """Registry for SDK-level provider configurations"""
    _configs: Dict[str, SDKProviderConfig] = {}
    _loaded = False
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None):
        """Load SDK provider configurations from JSON"""
        if cls._loaded:
            return
        
        if config_path is None:
            config_path = Path(__file__).parent / "sdk_providers.json"
        
        if not config_path.exists():
            cls._loaded = True
            return
        
        try:
            with open(config_path) as f:
                data = json.load(f)
            
            for slug, config_data in data.items():
                cls._configs[slug] = SDKProviderConfig(**config_data)
            
            cls._loaded = True
        except Exception as e:
            from litellm._logging import verbose_logger
            verbose_logger.error(f"Failed to load SDK provider configs: {e}")
            cls._loaded = True
    
    @classmethod
    def get(cls, provider_name: str) -> Optional[SDKProviderConfig]:
        """Get provider configuration"""
        return cls._configs.get(provider_name)
    
    @classmethod
    def list_providers(cls, provider_type: Optional[str] = None) -> List[str]:
        """List registered providers, optionally filtered by type"""
        if provider_type:
            return [
                name for name, config in cls._configs.items()
                if config.provider_type == provider_type
            ]
        return list(cls._configs.keys())

# Load on import
SDKProviderRegistry.load()
```

#### File: `litellm/llms/json_providers/transformation_engine.py`

```python
"""
Request/Response transformation engine for JSON-configured providers.
"""

import importlib
import json
from typing import Any, Dict, Optional
from jinja2 import Template
from jsonpath_ng import parse as jsonpath_parse

class TransformationEngine:
    """Engine for transforming requests and responses"""
    
    @staticmethod
    def transform_request(
        litellm_params: Dict[str, Any],
        transformation_config: "TransformationConfig"
    ) -> Dict[str, Any]:
        """
        Transform LiteLLM request parameters to provider format.
        
        Args:
            litellm_params: Standard LiteLLM parameters (prompt, n, size, etc.)
            transformation_config: Transformation configuration
        
        Returns:
            Provider-specific request body
        """
        if transformation_config.type == "jinja":
            return TransformationEngine._transform_with_jinja(
                litellm_params,
                transformation_config.template,
                transformation_config.filters
            )
        elif transformation_config.type == "jsonpath":
            # JSONPath is typically for responses, but can map request fields
            return TransformationEngine._transform_with_jsonpath(
                litellm_params,
                transformation_config.mappings
            )
        elif transformation_config.type == "function":
            return TransformationEngine._transform_with_function(
                litellm_params,
                transformation_config.module,
                transformation_config.function
            )
        else:
            raise ValueError(f"Unknown transformation type: {transformation_config.type}")
    
    @staticmethod
    def transform_response(
        provider_response: Dict[str, Any],
        transformation_config: "TransformationConfig"
    ) -> Dict[str, Any]:
        """
        Transform provider response to LiteLLM format.
        
        Args:
            provider_response: Provider's response
            transformation_config: Transformation configuration
        
        Returns:
            LiteLLM-formatted response
        """
        if transformation_config.type == "jsonpath":
            return TransformationEngine._extract_with_jsonpath(
                provider_response,
                transformation_config.mappings
            )
        elif transformation_config.type == "function":
            return TransformationEngine._transform_with_function(
                provider_response,
                transformation_config.module,
                transformation_config.function
            )
        else:
            raise ValueError(f"Unknown transformation type: {transformation_config.type}")
    
    @staticmethod
    def _transform_with_jinja(
        data: Dict[str, Any],
        template: Dict[str, Any],
        filters: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Any]:
        """Transform using Jinja2 templates"""
        # Convert template dict to JSON string, then render with Jinja2
        template_str = json.dumps(template)
        jinja_template = Template(template_str)
        
        # Apply custom filters
        if filters:
            for filter_name, filter_mappings in filters.items():
                def custom_filter(value):
                    return filter_mappings.get(value, value)
                jinja_template.globals[filter_name] = custom_filter
        
        # Render template
        rendered = jinja_template.render(**data)
        return json.loads(rendered)
    
    @staticmethod
    def _extract_with_jsonpath(
        data: Dict[str, Any],
        mappings: Dict[str, str]
    ) -> Dict[str, Any]:
        """Extract fields using JSONPath"""
        result = {}
        for key, jsonpath_expr in mappings.items():
            if key == "format":
                result[key] = jsonpath_expr  # Static value
                continue
            
            parser = jsonpath_parse(jsonpath_expr)
            matches = parser.find(data)
            
            if matches:
                # If multiple matches, return list
                values = [match.value for match in matches]
                result[key] = values if len(values) > 1 else values[0]
        
        return result
    
    @staticmethod
    def _transform_with_function(
        data: Dict[str, Any],
        module_name: str,
        function_name: str
    ) -> Dict[str, Any]:
        """Transform using Python function"""
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
        return func(data)
    
    @staticmethod
    def _transform_with_jsonpath_mapping(
        data: Dict[str, Any],
        mappings: Dict[str, str]
    ) -> Dict[str, Any]:
        """Transform using simple JSONPath field mappings"""
        result = {}
        for target_key, source_path in mappings.items():
            # Simple path traversal for request transformation
            keys = source_path.replace("$.", "").split(".")
            value = data
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    break
            if value is not None:
                result[target_key] = value
        return result
```

#### File: `litellm/llms/json_providers/cost_tracker.py`

```python
"""
Cost tracking for JSON-configured providers.
"""

from typing import Any, Dict, Optional
from litellm.types.utils import ImageResponse

class CostTracker:
    """Track costs for JSON-configured providers"""
    
    @staticmethod
    def calculate_image_generation_cost(
        response: ImageResponse,
        model: str,
        cost_config: "CostTrackingConfig"
    ) -> float:
        """
        Calculate cost for image generation.
        
        Args:
            response: LiteLLM image response
            model: Model name used
            cost_config: Cost configuration
        
        Returns:
            Total cost in USD
        """
        if not cost_config.enabled:
            return 0.0
        
        # Get number of images generated
        num_images = len(response.data) if response.data else 0
        
        # Get cost per image for this model
        cost_per_image = cost_config.cost_per_image.get(model, 0.0)
        
        # Calculate total cost
        total_cost = num_images * cost_per_image
        
        return total_cost
    
    @staticmethod
    def calculate_completion_cost(
        response: Any,
        model: str,
        cost_config: "CostTrackingConfig"
    ) -> float:
        """Calculate cost for completion endpoints (future)"""
        if not cost_config.enabled:
            return 0.0
        
        # Extract token usage
        usage = getattr(response, 'usage', None)
        if not usage:
            return 0.0
        
        prompt_tokens = getattr(usage, 'prompt_tokens', 0)
        completion_tokens = getattr(usage, 'completion_tokens', 0)
        
        # Get costs per token
        token_costs = cost_config.cost_per_token.get(model, {})
        prompt_cost = token_costs.get('prompt', 0.0)
        completion_cost = token_costs.get('completion', 0.0)
        
        # Calculate total cost
        total_cost = (prompt_tokens * prompt_cost) + (completion_tokens * completion_cost)
        
        return total_cost
```

### Phase 2: Integration with LiteLLM

#### File: `litellm/llms/json_providers/image_generation_handler.py`

```python
"""
Image generation handler for JSON-configured providers.
"""

import os
from typing import Any, Dict, Optional
import httpx

from litellm.types.utils import ImageResponse
from litellm.llms.json_providers.sdk_provider_registry import SDKProviderRegistry
from litellm.llms.json_providers.transformation_engine import TransformationEngine
from litellm.llms.json_providers.cost_tracker import CostTracker
from litellm.secret_managers.main import get_secret_str

class JSONProviderImageGeneration:
    """Image generation for JSON-configured providers"""
    
    @staticmethod
    async def aimage_generation(
        prompt: str,
        model: str,
        provider_config_name: str,
        optional_params: Optional[Dict] = None,
        **kwargs
    ) -> ImageResponse:
        """
        Generate images using JSON-configured provider.
        
        Args:
            prompt: Text prompt
            model: Model name
            provider_config_name: Name of provider in sdk_providers.json
            optional_params: Additional parameters (n, size, etc.)
        
        Returns:
            ImageResponse with generated images
        """
        # Load provider configuration
        config = SDKProviderRegistry.get(provider_config_name)
        if not config:
            raise ValueError(f"Provider configuration '{provider_config_name}' not found")
        
        # Prepare LiteLLM parameters
        litellm_params = {
            "prompt": prompt,
            "model": model,
            "n": optional_params.get("n", 1) if optional_params else 1,
            "size": optional_params.get("size", "1024x1024") if optional_params else "1024x1024",
            **(optional_params or {})
        }
        
        # Transform request to provider format
        request_body = TransformationEngine.transform_request(
            litellm_params,
            config.transformations["request"]
        )
        
        # Build API URL
        api_base = os.getenv(config.api_base_env) if config.api_base_env else None
        api_base = api_base or config.api_base
        
        endpoint_config = config.endpoints["generate"]
        url = api_base + endpoint_config.path.format(model=model)
        
        # Get authentication
        auth_config = config.authentication
        api_key = get_secret_str(auth_config.env_var)
        
        # Build request
        headers = {"Content-Type": "application/json"}
        params = {}
        
        if auth_config.type == "query_param":
            params[auth_config.param_name] = api_key
        elif auth_config.type == "bearer_token":
            headers["Authorization"] = f"Bearer {api_key}"
        elif auth_config.type == "custom_header":
            headers[auth_config.header_name] = api_key
        
        # Make request
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=request_body,
                headers=headers,
                params=params,
                timeout=30.0
            )
            response.raise_for_status()
            provider_response = response.json()
        
        # Transform response to LiteLLM format
        transformed = TransformationEngine.transform_response(
            provider_response,
            config.transformations["response"]
        )
        
        # Build ImageResponse
        from openai.types.image import Image
        
        images = []
        for img_data in transformed.get("images", []):
            images.append(Image(
                b64_json=img_data,
                url=None,
                revised_prompt=transformed.get("revised_prompt")
            ))
        
        image_response = ImageResponse(
            created=int(response.headers.get("date", 0)),
            data=images
        )
        
        # Calculate and set cost
        cost = CostTracker.calculate_image_generation_cost(
            image_response,
            model,
            config.cost_tracking
        )
        image_response._hidden_params = {"response_cost": cost}
        
        return image_response
    
    @staticmethod
    def image_generation(
        prompt: str,
        model: str,
        provider_config_name: str,
        optional_params: Optional[Dict] = None,
        **kwargs
    ) -> ImageResponse:
        """Sync version of aimage_generation"""
        import asyncio
        return asyncio.run(
            JSONProviderImageGeneration.aimage_generation(
                prompt, model, provider_config_name, optional_params, **kwargs
            )
        )
```

### Phase 3: Provider Configuration

#### File: `litellm/llms/json_providers/sdk_providers.json`

```json
{
  "_comment": "SDK-level provider configurations with transformations and cost tracking",
  "_version": "1.0.0",
  
  "google_imagen": {
    "provider_name": "google_imagen",
    "provider_type": "image_generation",
    "api_base": "https://generativelanguage.googleapis.com/v1beta",
    "api_base_env": "GOOGLE_IMAGEN_API_BASE",
    
    "authentication": {
      "type": "query_param",
      "env_var": "GOOGLE_API_KEY",
      "param_name": "key"
    },
    
    "endpoints": {
      "generate": {
        "path": "/models/{model}:predict",
        "method": "POST",
        "supported_models": [
          "imagen-3.0-generate-001",
          "imagen-3.0-fast-generate-001",
          "imagen-3.0-capability-generate-001"
        ]
      }
    },
    
    "transformations": {
      "request": {
        "type": "jinja",
        "template": {
          "instances": [
            {
              "prompt": "{{ prompt }}"
            }
          ],
          "parameters": {
            "sampleCount": "{{ n }}",
            "aspectRatio": "{{ aspect_ratio if aspect_ratio else '1:1' }}",
            "negativePrompt": "{{ negative_prompt if negative_prompt else none }}"
          }
        }
      },
      "response": {
        "type": "jsonpath",
        "mappings": {
          "images": "$.predictions[*].bytesBase64Encoded",
          "format": "b64_json"
        }
      }
    },
    
    "cost_tracking": {
      "enabled": true,
      "cost_per_image": {
        "imagen-3.0-generate-001": 0.04,
        "imagen-3.0-fast-generate-001": 0.02,
        "imagen-3.0-capability-generate-001": 0.04
      },
      "unit": "per_image"
    },
    
    "capabilities": {
      "max_images": 4,
      "supported_sizes": ["256x256", "512x512", "1024x1024"],
      "supports_negative_prompt": true,
      "supports_aspect_ratio": true
    }
  }
}
```

### Phase 4: Integration with Main LiteLLM

#### Modify: `litellm/main.py`

```python
# Add to image_generation function

def image_generation(
    prompt: str,
    model: Optional[str] = None,
    n: Optional[int] = None,
    **kwargs
) -> ImageResponse:
    """
    Generate images using various providers.
    
    Supports both hardcoded providers and JSON-configured providers.
    """
    # ... existing code ...
    
    # Check if this is a JSON-configured provider
    from litellm.llms.json_providers.sdk_provider_registry import SDKProviderRegistry
    from litellm.llms.json_providers.image_generation_handler import JSONProviderImageGeneration
    
    # Try to find JSON configuration for this provider
    custom_llm_provider = kwargs.get("custom_llm_provider")
    if custom_llm_provider:
        config = SDKProviderRegistry.get(custom_llm_provider)
        if config and config.provider_type == "image_generation":
            return JSONProviderImageGeneration.image_generation(
                prompt=prompt,
                model=model or "",
                provider_config_name=custom_llm_provider,
                optional_params={"n": n, **kwargs}
            )
    
    # ... rest of existing code ...
```

## Usage Examples

### Basic Usage

```python
import litellm

# Set API key
import os
os.environ["GOOGLE_API_KEY"] = "your-api-key"

# Generate image (automatic provider detection)
response = litellm.image_generation(
    prompt="A cute otter swimming in a pond",
    model="imagen-3.0-fast-generate-001",
    custom_llm_provider="google_imagen",
    n=1
)

print(response.data[0].b64_json)
print(f"Cost: ${response._hidden_params['response_cost']}")
```

### With Optional Parameters

```python
response = litellm.image_generation(
    prompt="Cyberpunk cityscape at sunset",
    model="imagen-3.0-generate-001",
    custom_llm_provider="google_imagen",
    n=4,
    aspect_ratio="16:9",
    negative_prompt="blurry, low quality"
)
```

### Cost Tracking

```python
response = litellm.image_generation(
    prompt="Mountain landscape",
    model="imagen-3.0-fast-generate-001",
    custom_llm_provider="google_imagen",
    n=2
)

# Cost automatically calculated based on config
cost = response._hidden_params["response_cost"]
print(f"Generated {len(response.data)} images for ${cost:.4f}")
```

## Adding OpenAI with JSON Config (Future)

```json
{
  "openai": {
    "provider_name": "openai",
    "provider_type": "image_generation",
    "api_base": "https://api.openai.com/v1",
    "api_base_env": "OPENAI_API_BASE",
    
    "authentication": {
      "type": "bearer_token",
      "env_var": "OPENAI_API_KEY"
    },
    
    "endpoints": {
      "generate": {
        "path": "/images/generations",
        "method": "POST",
        "supported_models": ["dall-e-2", "dall-e-3"]
      }
    },
    
    "transformations": {
      "request": {
        "type": "jinja",
        "template": {
          "model": "{{ model }}",
          "prompt": "{{ prompt }}",
          "n": "{{ n }}",
          "size": "{{ size }}",
          "response_format": "b64_json"
        }
      },
      "response": {
        "type": "jsonpath",
        "mappings": {
          "images": "$.data[*].b64_json",
          "format": "b64_json"
        }
      }
    },
    
    "cost_tracking": {
      "enabled": true,
      "cost_per_image": {
        "dall-e-2": 0.02,
        "dall-e-3": 0.04
      },
      "unit": "per_image"
    }
  }
}
```

## Benefits

### For Developers
- ✅ **Zero boilerplate** - Just call `litellm.image_generation()`
- ✅ **Automatic cost tracking** - Costs calculated per request
- ✅ **Consistent interface** - Same API for all providers
- ✅ **Easy to add providers** - Just JSON configuration

### For Contributors
- ✅ **No Python code** - Just JSON config
- ✅ **Declarative transformations** - Jinja/JSONPath templates
- ✅ **Built-in validation** - Pydantic schemas
- ✅ **Testable** - Easy to validate configurations

### For Maintainers
- ✅ **Centralized configs** - All providers in one place
- ✅ **Consistent patterns** - Enforced by schema
- ✅ **Easy updates** - Change cost/API without code
- ✅ **Version control** - Track config changes

## Next Steps

1. ✅ Implement core infrastructure
2. ✅ Add Google Imagen configuration
3. ✅ Test with `litellm.image_generation()`
4. ✅ Verify cost tracking works
5. ✅ Add OpenAI configuration (example)
6. ✅ Add more providers
7. ✅ Extend to chat/embeddings

## Conclusion

This SDK-level JSON configuration system provides:
- **Full SDK integration** with `litellm.image_generation()`
- **Automatic cost tracking** for billing
- **Request/response transformations** for any provider
- **Future-proof design** for adding any provider via JSON

**Result: Add new providers in 50 lines of JSON with full SDK support!** 🚀
