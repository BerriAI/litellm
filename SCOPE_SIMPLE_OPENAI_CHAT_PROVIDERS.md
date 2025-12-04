# Scope: Simple JSON Config for OpenAI-Compatible Chat Providers

## Problem
Adding an OpenAI-compatible chat provider requires editing 9+ files. The Amazon Nova PR got blocked because of implementation mistakes.

## Solution: Single JSON File

### Simplest Possible Case
For a provider that's fully OpenAI-compatible, you only need:

```json
{
  "providers": {
    "your_provider": {
      "base_url": "https://api.yourprovider.com/v1",
      "api_key_env": "YOUR_PROVIDER_API_KEY"
    }
  }
}
```

That's it. Everything else is optional.

## JSON Configuration File

Create `litellm/llms/openai_like/providers.json`:

```json
{
  "hyperbolic": {
    "base_url": "https://api.hyperbolic.xyz/v1",
    "api_key_env": "HYPERBOLIC_API_KEY"
  },
  
  "nscale": {
    "base_url": "https://inference.api.nscale.com/v1",
    "api_key_env": "NSCALE_API_KEY"
  },
  
  "moonshot": {
    "base_url": "https://api.moonshot.ai/v1",
    "api_key_env": "MOONSHOT_API_KEY",
    "api_base_env": "MOONSHOT_API_BASE",
    "param_mappings": {
      "max_completion_tokens": "max_tokens"
    },
    "constraints": {
      "temperature_max": 1.0,
      "temperature_min_with_n_gt_1": 0.3
    }
  }
}
```

## Complete JSON Schema

All fields except `base_url` and `api_key_env` are optional:

```json
{
  "provider_slug": {
    // REQUIRED
    "base_url": "https://api.provider.com/v1",
    "api_key_env": "PROVIDER_API_KEY",
    
    // OPTIONAL - for overriding base_url
    "api_base_env": "PROVIDER_API_BASE",
    
    // OPTIONAL - rename parameters
    "param_mappings": {
      "openai_param_name": "provider_param_name"
    },
    
    // OPTIONAL - parameter constraints
    "constraints": {
      "temperature_min": 0.0,
      "temperature_max": 1.0,
      "temperature_clamp": true,
      "temperature_min_with_n_gt_1": 0.3
    },
    
    // OPTIONAL - which base class to use
    "base_class": "openai_gpt"  // or "openai_like" (default: "openai_gpt")
  }
}
```

## Implementation

### File Structure
```
litellm/llms/openai_like/
  ├── providers.json          ← NEW: All provider configs
  ├── json_loader.py          ← NEW: Load and parse JSON
  ├── dynamic_config.py       ← NEW: Generate config classes
  └── chat/
      ├── handler.py          ← Existing
      └── transformation.py   ← Existing
```

### 1. JSON Loader (`litellm/llms/openai_like/json_loader.py`)

```python
import json
from pathlib import Path
from typing import Dict, Optional

class SimpleProviderConfig:
    """Simple data class for JSON provider config"""
    def __init__(self, slug: str, data: dict):
        self.slug = slug
        self.base_url = data["base_url"]
        self.api_key_env = data["api_key_env"]
        self.api_base_env = data.get("api_base_env")
        self.param_mappings = data.get("param_mappings", {})
        self.constraints = data.get("constraints", {})
        self.base_class = data.get("base_class", "openai_gpt")

class JSONProviderRegistry:
    """Load providers from JSON once on import"""
    _providers: Dict[str, SimpleProviderConfig] = {}
    _loaded = False
    
    @classmethod
    def load(cls):
        if cls._loaded:
            return
        
        json_path = Path(__file__).parent / "providers.json"
        with open(json_path) as f:
            data = json.load(f)
        
        for slug, config in data.items():
            cls._providers[slug] = SimpleProviderConfig(slug, config)
        
        cls._loaded = True
    
    @classmethod
    def get(cls, slug: str) -> Optional[SimpleProviderConfig]:
        return cls._providers.get(slug)
    
    @classmethod
    def exists(cls, slug: str) -> bool:
        return slug in cls._providers

# Load on import
JSONProviderRegistry.load()
```

### 2. Dynamic Config Generator (`litellm/llms/openai_like/dynamic_config.py`)

```python
from typing import Optional, Tuple
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig
from litellm.secret_managers.main import get_secret_str
from .json_loader import SimpleProviderConfig

def create_config_class(provider: SimpleProviderConfig):
    """Generate config class from JSON"""
    
    # Choose base class
    base_class = (
        OpenAIGPTConfig if provider.base_class == "openai_gpt" 
        else OpenAILikeChatConfig
    )
    
    class JSONProviderConfig(base_class):
        
        def _get_openai_compatible_provider_info(
            self, api_base: Optional[str], api_key: Optional[str]
        ) -> Tuple[Optional[str], Optional[str]]:
            """Get API base and key from JSON config"""
            
            # Resolve base URL
            resolved_base = api_base
            if not resolved_base and provider.api_base_env:
                resolved_base = get_secret_str(provider.api_base_env)
            if not resolved_base:
                resolved_base = provider.base_url
            
            # Resolve API key
            resolved_key = api_key or get_secret_str(provider.api_key_env)
            
            return resolved_base, resolved_key
        
        def map_openai_params(
            self, non_default_params: dict, optional_params: dict, 
            model: str, drop_params: bool
        ) -> dict:
            """Apply parameter mappings and constraints"""
            
            # Call parent first
            optional_params = super().map_openai_params(
                non_default_params, optional_params, model, drop_params
            )
            
            # Apply parameter mappings
            for openai_param, provider_param in provider.param_mappings.items():
                if openai_param in non_default_params:
                    optional_params[provider_param] = non_default_params[openai_param]
                    optional_params.pop(openai_param, None)
            
            # Apply temperature constraints
            if "temperature" in optional_params:
                temp = optional_params["temperature"]
                constraints = provider.constraints
                
                # Clamp to max
                if "temperature_max" in constraints:
                    temp = min(temp, constraints["temperature_max"])
                
                # Clamp to min
                if "temperature_min" in constraints:
                    temp = max(temp, constraints["temperature_min"])
                
                # Special case: temperature_min_with_n_gt_1
                if "temperature_min_with_n_gt_1" in constraints:
                    n = optional_params.get("n", 1)
                    if n > 1 and temp < constraints["temperature_min_with_n_gt_1"]:
                        temp = constraints["temperature_min_with_n_gt_1"]
                
                optional_params["temperature"] = temp
            
            return optional_params
        
        @property
        def custom_llm_provider(self) -> Optional[str]:
            return provider.slug
    
    return JSONProviderConfig
```

### 3. Wire into Provider Resolution (`litellm/litellm_core_utils/get_llm_provider_logic.py`)

Add to `_get_openai_compatible_provider_info()`:

```python
def _get_openai_compatible_provider_info(...):
    # ... existing code ...
    
    # Check JSON providers FIRST (before hardcoded ones)
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry
    from litellm.llms.openai_like.dynamic_config import create_config_class
    
    if JSONProviderRegistry.exists(custom_llm_provider):
        provider_config = JSONProviderRegistry.get(custom_llm_provider)
        config_class = create_config_class(provider_config)
        api_base, dynamic_api_key = config_class()._get_openai_compatible_provider_info(
            api_base, api_key
        )
        return model, custom_llm_provider, dynamic_api_key, api_base
    
    # ... existing provider checks ...
```

### 4. Wire into ProviderConfigManager (`litellm/utils.py`)

Add to `get_provider_chat_config()`:

```python
@staticmethod
def get_provider_chat_config(model: str, provider: LlmProviders) -> Optional[BaseConfig]:
    # Check JSON providers FIRST
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry
    from litellm.llms.openai_like.dynamic_config import create_config_class
    
    if JSONProviderRegistry.exists(provider.value):
        provider_config = JSONProviderRegistry.get(provider.value)
        return create_config_class(provider_config)()
    
    # ... existing provider checks ...
```

### 5. Add to LlmProviders Enum

The only manual step - add to `litellm/types/utils.py`:

```python
class LlmProviders(str, Enum):
    # ... existing providers ...
    HYPERBOLIC = "hyperbolic"
    NSCALE = "nscale"
    MOONSHOT = "moonshot"
```

## To Add a New Provider

### For 99% of providers (fully OpenAI-compatible):

1. Edit `litellm/llms/openai_like/providers.json`:
```json
{
  "newprovider": {
    "base_url": "https://api.newprovider.com/v1",
    "api_key_env": "NEWPROVIDER_API_KEY"
  }
}
```

2. Add to enum in `litellm/types/utils.py`:
```python
NEWPROVIDER = "newprovider"
```

3. Add to `provider_list` in `litellm/__init__.py` (if not auto-generated)

**That's it. 3 lines total.**

### For the 1% with quirks (like Moonshot):

Add optional fields to the JSON:

```json
{
  "quirkyprovider": {
    "base_url": "https://api.quirky.com/v1",
    "api_key_env": "QUIRKY_API_KEY",
    "param_mappings": {
      "max_completion_tokens": "max_tokens"
    },
    "constraints": {
      "temperature_max": 1.0
    }
  }
}
```

## What About Complex Providers?

If a provider needs more than param mappings and simple constraints, keep using Python:
- Custom authentication (OAuth, rotating tokens)
- Request/response transformations
- Special streaming logic
- Tool calling transformations

**Rule of thumb**: If you need more than 5 optional JSON fields, use Python.

## Implementation Plan

### Week 1: Core Infrastructure
- Day 1-2: Implement `json_loader.py` and `dynamic_config.py`
- Day 3: Wire into provider resolution
- Day 4: Write tests
- Day 5: Documentation

### Week 2: Migration & Validation
- Migrate 3-5 simple providers to JSON
- Add validation script
- Test thoroughly

**Total: 2 weeks**

## Testing

```python
# tests/test_litellm/llms/openai_like/test_json_providers.py

def test_load_json_providers():
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry
    
    assert JSONProviderRegistry.exists("hyperbolic")
    assert JSONProviderRegistry.exists("moonshot")
    
    hyperbolic = JSONProviderRegistry.get("hyperbolic")
    assert hyperbolic.base_url == "https://api.hyperbolic.xyz/v1"

def test_dynamic_config_creation():
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry
    from litellm.llms.openai_like.dynamic_config import create_config_class
    
    provider = JSONProviderRegistry.get("moonshot")
    config_class = create_config_class(provider)
    config = config_class()
    
    # Test parameter mapping
    optional_params = {}
    non_default_params = {"max_completion_tokens": 100}
    result = config.map_openai_params(non_default_params, optional_params, "moonshot-v1-8k", False)
    
    assert result["max_tokens"] == 100
    assert "max_completion_tokens" not in result

def test_temperature_constraints():
    # Test temperature clamping works
    pass

def test_provider_resolution():
    # Test that completion() works with JSON provider
    pass
```

## Files Changed

### New Files (3)
- `litellm/llms/openai_like/providers.json`
- `litellm/llms/openai_like/json_loader.py`
- `litellm/llms/openai_like/dynamic_config.py`

### Modified Files (3)
- `litellm/litellm_core_utils/get_llm_provider_logic.py` (add 10 lines)
- `litellm/utils.py` (add 10 lines)
- `litellm/types/utils.py` (add 1 line per provider to enum)

**Total: 6 files vs 9+ files before**

## Benefits

✅ **Simple**: 2-3 lines of JSON for most providers
✅ **Safe**: No Python code to mess up
✅ **Fast**: Add provider in 5 minutes
✅ **Flexible**: Can still use Python for complex cases
✅ **Backward Compatible**: All existing providers work unchanged

## Example: Adding "DeepInfra" Style Provider

**Before** (9+ files):
```
litellm/llms/deepinfra/
  chat/transformation.py (50 lines)
litellm/__init__.py (5 changes)
litellm/types/utils.py (1 change)
litellm/utils.py (3 changes)
litellm/litellm_core_utils/get_llm_provider_logic.py (10 lines)
provider_endpoints_support.json (20 lines)
docs/providers/deepinfra.md (100 lines)
```

**After** (1 file + enum):
```json
{
  "deepinfra": {
    "base_url": "https://api.deepinfra.com/v1/openai",
    "api_key_env": "DEEPINFRA_API_KEY"
  }
}
```
Plus 1 line in LlmProviders enum.

**Reduction**: 200+ lines → 3 lines (98% reduction)
