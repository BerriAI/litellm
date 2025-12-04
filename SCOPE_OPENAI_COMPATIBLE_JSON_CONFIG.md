# Scope: JSON-Based OpenAI-Compatible Provider Configuration

## Problem Statement

Currently, adding an OpenAI-compatible provider to LiteLLM requires making changes across **9+ files** and understanding complex internal patterns. This creates friction for contributors and slows down provider onboarding. For example, the Amazon Nova PR from the OSS community got blocked due to implementation mistakes.

## Current Implementation Pattern (Moonshot AI Example)

To add a provider like Moonshot AI, you currently need to:

### 1. Create Provider Directory Structure
```
litellm/llms/moonshot/
└── chat/
    └── transformation.py
```

### 2. Create Transformation Config Class
```python
class MoonshotChatConfig(OpenAIGPTConfig):
    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("MOONSHOT_API_BASE") or "https://api.moonshot.ai/v1"
        dynamic_api_key = api_key or get_secret_str("MOONSHOT_API_KEY")
        return api_base, dynamic_api_key
    
    def get_supported_openai_params(self, model: str) -> list:
        # Define supported parameters
        pass
```

### 3. Modify Core Files

| File | Change Required | Example |
|------|----------------|---------|
| `litellm/__init__.py` | Import config class | `from .llms.moonshot.chat.transformation import MoonshotChatConfig` |
| `litellm/__init__.py` | Add model set | `moonshot_models: Set = set()` |
| `litellm/__init__.py` | Add to model_list | `moonshot_models \|` |
| `litellm/__init__.py` | Add to models_by_provider | `"moonshot": moonshot_models,` |
| `litellm/types/utils.py` | Add to LlmProviders enum | `MOONSHOT = "moonshot"` |
| `litellm/utils.py` | Add to ProviderConfigManager | `elif litellm.LlmProviders.MOONSHOT == provider: return litellm.MoonshotChatConfig()` |
| `litellm/litellm_core_utils/get_llm_provider_logic.py` | Add provider resolution | `elif custom_llm_provider == "moonshot": ...` |
| `provider_endpoints_support.json` | Add provider metadata | JSON entry with endpoints |
| `docs/my-website/docs/providers/moonshot.md` | Create documentation | Full provider docs |

### 4. Key Patterns

**Simple Provider (Nscale):**
```python
class NscaleConfig(OpenAIGPTConfig):
    API_BASE_URL = "https://inference.api.nscale.com/v1"
    
    def _get_openai_compatible_provider_info(self, api_base, api_key):
        resolved_api_base = api_base or get_secret_str("NSCALE_API_BASE") or self.API_BASE_URL
        resolved_api_key = api_key or get_secret_str("NSCALE_API_KEY")
        return resolved_api_base, resolved_api_key
```

**OpenAI-Like Provider (Hyperbolic):**
```python
class HyperbolicChatConfig(OpenAILikeChatConfig):
    def _get_openai_compatible_provider_info(self, api_base, api_key):
        api_base = api_base or get_secret_str("HYPERBOLIC_API_BASE") or "https://api.hyperbolic.xyz/v1"
        dynamic_api_key = api_key or get_secret_str("HYPERBOLIC_API_KEY")
        return api_base, dynamic_api_key
```

## Proposed Solution: JSON-Based Configuration

### Design Goals

1. **Single File Addition**: Adding a simple OpenAI-compatible provider should require editing only ONE JSON file
2. **Zero Python Code**: Simple providers shouldn't need any Python code
3. **Backward Compatible**: Existing providers continue working as-is
4. **Progressive Enhancement**: Complex providers can still use Python configs
5. **Auto-Registration**: Providers defined in JSON are automatically registered

### JSON Configuration Schema

Create `litellm/openai_compatible_providers.json`:

```json
{
  "$schema": "./openai_compatible_providers_schema.json",
  "providers": {
    "moonshot": {
      "display_name": "Moonshot AI",
      "description": "Moonshot AI provides large language models including moonshot-v1 and kimi models",
      "provider_type": "openai_gpt",
      "api_config": {
        "base_url": "https://api.moonshot.ai/v1",
        "api_key_env_var": "MOONSHOT_API_KEY",
        "api_base_env_var": "MOONSHOT_API_BASE",
        "endpoint": "/chat/completions"
      },
      "supported_endpoints": {
        "chat_completions": true,
        "embeddings": false,
        "image_generations": false
      },
      "supported_params": [
        "messages",
        "model",
        "max_tokens",
        "temperature",
        "top_p",
        "n",
        "stream",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
        "user",
        "tools",
        "tool_choice",
        "response_format",
        "seed"
      ],
      "excluded_params": ["functions"],
      "param_mappings": {
        "max_completion_tokens": "max_tokens"
      },
      "constraints": {
        "temperature": {
          "min": 0,
          "max": 1,
          "clamp": true
        },
        "temperature_with_n": {
          "condition": "n > 1",
          "min_temperature": 0.3,
          "adjust_to_min": true
        }
      },
      "special_handling": {
        "tool_choice_required": {
          "strategy": "append_message",
          "message": {
            "role": "user",
            "content": "Please select a tool to handle the current issue."
          },
          "remove_param": "tool_choice"
        },
        "content_format": "string_only"
      },
      "models": [
        "moonshot-v1-8k",
        "moonshot-v1-32k",
        "moonshot-v1-128k",
        "kimi-thinking-preview"
      ],
      "model_specific_overrides": {
        "kimi-thinking-preview": {
          "excluded_params": ["functions", "tools", "tool_choice"]
        }
      },
      "documentation": {
        "url": "https://platform.moonshot.ai/",
        "provider_docs": "https://docs.litellm.ai/docs/providers/moonshot"
      }
    },
    "nscale": {
      "display_name": "Nscale",
      "description": "Nscale provides inference API for various LLMs",
      "provider_type": "openai_gpt",
      "api_config": {
        "base_url": "https://inference.api.nscale.com/v1",
        "api_key_env_var": "NSCALE_API_KEY",
        "api_base_env_var": "NSCALE_API_BASE"
      },
      "supported_endpoints": {
        "chat_completions": true
      },
      "supported_params": [
        "max_tokens",
        "n",
        "temperature",
        "top_p",
        "stream",
        "logprobs",
        "top_logprobs",
        "frequency_penalty",
        "presence_penalty",
        "response_format",
        "stop",
        "logit_bias"
      ],
      "models": []
    },
    "hyperbolic": {
      "display_name": "Hyperbolic",
      "description": "Hyperbolic provides accelerated inference for AI models",
      "provider_type": "openai_like",
      "api_config": {
        "base_url": "https://api.hyperbolic.xyz/v1",
        "api_key_env_var": "HYPERBOLIC_API_KEY",
        "api_base_env_var": "HYPERBOLIC_API_BASE"
      },
      "supported_endpoints": {
        "chat_completions": true
      },
      "supported_params": [
        "messages",
        "model",
        "stream",
        "temperature",
        "top_p",
        "max_tokens",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "n",
        "tools",
        "tool_choice",
        "response_format",
        "seed",
        "user"
      ],
      "models": []
    }
  }
}
```

### JSON Schema Definition

Create `litellm/openai_compatible_providers_schema.json`:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "OpenAI-Compatible Provider Configuration",
  "type": "object",
  "properties": {
    "providers": {
      "type": "object",
      "patternProperties": {
        "^[a-z][a-z0-9_]*$": {
          "type": "object",
          "required": ["display_name", "provider_type", "api_config"],
          "properties": {
            "display_name": {
              "type": "string",
              "description": "Human-readable provider name"
            },
            "description": {
              "type": "string",
              "description": "Provider description"
            },
            "provider_type": {
              "type": "string",
              "enum": ["openai_gpt", "openai_like"],
              "description": "Base config type: openai_gpt (OpenAIGPTConfig) or openai_like (OpenAILikeChatConfig)"
            },
            "api_config": {
              "type": "object",
              "required": ["base_url", "api_key_env_var"],
              "properties": {
                "base_url": {
                  "type": "string",
                  "format": "uri"
                },
                "api_key_env_var": {
                  "type": "string"
                },
                "api_base_env_var": {
                  "type": "string"
                },
                "endpoint": {
                  "type": "string",
                  "default": "/chat/completions"
                }
              }
            },
            "supported_endpoints": {
              "type": "object",
              "properties": {
                "chat_completions": {"type": "boolean"},
                "embeddings": {"type": "boolean"},
                "image_generations": {"type": "boolean"}
              }
            },
            "supported_params": {
              "type": "array",
              "items": {"type": "string"}
            },
            "excluded_params": {
              "type": "array",
              "items": {"type": "string"}
            },
            "param_mappings": {
              "type": "object",
              "patternProperties": {
                ".*": {"type": "string"}
              }
            },
            "models": {
              "type": "array",
              "items": {"type": "string"}
            }
          }
        }
      }
    }
  }
}
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)

#### 1.1 JSON Configuration Loader
Create `litellm/llms/openai_compatible/json_config_loader.py`:

```python
from typing import Dict, Optional, List
import json
from pathlib import Path

class OpenAICompatibleProviderConfig:
    """Represents a provider configured via JSON"""
    
    def __init__(self, provider_slug: str, config_dict: dict):
        self.provider_slug = provider_slug
        self.display_name = config_dict["display_name"]
        self.provider_type = config_dict["provider_type"]
        self.api_config = config_dict["api_config"]
        self.supported_endpoints = config_dict.get("supported_endpoints", {})
        self.supported_params = config_dict.get("supported_params", [])
        self.excluded_params = config_dict.get("excluded_params", [])
        self.param_mappings = config_dict.get("param_mappings", {})
        self.models = config_dict.get("models", [])
        self.constraints = config_dict.get("constraints", {})
        self.special_handling = config_dict.get("special_handling", {})

class JSONProviderRegistry:
    """Singleton registry for JSON-configured providers"""
    
    _instance = None
    _providers: Dict[str, OpenAICompatibleProviderConfig] = {}
    
    @classmethod
    def load(cls, json_path: Optional[Path] = None):
        """Load providers from JSON configuration"""
        if json_path is None:
            json_path = Path(__file__).parent.parent.parent / "openai_compatible_providers.json"
        
        with open(json_path) as f:
            config = json.load(f)
        
        for provider_slug, provider_config in config.get("providers", {}).items():
            cls._providers[provider_slug] = OpenAICompatibleProviderConfig(
                provider_slug, provider_config
            )
    
    @classmethod
    def get_provider(cls, provider_slug: str) -> Optional[OpenAICompatibleProviderConfig]:
        """Get a provider configuration by slug"""
        return cls._providers.get(provider_slug)
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered provider slugs"""
        return list(cls._providers.keys())
    
    @classmethod
    def is_json_provider(cls, provider_slug: str) -> bool:
        """Check if a provider is defined via JSON"""
        return provider_slug in cls._providers
```

#### 1.2 Dynamic Config Class Generator
Create `litellm/llms/openai_compatible/dynamic_config.py`:

```python
from typing import Optional, Tuple, List
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig
from litellm.secret_managers.main import get_secret_str
from .json_config_loader import OpenAICompatibleProviderConfig

class DynamicOpenAICompatibleConfig:
    """Factory for creating dynamic config classes from JSON"""
    
    @staticmethod
    def create_config_class(provider_config: OpenAICompatibleProviderConfig):
        """Generate a config class dynamically from JSON configuration"""
        
        base_class = (
            OpenAIGPTConfig 
            if provider_config.provider_type == "openai_gpt" 
            else OpenAILikeChatConfig
        )
        
        class JSONBasedConfig(base_class):
            def _get_openai_compatible_provider_info(
                self, api_base: Optional[str], api_key: Optional[str]
            ) -> Tuple[Optional[str], Optional[str]]:
                resolved_api_base = (
                    api_base 
                    or get_secret_str(provider_config.api_config.get("api_base_env_var"))
                    or provider_config.api_config["base_url"]
                )
                resolved_api_key = (
                    api_key 
                    or get_secret_str(provider_config.api_config["api_key_env_var"])
                )
                return resolved_api_base, resolved_api_key
            
            def get_supported_openai_params(self, model: str) -> list:
                # Handle model-specific overrides
                if model in provider_config.get("model_specific_overrides", {}):
                    overrides = provider_config["model_specific_overrides"][model]
                    excluded = set(overrides.get("excluded_params", []))
                else:
                    excluded = set(provider_config.excluded_params)
                
                # Return supported params minus excluded ones
                return [
                    param for param in provider_config.supported_params 
                    if param not in excluded
                ]
            
            def map_openai_params(
                self, non_default_params: dict, optional_params: dict, 
                model: str, drop_params: bool
            ) -> dict:
                # Apply parameter mappings
                for source_param, target_param in provider_config.param_mappings.items():
                    if source_param in non_default_params:
                        optional_params[target_param] = non_default_params[source_param]
                        optional_params.pop(source_param, None)
                
                # Apply constraints
                if "temperature" in optional_params and "temperature" in provider_config.constraints:
                    temp_constraint = provider_config.constraints["temperature"]
                    if temp_constraint.get("clamp"):
                        optional_params["temperature"] = max(
                            temp_constraint.get("min", 0),
                            min(temp_constraint.get("max", 2), optional_params["temperature"])
                        )
                
                return super().map_openai_params(
                    non_default_params, optional_params, model, drop_params
                )
            
            @property
            def custom_llm_provider(self) -> Optional[str]:
                return provider_config.provider_slug
        
        return JSONBasedConfig
```

#### 1.3 Integration with Provider Resolution
Modify `litellm/litellm_core_utils/get_llm_provider_logic.py`:

```python
from litellm.llms.openai_compatible.json_config_loader import JSONProviderRegistry
from litellm.llms.openai_compatible.dynamic_config import DynamicOpenAICompatibleConfig

def _get_openai_compatible_provider_info(...):
    # ... existing code ...
    
    # Check if provider is defined in JSON
    if JSONProviderRegistry.is_json_provider(custom_llm_provider):
        provider_config = JSONProviderRegistry.get_provider(custom_llm_provider)
        config_class = DynamicOpenAICompatibleConfig.create_config_class(provider_config)
        api_base, dynamic_api_key = config_class()._get_openai_compatible_provider_info(
            api_base, api_key
        )
        return model, custom_llm_provider, dynamic_api_key, api_base
    
    # ... existing provider checks ...
```

#### 1.4 Integration with ProviderConfigManager
Modify `litellm/utils.py`:

```python
@staticmethod
def get_provider_chat_config(model: str, provider: LlmProviders) -> Optional[BaseConfig]:
    # Check JSON providers first
    if JSONProviderRegistry.is_json_provider(provider.value):
        provider_config = JSONProviderRegistry.get_provider(provider.value)
        return DynamicOpenAICompatibleConfig.create_config_class(provider_config)()
    
    # ... existing provider checks ...
```

#### 1.5 Auto-register JSON Providers
Modify `litellm/__init__.py` to load JSON providers on import:

```python
# Near the top of __init__.py, after imports
from litellm.llms.openai_compatible.json_config_loader import JSONProviderRegistry
from litellm.llms.openai_compatible.dynamic_config import DynamicOpenAICompatibleConfig

# Load JSON-configured providers
try:
    JSONProviderRegistry.load()
    
    # Auto-register JSON providers in LlmProviders enum
    # (This requires making LlmProviders dynamically extensible OR
    # having a separate registry that provider_list checks)
except Exception as e:
    print(f"Warning: Failed to load JSON provider configs: {e}")
```

### Phase 2: Enhanced Features (Week 2)

#### 2.1 Special Handling Support
Extend `DynamicOpenAICompatibleConfig` to support:
- `tool_choice_required` → append message strategy
- Content format conversions (list to string)
- Custom parameter validation

#### 2.2 Model-Specific Overrides
Support model-specific behavior changes (like `kimi-thinking-preview` excluding tools)

#### 2.3 Validation & Tooling
Create validation script:

```bash
python -m litellm.llms.openai_compatible.validate_config
```

This should:
- Validate JSON schema
- Check for duplicate provider names
- Verify environment variable naming conventions
- Test provider instantiation

### Phase 3: Migration & Documentation (Week 3)

#### 3.1 Migrate Simple Providers
Migrate these providers to JSON (as they're simple):
- `nscale`
- `hyperbolic`
- `novita`
- `featherless_ai`
- `aiml`

#### 3.2 Documentation
Create `docs/my-website/docs/contributing/adding-openai-compatible-providers.md`:

```markdown
# Adding OpenAI-Compatible Providers

For simple OpenAI-compatible providers, you can add support by editing a single JSON file.

## Quick Start

1. Add your provider to `litellm/openai_compatible_providers.json`
2. Test with: `litellm --model your_provider/model-name --test`
3. Submit PR

## Configuration Example
[Full example with explanations]

## When to Use Python Config Instead
- Custom authentication flows
- Complex parameter transformations
- Provider-specific streaming logic
- Response format transformations
```

## Benefits

### For Contributors
- **90% reduction** in files to modify (1 file vs 9+ files)
- **No Python knowledge** required for simple providers
- **Self-documenting** configuration format
- **Validation** catches errors before runtime

### For Maintainers
- **Easier review** - single JSON diff instead of multi-file changes
- **Consistent patterns** - all simple providers follow same structure
- **Less maintenance** - provider logic centralized
- **Faster onboarding** - reduced PR review time

### For Users
- **More providers** added faster
- **Better documentation** - generated from JSON
- **Consistent behavior** across similar providers

## Testing Strategy

### Unit Tests
Create `tests/test_litellm/llms/openai_compatible/test_json_config_loader.py`:

```python
def test_load_json_providers():
    """Test that JSON providers load correctly"""
    
def test_dynamic_config_generation():
    """Test dynamic config class creation"""
    
def test_parameter_mapping():
    """Test parameter mapping works"""
    
def test_constraints():
    """Test parameter constraints are applied"""
```

### Integration Tests
Create `tests/test_litellm/llms/openai_compatible/test_json_providers_integration.py`:

```python
def test_moonshot_via_json_config():
    """Test Moonshot provider works via JSON config"""
    
def test_provider_resolution():
    """Test provider resolution finds JSON providers"""
```

## Migration Path

### Backward Compatibility
- **All existing providers continue working** as-is
- JSON providers are **additive only**
- Gradual migration of simple providers

### Version 1.0 (Initial Release)
- Core infrastructure
- 3-5 providers migrated to JSON
- Documentation

### Version 2.0 (6 months later)
- All simple OpenAI-compatible providers in JSON
- Advanced features (complex constraints, custom streaming)
- Provider marketplace/directory

## Open Questions

1. **LlmProviders Enum**: Should we make it dynamically extensible, or maintain a separate JSON provider list?
   - **Recommendation**: Separate list initially, merge in v2.0

2. **Provider Validation**: How strict should schema validation be?
   - **Recommendation**: Strict validation with helpful error messages

3. **Complex Providers**: Where's the line between JSON and Python?
   - **Recommendation**: If it needs >50 lines of custom logic, use Python

4. **Testing**: Should JSON providers have auto-generated tests?
   - **Recommendation**: Yes, basic smoke tests auto-generated

## Success Metrics

- **Time to add provider**: <30 minutes (from hours)
- **Lines of code per provider**: ~50 lines (from 200+)
- **PR review time**: <1 day (from several days)
- **Community contributions**: 3x increase in provider PRs

## Alternative Approaches Considered

### 1. Python Dataclass Configuration
**Pros**: Type-safe, IDE support
**Cons**: Still requires Python knowledge, not as simple as JSON

### 2. YAML Configuration
**Pros**: More human-readable than JSON
**Cons**: No schema validation, parsing complexity

### 3. Provider Plugin System
**Pros**: Most flexible
**Cons**: Over-engineered for simple use case, security concerns

**Decision**: JSON with JSON Schema provides the best balance of simplicity, validation, and familiarity.

## Next Steps

1. **Review & Approve Scope** (1 day)
2. **Phase 1 Implementation** (1 week)
3. **Testing & Validation** (2 days)
4. **Phase 2 Implementation** (1 week)
5. **Documentation & Migration** (1 week)
6. **Release & Announce** (1 day)

**Total Timeline**: ~3-4 weeks

## Questions for Review

1. Is the JSON schema too complex or just right?
2. Should we support YAML as an alternative to JSON?
3. What providers should we prioritize for migration?
4. Should this be released as experimental first?
