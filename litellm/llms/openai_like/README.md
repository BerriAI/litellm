## Adding a New Provider

### For Simple OpenAI-Compatible Providers

**That's it!** Just edit `providers.json` and add your provider:

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY"
  }
}
```

The provider will be automatically:
- ✅ Loaded and available for use
- ✅ Added to provider resolution logic
- ✅ Included in endpoint detection
- ✅ Added to OpenAI-compatible provider lists
- ✅ Available in the LlmProviders enum (via dynamic lookup)
- ✅ Supported in all OpenAI-compatible endpoints

**No other files need to be modified!** Everything is handled automatically.

### 验证 Provider 是否成功加载

添加 provider 后，你可以通过以下方式验证是否成功加载：

#### 方法 1: 使用检查脚本

运行项目根目录下的检查脚本：

```bash
poetry run python check_json_providers.py
```

或者检查特定 provider：

```bash
poetry run python check_json_providers.py publicai
```

#### 方法 2: 在代码中检查

```python
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.constants import openai_compatible_providers
from litellm.types.utils import get_llm_provider_enum

# 确保加载
JSONProviderRegistry.load()

# 1. 检查 provider 是否存在
provider_slug = "publicai"
if JSONProviderRegistry.exists(provider_slug):
    print(f"✅ Provider '{provider_slug}' 已加载")
    
    # 获取配置信息
    config = JSONProviderRegistry.get(provider_slug)
    print(f"   Base URL: {config.base_url}")
    print(f"   API Key Env: {config.api_key_env}")

# 2. 列出所有已加载的 JSON providers
all_providers = JSONProviderRegistry.list_providers()
print(f"已加载的 providers: {all_providers}")

# 3. 检查是否在 openai_compatible_providers 中
if provider_slug in openai_compatible_providers:
    print(f"✅ Provider '{provider_slug}' 已集成到 openai_compatible_providers")

# 4. 测试 provider enum 解析
try:
    enum_obj = get_llm_provider_enum(provider_slug)
    print(f"✅ Provider enum 解析成功: {enum_obj}")
except ValueError as e:
    print(f"❌ Provider enum 解析失败: {e}")

# 5. 测试 provider 解析逻辑
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

config = JSONProviderRegistry.get(provider_slug)
if config:
    model, resolved_provider, api_key, api_base = get_llm_provider(
        model="gpt-3.5-turbo",
        api_base=config.base_url
    )
    if resolved_provider == provider_slug:
        print(f"✅ Provider 解析成功: {resolved_provider}")
    else:
        print(f"⚠️  Provider 解析为: {resolved_provider} (期望: {provider_slug})")
```

#### 方法 3: 实际使用测试

最简单的方式是直接使用 provider：

```python
import litellm

# 使用 JSON 配置的 provider
response = litellm.completion(
    model="publicai/gpt-3.5-turbo",  # 格式: provider_slug/model_name
    messages=[{"role": "user", "content": "Hello!"}],
    api_key="your-api-key"
)
```

如果 provider 未正确加载，会抛出相应的错误。

### Optional Configuration Fields

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY",
    
    // Optional: Override base_url via environment variable
    "api_base_env": "YOUR_PROVIDER_API_BASE",
    
    // Optional: Which base class to use (default: "openai_gpt")
    "base_class": "openai_gpt",  // or "openai_like"
    
    // Optional: Parameter name mappings
    "param_mappings": {
      "max_completion_tokens": "max_tokens"
    },
    
    // Optional: Parameter constraints
    "constraints": {
      "temperature_max": 1.0,
      "temperature_min": 0.0,
      "temperature_min_with_n_gt_1": 0.3
    },
    
    // Optional: Special handling flags
    "special_handling": {
      "convert_content_list_to_string": true
    }
  }
}
```

## Example: PublicAI

The first JSON-configured provider:

```json
{
  "publicai": {
    "base_url": "https://api.publicai.co/v1",
    "api_key_env": "PUBLICAI_API_KEY",
    "api_base_env": "PUBLICAI_API_BASE",
    "base_class": "openai_gpt",
    "param_mappings": {
      "max_completion_tokens": "max_tokens"
    },
    "special_handling": {
      "convert_content_list_to_string": true
    }
  }
}
```

## Usage

```python
import litellm

response = litellm.completion(
    model="publicai/swiss-ai/apertus-8b-instruct",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Benefits

- **Simple**: 2-5 lines of JSON vs 100+ lines of Python
- **Fast**: Add a provider in 5 minutes
- **Safe**: No Python code to mess up
- **Consistent**: All providers follow the same pattern
- **Maintainable**: Centralized configuration

## When to Use Python Instead

Use a Python config class if you need:
- Custom authentication (OAuth, rotating tokens, etc.)
- Complex request/response transformations
- Provider-specific streaming logic
- Advanced tool calling transformations

## Implementation Details

### How It Works

1. `json_loader.py` loads `providers.json` on import
2. `dynamic_config.py` generates config classes on-demand
3. Provider resolution checks JSON registry first
4. ProviderConfigManager returns JSON-based configs

### Integration Points

The JSON system is automatically integrated at:
- `litellm/litellm_core_utils/get_llm_provider_logic.py` - Provider resolution and endpoint detection
- `litellm/utils.py` - ProviderConfigManager
- `litellm/constants.py` - Dynamic generation of `openai_compatible_providers`, `openai_compatible_endpoints`, and `openai_text_completion_compatible_providers`
- `litellm/types/utils.py` - Dynamic `LlmProvidersSet` that includes JSON providers
- `litellm/proxy/vector_store_files_endpoints/endpoints.py` - Enum validation with JSON provider support

All integration happens automatically - no manual updates required!