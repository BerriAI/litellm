# RAGFlow Provider Implementation for LiteLLM

This document describes the implementation of RAGFlow provider support in LiteLLM.

## Overview

RAGFlow is an open-source RAG (Retrieval-Augmented Generation) engine based on deep document understanding. This implementation adds first-class support for RAGFlow within LiteLLM, allowing users to interact with RAGFlow agents and datasets using the standard `litellm.completion()` interface.

## Implementation Details

### Pattern Followed

The implementation follows the same pattern as the Moonshot AI provider, which is also OpenAI-compatible:

1. Created transformation class extending `OpenAIGPTConfig`
2. Added provider to constants and routing logic
3. Created comprehensive tests
4. Added documentation

### Files Created

1. **Provider Implementation**
   - `litellm/llms/ragflow/__init__.py` - Package initialization
   - `litellm/llms/ragflow/chat/__init__.py` - Chat module initialization
   - `litellm/llms/ragflow/chat/transformation.py` - Main transformation class

2. **Tests**
   - `tests/test_litellm/llms/ragflow/__init__.py` - Test package initialization
   - `tests/test_litellm/llms/ragflow/test_ragflow_chat_transformation.py` - Comprehensive unit tests

3. **Documentation**
   - `docs/my-website/docs/providers/ragflow.md` - Provider documentation with usage examples

4. **Examples**
   - `examples/ragflow_example.py` - Python examples showing various usage patterns

### Files Modified

1. **Core Integration**
   - `litellm/constants.py`
     - Added "ragflow" to `LITELLM_CHAT_PROVIDERS`
     - Added "http://localhost:9380/v1" to `openai_compatible_endpoints`
     - Added "ragflow" to `openai_compatible_providers`
     - Added "ragflow" to `openai_text_completion_compatible_providers`

2. **Provider Configuration**
   - `litellm/__init__.py`
     - Added import: `from .llms.ragflow.chat.transformation import RAGFlowChatConfig`

3. **Provider Logic**
   - `litellm/litellm_core_utils/get_llm_provider_logic.py`
     - Added endpoint detection for "localhost:9380/v1"
     - Added provider info retrieval using `RAGFlowChatConfig`

4. **Documentation**
   - `docs/my-website/sidebars.js`
     - Added "providers/ragflow" to sidebar navigation

## Usage

### Basic Usage

```python
import os
from litellm import completion

os.environ["RAGFLOW_API_KEY"] = "your-api-key"
os.environ["RAGFLOW_API_BASE"] = "http://localhost:9380/v1"  # Optional

response = completion(
    model="ragflow/<agent_id>",
    messages=[{"role": "user", "content": "Your question here"}]
)
```

### Streaming

```python
response = completion(
    model="ragflow/<agent_id>",
    messages=[{"role": "user", "content": "Your question here"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

### With LiteLLM Proxy

```yaml
model_list:
  - model_name: my-rag-agent
    litellm_params:
      model: ragflow/<agent_id>
      api_key: os.environ/RAGFLOW_API_KEY
      api_base: http://localhost:9380/v1
```

## Features

### Supported

- ✅ Chat completions
- ✅ Streaming responses
- ✅ Custom temperature and max_tokens
- ✅ OpenAI-compatible parameters
- ✅ Message history (multi-turn conversations)
- ✅ Custom API base URL
- ✅ Environment variable configuration

### RAGFlow-Specific Features

- **Deep Document Understanding**: Advanced document parsing and chunking
- **Multiple Agent Support**: Use different RAGFlow agents as different "models"
- **Knowledge Base Integration**: Query documents through RAGFlow's retrieval system
- **Self-hosted or Cloud**: Works with both self-hosted and cloud RAGFlow instances

## Configuration

### Environment Variables

- `RAGFLOW_API_KEY`: Your RAGFlow API key (required)
- `RAGFLOW_API_BASE`: RAGFlow API base URL (optional, defaults to `http://localhost:9380/v1`)

### Model Naming Convention

Use the format: `ragflow/<agent_id>` or `ragflow/<dataset_id>`

Where:
- `agent_id` is the ID of your RAGFlow conversational agent
- `dataset_id` is the ID of your RAGFlow dataset

## Testing

The implementation includes comprehensive unit tests covering:

- Default API base configuration
- Custom API base handling
- URL construction and path handling
- Parameter mapping and transformation
- Message format handling
- Tools and tool_choice support
- Content list to string conversion

### Running Tests

```bash
# Run RAGFlow-specific tests
pytest tests/test_litellm/llms/ragflow/test_ragflow_chat_transformation.py -v

# Run all unit tests
make test-unit
```

## Architecture

### RAGFlowChatConfig Class

The main transformation class extends `OpenAIGPTConfig` and provides:

1. **Message Transformation**
   - Converts content lists to strings (RAGFlow doesn't support multimodal content format)
   - Maintains compatibility with OpenAI message format

2. **Provider Information**
   - Manages API base URL (default: http://localhost:9380/v1)
   - Handles API key retrieval from environment

3. **URL Construction**
   - Automatically appends `/chat/completions` to API base
   - Prevents duplicate path segments

4. **Parameter Mapping**
   - Maps `max_completion_tokens` to `max_tokens`
   - Passes through standard OpenAI parameters

## OpenAI Compatibility

RAGFlow exposes an OpenAI-compatible `/v1/chat/completions` endpoint, making it straightforward to integrate:

- Same request/response format as OpenAI
- Standard message structure
- Compatible parameter names
- Streaming support using SSE

## Differences from Standard OpenAI

1. **Model Parameter**: Instead of standard model names, RAGFlow uses agent/dataset IDs
2. **RAG Context**: Responses are augmented with retrieved context from knowledge bases
3. **Self-hosted**: Typically runs on localhost or private infrastructure

## Future Enhancements

Potential improvements for future versions:

1. Add RAG-specific parameters (similarity_threshold, top_k, etc.)
2. Support for RAGFlow's document upload API
3. Integration with RAGFlow's dataset management
4. Cost tracking for self-hosted instances
5. Support for RAGFlow's advanced search features

## References

- [RAGFlow GitHub Repository](https://github.com/infiniflow/ragflow)
- [RAGFlow Documentation](https://github.com/infiniflow/ragflow#-documentation)
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Issue #17112](https://github.com/BerriAI/litellm/issues/17112) - Original feature request

## Related Code

The implementation pattern follows these existing providers:

- Moonshot AI (`litellm/llms/moonshot/`) - Primary reference
- DashScope (`litellm/llms/dashscope/`) - Similar OpenAI-compatible provider
- Docker Model Runner (`litellm/llms/docker_model_runner/`) - Recent similar addition

## Verification

All changes have been verified:

- ✅ Syntax validation passed for all Python files
- ✅ Provider added to all necessary constants
- ✅ Import paths configured correctly
- ✅ Documentation added and linked in sidebar
- ✅ Example code provided
- ✅ Comprehensive unit tests created

## Notes

- RAGFlow is typically self-hosted, so the default API base uses localhost
- Users can override the API base to point to cloud or remote instances
- The implementation supports all RAGFlow agents and datasets as "models"
- No pricing information added to `model_prices_and_context_window.json` as RAGFlow is self-hosted with variable costs
