# Vertex AI Partner Models - Token Counting Analysis

## Executive Summary

LiteLLM has a well-structured implementation for Vertex AI partner models (Anthropic Claude, Mistral, Llama, etc.). The codebase already supports Anthropic models on Vertex AI with token counting capabilities. This document provides a comprehensive analysis of the existing structure and recommendations for leveraging the token counting API.

---

## 1. Directory Structure

### Location
```
/Users/stevegore/Code/litellm/litellm/llms/vertex_ai/vertex_ai_partner_models/
```

### Files Present
```
vertex_ai_partner_models/
├── __init__.py                          # Factory function for getting provider configs
├── main.py                              # VertexAIPartnerModels class with routing logic
├── anthropic/
│   ├── transformation.py                # VertexAIAnthropicConfig - transformation logic
│   └── experimental_pass_through/
│       └── transformation.py            # VertexAIPartnerModelsAnthropicMessagesConfig
├── ai21/
│   └── transformation.py                # VertexAIAi21Config
├── gpt_oss/
│   └── transformation.py                # VertexAIGptOssConfig
└── llama3/
    └── transformation.py                # VertexAILlama3Config
```

---

## 2. How Partner Models Are Detected & Routed

### Detection Mechanism

**Location:** `/Users/stevegore/Code/litellm/litellm/llms/vertex_ai/vertex_ai_partner_models/main.py`

The `VertexAIPartnerModels` class has a static method `is_vertex_partner_model()` that checks model prefixes:

```python
class PartnerModelPrefixes(str, Enum):
    META_PREFIX = "meta/"
    DEEPSEEK_PREFIX = "deepseek-ai"
    MISTRAL_PREFIX = "mistral"
    CODERESTAL_PREFIX = "codestral"
    JAMBA_PREFIX = "jamba"
    CLAUDE_PREFIX = "claude"          # Detects all Claude models
    QWEN_PREFIX = "qwen"
    GPT_OSS_PREFIX = "openai/gpt-oss-"

@staticmethod
def is_vertex_partner_model(model: str) -> bool:
    """Returns True if model starts with any partner prefix"""
    return any(model.startswith(prefix) for prefix in PartnerModelPrefixes)
```

### Routing Detection Points

**1. In `litellm/utils.py`:**
- Used in `get_llm_provider_chat_config()` to determine which config class to return
- If it's a partner model, returns `None` to skip the Google GenAI adapter
- This allows the request to fall through to the `litellm.completion()` adapter

**2. In `litellm/llms/vertex_ai/common_utils.py`:**
- Used to identify if model should use partner model routing

**3. Model Model Detection Flow:**
```
Request with "claude-3-5-sonnet" on vertex_ai
    ↓
is_vertex_partner_model("claude-3-5-sonnet") == True
    ↓
Route to VertexAIPartnerModels.completion()
    ↓
Detect "claude" in model name
    ↓
Use AnthropicChatCompletion handler with Vertex-specific headers
```

---

## 3. Existing Transformation Logic for Vertex AI Anthropic

### VertexAIAnthropicConfig
**File:** `/Users/stevegore/Code/litellm/litellm/llms/vertex_ai/vertex_ai_partner_models/anthropic/transformation.py`

**Key Features:**
- Extends `AnthropicConfig` - reuses all Anthropic transformation logic
- **Removes the `model` parameter** from request body (Vertex AI specifies model in URL)
- Inherits all message transformation from Anthropic SDK
- Uses direct Anthropic format - NO transformation needed

**Request Transformation:**
```python
def transform_request(self, model, messages, optional_params, ...):
    # First, transform using parent Anthropic config
    data = super().transform_request(...)

    # Then remove 'model' field (specified in Vertex URL instead)
    data.pop("model", None)

    return data
```

**Key Header Setup:**
```python
optional_params.update({
    "anthropic_version": "vertex-2023-10-16",  # Vertex-specific version
    "is_vertex_request": True,
})
```

### Completion Flow for Anthropic on Vertex AI

**File:** `/Users/stevegore/Code/litellm/litellm/llms/vertex_ai/vertex_ai_partner_models/main.py`

```python
def completion(self, model, messages, api_base, headers, optional_params, ...):
    if "claude" in model:
        # Get Vertex AI access token
        access_token, project_id = self._ensure_access_token(...)

        # Set up Vertex-specific headers
        headers.update({"Authorization": f"Bearer {access_token}"})
        optional_params.update({
            "anthropic_version": "vertex-2023-10-16",
            "is_vertex_request": True,
        })

        # Use standard AnthropicChatCompletion handler with Vertex API base
        return AnthropicChatCompletion().completion(
            model=model,
            messages=messages,
            api_base=api_base,  # Vertex endpoint
            api_key=access_token,
            custom_llm_provider="vertex_ai",
            ...
        )
```

---

## 4. Vertex AI Partner Model Count Tokens API

### API Endpoint Details

**Endpoint URL:**
```
POST https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/LOCATION/publishers/anthropic/models/count-tokens:rawPredict
```

**Region Support:**
- us-east5
- europe-west1
- asia-east1
- (and other supported regions)

### Request Format

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [
    {
      "role": "user",
      "content": "your text here"
    },
    {
      "role": "assistant",
      "content": "response here"
    }
  ]
}
```

**Note:** Uses standard Anthropic Messages API format - same as native Anthropic!

### Response Format

```json
{
  "input_tokens": 14
}
```

### Key Characteristics
- **Cost:** FREE - no cost for token counting
- **Rate Limit:** 2000 requests per minute (default)
- **Supported Models:** All Claude 3 variants and newer (Opus, Sonnet, Haiku)
- **Format:** Anthropic Messages API format (no transformation needed!)

---

## 5. Existing Token Counting Implementation

### Current Structure

**Location:** `/Users/stevegore/Code/litellm/litellm/llms/vertex_ai/count_tokens/`

```
count_tokens/
└── handler.py          # VertexAITokenCounter class
```

**VertexAITokenCounter:**
- Extends `GoogleAIStudioTokenCounter` (reuses Gemini token counting patterns)
- Extends `VertexBase` (for Vertex AI auth & URL building)
- Method: `async def validate_environment()` - returns headers and endpoint URL

### Current Implementation

```python
class VertexAITokenCounter(GoogleAIStudioTokenCounter, VertexBase):
    async def validate_environment(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, Any]] = None,
        model: str = "",
        litellm_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """Returns headers and URL for Vertex AI countTokens endpoint"""

        # Get credentials & auth
        vertex_credentials = self.get_vertex_ai_credentials(litellm_params)
        vertex_project = self.get_vertex_ai_project(litellm_params)
        vertex_location = self.get_vertex_ai_location(litellm_params)

        # Ensure access token
        _auth_header, vertex_project = await self._ensure_access_token_async(...)

        # Build endpoint URL
        auth_header, api_base = self._get_token_and_url(
            model=model,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            mode="count_tokens",
        )

        headers = {"Authorization": f"Bearer {auth_header}"}
        return headers, api_base
```

### How Vertex AI Token Counting is Called

**In `litellm/llms/vertex_ai/common_utils.py`:**

```python
async def count_tokens(
    self,
    model_to_use: str,
    messages: Optional[List[Dict[str, Any]]],
    contents: Optional[List[Dict[str, Any]]],
    deployment: Optional[Dict[str, Any]] = None,
    request_model: str = "",
) -> Optional[TokenCountResponse]:

    from litellm.llms.vertex_ai.count_tokens.handler import VertexAITokenCounter

    # Prepare parameters
    count_tokens_params_request = copy.deepcopy(
        deployment.get("litellm_params", {})
    )
    count_tokens_params = {
        "model": model_to_use,
        "contents": contents,
        "messages": messages,  # Pass messages for transformation
    }
    count_tokens_params_request.update(count_tokens_params)

    # Call token counter
    result = await VertexAITokenCounter().acount_tokens(
        **count_tokens_params_request,
    )

    # Return in LiteLLM format
    if result is not None:
        return TokenCountResponse(
            total_tokens=result.get("totalTokens", 0),
            request_model=request_model,
            model_used=model_to_use,
            tokenizer_type=result.get("tokenizer_used", ""),
            original_response=result,
        )
```

---

## 6. Reference: Bedrock Anthropic Token Counting Pattern

### Why Bedrock as Reference
Bedrock also handles Anthropic (Claude) models and has token counting implemented:

**Location:** `/Users/stevegore/Code/litellm/litellm/llms/bedrock/count_tokens/`

### Bedrock Implementation Pattern

```
count_tokens/
├── handler.py          # BedrockCountTokensHandler
└── transformation.py   # BedrockCountTokensConfig
```

**Key Features:**
1. **Transformation Logic** (`transformation.py`):
   - Converts Anthropic format to Bedrock count-tokens format
   - Handles both Converse and InvokeModel input types
   - Transforms response back to Anthropic format

2. **Handler** (`handler.py`):
   - Orchestrates the request/response flow
   - Gets AWS credentials and region
   - Signs requests with AWS SigV4
   - Handles HTTP communication

3. **Response Transformation:**
   ```python
   # Bedrock returns: {"inputTokens": 123}
   # Transformed to:  {"input_tokens": 123}
   ```

### Key Difference from Vertex AI
- Bedrock requires **request transformation** (AWS format)
- Vertex AI Claude models accept **Anthropic format directly** (no transformation!)

---

## 7. Anthropic Native Token Counting (for comparison)

**Location:** `/Users/stevegore/Code/litellm/litellm/llms/anthropic/common_utils.py`

```python
async def count_tokens(
    self,
    model_to_use: str,
    messages: Optional[List[Dict[str, Any]]],
    contents: Optional[List[Dict[str, Any]]],
    deployment: Optional[Dict[str, Any]] = None,
    request_model: str = "",
) -> Optional[TokenCountResponse]:

    from litellm.proxy.utils import count_tokens_with_anthropic_api

    # Simply call Anthropic API
    result = await count_tokens_with_anthropic_api(
        model_to_use=model_to_use,
        messages=messages,
        deployment=deployment,
    )

    if result is not None:
        return TokenCountResponse(
            total_tokens=result.get("total_tokens", 0),
            request_model=request_model,
            model_used=model_to_use,
            tokenizer_type=result.get("tokenizer_used", ""),
            original_response=result,
        )
```

---

## 8. Recommended Approach for Vertex AI Partner Models Token Counting

### Strategy: Leverage Existing Infrastructure

Given the analysis, here's the recommended approach:

### Option 1: Create Minimal Partner Model Token Counter (RECOMMENDED)

**Why:**
- Partner models (especially Claude) accept Anthropic format directly
- No request transformation needed
- Can be done with minimal code

**Implementation Steps:**

1. **Create `/litellm/llms/vertex_ai/vertex_ai_partner_models/count_tokens/`**
   ```
   count_tokens/
   ├── __init__.py
   ├── handler.py          # VertexAIPartnerModelsTokenCounter
   └── transformation.py   # VertexAIPartnerModelsCountTokensConfig
   ```

2. **transformation.py** - Minimal transformation layer:
   ```python
   class VertexAIPartnerModelsCountTokensConfig:
       def validate_count_tokens_request(self, request_data):
           # Validate model and messages
           if not request_data.get("model"):
               raise ValueError("model required")
           if not request_data.get("messages"):
               raise ValueError("messages required")

       def get_vertex_partner_count_tokens_endpoint(
           self, model, project_id, location
       ) -> str:
           """Build count-tokens endpoint for partner model"""
           # For Claude models: use anthropic publisher
           if "claude" in model:
               return f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/anthropic/models/count-tokens:rawPredict"
   ```

3. **handler.py** - Handler class:
   ```python
   class VertexAIPartnerModelsTokenCounter(VertexBase):
       async def handle_count_tokens_request(
           self,
           request_data: Dict[str, Any],
           litellm_params: Dict[str, Any],
           model: str,
       ) -> Dict[str, Any]:
           # Get Vertex AI auth
           access_token, project_id = await self._ensure_access_token_async(...)

           # Build endpoint
           endpoint_url = self.get_vertex_partner_count_tokens_endpoint(
               model=model,
               project_id=project_id,
               location=vertex_location,
           )

           # Make request (Anthropic format directly)
           response = await async_client.post(
               endpoint_url,
               headers={"Authorization": f"Bearer {access_token}"},
               json=request_data,  # Already in Anthropic format!
               timeout=30.0,
           )

           # Return response
           return {
               "input_tokens": response.json().get("input_tokens")
           }
   ```

4. **Wire into VertexAIPartnerModels.completion()**:
   - Add `count_tokens()` method to main class
   - Check if model is Claude and route to partner models token counter

5. **Wire into `litellm/llms/vertex_ai/common_utils.py`**:
   - Modify the existing `count_tokens()` method
   - Check if it's a partner model using `VertexAIPartnerModels.is_vertex_partner_model()`
   - Route to partner models token counter

### Option 2: Extend Existing VertexAITokenCounter (Alternative)

**Why:** Reuses more existing code

**How:**
- Add partner model detection to `VertexAITokenCounter`
- Add conditional endpoint building for different publishers
- Handle model-specific formatting

**Drawback:**
- Mixes Gemini and partner model logic in one class
- Requires more conditional branching

---

## 9. Implementation Checklist

### Phase 1: Create Partner Models Token Counter
- [ ] Create `/litellm/llms/vertex_ai/vertex_ai_partner_models/count_tokens/` directory
- [ ] Create `transformation.py` with config and validation
- [ ] Create `handler.py` with token counter logic
- [ ] Create `__init__.py` with factory function
- [ ] Support Anthropic (claude) models first
- [ ] Add support for other partners (mistral, llama, etc.) - can follow similar pattern

### Phase 2: Wire into Routing
- [ ] Update `VertexAIPartnerModels.completion()` method to detect and handle token counting
- [ ] Update `litellm/llms/vertex_ai/common_utils.py` count_tokens() method
- [ ] Add logic to route partner models to new token counter

### Phase 3: Testing
- [ ] Unit tests for token counter with mocked Vertex AI API
- [ ] Integration tests with real Vertex AI credentials
- [ ] Test all Claude model variants
- [ ] Test error handling (auth, malformed requests, etc.)

### Phase 4: Documentation
- [ ] Add docstrings explaining partner model token counting
- [ ] Update CLAUDE.md with partner models info
- [ ] Add example usage

---

## 10. Key Code References

### Files to Review
1. **Partner Model Detection:**
   - `/Users/stevegore/Code/litellm/litellm/llms/vertex_ai/vertex_ai_partner_models/main.py`

2. **Anthropic Transformation (reuse pattern):**
   - `/Users/stevegore/Code/litellm/litellm/llms/vertex_ai/vertex_ai_partner_models/anthropic/transformation.py`

3. **Existing Token Counter Pattern:**
   - `/Users/stevegore/Code/litellm/litellm/llms/vertex_ai/count_tokens/handler.py`

4. **Bedrock Reference (for structure):**
   - `/Users/stevegore/Code/litellm/litellm/llms/bedrock/count_tokens/handler.py`
   - `/Users/stevegore/Code/litellm/litellm/llms/bedrock/count_tokens/transformation.py`

5. **Integration Point:**
   - `/Users/stevegore/Code/litellm/litellm/llms/vertex_ai/common_utils.py` - `count_tokens()` method

---

## 11. Critical Observations

### Advantages of Vertex AI Partner Models
1. **Direct Anthropic Format Support**
   - No transformation needed (unlike Bedrock)
   - Same request/response format as native Anthropic
   - Simplified implementation

2. **Unified Routing**
   - Already have `is_vertex_partner_model()` detection
   - `VertexAIPartnerModels` class already handles routing
   - Just needs token counting addition

3. **Existing Infrastructure**
   - Vertex AI auth already implemented
   - URL building already implemented
   - Just need to add `count-tokens:rawPredict` endpoint

### What Already Works
- Anthropic model completion on Vertex AI (working in production)
- All transformation logic (already handles Vertex-specific headers)
- Authentication and access token management
- URL building for API endpoints

### What Needs to be Added
- Specific endpoint URL building for `count-tokens:rawPredict`
- Token counting request handler
- Response transformation (minimal - just extract input_tokens)
- Integration point to route count_tokens calls for partner models

---

## 12. Simplified Implementation Example

### Minimal Code to Add

```python
# File: litellm/llms/vertex_ai/vertex_ai_partner_models/count_tokens/handler.py

from typing import Any, Dict, Optional, Tuple
import httpx

from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
import litellm

class VertexAIPartnerModelsTokenCounter(VertexBase):
    """Token counter for Vertex AI Partner Models (Claude, Mistral, etc.)"""

    async def handle_count_tokens_request(
        self,
        model: str,
        request_data: Dict[str, Any],
        litellm_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle count_tokens request for Vertex AI partner models.
        Uses Anthropic Messages API format directly (no transformation needed).
        """

        # Validate
        if "messages" not in request_data:
            raise ValueError("messages required for token counting")

        # Get Vertex AI credentials
        vertex_credentials = self.get_vertex_ai_credentials(litellm_params)
        vertex_project = self.get_vertex_ai_project(litellm_params)
        vertex_location = self.get_vertex_ai_location(litellm_params)

        # Get access token
        access_token, project_id = await self._ensure_access_token_async(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )

        # Determine publisher based on model
        if "claude" in model:
            publisher = "anthropic"
        elif "mistral" in model or "codestral" in model:
            publisher = "mistralai"
        elif "llama" in model:
            publisher = "meta"
        else:
            raise ValueError(f"Unknown partner model: {model}")

        # Build endpoint
        endpoint_url = (
            f"https://{vertex_location}-aiplatform.googleapis.com/"
            f"v1/projects/{project_id}/locations/{vertex_location}/"
            f"publishers/{publisher}/models/count-tokens:rawPredict"
        )

        # Make request (Anthropic format directly!)
        headers = {"Authorization": f"Bearer {access_token}"}
        async_client = get_async_httpx_client(llm_provider=litellm.LlmProviders.VERTEX_AI)

        response = await async_client.post(
            endpoint_url,
            headers=headers,
            json=request_data,
            timeout=30.0,
        )

        if response.status_code != 200:
            raise ValueError(f"Token counting failed: {response.text}")

        result = response.json()
        return {"input_tokens": result.get("input_tokens", 0)}
```

---

## Summary

The Vertex AI Partner Models infrastructure in LiteLLM is well-designed and modular. Token counting for Claude on Vertex AI can be implemented by:

1. **Creating a minimal partner models token counter** that handles `count-tokens:rawPredict` endpoint
2. **Leveraging the existing detection logic** (`is_vertex_partner_model()`)
3. **Reusing Vertex AI auth and URL building** utilities
4. **No transformation needed** - Anthropic format is accepted directly

The implementation is straightforward because Vertex AI partner models use the standard Anthropic Messages API format, unlike Bedrock which requires format conversion.
