# Anthropic /v1/messages API Support for Bedrock and Vertex AI

This implementation provides /v1/messages API support for Anthropic Claude models on Vertex AI and AWS Bedrock (Invoke & Converse).

## Implementation Status

| Provider | Family    | Streaming | Model String Example                                       |
| -------- | --------- | --------- | ---------------------------------------------------------- |
| Vertex   | Anthropic | 🔄        | vertex_ai/claude-3-sonnet@20240229                         |
| Bedrock  | Invoke    | 🔄        | bedrock/invoke/anthropic.claude-3-5-sonnet-20240620-v1:0   |
| Bedrock  | Converse  | 🔄        | bedrock/converse/anthropic.claude-3-5-sonnet-20240620-v1:0 |

Legend:

- ✅ = Implemented and tested
- 🔄 = In progress
- ❌ = Not implemented

## Implementation Plan

### Day 1: Setup and Test Updates

- ✅ Update tests to validate proper Anthropic Messages API response structure
- ✅ Refactor transformation.py to split logic into modular functions
- ✅ Create stubs for Bedrock and Vertex AI implementations

### Day 2: Refactor Current Implementation

- 🔄 Move Anthropic sync/async calls to use BaseLLMHTTPHandler
- 🔄 Implement transform_request and transform_response functions
- 🔄 Ensure streaming response handling works correctly

### Day 3: Implement Bedrock Invoke Support

- 🔄 Complete Bedrock Invoke implementation
- 🔄 Add support for streaming responses
- 🔄 Test with real AWS credentials

### Day 4: Implement Bedrock Converse and Vertex AI

- 🔄 Complete Bedrock Converse implementation
- 🔄 Complete Vertex AI implementation
- 🔄 Test both implementations with real credentials

### Day 5: Integration Testing and Documentation

- 🔄 Perform integration testing across all implementations
- 🔄 Update documentation and examples
- 🔄 Create final changelog

## Usage

Once implemented, users will be able to call Anthropic models through the OpenAI-style `/v1/messages` endpoint:

```python
# Bedrock Invoke example
response = await litellm.anthropic.messages.acreate(
    messages=[{"role": "user", "content": "Hello, Claude!"}],
    model="bedrock/invoke/anthropic.claude-3-5-sonnet-20240620-v1:0",
    max_tokens=100,
    stream=True,
)

# Bedrock Converse example
response = await litellm.anthropic.messages.acreate(
    messages=[{"role": "user", "content": "Hello, Claude!"}],
    model="bedrock/converse/anthropic.claude-3-5-sonnet-20240620-v1:0",
    max_tokens=100,
)

# Vertex AI example
response = await litellm.anthropic.messages.acreate(
    messages=[{"role": "user", "content": "Hello, Claude!"}],
    model="vertex_ai/claude-3-sonnet@20240229",
    max_tokens=100,
)
```

## Architecture

The implementation uses the following components:

1. `transformation.py` - Contains the core transformation logic for Anthropic Messages
2. `invoke_transformation.py` - Bedrock Invoke specific transformations
3. `converse_transformation.py` - Bedrock Converse specific transformations
4. `vertex_ai/anthropic_messages/transformation.py` - Vertex AI specific transformations
5. `handler.py` - Entry point for API calls, moved to use BaseLLMHTTPHandler

All implementations follow the same pattern:

- `transform_request` - Transforms the request to the provider's format
- `transform_response` - Transforms the response back to Anthropic's format
- `map_openai_params` - Maps OpenAI-style parameters to Anthropic parameters
- `transform_streaming_response` - Handles streaming responses
