# Add Model to LiteLLM

Add a new model to `model_prices_and_context_window.json` with correct configuration.

## Workflow

1. **Get the model information** from the user or from a provided URL (AWS blog post, provider documentation, etc.). Extract:
   - Model name/ID
   - Provider (bedrock, openai, anthropic, etc.)
   - Pricing (input/output per token)
   - Context window (max input tokens, max output tokens)
   - Supported features (vision, function calling, etc.)

2. **Determine the correct model key format** based on provider:
   - **Bedrock**: Use the model ID directly, e.g., `anthropic.claude-opus-4-5-20251101-v1:0`, `openai.gpt-5-5-v1:0`
   - **OpenAI**: Use model name, e.g., `gpt-4o`, `o1-preview`
   - **Anthropic**: Use model name, e.g., `claude-sonnet-4-20250514`
   - **Azure**: Prefix with `azure/`, e.g., `azure/gpt-4o`
   - **Other providers**: Check existing entries for that provider's pattern

3. **Set required fields**:
   - `litellm_provider`: Provider identifier. For Bedrock models using the Converse API (most modern models), use `bedrock_converse`. For older Bedrock models, use `bedrock`.
   - `mode`: Usually `chat` for LLMs. Other options: `embedding`, `completion`, `image_generation`, `audio_transcription`, `audio_speech`, `moderation`, `rerank`, `search`
   - `input_cost_per_token`: Cost per input token (convert from "per 1M tokens" by dividing by 1,000,000)
   - `output_cost_per_token`: Cost per output token

4. **Set token limits**:
   - `max_input_tokens`: Maximum input context window
   - `max_output_tokens`: Maximum output tokens the model can generate
   - `max_tokens`: Legacy field; set to `max_output_tokens` value

5. **Set feature flags** (only include if `true`):
   - `supports_function_calling`: Tool/function calling
   - `supports_vision`: Image input support
   - `supports_prompt_caching`: Prompt caching support
   - `supports_response_schema`: Structured output/JSON schema support
   - `supports_tool_choice`: Ability to force tool selection
   - `supports_system_messages`: System message support
   - `supports_audio_input`: Audio input
   - `supports_audio_output`: Audio output
   - `supports_pdf_input`: PDF document input
   - `supports_reasoning`: Extended thinking/reasoning
   - `supports_computer_use`: Computer use capability
   - `supports_assistant_prefill`: Assistant prefill
   - `supports_parallel_function_calling`: Parallel tool calls
   - `supports_web_search`: Web search integration
   - `supports_native_structured_output`: Native structured outputs

6. **Set caching costs** (if model supports prompt caching):
   - `cache_creation_input_token_cost`: Cost to create cache
   - `cache_read_input_token_cost`: Cost to read from cache
   - `cache_creation_input_token_cost_above_1hr`: Higher tier for >1 hour cache (if applicable)

7. **Add source URL**:
   - `source`: URL to the pricing page or announcement

8. **Insert the model entry** into `model_prices_and_context_window.json`:
   - The file is sorted alphabetically by key
   - Find the correct position and insert
   - Ensure proper JSON formatting (no trailing commas, correct indentation)

## Example: Adding a Bedrock model

For a new Bedrock Claude model with:
- Model ID: `anthropic.claude-new-model-v1:0`
- Input: $3/1M tokens, Output: $15/1M tokens
- Context: 200K input, 64K output
- Supports: vision, function calling, caching, reasoning

```json
"anthropic.claude-new-model-v1:0": {
    "cache_creation_input_token_cost": 3.75e-06,
    "cache_read_input_token_cost": 3e-07,
    "input_cost_per_token": 3e-06,
    "litellm_provider": "bedrock_converse",
    "max_input_tokens": 200000,
    "max_output_tokens": 64000,
    "max_tokens": 64000,
    "mode": "chat",
    "output_cost_per_token": 1.5e-05,
    "source": "https://aws.amazon.com/bedrock/pricing/",
    "supports_assistant_prefill": true,
    "supports_function_calling": true,
    "supports_prompt_caching": true,
    "supports_reasoning": true,
    "supports_response_schema": true,
    "supports_tool_choice": true,
    "supports_vision": true
}
```

## Example: Adding OpenAI models on Bedrock

For OpenAI GPT models available on Bedrock:
- Use the Bedrock model ID format: `openai.gpt-5-5-v1:0`
- Set `litellm_provider` to `bedrock_converse`
- Include the Bedrock-specific pricing (may differ from OpenAI direct)

```json
"openai.gpt-5-5-v1:0": {
    "input_cost_per_token": 2.5e-06,
    "litellm_provider": "bedrock_converse",
    "max_input_tokens": 128000,
    "max_output_tokens": 32768,
    "max_tokens": 32768,
    "mode": "chat",
    "output_cost_per_token": 1e-05,
    "source": "https://aws.amazon.com/bedrock/pricing/",
    "supports_function_calling": true,
    "supports_response_schema": true,
    "supports_tool_choice": true,
    "supports_vision": true
}
```

## Price conversion reference

AWS and many providers list prices per 1 million tokens. Convert to per-token:
- $3.00 per 1M tokens = 3 / 1,000,000 = 0.000003 = 3e-06
- $0.25 per 1M tokens = 0.25 / 1,000,000 = 0.00000025 = 2.5e-07
- $15.00 per 1M tokens = 15 / 1,000,000 = 0.000015 = 1.5e-05

Cache pricing typically follows these patterns:
- Cache creation cost: Usually 1.25x the base input cost
- Cache read cost: Usually 0.1x the base input cost (10% of base)

## Validation checklist

Before committing:
- [ ] Model key is in correct format for the provider
- [ ] `litellm_provider` matches provider conventions
- [ ] Prices are per-token (not per-1M-tokens)
- [ ] `max_tokens` equals `max_output_tokens`
- [ ] Feature flags match actual model capabilities (don't guess)
- [ ] Entry is inserted in alphabetical order
- [ ] JSON is valid (no trailing commas, proper formatting)
- [ ] Source URL is included

## Testing

After adding the model, test with:

```bash
python litellm/proxy/proxy_cli.py --config litellm/proxy/dev_config.yaml --detailed_debug --reload
```

Then curl the proxy to verify the model works:

```bash
curl -X POST http://localhost:4000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bedrock/anthropic.claude-new-model-v1:0",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```
