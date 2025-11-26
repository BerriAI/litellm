---
title: "Add Model Pricing & Context Window"
---

To add pricing or context window information for a model, simply make a PR to this file:

**[model_prices_and_context_window.json](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)**

### Sample Spec

Here's the full specification with all available fields:

```json
{
    "sample_spec": {
        "code_interpreter_cost_per_session": 0.0,
        "computer_use_input_cost_per_1k_tokens": 0.0,
        "computer_use_output_cost_per_1k_tokens": 0.0,
        "deprecation_date": "date when the model becomes deprecated in the format YYYY-MM-DD",
        "file_search_cost_per_1k_calls": 0.0,
        "file_search_cost_per_gb_per_day": 0.0,
        "input_cost_per_audio_token": 0.0,
        "input_cost_per_token": 0.0,
        "litellm_provider": "one of https://docs.litellm.ai/docs/providers",
        "max_input_tokens": "max input tokens, if the provider specifies it. if not default to max_tokens",
        "max_output_tokens": "max output tokens, if the provider specifies it. if not default to max_tokens",
        "max_tokens": "LEGACY parameter. set to max_output_tokens if provider specifies it. IF not set to max_input_tokens, if provider specifies it.",
        "mode": "one of: chat, embedding, completion, image_generation, audio_transcription, audio_speech, image_generation, moderation, rerank, search",
        "output_cost_per_reasoning_token": 0.0,
        "output_cost_per_token": 0.0,
        "search_context_cost_per_query": {
            "search_context_size_high": 0.0,
            "search_context_size_low": 0.0,
            "search_context_size_medium": 0.0
        },
        "supported_regions": [
            "global",
            "us-west-2",
            "eu-west-1",
            "ap-southeast-1",
            "ap-northeast-1"
        ],
        "supports_audio_input": true,
        "supports_audio_output": true,
        "supports_function_calling": true,
        "supports_parallel_function_calling": true,
        "supports_prompt_caching": true,
        "supports_reasoning": true,
        "supports_response_schema": true,
        "supports_system_messages": true,
        "supports_vision": true,
        "supports_web_search": true,
        "vector_store_cost_per_gb_per_day": 0.0
    }
}
```

### Examples

#### Anthropic Claude

```json
{
    "claude-3-5-haiku-20241022": {
        "cache_creation_input_token_cost": 1e-06,
        "cache_creation_input_token_cost_above_1hr": 6e-06,
        "cache_read_input_token_cost": 8e-08,
        "deprecation_date": "2025-10-01",
        "input_cost_per_token": 8e-07,
        "litellm_provider": "anthropic",
        "max_input_tokens": 200000,
        "max_output_tokens": 8192,
        "max_tokens": 8192,
        "mode": "chat",
        "output_cost_per_token": 4e-06,
        "search_context_cost_per_query": {
            "search_context_size_high": 0.01,
            "search_context_size_low": 0.01,
            "search_context_size_medium": 0.01
        },
        "supports_assistant_prefill": true,
        "supports_function_calling": true,
        "supports_pdf_input": true,
        "supports_prompt_caching": true,
        "supports_vision": true
    }
}
```

#### Vertex AI Gemini

```json
{
    "vertex_ai/gemini-3-pro-preview": {
        "cache_read_input_token_cost": 2e-07,
        "cache_read_input_token_cost_above_200k_tokens": 4e-07,
        "cache_creation_input_token_cost_above_200k_tokens": 2.5e-07,
        "input_cost_per_token": 2e-06,
        "input_cost_per_token_above_200k_tokens": 4e-06,
        "input_cost_per_token_batches": 1e-06,
        "litellm_provider": "vertex_ai",
        "max_audio_length_hours": 8.4,
        "max_audio_per_prompt": 1,
        "max_images_per_prompt": 3000,
        "max_input_tokens": 1048576,
        "max_output_tokens": 65535,
        "max_pdf_size_mb": 30,
        "max_tokens": 65535,
        "max_video_length": 1,
        "max_videos_per_prompt": 10,
        "mode": "chat",
        "output_cost_per_token": 1.2e-05,
        "output_cost_per_token_above_200k_tokens": 1.8e-05,
        "output_cost_per_token_batches": 6e-06,
        "supports_function_calling": true,
        "supports_parallel_function_calling": true,
        "supports_prompt_caching": true,
        "supports_system_messages": true,
        "supports_vision": true
    }
}
```

That's it! Your PR will be reviewed and merged.
