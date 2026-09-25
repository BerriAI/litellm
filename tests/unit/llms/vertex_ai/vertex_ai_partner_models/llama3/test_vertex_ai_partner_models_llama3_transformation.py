import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


from litellm.llms.vertex_ai.vertex_ai_partner_models.llama3.transformation import (
    VertexAILlama3Config,
    VertexAILlama3StreamingHandler,
)


OPENAI_PLATFORM_PARAMS = (
    "prompt_cache_key",
    "prompt_cache_retention",
    "safety_identifier",
    "service_tier",
    "store",
    "web_search_options",
    "modalities",
    "prediction",
    "audio",
)

SELF_DEPLOYED_ENDPOINT_MODELS = (
    "gemma/gemma-2-2b-it",
    "vertex_ai/gemma/gemma-2-2b-it",
    "openai/mg-endpoint-lit8592",
    "vertex_ai/openai/mg-endpoint-lit8592",
    "openai/5464397967697903616",
)

MAAS_MODELS = (
    "meta/llama-4-maverick-17b-128e-instruct-maas",
    "vertex_ai/meta/llama-4-maverick-17b-128e-instruct-maas",
    "moonshotai/kimi-k2-thinking-maas",
    "qwen/qwen3-next-80b-a3b-instruct-maas",
    "google/gemma-4-26b-a4b-it-maas",
    "xai/grok-4.1-fast-non-reasoning",
    "openai/xai/grok-4.1-fast-reasoning",
    "1984786713414729728",
    "llama3",
)


class TestVertexAILlama3Config:
    def test_transform_choices(self):
        """
        Relevant Issue: https://github.com/BerriAI/litellm/issues/10441#issuecomment-2844975599
        """
        config = VertexAILlama3Config()

        choices = [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": '{"type": "function", "name": "get_weather", "parameters": {"location": "Boston, MA"}}',
                    "role": "assistant",
                },
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current temperature for a given location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City and country e.g. Bogotá, Colombia",
                            }
                        },
                        "required": ["location"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        optional_params = {"tools": tools}
        response = config._transform_choices(
            choices=choices, json_mode=False, optional_params=optional_params
        )
        assert response[0].message.tool_calls is not None
        assert response[0].finish_reason == "tool_calls"

    @pytest.mark.parametrize("model", SELF_DEPLOYED_ENDPOINT_MODELS)
    @pytest.mark.parametrize("param", OPENAI_PLATFORM_PARAMS)
    def test_get_supported_openai_params_omits_platform_params_for_self_deployed_endpoints(
        self, model: str, param: str
    ):
        assert param not in VertexAILlama3Config().get_supported_openai_params(model=model)

    @pytest.mark.parametrize("model", MAAS_MODELS)
    @pytest.mark.parametrize("param", OPENAI_PLATFORM_PARAMS)
    def test_get_supported_openai_params_keeps_platform_params_for_maas_models(self, model: str, param: str):
        assert param in VertexAILlama3Config().get_supported_openai_params(model=model)

    @pytest.mark.parametrize("model", [*SELF_DEPLOYED_ENDPOINT_MODELS, *MAAS_MODELS])
    def test_get_supported_openai_params_never_lists_max_retries(self, model: str):
        assert "max_retries" not in VertexAILlama3Config().get_supported_openai_params(model=model)

    @pytest.mark.parametrize("model", [*SELF_DEPLOYED_ENDPOINT_MODELS, *MAAS_MODELS])
    @pytest.mark.parametrize(
        "param",
        ["max_completion_tokens", "tools", "tool_choice", "response_format", "seed", "logprobs", "parallel_tool_calls"],
    )
    def test_get_supported_openai_params_keeps_params_every_vertex_openai_endpoint_accepts(
        self, model: str, param: str
    ):
        assert param in VertexAILlama3Config().get_supported_openai_params(model=model)

    @pytest.mark.parametrize("model", SELF_DEPLOYED_ENDPOINT_MODELS)
    def test_map_openai_params_drops_prompt_cache_key_for_self_deployed_endpoints(self, model: str):
        mapped = VertexAILlama3Config().map_openai_params(
            {"prompt_cache_key": "session-lit8592", "max_completion_tokens": 10},
            {},
            model,
            drop_params=True,
        )
        assert mapped == {"max_tokens": 10}

    @pytest.mark.parametrize("model", MAAS_MODELS)
    def test_map_openai_params_forwards_prompt_cache_key_for_maas_models(self, model: str):
        mapped = VertexAILlama3Config().map_openai_params(
            {"prompt_cache_key": "session-lit8592", "max_completion_tokens": 10},
            {},
            model,
            drop_params=True,
        )
        assert mapped == {"prompt_cache_key": "session-lit8592", "max_tokens": 10}


class TestVertexAILlama3StreamingHandler:
    def test_first_chunk_has_role_assistant_when_missing(self):
        """
        Vertex AI Llama streaming may return chunks without role in delta.
        The handler should inject role='assistant' on the first chunk.
        """
        handler = VertexAILlama3StreamingHandler(
            streaming_response=iter([]),
            sync_stream=True,
        )
        chunk = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "meta/llama-4-scout-17b-16e-instruct-maas",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": None, "role": None},
                    "finish_reason": None,
                }
            ],
        }
        result = handler.chunk_parser(chunk)
        assert result.choices[0].delta.role == "assistant"

    def test_subsequent_chunks_no_role_override(self):
        """
        Only the first chunk should have role injected.
        """
        handler = VertexAILlama3StreamingHandler(
            streaming_response=iter([]),
            sync_stream=True,
        )
        first_chunk = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "meta/llama-4-scout-17b-16e-instruct-maas",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello", "role": None},
                    "finish_reason": None,
                }
            ],
        }
        second_chunk = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "meta/llama-4-scout-17b-16e-instruct-maas",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " world", "role": None},
                    "finish_reason": None,
                }
            ],
        }
        first_result = handler.chunk_parser(first_chunk)
        second_result = handler.chunk_parser(second_chunk)
        assert first_result.choices[0].delta.role == "assistant"
        assert second_result.choices[0].delta.role is None

    def test_first_chunk_preserves_existing_role(self):
        """
        If the API already provides role, don't overwrite it.
        """
        handler = VertexAILlama3StreamingHandler(
            streaming_response=iter([]),
            sync_stream=True,
        )
        chunk = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "meta/llama-4-scout-17b-16e-instruct-maas",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": None, "role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }
        result = handler.chunk_parser(chunk)
        assert result.choices[0].delta.role == "assistant"
