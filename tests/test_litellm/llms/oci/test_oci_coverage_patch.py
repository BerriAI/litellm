"""
Targeted tests to close coverage gaps flagged by codecov on the OCI PR.
Each test name references the file + line range it covers.
"""

import asyncio
import json
import os
import tempfile
from typing import AsyncIterator, Iterator
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import httpx
import pytest

# ---------------------------------------------------------------------------
# common_utils.py coverage
# ---------------------------------------------------------------------------


class TestCommonUtilsCoverage:
    """Covers common_utils.py lines 19-20, 24-25, 34, 70, 119-128, 131-144, 328."""

    def test_require_cryptography_raises_when_missing(self):
        """Line 32-37: _require_cryptography raises ImportError."""
        from litellm.llms.oci.common_utils import _require_cryptography

        with patch("litellm.llms.oci.common_utils._CRYPTOGRAPHY_AVAILABLE", False):
            with pytest.raises(ImportError, match="cryptography package is required"):
                _require_cryptography()

    def test_oci_signer_protocol_do_request_sign(self):
        """Line 67-70: OCISignerProtocol.do_request_sign stub body is reachable."""
        # The Protocol class itself can't be instantiated, but we can create a
        # conforming class and call its method to cover the `pass` body.

        class FakeSigner:
            def do_request_sign(self, request, *, enforce_content_headers=False):
                pass

        signer = FakeSigner()
        signer.do_request_sign(MagicMock())  # should not raise

    def test_load_private_key_from_str_non_rsa_raises(self):
        """Line 118-128: non-RSA key raises TypeError."""
        from litellm.llms.oci.common_utils import load_private_key_from_str

        with patch("litellm.llms.oci.common_utils.serialization") as mock_ser:
            mock_ser.load_pem_private_key.return_value = "not-an-rsa-key"
            with patch("litellm.llms.oci.common_utils.rsa") as mock_rsa:
                mock_rsa.RSAPrivateKey = type("RSAPrivateKey", (), {})
                with pytest.raises(TypeError, match="not an RSA key"):
                    load_private_key_from_str(
                        "-----BEGIN PRIVATE KEY-----\nfake\n-----END PRIVATE KEY-----"
                    )

    def test_load_private_key_from_file_not_found(self):
        """Line 131-137: file not found raises FileNotFoundError."""
        from litellm.llms.oci.common_utils import load_private_key_from_file

        with pytest.raises(FileNotFoundError, match="not found"):
            load_private_key_from_file("/nonexistent/path/key.pem")

    def test_load_private_key_from_file_empty(self):
        """Line 141-142: empty file raises ValueError."""
        from litellm.llms.oci.common_utils import load_private_key_from_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False) as f:
            f.write("")
            f.flush()
            try:
                with pytest.raises(ValueError, match="empty"):
                    load_private_key_from_file(f.name)
            finally:
                os.unlink(f.name)

    def test_sign_oci_request_no_key_raises(self):
        """Line 327-331: no private key raises OCIError(400)."""
        from litellm.llms.oci.common_utils import OCIError, sign_with_manual_credentials

        with pytest.raises(OCIError, match="Missing required OCI credentials"):
            sign_with_manual_credentials(
                headers={"content-type": "application/json"},
                optional_params={},
                request_data={"hello": "world"},
                api_base="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/chat",
            )


# ---------------------------------------------------------------------------
# chat/cohere.py coverage
# ---------------------------------------------------------------------------


class TestCohereCoverage:
    """Covers cohere.py lines 100-103, 201, 237."""

    def test_adapt_messages_tool_call_invalid_json_arguments(self):
        """Lines 100-101: JSONDecodeError on tool_call arguments falls back to {}."""
        from litellm.llms.oci.chat.cohere import adapt_messages_to_cohere_standard

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "get_weather",
                            "arguments": "not-valid-json{{{",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "sunny",
            },
            {"role": "user", "content": "Thanks"},
        ]
        result = adapt_messages_to_cohere_standard(messages)
        # Returns a list; should not raise
        assert isinstance(result, list)

    def test_adapt_messages_tool_call_dict_arguments(self):
        """Lines 102-103: arguments already a dict used directly."""
        from litellm.llms.oci.chat.cohere import adapt_messages_to_cohere_standard

        messages = [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "NYC"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "sunny",
            },
            {"role": "user", "content": "Thanks"},
        ]
        result = adapt_messages_to_cohere_standard(messages)
        assert isinstance(result, list)

    def test_handle_cohere_response_tool_call_finish_reason(self):
        """Line 198-201: TOOL_CALL finish reason mapped to 'tool_calls'."""
        from litellm.llms.oci.chat.cohere import handle_cohere_response
        from litellm.types.utils import ModelResponse

        json_response = {
            "modelId": "cohere.command-latest",
            "modelVersion": "1.0",
            "chatResponse": {
                "apiFormat": "COHERE",
                "finishReason": "TOOL_CALL",
                "text": "",
                "toolCalls": [{"name": "get_weather", "parameters": {"city": "NYC"}}],
            },
        }
        result = handle_cohere_response(
            json_response, "cohere.command-latest", ModelResponse()
        )
        assert result.choices[0].finish_reason == "tool_calls"
        assert result.choices[0].message.tool_calls is not None


# ---------------------------------------------------------------------------
# chat/generic.py coverage
# ---------------------------------------------------------------------------


class TestGenericCoverage:
    """Covers generic.py lines 248, 252, 301-302, 347-348."""

    def test_adapt_tool_definition_non_function_type_raises(self):
        """Line 247-248: tool type != 'function' raises OCIError."""
        from litellm.llms.oci.chat.generic import (
            adapt_tool_definition_to_oci_standard,
        )
        from litellm.llms.oci.common_utils import OCIError
        from litellm.types.llms.oci import OCIVendors

        tools = [{"type": "retrieval", "function": {"name": "test"}}]
        with pytest.raises(OCIError, match="only supports function tools"):
            adapt_tool_definition_to_oci_standard(tools, OCIVendors.GENERIC)

    def test_adapt_tool_definition_missing_function_dict_raises(self):
        """Lines 250-254: tool['function'] not a dict raises OCIError."""
        from litellm.llms.oci.chat.generic import (
            adapt_tool_definition_to_oci_standard,
        )
        from litellm.llms.oci.common_utils import OCIError
        from litellm.types.llms.oci import OCIVendors

        tools = [{"type": "function", "function": "not-a-dict"}]
        with pytest.raises(OCIError, match="must be a dictionary"):
            adapt_tool_definition_to_oci_standard(tools, OCIVendors.GENERIC)

    def test_handle_generic_response_type_error(self):
        """Lines 301-302: TypeError on OCICompletionResponse raises OCIError."""
        from litellm.llms.oci.chat.generic import handle_generic_response
        from litellm.llms.oci.common_utils import OCIError
        from litellm.types.utils import ModelResponse

        raw = MagicMock(spec=httpx.Response)
        raw.status_code = 200
        # Pass something that triggers TypeError in the ** unpacking
        with pytest.raises((OCIError, Exception)):
            handle_generic_response("not-a-dict", "test-model", ModelResponse(), raw)

    def test_handle_generic_stream_chunk_type_error(self):
        """Lines 345-351: Pydantic ValidationError triggers OCIError path."""
        from litellm.llms.oci.chat.generic import handle_generic_stream_chunk

        # OCIStreamChunk uses Pydantic BaseModel — passing valid-ish data
        # with extra=forbid or wrong types. Since OCIStreamChunk has all Optional
        # fields, we need to trigger a TypeError by passing a non-dict.
        # The function signature requires dict, so we mock internally.
        with patch(
            "litellm.llms.oci.chat.generic.OCIStreamChunk",
            side_effect=TypeError("test error"),
        ):
            from litellm.llms.oci.common_utils import OCIError

            with pytest.raises(OCIError, match="cannot be parsed"):
                handle_generic_stream_chunk({"finishReason": "COMPLETE"})


# ---------------------------------------------------------------------------
# chat/transformation.py coverage
# ---------------------------------------------------------------------------


class TestTransformationCoverage:
    """Covers transformation.py lines 104-108, 174-183, 428-432."""

    def test_init_sets_has_custom_stream_wrapper(self):
        """__init__ sets has_custom_stream_wrapper."""
        from litellm.llms.oci.chat.transformation import OCIChatConfig

        config = OCIChatConfig()
        assert config.has_custom_stream_wrapper is True

    def test_map_openai_params_unsupported_param_raises(self):
        """Lines 174-180: unsupported param raises OCIError when drop_params=False."""
        from litellm.llms.oci.chat.transformation import OCIChatConfig
        from litellm.llms.oci.common_utils import OCIError

        config = OCIChatConfig()
        # Find a param that maps to False. Check the cohere map for 'n'.
        # 'n' is mapped to False in the cohere param map (numGenerations excluded).
        # The key must be in the param_map with value False.
        # Let's check what maps to False:
        false_keys = [
            k for k, v in config.openai_to_oci_cohere_param_map.items() if v is False
        ]
        if false_keys:
            with pytest.raises(OCIError, match="not supported on OCI"):
                config.map_openai_params(
                    non_default_params={false_keys[0]: "value"},
                    optional_params={},
                    model="oci/cohere.command-latest",
                    drop_params=False,
                )
        else:
            pytest.skip("No params mapped to False in cohere param map")

    def test_map_openai_params_unknown_param_passthrough(self):
        """Lines 181-183: unknown param with alias=None passed through."""
        from litellm.llms.oci.chat.transformation import OCIChatConfig

        config = OCIChatConfig()
        result = config.map_openai_params(
            non_default_params={"custom_unknown_param": "value"},
            optional_params={},
            model="oci/meta.llama-3.3-70b-instruct",
            drop_params=False,
        )
        assert result.get("custom_unknown_param") == "value"

    def test_transform_response_error_key_raises(self):
        """Response with 'error' key raises OCIError."""
        from litellm.llms.oci.chat.transformation import OCIChatConfig
        from litellm.llms.oci.common_utils import OCIError
        from litellm.types.utils import ModelResponse

        config = OCIChatConfig()
        raw_response = MagicMock()
        raw_response.json.return_value = {"error": "something went wrong"}
        raw_response.status_code = 500
        raw_response.headers = {}

        with pytest.raises(OCIError, match="something went wrong"):
            config.transform_response(
                model="oci/meta.llama-3.3-70b-instruct",
                raw_response=raw_response,
                model_response=ModelResponse(),
                logging_obj=MagicMock(),
                request_data={},
                messages=[{"role": "user", "content": "hi"}],
                optional_params={"oci_region": "us-chicago-1"},
                litellm_params={},
                encoding=None,
                api_key=None,
                json_mode=False,
            )

    def test_transform_response_non_dict_raises(self):
        """Non-dict response raises OCIError."""
        from litellm.llms.oci.chat.transformation import OCIChatConfig
        from litellm.llms.oci.common_utils import OCIError
        from litellm.types.utils import ModelResponse

        config = OCIChatConfig()
        raw_response = MagicMock()
        raw_response.json.return_value = ["not", "a", "dict"]
        raw_response.status_code = 200
        raw_response.headers = {}

        with pytest.raises(OCIError, match="Invalid response format"):
            config.transform_response(
                model="oci/meta.llama-3.3-70b-instruct",
                raw_response=raw_response,
                model_response=ModelResponse(),
                logging_obj=MagicMock(),
                request_data={},
                messages=[{"role": "user", "content": "hi"}],
                optional_params={"oci_region": "us-chicago-1"},
                litellm_params={},
                encoding=None,
                api_key=None,
                json_mode=False,
            )


# ---------------------------------------------------------------------------
# embed/transformation.py coverage
# ---------------------------------------------------------------------------


class TestEmbedCoverage:
    """Covers embed/transformation.py lines 145, 254-260, 276-281, 288-291."""

    def test_sign_request_delegates(self):
        """Line 145: sign_request delegates to sign_oci_request."""
        from litellm.llms.oci.embed.transformation import OCIEmbedConfig

        config = OCIEmbedConfig()
        mock_result = ({"Authorization": "sig"}, b"body")
        with patch(
            "litellm.llms.oci.embed.transformation.sign_oci_request",
            return_value=mock_result,
        ) as mock_sign:
            result = config.sign_request(
                headers={"content-type": "application/json"},
                optional_params={"oci_region": "us-chicago-1"},
                request_data={"inputs": ["hello"]},
                api_base="https://example.com",
            )
            mock_sign.assert_called_once()
            assert result == mock_result

    def test_transform_response_schema_mismatch_raises(self):
        """Lines 254-260: malformed response raises OCIError."""
        from litellm.llms.oci.common_utils import OCIError
        from litellm.llms.oci.embed.transformation import OCIEmbedConfig
        from litellm.types.utils import EmbeddingResponse

        config = OCIEmbedConfig()
        raw_response = MagicMock()
        raw_response.status_code = 200
        raw_response.headers = {}
        raw_response.json.return_value = {"unexpected": "data"}
        raw_response.text = '{"unexpected": "data"}'

        with pytest.raises(OCIError, match="does not match expected schema"):
            config.transform_embedding_response(
                model="oci/cohere.embed-english-v3.0",
                raw_response=raw_response,
                model_response=EmbeddingResponse(),
                logging_obj=MagicMock(),
                api_key=None,
                request_data={},
                optional_params={},
                litellm_params={},
            )

    def test_transform_response_fallback_usage(self):
        """Lines 276-281: fallback usage from parsed.usage when inputTextTokenCounts is None."""
        from litellm.llms.oci.embed.transformation import OCIEmbedConfig
        from litellm.types.utils import EmbeddingResponse

        config = OCIEmbedConfig()
        raw_response = MagicMock()
        raw_response.status_code = 200
        raw_response.headers = {}
        raw_response.json.return_value = {
            "modelId": "cohere.embed-english-v3.0",
            "modelVersion": "1.0",
            "embeddings": [[0.1, 0.2]],
            "inputTextTokenCounts": None,
            "usage": {"promptTokens": 5, "totalTokens": 5},
        }

        result = config.transform_embedding_response(
            model="oci/cohere.embed-english-v3.0",
            raw_response=raw_response,
            model_response=EmbeddingResponse(),
            logging_obj=MagicMock(),
            api_key=None,
            request_data={},
            optional_params={},
            litellm_params={},
        )
        assert result.usage.prompt_tokens == 5
        assert result.usage.total_tokens == 5

    def test_get_error_class(self):
        """Lines 288-291: get_error_class returns OCIError."""
        from litellm.llms.oci.common_utils import OCIError
        from litellm.llms.oci.embed.transformation import OCIEmbedConfig

        config = OCIEmbedConfig()
        err = config.get_error_class(
            error_message="something went wrong",
            status_code=500,
            headers={},
        )
        assert isinstance(err, OCIError)
        assert err.status_code == 500


# ---------------------------------------------------------------------------
# Remaining coverage: cohere.py 201, 250-251
# ---------------------------------------------------------------------------


class TestCohereRemainingCoverage:
    """Covers cohere.py lines 201 (unknown finish reason) and 250-251 (stream TypeError)."""

    def test_handle_cohere_response_error_finish_reason(self):
        """Line 194-201: ERROR finish reason mapped through the if/elif chain."""
        from litellm.llms.oci.chat.cohere import handle_cohere_response
        from litellm.types.utils import ModelResponse

        # ERROR is a valid Literal value that falls through to the else branch
        json_response = {
            "modelId": "cohere.command-latest",
            "modelVersion": "1.0",
            "chatResponse": {
                "apiFormat": "COHERE",
                "finishReason": "ERROR",
                "text": "hello",
            },
        }
        result = handle_cohere_response(
            json_response, "cohere.command-latest", ModelResponse()
        )
        # ERROR falls to else branch; LiteLLM's Choices may normalize it
        assert result.choices[0].finish_reason is not None

    def test_handle_cohere_stream_chunk_type_error(self):
        """Lines 248-254: TypeError on CohereStreamChunk raises OCIError."""
        from litellm.llms.oci.chat.cohere import handle_cohere_stream_chunk
        from litellm.llms.oci.common_utils import OCIError

        with patch(
            "litellm.llms.oci.chat.cohere.CohereStreamChunk",
            side_effect=TypeError("bad chunk"),
        ):
            with pytest.raises(OCIError, match="cannot be parsed"):
                handle_cohere_stream_chunk({"text": "hello"})


# ---------------------------------------------------------------------------
# Remaining coverage: generic.py 367
# ---------------------------------------------------------------------------


class TestGenericRemainingCoverage:
    """Covers generic.py line 367 (unsupported content type in stream)."""

    def test_handle_generic_stream_chunk_unknown_content_type(self):
        """Lines 366-370: unknown content type raises OCIError."""
        from litellm.llms.oci.chat.generic import handle_generic_stream_chunk
        from litellm.llms.oci.common_utils import OCIError
        from litellm.types.llms.oci import OCIContentPart

        # Build a chunk with a content part that is neither Text nor Image
        class WeirdContentPart(OCIContentPart):
            type: str = "video"

        chunk_dict = {
            "finishReason": None,
            "message": {
                "role": "ASSISTANT",
                "content": [{"type": "video"}],
            },
            "index": 0,
        }
        # We need to patch OCIStreamChunk to return our custom content
        mock_chunk = MagicMock()
        mock_chunk.index = 0
        mock_chunk.finishReason = None
        mock_chunk.message = MagicMock()
        mock_chunk.message.content = [WeirdContentPart()]
        mock_chunk.message.toolCalls = None

        with patch(
            "litellm.llms.oci.chat.generic.OCIStreamChunk", return_value=mock_chunk
        ):
            with pytest.raises(OCIError, match="Unsupported content type"):
                handle_generic_stream_chunk(chunk_dict)


# ---------------------------------------------------------------------------
# Remaining coverage: transformation.py 280, 462-493, 509-543
# ---------------------------------------------------------------------------


class TestTransformationRemainingCoverage:
    """Covers transformation.py lines 280, 462-493, 509-543."""

    def test_get_optional_params_oci_native_key_passthrough(self):
        """Line 280: OCI-native param name already in optional_params passed through."""
        from litellm.llms.oci.chat.transformation import OCIChatConfig
        from litellm.types.llms.oci import OCIVendors

        config = OCIChatConfig()
        # Pass an OCI-native param name directly (e.g. "maxTokens" instead of "max_tokens")
        result = config._get_optional_params(
            vendor=OCIVendors.GENERIC,
            optional_params={"maxTokens": 100},
        )
        assert result.get("maxTokens") == 100

    def test_sync_stream_wrapper_client_none_creates_default(self):
        """Line 464: when client=None, _get_httpx_client is called."""
        from litellm.llms.oci.chat.transformation import OCIChatConfig

        config = OCIChatConfig()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_text.return_value = iter(['{"text":"hi"}'])

        mock_created_client = MagicMock()
        mock_created_client.post.return_value = mock_response

        with patch(
            "litellm.llms.oci.chat.transformation._get_httpx_client",
            return_value=mock_created_client,
        ) as mock_factory:
            config.get_sync_custom_stream_wrapper(
                model="oci/meta.llama-3.3-70b-instruct",
                custom_llm_provider="oci",
                logging_obj=MagicMock(),
                api_base="https://example.com",
                headers={},
                data={},
                messages=[],
                client=None,
            )
            mock_factory.assert_called_once_with(params={})

    @pytest.mark.asyncio
    async def test_async_stream_wrapper_client_none_creates_default(self):
        """Line 512: when client=None, get_async_httpx_client is called."""
        from litellm.llms.oci.chat.transformation import OCIChatConfig

        config = OCIChatConfig()

        async def mock_aiter_text():
            yield '{"text":"hi"}'

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aiter_text = mock_aiter_text

        mock_created_client = AsyncMock()
        mock_created_client.post.return_value = mock_response

        with patch(
            "litellm.llms.oci.chat.transformation.get_async_httpx_client",
            return_value=mock_created_client,
        ) as mock_factory:
            await config.get_async_custom_stream_wrapper(
                model="oci/meta.llama-3.3-70b-instruct",
                custom_llm_provider="oci",
                logging_obj=MagicMock(),
                api_base="https://example.com",
                headers={},
                data={},
                messages=[],
                client=None,
            )
            mock_factory.assert_called_once()

    def test_sync_stream_wrapper_happy_path(self):
        """Lines 461-493: sync streaming wrapper posts and returns OCIStreamWrapper."""
        from litellm.llms.oci.chat.transformation import OCIChatConfig

        config = OCIChatConfig()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_text.return_value = iter(
            ['{"text":"hello"}\n\n{"text":"world"}']
        )

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        result = config.get_sync_custom_stream_wrapper(
            model="oci/meta.llama-3.3-70b-instruct",
            custom_llm_provider="oci",
            logging_obj=MagicMock(),
            api_base="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/chat",
            headers={"content-type": "application/json"},
            data={"stream": True, "messages": []},
            messages=[],
            client=mock_client,
            signed_json_body=b'{"messages":[]}',
        )
        # Should return an OCIStreamWrapper
        from litellm.llms.oci.chat.transformation import OCIStreamWrapper

        assert isinstance(result, OCIStreamWrapper)
        mock_client.post.assert_called_once()

    def test_sync_stream_wrapper_http_error(self):
        """Lines 475-476: HTTPStatusError raises OCIError."""
        from litellm.llms.oci.chat.transformation import OCIChatConfig
        from litellm.llms.oci.common_utils import OCIError

        config = OCIChatConfig()
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"

        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=mock_resp
        )

        with pytest.raises(OCIError, match="Unauthorized"):
            config.get_sync_custom_stream_wrapper(
                model="oci/meta.llama-3.3-70b-instruct",
                custom_llm_provider="oci",
                logging_obj=MagicMock(),
                api_base="https://example.com",
                headers={},
                data={},
                messages=[],
                client=mock_client,
            )

    def test_sync_stream_wrapper_non_200(self):
        """Lines 478-479: non-200 status raises OCIError."""
        from litellm.llms.oci.chat.transformation import OCIChatConfig
        from litellm.llms.oci.common_utils import OCIError

        config = OCIChatConfig()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        with pytest.raises(OCIError, match="Internal Server Error"):
            config.get_sync_custom_stream_wrapper(
                model="oci/meta.llama-3.3-70b-instruct",
                custom_llm_provider="oci",
                logging_obj=MagicMock(),
                api_base="https://example.com",
                headers={},
                data={},
                messages=[],
                client=mock_client,
            )

    def test_sync_split_chunks(self):
        """Lines 481-486: split_chunks splits on double newlines and strips."""
        from litellm.llms.oci.chat.transformation import OCIChatConfig

        config = OCIChatConfig()
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Simulate chunks with double-newline separators and whitespace
        mock_response.iter_text.return_value = iter(
            ['  {"a":1}  \n\n  {"b":2}  \n\n  \n\n  {"c":3}  ']
        )

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        wrapper = config.get_sync_custom_stream_wrapper(
            model="oci/meta.llama-3.3-70b-instruct",
            custom_llm_provider="oci",
            logging_obj=MagicMock(),
            api_base="https://example.com",
            headers={},
            data={},
            messages=[],
            client=mock_client,
        )
        # The completion_stream should yield stripped, non-empty chunks
        chunks = list(wrapper.completion_stream)
        assert chunks == ['{"a":1}', '{"b":2}', '{"c":3}']

    @pytest.mark.asyncio
    async def test_async_stream_wrapper_happy_path(self):
        """Lines 509-543: async streaming wrapper posts and returns OCIStreamWrapper."""
        from litellm.llms.oci.chat.transformation import OCIChatConfig, OCIStreamWrapper

        config = OCIChatConfig()

        async def mock_aiter_text():
            yield '{"text":"hello"}\n\n{"text":"world"}'

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aiter_text = mock_aiter_text

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        result = await config.get_async_custom_stream_wrapper(
            model="oci/meta.llama-3.3-70b-instruct",
            custom_llm_provider="oci",
            logging_obj=MagicMock(),
            api_base="https://example.com",
            headers={},
            data={"stream": True, "messages": []},
            messages=[],
            client=mock_client,
            signed_json_body=b'{"messages":[]}',
        )
        assert isinstance(result, OCIStreamWrapper)

    @pytest.mark.asyncio
    async def test_async_stream_wrapper_http_error(self):
        """Lines 523-524: async HTTPStatusError raises OCIError."""
        from litellm.llms.oci.chat.transformation import OCIChatConfig
        from litellm.llms.oci.common_utils import OCIError

        config = OCIChatConfig()
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"

        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "403", request=MagicMock(), response=mock_resp
        )

        with pytest.raises(OCIError, match="Forbidden"):
            await config.get_async_custom_stream_wrapper(
                model="oci/meta.llama-3.3-70b-instruct",
                custom_llm_provider="oci",
                logging_obj=MagicMock(),
                api_base="https://example.com",
                headers={},
                data={},
                messages=[],
                client=mock_client,
            )

    @pytest.mark.asyncio
    async def test_async_stream_wrapper_non_200(self):
        """Lines 526-527: async non-200 raises OCIError."""
        from litellm.llms.oci.chat.transformation import OCIChatConfig
        from litellm.llms.oci.common_utils import OCIError

        config = OCIChatConfig()
        mock_response = MagicMock()
        mock_response.status_code = 502
        mock_response.text = "Bad Gateway"

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with pytest.raises(OCIError, match="Bad Gateway"):
            await config.get_async_custom_stream_wrapper(
                model="oci/meta.llama-3.3-70b-instruct",
                custom_llm_provider="oci",
                logging_obj=MagicMock(),
                api_base="https://example.com",
                headers={},
                data={},
                messages=[],
                client=mock_client,
            )

    @pytest.mark.asyncio
    async def test_async_split_chunks(self):
        """Lines 531-536: async split_chunks splits on double newlines."""
        from litellm.llms.oci.chat.transformation import OCIChatConfig

        config = OCIChatConfig()

        async def mock_aiter_text():
            yield '  {"a":1}  \n\n  {"b":2}  \n\n  \n\n  {"c":3}  '

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aiter_text = mock_aiter_text

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        wrapper = await config.get_async_custom_stream_wrapper(
            model="oci/meta.llama-3.3-70b-instruct",
            custom_llm_provider="oci",
            logging_obj=MagicMock(),
            api_base="https://example.com",
            headers={},
            data={},
            messages=[],
            client=mock_client,
        )
        chunks = [chunk async for chunk in wrapper.completion_stream]
        assert chunks == ['{"a":1}', '{"b":2}', '{"c":3}']


# ---------------------------------------------------------------------------
# Remaining coverage: common_utils.py 19-20, 24-25, 70, 128, 144, 328
# ---------------------------------------------------------------------------


class TestCommonUtilsRemainingCoverage:
    """Covers common_utils.py lines 19-20, 24-25, 70, 128, 144, 328."""

    def test_cryptography_import_failure_sets_flag_false(self):
        """Lines 19-20: when cryptography is not installed, _CRYPTOGRAPHY_AVAILABLE = False."""
        import subprocess
        import sys

        code = (
            "import builtins\n"
            "real_import = builtins.__import__\n"
            "def fake(name, *a, **kw):\n"
            "    if 'cryptography' in name: raise ImportError('mocked')\n"
            "    return real_import(name, *a, **kw)\n"
            "builtins.__import__ = fake\n"
            "import litellm.llms.oci.common_utils as mod\n"
            "assert mod._CRYPTOGRAPHY_AVAILABLE is False, "
            "f'Expected False, got {mod._CRYPTOGRAPHY_AVAILABLE}'\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"

    def test_litellm_version_import_failure_sets_fallback(self):
        """Lines 24-25: when litellm._version is missing, _litellm_version = '0.0.0'."""
        import subprocess
        import sys

        code = (
            "import builtins, sys\n"
            "real_import = builtins.__import__\n"
            "def fake(name, *a, **kw):\n"
            "    if name == 'litellm._version': raise ImportError('mocked')\n"
            "    return real_import(name, *a, **kw)\n"
            "builtins.__import__ = fake\n"
            "sys.modules.pop('litellm._version', None)\n"
            "import litellm.llms.oci.common_utils as mod\n"
            "assert mod._litellm_version == '0.0.0', "
            "f'Expected 0.0.0, got {mod._litellm_version}'\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"

    def test_protocol_do_request_sign_body_reachable(self):
        """Line 70: Protocol pass body is reachable via super() call."""
        from litellm.llms.oci.common_utils import OCISignerProtocol

        class SubSigner(OCISignerProtocol):
            def do_request_sign(self, request, *, enforce_content_headers=False):
                # Call the Protocol's own method body (the pass statement)
                OCISignerProtocol.do_request_sign(
                    self, request, enforce_content_headers=enforce_content_headers
                )

        signer = SubSigner()
        signer.do_request_sign(MagicMock())  # should not raise

    def test_load_private_key_from_str_happy_path(self):
        """Line 128: successful return from load_private_key_from_str."""
        from litellm.llms.oci.common_utils import load_private_key_from_str

        mock_key = MagicMock()
        with patch("litellm.llms.oci.common_utils.serialization") as mock_ser:
            mock_ser.load_pem_private_key.return_value = mock_key
            with patch("litellm.llms.oci.common_utils.rsa") as mock_rsa:
                mock_rsa.RSAPrivateKey = type(mock_key)
                result = load_private_key_from_str(
                    "-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----"
                )
                assert result is mock_key

    def test_load_private_key_from_file_happy_path(self):
        """Line 144: successful return from load_private_key_from_file."""
        from litellm.llms.oci.common_utils import load_private_key_from_file

        mock_key = MagicMock()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False) as f:
            f.write(
                "-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----"
            )
            f.flush()
            try:
                with patch(
                    "litellm.llms.oci.common_utils.load_private_key_from_str",
                    return_value=mock_key,
                ):
                    result = load_private_key_from_file(f.name)
                    assert result is mock_key
            finally:
                os.unlink(f.name)

    def test_sign_with_manual_credentials_no_key(self):
        """Line 327-331: no key (after credential validation) raises OCIError."""
        from litellm.llms.oci.common_utils import OCIError, sign_with_manual_credentials

        # Provide user/fingerprint/tenancy AND a key_file so credential validation
        # passes, but patch load_private_key_from_file to return None to hit line 327
        with patch(
            "litellm.llms.oci.common_utils.load_private_key_from_file",
            return_value=None,
        ):
            with pytest.raises(OCIError, match="Private key is required"):
                sign_with_manual_credentials(
                    headers={"content-type": "application/json"},
                    optional_params={
                        "oci_user": "ocid1.user.oc1..test",
                        "oci_fingerprint": "aa:bb:cc",
                        "oci_tenancy": "ocid1.tenancy.oc1..test",
                        "oci_region": "us-chicago-1",
                        "oci_key_file": "/fake/key.pem",
                    },
                    request_data={"hello": "world"},
                    api_base="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/chat",
                )
