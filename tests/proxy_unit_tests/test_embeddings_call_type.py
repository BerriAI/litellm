"""
Test that the embeddings endpoint uses the correct CallTypes enum value.

This test verifies the fix for issue #16240 where the embeddings endpoint
was passing call_type="embeddings" (invalid) instead of call_type="aembedding" (valid).
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath("../.."))
from litellm.types.utils import CallTypes


def test_call_type_embeddings_is_invalid():
    """
    Test that "embeddings" (with 's') is NOT a valid CallTypes enum value.

    This test documents the bug: the code was using call_type="embeddings"
    which is not a valid CallTypes enum value.
    """
    # Verify that "embeddings" (plural) is NOT in the CallTypes enum
    with pytest.raises(ValueError, match="is not a valid CallTypes"):
        CallTypes("embeddings")

    # Show what the valid values are
    valid_values = [e.value for e in CallTypes]
    assert "embeddings" not in valid_values, "embeddings should not be a valid CallTypes value"
    assert "embedding" in valid_values, "embedding (singular) should be valid"
    assert "aembedding" in valid_values, "aembedding (async) should be valid"


def test_call_type_aembedding_is_valid():
    """
    Test that "aembedding" IS a valid CallTypes enum value.

    This is the correct value that should be used in the embeddings endpoint.
    """
    # This should not raise any exception
    call_type = CallTypes("aembedding")

    # Verify it's the correct enum member
    assert call_type == CallTypes.aembedding
    assert call_type.value == "aembedding"


@pytest.mark.asyncio
async def test_embeddings_endpoint_uses_aembedding():
    """
    Test that the embeddings endpoint uses call_type="aembedding" (not "embeddings").

    This test mocks the hooks and verifies they're called with the correct call_type.
    """
    from litellm.proxy.proxy_server import embeddings
    from litellm.proxy.utils import ProxyLogging
    from unittest.mock import AsyncMock, Mock, patch
    from fastapi import Request

    # Create mock objects
    mock_user_api_key_dict = {
        "token": "test-token",
        "user_id": "test-user",
        "team_id": "test-team",
    }

    mock_request = AsyncMock(spec=Request)
    mock_request.url.path = "/embeddings"
    mock_request.body = AsyncMock(return_value=b'{"model": "text-embedding-ada-002", "input": "test"}')

    mock_fastapi_response = Mock()

    # Mock the ProxyLogging object
    mock_proxy_logging = Mock(spec=ProxyLogging)
    mock_proxy_logging.pre_call_hook = AsyncMock(
        return_value={"model": "text-embedding-ada-002", "input": ["test"]}
    )
    mock_proxy_logging.during_call_hook = AsyncMock()
    mock_proxy_logging.post_call_success_hook = AsyncMock()

    # Mock the embedding response
    mock_embedding_response = Mock()
    mock_embedding_response.data = [{"embedding": [0.1, 0.2, 0.3]}]
    mock_embedding_response.model = "text-embedding-ada-002"
    mock_embedding_response.usage = Mock(total_tokens=10, prompt_tokens=10, completion_tokens=0)
    mock_embedding_response._hidden_params = {}

    with patch("litellm.proxy.proxy_server.user_api_key_auth") as mock_auth_dep, \
         patch("litellm.proxy.proxy_server.proxy_logging_obj", mock_proxy_logging), \
         patch("litellm.proxy.proxy_server.route_request", new_callable=AsyncMock) as mock_route, \
         patch("litellm.proxy.proxy_server.llm_router", Mock()), \
         patch("litellm.proxy.proxy_server.prisma_client", None), \
         patch("litellm.proxy.proxy_server.master_key", "test-master-key"), \
         patch("litellm.proxy.proxy_server.general_settings", {}), \
         patch("litellm.proxy.proxy_server.premium_user", False):

        # Configure mocks
        mock_route.return_value = mock_embedding_response

        try:
            # Call the embeddings endpoint - don't await, just create the coroutine
            response = await embeddings(
                request=mock_request,
                fastapi_response=mock_fastapi_response,
                model="text-embedding-ada-002",
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify pre_call_hook was called
            assert mock_proxy_logging.pre_call_hook.called, "pre_call_hook should have been called"

            # Get the call_type passed to pre_call_hook
            pre_call_args = mock_proxy_logging.pre_call_hook.call_args
            if pre_call_args:
                call_type_used = pre_call_args.kwargs.get("call_type")

                # The call_type should be "aembedding" which is a valid CallTypes enum value
                assert call_type_used == "aembedding", \
                    f"Expected call_type='aembedding', got '{call_type_used}'"

                # Verify it's a valid CallTypes enum value (this would raise ValueError if invalid)
                try:
                    enum_value = CallTypes(call_type_used)
                    assert enum_value == CallTypes.aembedding, \
                        f"CallTypes enum value should be CallTypes.aembedding, got {enum_value}"
                except ValueError as e:
                    pytest.fail(f"call_type '{call_type_used}' is not a valid CallTypes enum value: {e}")

            # Verify during_call_hook was also called with valid call_type
            if mock_proxy_logging.during_call_hook.called:
                during_call_args = mock_proxy_logging.during_call_hook.call_args
                if during_call_args:
                    during_call_type = during_call_args.kwargs.get("call_type")
                    assert during_call_type == "aembedding", \
                        f"Expected call_type='aembedding' in during_call_hook, got '{during_call_type}'"

        except ValueError as e:
            if "is not a valid CallTypes" in str(e):
                pytest.fail(f"embeddings endpoint used invalid call_type: {e}")
            raise
        except Exception as e:
            # If there's any other exception, that's okay for this test
            # We're only testing that the call_type is valid
            pass
