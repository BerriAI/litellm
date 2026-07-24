import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from litellm.proxy.proxy_server import chat_completion, completion, embeddings
from litellm.proxy._types import UserAPIKeyAuth
from fastapi import Request, Response


@pytest.mark.asyncio
async def test_chat_completion_metadata_population():
    # Setup
    request = MagicMock(spec=Request)
    # Mock _read_request_body to return a dict
    with patch(
        "litellm.proxy.proxy_server._read_request_body", new_callable=AsyncMock
    ) as mock_read_body:
        mock_read_body.return_value = {"model": "gpt-3.5-turbo", "messages": []}

        user_api_key_dict = UserAPIKeyAuth(
            user_id="test_user_id", team_id="test_team_id", org_id="test_org_id"
        )

        fastapi_response = MagicMock(spec=Response)

        # Mock ProxyBaseLLMRequestProcessing
        with patch(
            "litellm.proxy.proxy_server.ProxyBaseLLMRequestProcessing"
        ) as MockProcessor:
            mock_instance = MockProcessor.return_value
            mock_instance.base_process_llm_request = AsyncMock(
                return_value={"choices": []}
            )

            # Execute
            await chat_completion(
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
            )

            # Verify
            # Check if ProxyBaseLLMRequestProcessing was initialized with data containing metadata
            call_args = MockProcessor.call_args
            assert call_args is not None
            data_arg = call_args.kwargs.get("data")
            assert data_arg is not None

            assert "metadata" in data_arg
            assert data_arg["metadata"]["user_api_key_user_id"] == "test_user_id"
            assert data_arg["metadata"]["user_api_key_team_id"] == "test_team_id"
            assert data_arg["metadata"]["user_api_key_org_id"] == "test_org_id"


@pytest.mark.asyncio
async def test_embedding_metadata_population():
    """
    Test that the embedding endpoint correctly populates metadata
    from UserAPIKeyAuth.
    """
    # Setup
    with patch(
        "litellm.proxy.proxy_server.ProxyBaseLLMRequestProcessing.base_process_llm_request"
    ):
        with patch(
            "litellm.proxy.proxy_server.ProxyBaseLLMRequestProcessing.__init__",
            return_value=None,
        ) as mock_base_process_init:
            # Create a mock UserAPIKeyAuth object
            mock_user_auth = MagicMock(spec=UserAPIKeyAuth)
            mock_user_auth.user_id = "test_user_id_emb"
            mock_user_auth.team_id = "test_team_id_emb"
            mock_user_auth.org_id = "test_org_id_emb"

            # Create a mock Request object
            mock_request = MagicMock(spec=Request)
            mock_request.json = AsyncMock(
                return_value={"model": "gpt-3.5-turbo", "input": "hello"}
            )
            # Mock _read_request_body to return our data
            with patch(
                "litellm.proxy.proxy_server._read_request_body",
                new=AsyncMock(
                    return_value={"model": "gpt-3.5-turbo", "input": "hello"}
                ),
            ):
                # Call the endpoint function directly
                await embeddings(
                    request=mock_request,
                    fastapi_response=MagicMock(spec=Response),
                    user_api_key_dict=mock_user_auth,
                )

                # Check if ProxyBaseLLMRequestProcessing was initialized with the correct metadata
                mock_base_process_init.assert_called_once()
                call_args = mock_base_process_init.call_args
                # handle both positional and keyword args for data
                if "data" in call_args.kwargs:
                    data_arg = call_args.kwargs["data"]
                else:
                    data_arg = call_args.args[0]

                assert (
                    data_arg["metadata"]["user_api_key_user_id"] == "test_user_id_emb"
                )
                assert (
                    data_arg["metadata"]["user_api_key_team_id"] == "test_team_id_emb"
                )
                assert data_arg["metadata"]["user_api_key_org_id"] == "test_org_id_emb"


@pytest.mark.asyncio
async def test_completion_metadata_population():
    # Setup
    request = MagicMock(spec=Request)
    # Mock _read_request_body to return a dict
    with patch(
        "litellm.proxy.proxy_server._read_request_body", new_callable=AsyncMock
    ) as mock_read_body:
        mock_read_body.return_value = {
            "model": "gpt-3.5-turbo-instruct",
            "prompt": "test",
        }

        user_api_key_dict = UserAPIKeyAuth(
            user_id="test_user_id_2", team_id="test_team_id_2", org_id="test_org_id_2"
        )

        fastapi_response = MagicMock(spec=Response)

        # Mock ProxyBaseLLMRequestProcessing
        with patch(
            "litellm.proxy.proxy_server.ProxyBaseLLMRequestProcessing"
        ) as MockProcessor:
            mock_instance = MockProcessor.return_value
            mock_instance.base_process_llm_request = AsyncMock(
                return_value={"choices": []}
            )

            # Execute
            await completion(
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
            )

            # Verify
            call_args = MockProcessor.call_args
            assert call_args is not None
            data_arg = call_args.kwargs.get("data")
            assert data_arg is not None

            assert "metadata" in data_arg
            assert data_arg["metadata"]["user_api_key_user_id"] == "test_user_id_2"
            assert data_arg["metadata"]["user_api_key_team_id"] == "test_team_id_2"
            assert data_arg["metadata"]["user_api_key_org_id"] == "test_org_id_2"
