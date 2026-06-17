"""
Test router.acancel_batch() functionality

This ensures the router's batch cancellation method has test coverage.
"""

import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from litellm import Router
import litellm
from litellm.types.utils import CredentialItem


@pytest.fixture
def router():
    """Create a router with a mock deployment"""
    return Router(
        model_list=[
            {
                "model_name": "gpt-5.5",
                "litellm_params": {
                    "model": "gpt-5.5",
                    "api_key": "fake-key",
                },
            }
        ]
    )


@pytest.mark.asyncio
async def test_router_acancel_batch(router):
    """Test that router.acancel_batch() calls litellm.acancel_batch with correct params"""
    mock_response = MagicMock()
    mock_response.id = "batch_123"
    mock_response.status = "cancelled"

    with patch.object(litellm, "acancel_batch", new_callable=AsyncMock) as mock_cancel:
        mock_cancel.return_value = mock_response

        # This tests that the router method exists and can be called
        # The actual API call is mocked
        response = await router.acancel_batch(
            model="gpt-5.5",
            batch_id="batch_123",
        )

        # Verify the mock was called
        assert mock_cancel.called
        assert response.id == "batch_123"
        assert response.status == "cancelled"


@pytest.mark.asyncio
async def test_router_acancel_batch_resolves_credential_name():
    litellm.credential_list = [
        CredentialItem(
            credential_name="openai-test-credential",
            credential_info={"custom_llm_provider": "openai"},
            credential_values={"api_key": "resolved-openai-key"},
        )
    ]
    router = Router(
        model_list=[
            {
                "model_name": "gpt-5.5",
                "litellm_params": {
                    "model": "openai/gpt-5.5",
                    "litellm_credential_name": "openai-test-credential",
                },
            }
        ]
    )
    mock_response = MagicMock()
    mock_response.id = "batch_123"
    mock_response.status = "cancelled"

    try:
        with patch.object(
            litellm, "acancel_batch", new_callable=AsyncMock
        ) as mock_cancel:
            mock_cancel.return_value = mock_response

            await router.acancel_batch(
                model="gpt-5.5",
                batch_id="batch_123",
            )

        call_kwargs = mock_cancel.call_args.kwargs
        assert call_kwargs["api_key"] == "resolved-openai-key"
        assert "litellm_credential_name" not in call_kwargs
    finally:
        litellm.credential_list = []


@pytest.mark.asyncio
async def test_router_acancel_batch_removes_unresolved_credential_name():
    router = Router(
        model_list=[
            {
                "model_name": "gpt-5.5",
                "litellm_params": {
                    "model": "openai/gpt-5.5",
                    "litellm_credential_name": "missing-openai-credential",
                },
            }
        ]
    )
    mock_response = MagicMock()
    mock_response.id = "batch_123"
    mock_response.status = "cancelled"

    with (
        patch.object(
            router, "get_deployment_credentials_with_provider", return_value=None
        ),
        patch.object(litellm, "acancel_batch", new_callable=AsyncMock) as mock_cancel,
    ):
        mock_cancel.return_value = mock_response

        await router.acancel_batch(
            model="gpt-5.5",
            batch_id="batch_123",
        )

    call_kwargs = mock_cancel.call_args.kwargs
    assert "litellm_credential_name" not in call_kwargs
