import pytest
from unittest.mock import patch, MagicMock
import litellm
import httpx

@pytest.mark.asyncio
async def test_ssl_verify_false():
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.post.return_value = MagicMock(status_code=200, json=lambda: {"choices": [{"message": {"content": "hello"}}]})
        
        response = await litellm.acompletion(
            model="openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            ssl_verify=False,
            api_key="sk-123"
        )
        # Check that AsyncClient was initialized with verify=False
        called_kwargs = mock_client.call_args[1]
        assert called_kwargs.get("verify") is False

@pytest.mark.asyncio
async def test_ssl_verify_custom_ca():
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.post.return_value = MagicMock(status_code=200, json=lambda: {"choices": [{"message": {"content": "hello"}}]})
        
        custom_ca_path = "/path/to/custom-ca.pem"
        response = await litellm.acompletion(
            model="openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            ssl_verify=custom_ca_path,
            api_key="sk-123"
        )
        # Check that AsyncClient was initialized with verify=custom_ca_path
        called_kwargs = mock_client.call_args[1]
        assert called_kwargs.get("verify") == custom_ca_path

@pytest.mark.asyncio
async def test_ssl_verify_default():
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.post.return_value = MagicMock(status_code=200, json=lambda: {"choices": [{"message": {"content": "hello"}}]})
        
        response = await litellm.acompletion(
            model="openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            api_key="sk-123"
        )
        # By default, should not pass verify=False (usually defaults to True or SSLContext depending on get_ssl_configuration)
        called_kwargs = mock_client.call_args[1]
        verify_arg = called_kwargs.get("verify")
        assert verify_arg is not False and verify_arg is not None
