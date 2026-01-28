import pytest
from unittest.mock import MagicMock, patch
from litellm._redis import _init_redis_sentinel, _init_async_redis_sentinel

def test_init_redis_sentinel_with_auth_and_ssl():
    """
    Test that _init_redis_sentinel correctly passes password and ssl params
    to both Sentinel and master_for.
    """
    mock_sentinel_instance = MagicMock()
    mock_sentinel_class = MagicMock(return_value=mock_sentinel_instance)
    
    redis_kwargs = {
        "sentinel_nodes": [("localhost", 26379)],
        "service_name": "mymaster",
        "sentinel_password": "sentinel_pass",
        "password": "redis_pass",
        "ssl": True,
        "ssl_cert_reqs": "required"
    }
    
    with patch("redis.Sentinel", mock_sentinel_class):
        _init_redis_sentinel(redis_kwargs)
        
        # Check Sentinel init
        mock_sentinel_class.assert_called_once()
        call_kwargs = mock_sentinel_class.call_args[1]
        assert call_kwargs["sentinel_kwargs"] == {
            "password": "sentinel_pass",
            "ssl": True,
            "ssl_cert_reqs": "required"
        }
        
        # Check master_for call
        mock_sentinel_instance.master_for.assert_called_once()
        master_call_kwargs = mock_sentinel_instance.master_for.call_args[1]
        assert master_call_kwargs["password"] == "redis_pass"
        assert master_call_kwargs["ssl"] is True
        assert master_call_kwargs["ssl_cert_reqs"] == "required"

@pytest.mark.asyncio
async def test_init_async_redis_sentinel_with_auth_and_ssl():
    """
    Test that _init_async_redis_sentinel correctly passes password and ssl params
    to both Sentinel and master_for.
    """
    mock_sentinel_instance = MagicMock()
    mock_sentinel_class = MagicMock(return_value=mock_sentinel_instance)
    
    redis_kwargs = {
        "sentinel_nodes": [("localhost", 26379)],
        "service_name": "mymaster",
        "sentinel_password": "sentinel_pass",
        "password": "redis_pass",
        "ssl": True,
        "ssl_cert_reqs": "required"
    }
    
    with patch("redis.asyncio.Sentinel", mock_sentinel_class):
        _init_async_redis_sentinel(redis_kwargs)
        
        # Check Sentinel init
        mock_sentinel_class.assert_called_once()
        call_kwargs = mock_sentinel_class.call_args[1]
        assert call_kwargs["sentinel_kwargs"] == {
            "password": "sentinel_pass",
            "ssl": True,
            "ssl_cert_reqs": "required"
        }
        
        # Check master_for call
        mock_sentinel_instance.master_for.assert_called_once()
        master_call_kwargs = mock_sentinel_instance.master_for.call_args[1]
        assert master_call_kwargs["password"] == "redis_pass"
        assert master_call_kwargs["ssl"] is True
        assert master_call_kwargs["ssl_cert_reqs"] == "required"
