import pytest
from unittest.mock import Mock, patch


def create_mock_router(
    fallbacks=None, context_window_fallbacks=None, content_policy_fallbacks=None
):
    """Helper function to create a mock router with fallback configurations."""
    router = Mock()
    router.fallbacks = fallbacks or []
    router.context_window_fallbacks = context_window_fallbacks or []
    router.content_policy_fallbacks = content_policy_fallbacks or []
    return router


def test_no_router_returns_empty_list():
    """Test that None router returns empty list."""
    from litellm.proxy.auth.model_checks import get_all_fallbacks
    
    result = get_all_fallbacks("claude-4-sonnet", llm_router=None)
    assert result == []


def test_no_fallbacks_config_returns_empty_list():
    """Test that empty fallbacks config returns empty list."""
    from litellm.proxy.auth.model_checks import get_all_fallbacks
    
    router = create_mock_router(fallbacks=[])
    result = get_all_fallbacks("claude-4-sonnet", llm_router=router)
    assert result == []


def test_model_with_fallbacks_returns_complete_list():
    """Test that model with fallbacks returns complete fallback list."""
    from litellm.proxy.auth.model_checks import get_all_fallbacks
    
    fallbacks_config = [
        {"claude-4-sonnet": ["bedrock-claude-sonnet-4", "google-claude-sonnet-4"]}
    ]
    router = create_mock_router(fallbacks=fallbacks_config)
    
    with patch(
        'litellm.proxy.auth.model_checks.get_fallback_model_group'
    ) as mock_get_fallback:
        mock_get_fallback.return_value = (
            ["bedrock-claude-sonnet-4", "google-claude-sonnet-4"], None
        )
        
        result = get_all_fallbacks("claude-4-sonnet", llm_router=router)
        assert result == ["bedrock-claude-sonnet-4", "google-claude-sonnet-4"]


def test_model_without_fallbacks_returns_empty_list():
    """Test that model without fallbacks returns empty list."""
    from litellm.proxy.auth.model_checks import get_all_fallbacks
    
    fallbacks_config = [
        {"claude-4-sonnet": ["bedrock-claude-sonnet-4", "google-claude-sonnet-4"]}
    ]
    router = create_mock_router(fallbacks=fallbacks_config)
    
    with patch(
        'litellm.proxy.auth.model_checks.get_fallback_model_group'
    ) as mock_get_fallback:
        mock_get_fallback.return_value = (None, None)
        
        result = get_all_fallbacks("bedrock-claude-sonnet-4", llm_router=router)
        assert result == []


def test_general_fallback_type():
    """Test general fallback type uses router.fallbacks."""
    from litellm.proxy.auth.model_checks import get_all_fallbacks
    
    fallbacks_config = [
        {"claude-4-sonnet": ["bedrock-claude-sonnet-4"]}
    ]
    router = create_mock_router(fallbacks=fallbacks_config)
    
    with patch(
        'litellm.proxy.auth.model_checks.get_fallback_model_group'
    ) as mock_get_fallback:
        mock_get_fallback.return_value = (["bedrock-claude-sonnet-4"], None)
        
        result = get_all_fallbacks("claude-4-sonnet", llm_router=router, fallback_type="general")
        assert result == ["bedrock-claude-sonnet-4"]
        
        # Verify it used the general fallbacks config
        mock_get_fallback.assert_called_once_with(
            fallbacks=fallbacks_config,
            model_group="claude-4-sonnet"
        )


def test_context_window_fallback_type():
    """Test context_window fallback type uses router.context_window_fallbacks."""
    from litellm.proxy.auth.model_checks import get_all_fallbacks
    
    context_fallbacks_config = [
        {"gpt-4": ["gpt-3.5-turbo"]}
    ]
    router = create_mock_router(context_window_fallbacks=context_fallbacks_config)
    
    with patch(
        'litellm.proxy.auth.model_checks.get_fallback_model_group'
    ) as mock_get_fallback:
        mock_get_fallback.return_value = (["gpt-3.5-turbo"], None)
        
        result = get_all_fallbacks("gpt-4", llm_router=router, fallback_type="context_window")
        assert result == ["gpt-3.5-turbo"]
        
        # Verify it used the context window fallbacks config
        mock_get_fallback.assert_called_once_with(
            fallbacks=context_fallbacks_config,
            model_group="gpt-4"
        )


def test_content_policy_fallback_type():
    """Test content_policy fallback type uses router.content_policy_fallbacks."""
    from litellm.proxy.auth.model_checks import get_all_fallbacks
    
    content_fallbacks_config = [
        {"claude-4": ["claude-3"]}
    ]
    router = create_mock_router(content_policy_fallbacks=content_fallbacks_config)
    
    with patch(
        'litellm.proxy.auth.model_checks.get_fallback_model_group'
    ) as mock_get_fallback:
        mock_get_fallback.return_value = (["claude-3"], None)
        
        result = get_all_fallbacks("claude-4", llm_router=router, fallback_type="content_policy")
        assert result == ["claude-3"]
        
        # Verify it used the content policy fallbacks config
        mock_get_fallback.assert_called_once_with(
            fallbacks=content_fallbacks_config,
            model_group="claude-4"
        )


def test_invalid_fallback_type_returns_empty_list():
    """Test that invalid fallback type returns empty list and logs warning."""
    from litellm.proxy.auth.model_checks import get_all_fallbacks
    
    router = create_mock_router(fallbacks=[])
    
    with patch('litellm.proxy.auth.model_checks.verbose_proxy_logger') as mock_logger:
        result = get_all_fallbacks("claude-4-sonnet", llm_router=router, fallback_type="invalid")
        
        assert result == []
        mock_logger.warning.assert_called_once_with("Unknown fallback_type: invalid")


def test_exception_handling_returns_empty_list():
    """Test that exceptions are handled gracefully and return empty list."""
    from litellm.proxy.auth.model_checks import get_all_fallbacks
    
    router = create_mock_router(fallbacks=[{"claude-4-sonnet": ["fallback"]}])
    
    with patch(
        'litellm.proxy.auth.model_checks.get_fallback_model_group'
    ) as mock_get_fallback:
        mock_get_fallback.side_effect = Exception("Test exception")
        
        with patch('litellm.proxy.auth.model_checks.verbose_proxy_logger') as mock_logger:
            result = get_all_fallbacks("claude-4-sonnet", llm_router=router)
            
            assert result == []
            mock_logger.error.assert_called_once()
            error_call_args = mock_logger.error.call_args[0][0]
            assert "Error getting fallbacks for model claude-4-sonnet" in error_call_args


def test_multiple_fallbacks_complete_list():
    """Test model with multiple fallbacks returns the complete list."""
    from litellm.proxy.auth.model_checks import get_all_fallbacks
    
    fallbacks_config = [
        {"gpt-4": ["gpt-4-turbo", "gpt-3.5-turbo", "claude-3-haiku"]}
    ]
    router = create_mock_router(fallbacks=fallbacks_config)
    
    with patch(
        'litellm.proxy.auth.model_checks.get_fallback_model_group'
    ) as mock_get_fallback:
        mock_get_fallback.return_value = (["gpt-4-turbo", "gpt-3.5-turbo", "claude-3-haiku"], None)
        
        result = get_all_fallbacks("gpt-4", llm_router=router)
        assert result == ["gpt-4-turbo", "gpt-3.5-turbo", "claude-3-haiku"]


def test_wildcard_and_specific_fallbacks():
    """Test fallbacks with wildcard and specific model configurations."""
    from litellm.proxy.auth.model_checks import get_all_fallbacks
    
    fallbacks_config = [
        {"*": ["gpt-3.5-turbo"]},
        {"claude-4-sonnet": ["bedrock-claude-sonnet-4", "google-claude-sonnet-4"]}
    ]
    router = create_mock_router(fallbacks=fallbacks_config)
    
    with patch(
        'litellm.proxy.auth.model_checks.get_fallback_model_group'
    ) as mock_get_fallback:
        # Test specific model fallbacks
        mock_get_fallback.return_value = (
            ["bedrock-claude-sonnet-4", "google-claude-sonnet-4"], None
        )
        result = get_all_fallbacks("claude-4-sonnet", llm_router=router)
        assert result == ["bedrock-claude-sonnet-4", "google-claude-sonnet-4"]
        
        # Test wildcard fallbacks
        mock_get_fallback.return_value = (["gpt-3.5-turbo"], 0)
        result = get_all_fallbacks("some-unknown-model", llm_router=router)
        assert result == ["gpt-3.5-turbo"]


def test_default_fallback_type_is_general():
    """Test that default fallback_type is 'general'."""
    from litellm.proxy.auth.model_checks import get_all_fallbacks
    
    fallbacks_config = [
        {"claude-4-sonnet": ["bedrock-claude-sonnet-4"]}
    ]
    router = create_mock_router(fallbacks=fallbacks_config)
    
    with patch(
        'litellm.proxy.auth.model_checks.get_fallback_model_group'
    ) as mock_get_fallback:
        mock_get_fallback.return_value = (["bedrock-claude-sonnet-4"], None)
        
        # Call without specifying fallback_type
        result = get_all_fallbacks("claude-4-sonnet", llm_router=router)
        
        # Should use general fallbacks (router.fallbacks)
        mock_get_fallback.assert_called_once_with(
            fallbacks=fallbacks_config,
            model_group="claude-4-sonnet"
        )
        assert result == ["bedrock-claude-sonnet-4"]