import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from litellm.proxy.proxy_server import app, user_api_key_auth
from litellm.proxy._types import UserAPIKeyAuth
from litellm.router import Router


@pytest.fixture
def mock_user_api_key_auth():
    """Mock user API key authentication."""
    mock_auth = UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
        team_id="test-team",
        team_models=[],
        models=[]
    )
    return mock_auth


@pytest.fixture
def mock_router_with_fallbacks():
    """Create a mock router with fallback configurations."""
    router = Mock(spec=Router)
    router.fallbacks = [
        {"claude-4-sonnet": ["bedrock-claude-sonnet-4", "google-claude-sonnet-4"]},
        {"gpt-4": ["gpt-4-turbo", "gpt-3.5-turbo"]}
    ]
    router.context_window_fallbacks = [
        {"claude-4-sonnet": ["claude-3-sonnet"]},
        {"gpt-4": ["gpt-3.5-turbo"]}
    ]
    router.content_policy_fallbacks = [
        {"claude-4-sonnet": ["claude-3-haiku"]}
    ]
    router.get_model_names.return_value = [
        "claude-4-sonnet", "bedrock-claude-sonnet-4", "google-claude-sonnet-4", 
        "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"
    ]
    router.get_model_access_groups.return_value = {}
    return router


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestModelsEndpointWithFallbacks:
    """Test the enhanced /models endpoint with fallback functionality."""

    @patch('litellm.proxy.proxy_server.llm_router')
    @patch('litellm.proxy.proxy_server.user_api_key_auth')
    @patch('litellm.proxy.proxy_server.get_complete_model_list')
    @patch('litellm.proxy.proxy_server.get_key_models')
    @patch('litellm.proxy.proxy_server.get_team_models')
    def test_models_endpoint_without_metadata(
        self, mock_get_team_models, mock_get_key_models, mock_get_complete_model_list,
        mock_auth, mock_router, client, mock_user_api_key_auth, mock_router_with_fallbacks
    ):
        """Test /models endpoint without metadata returns standard response."""
        mock_auth.return_value = mock_user_api_key_auth
        mock_router.return_value = mock_router_with_fallbacks
        mock_get_key_models.return_value = []
        mock_get_team_models.return_value = []
        mock_get_complete_model_list.return_value = ["claude-4-sonnet", "gpt-4"]

        response = client.get("/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "data" in data
        assert "object" in data
        assert data["object"] == "list"
        
        # Should not have metadata in any model
        for model in data["data"]:
            assert "metadata" not in model
            assert all(key in model for key in ["id", "object", "created", "owned_by"])

    @patch('litellm.proxy.proxy_server.llm_router')
    @patch('litellm.proxy.proxy_server.user_api_key_auth')
    @patch('litellm.proxy.proxy_server.get_complete_model_list')
    @patch('litellm.proxy.proxy_server.get_key_models')
    @patch('litellm.proxy.proxy_server.get_team_models')
    @patch('litellm.proxy.proxy_server.get_all_fallbacks')
    def test_models_endpoint_with_metadata_defaults_to_general(
        self, mock_get_all_fallbacks, mock_get_team_models, mock_get_key_models, 
        mock_get_complete_model_list, mock_auth, mock_router, client, 
        mock_user_api_key_auth, mock_router_with_fallbacks
    ):
        """Test /models endpoint with metadata=true defaults to general fallback_type."""
        mock_auth.return_value = mock_user_api_key_auth
        mock_router.return_value = mock_router_with_fallbacks
        mock_get_key_models.return_value = []
        mock_get_team_models.return_value = []
        mock_get_complete_model_list.return_value = ["claude-4-sonnet", "gpt-4"]
        
        # Mock fallback responses
        def fallback_side_effect(model, llm_router, fallback_type):
            if model == "claude-4-sonnet" and fallback_type == "general":
                return ["bedrock-claude-sonnet-4", "google-claude-sonnet-4"]
            return []
        
        mock_get_all_fallbacks.side_effect = fallback_side_effect

        response = client.get("/models?include_metadata=true")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have metadata with general fallbacks by default
        claude_model = next((m for m in data["data"] if m["id"] == "claude-4-sonnet"), None)
        assert claude_model is not None
        assert "metadata" in claude_model
        assert "fallbacks" in claude_model["metadata"]
        assert claude_model["metadata"]["fallbacks"] == ["bedrock-claude-sonnet-4", "google-claude-sonnet-4"]
        
        # Verify get_all_fallbacks was called with "general" as default
        mock_get_all_fallbacks.assert_any_call(
            model="claude-4-sonnet",
            llm_router=mock_router_with_fallbacks,
            fallback_type="general"
        )

    @patch('litellm.proxy.proxy_server.llm_router')
    @patch('litellm.proxy.proxy_server.user_api_key_auth')
    @patch('litellm.proxy.proxy_server.get_complete_model_list')
    @patch('litellm.proxy.proxy_server.get_key_models')
    @patch('litellm.proxy.proxy_server.get_team_models')
    @patch('litellm.proxy.proxy_server.get_all_fallbacks')
    def test_models_endpoint_with_general_fallbacks(
        self, mock_get_all_fallbacks, mock_get_team_models, mock_get_key_models, 
        mock_get_complete_model_list, mock_auth, mock_router, client, 
        mock_user_api_key_auth, mock_router_with_fallbacks
    ):
        """Test /models endpoint with general fallbacks."""
        mock_auth.return_value = mock_user_api_key_auth
        mock_router.return_value = mock_router_with_fallbacks
        mock_get_key_models.return_value = []
        mock_get_team_models.return_value = []
        mock_get_complete_model_list.return_value = ["claude-4-sonnet", "bedrock-claude-sonnet-4"]
        
        # Mock fallback responses
        def fallback_side_effect(model, llm_router, fallback_type):
            if model == "claude-4-sonnet":
                return ["bedrock-claude-sonnet-4", "google-claude-sonnet-4"]
            return []
        
        mock_get_all_fallbacks.side_effect = fallback_side_effect

        response = client.get("/models?include_metadata=true&fallback_type=general")
        
        assert response.status_code == 200
        data = response.json()
        
        # Find claude-4-sonnet in response
        claude_model = next((m for m in data["data"] if m["id"] == "claude-4-sonnet"), None)
        assert claude_model is not None
        assert "metadata" in claude_model
        assert "fallbacks" in claude_model["metadata"]
        assert claude_model["metadata"]["fallbacks"] == ["bedrock-claude-sonnet-4", "google-claude-sonnet-4"]
        
        # Find bedrock-claude-sonnet-4 in response (should have no fallbacks)
        bedrock_model = next((m for m in data["data"] if m["id"] == "bedrock-claude-sonnet-4"), None)
        assert bedrock_model is not None
        assert "metadata" in bedrock_model
        assert "fallbacks" in bedrock_model["metadata"]
        assert bedrock_model["metadata"]["fallbacks"] == []

    @patch('litellm.proxy.proxy_server.llm_router')
    @patch('litellm.proxy.proxy_server.user_api_key_auth')
    @patch('litellm.proxy.proxy_server.get_complete_model_list')
    @patch('litellm.proxy.proxy_server.get_key_models')
    @patch('litellm.proxy.proxy_server.get_team_models')
    @patch('litellm.proxy.proxy_server.get_all_fallbacks')
    def test_models_endpoint_with_context_window_fallbacks(
        self, mock_get_all_fallbacks, mock_get_team_models, mock_get_key_models, 
        mock_get_complete_model_list, mock_auth, mock_router, client, 
        mock_user_api_key_auth, mock_router_with_fallbacks
    ):
        """Test /models endpoint with context window fallbacks."""
        mock_auth.return_value = mock_user_api_key_auth
        mock_router.return_value = mock_router_with_fallbacks
        mock_get_key_models.return_value = []
        mock_get_team_models.return_value = []
        mock_get_complete_model_list.return_value = ["claude-4-sonnet"]
        
        # Mock fallback responses for context window
        def fallback_side_effect(model, llm_router, fallback_type):
            if model == "claude-4-sonnet" and fallback_type == "context_window":
                return ["claude-3-sonnet"]
            return []
        
        mock_get_all_fallbacks.side_effect = fallback_side_effect

        response = client.get("/models?include_metadata=true&fallback_type=context_window")
        
        assert response.status_code == 200
        data = response.json()
        
        claude_model = next((m for m in data["data"] if m["id"] == "claude-4-sonnet"), None)
        assert claude_model is not None
        assert "metadata" in claude_model
        assert "fallbacks" in claude_model["metadata"]
        assert claude_model["metadata"]["fallbacks"] == ["claude-3-sonnet"]

    @patch('litellm.proxy.proxy_server.llm_router')
    @patch('litellm.proxy.proxy_server.user_api_key_auth')
    @patch('litellm.proxy.proxy_server.get_complete_model_list')
    @patch('litellm.proxy.proxy_server.get_key_models')
    @patch('litellm.proxy.proxy_server.get_team_models')
    @patch('litellm.proxy.proxy_server.get_all_fallbacks')
    def test_models_endpoint_with_content_policy_fallbacks(
        self, mock_get_all_fallbacks, mock_get_team_models, mock_get_key_models, 
        mock_get_complete_model_list, mock_auth, mock_router, client, 
        mock_user_api_key_auth, mock_router_with_fallbacks
    ):
        """Test /models endpoint with content policy fallbacks."""
        mock_auth.return_value = mock_user_api_key_auth
        mock_router.return_value = mock_router_with_fallbacks
        mock_get_key_models.return_value = []
        mock_get_team_models.return_value = []
        mock_get_complete_model_list.return_value = ["claude-4-sonnet"]
        
        # Mock fallback responses for content policy
        def fallback_side_effect(model, llm_router, fallback_type):
            if model == "claude-4-sonnet" and fallback_type == "content_policy":
                return ["claude-3-haiku"]
            return []
        
        mock_get_all_fallbacks.side_effect = fallback_side_effect

        response = client.get("/models?include_metadata=true&fallback_type=content_policy")
        
        assert response.status_code == 200
        data = response.json()
        
        claude_model = next((m for m in data["data"] if m["id"] == "claude-4-sonnet"), None)
        assert claude_model is not None
        assert "metadata" in claude_model
        assert "fallbacks" in claude_model["metadata"]
        assert claude_model["metadata"]["fallbacks"] == ["claude-3-haiku"]

    @patch('litellm.proxy.proxy_server.llm_router')
    @patch('litellm.proxy.proxy_server.user_api_key_auth')
    @patch('litellm.proxy.proxy_server.get_complete_model_list')
    @patch('litellm.proxy.proxy_server.get_key_models')
    @patch('litellm.proxy.proxy_server.get_team_models')
    def test_models_endpoint_invalid_fallback_type(
        self, mock_get_team_models, mock_get_key_models, mock_get_complete_model_list,
        mock_auth, mock_router, client, mock_user_api_key_auth, mock_router_with_fallbacks
    ):
        """Test /models endpoint with invalid fallback_type returns 400."""
        mock_auth.return_value = mock_user_api_key_auth
        mock_router.return_value = mock_router_with_fallbacks
        mock_get_key_models.return_value = []
        mock_get_team_models.return_value = []
        mock_get_complete_model_list.return_value = ["claude-4-sonnet"]

        response = client.get("/models?include_metadata=true&fallback_type=invalid")
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "Invalid fallback_type" in data["detail"]
        assert "general" in data["detail"]
        assert "context_window" in data["detail"]
        assert "content_policy" in data["detail"]

    @patch('litellm.proxy.proxy_server.llm_router')
    @patch('litellm.proxy.proxy_server.user_api_key_auth')
    @patch('litellm.proxy.proxy_server.get_complete_model_list')
    @patch('litellm.proxy.proxy_server.get_key_models')
    @patch('litellm.proxy.proxy_server.get_team_models')
    @patch('litellm.proxy.proxy_server.get_all_fallbacks')
    def test_models_endpoint_multiple_models_different_fallbacks(
        self, mock_get_all_fallbacks, mock_get_team_models, mock_get_key_models, 
        mock_get_complete_model_list, mock_auth, mock_router, client, 
        mock_user_api_key_auth, mock_router_with_fallbacks
    ):
        """Test /models endpoint with multiple models having different fallback configurations."""
        mock_auth.return_value = mock_user_api_key_auth
        mock_router.return_value = mock_router_with_fallbacks
        mock_get_key_models.return_value = []
        mock_get_team_models.return_value = []
        mock_get_complete_model_list.return_value = [
            "claude-4-sonnet", "gpt-4", "bedrock-claude-sonnet-4", "gpt-3.5-turbo"
        ]
        
        # Mock different fallback responses for different models
        def fallback_side_effect(model, llm_router, fallback_type):
            if model == "claude-4-sonnet":
                return ["bedrock-claude-sonnet-4", "google-claude-sonnet-4"]
            elif model == "gpt-4":
                return ["gpt-4-turbo", "gpt-3.5-turbo"]
            return []  # No fallbacks for other models
        
        mock_get_all_fallbacks.side_effect = fallback_side_effect

        response = client.get("/models?include_metadata=true&fallback_type=general")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check claude-4-sonnet fallbacks
        claude_model = next((m for m in data["data"] if m["id"] == "claude-4-sonnet"), None)
        assert claude_model is not None
        assert claude_model["metadata"]["fallbacks"] == ["bedrock-claude-sonnet-4", "google-claude-sonnet-4"]
        
        # Check gpt-4 fallbacks
        gpt4_model = next((m for m in data["data"] if m["id"] == "gpt-4"), None)
        assert gpt4_model is not None
        assert gpt4_model["metadata"]["fallbacks"] == ["gpt-4-turbo", "gpt-3.5-turbo"]
        
        # Check models without fallbacks
        bedrock_model = next((m for m in data["data"] if m["id"] == "bedrock-claude-sonnet-4"), None)
        assert bedrock_model is not None
        assert bedrock_model["metadata"]["fallbacks"] == []
        
        gpt35_model = next((m for m in data["data"] if m["id"] == "gpt-3.5-turbo"), None)
        assert gpt35_model is not None
        assert gpt35_model["metadata"]["fallbacks"] == []

    @patch('litellm.proxy.proxy_server.llm_router')
    @patch('litellm.proxy.proxy_server.user_api_key_auth')
    @patch('litellm.proxy.proxy_server.get_complete_model_list')
    @patch('litellm.proxy.proxy_server.get_key_models')
    @patch('litellm.proxy.proxy_server.get_team_models')
    def test_models_endpoint_preserves_existing_functionality(
        self, mock_get_team_models, mock_get_key_models, mock_get_complete_model_list,
        mock_auth, mock_router, client, mock_user_api_key_auth, mock_router_with_fallbacks
    ):
        """Test that existing query parameters still work correctly."""
        mock_auth.return_value = mock_user_api_key_auth
        mock_router.return_value = mock_router_with_fallbacks
        mock_get_key_models.return_value = []
        mock_get_team_models.return_value = []
        mock_get_complete_model_list.return_value = ["claude-4-sonnet", "gpt-4"]

        # Test with existing parameters
        response = client.get("/models?return_wildcard_routes=true&include_model_access_groups=true")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have standard OpenAI format
        assert "data" in data
        assert "object" in data
        assert data["object"] == "list"
        
        # Should not have metadata (backward compatibility)
        for model in data["data"]:
            assert "metadata" not in model
            assert all(key in model for key in ["id", "object", "created", "owned_by"])