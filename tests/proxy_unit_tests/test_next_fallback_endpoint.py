"""
Integration tests for the /v1/models/{model}/next-fallback API endpoint

Tests the HTTP endpoint functionality including authentication, authorization, 
request/response handling, and error cases.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from fastapi.testclient import TestClient

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.proxy_server import app
from litellm.router import Router


class TestNextFallbackEndpoint:
    """Test cases for the /v1/models/{model}/next-fallback endpoint"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.client = TestClient(app)
        self.test_token = "Bearer sk-test-key"
        self.base_url = "/v1/models"
        
        # Mock user API key dict
        self.mock_user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-test-key",
            user_id="test-user",
            models=["claude-4-sonnet", "bedrock-claude-sonnet-4", "google-claude-sonnet-4"],
            team_models=[],
        )
        
        # Sample fallback configuration
        self.sample_fallbacks = [
            {"claude-4-sonnet": ["bedrock-claude-sonnet-4", "google-claude-sonnet-4"]},
            {"gpt-4o": ["azure-gpt-4o", "github-gpt-4o"]},
        ]

    def create_mock_router(self):
        """Create a mock router with test configuration"""
        mock_router = Mock(spec=Router)
        mock_router.fallbacks = self.sample_fallbacks
        mock_router.context_window_fallbacks = []
        mock_router.content_policy_fallbacks = []
        mock_router.get_model_names.return_value = [
            "claude-4-sonnet", "bedrock-claude-sonnet-4", "google-claude-sonnet-4",
            "gpt-4o", "azure-gpt-4o", "github-gpt-4o"
        ]
        mock_router.get_model_access_groups.return_value = {}
        return mock_router

    @patch('litellm.proxy.proxy_server.user_api_key_auth')
    @patch('litellm.proxy.proxy_server.llm_router')
    @patch('litellm.proxy.proxy_server.general_settings')
    @patch('litellm.proxy.proxy_server.get_key_models')
    @patch('litellm.proxy.proxy_server.get_team_models')
    @patch('litellm.proxy.proxy_server.get_complete_model_list')
    def test_successful_fallback_request(
        self, mock_get_complete_model_list, mock_get_team_models, 
        mock_get_key_models, mock_general_settings, mock_router, mock_auth
    ):
        """Test successful fallback request returns correct next fallback"""
        # Setup mocks
        mock_auth.return_value = self.mock_user_api_key_dict
        mock_router = self.create_mock_router()
        
        # Mock the model access functions
        accessible_models = ["claude-4-sonnet", "bedrock-claude-sonnet-4", "google-claude-sonnet-4"]
        mock_get_key_models.return_value = []
        mock_get_team_models.return_value = []
        mock_get_complete_model_list.return_value = accessible_models
        mock_general_settings.get.return_value = False
        
        with patch('litellm.proxy.proxy_server.llm_router', mock_router):
            response = self.client.get(
                f"{self.base_url}/claude-4-sonnet/next-fallback",
                headers={"Authorization": self.test_token}
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["current_model"] == "claude-4-sonnet"
        assert data["next_fallback"] == "bedrock-claude-sonnet-4"
        assert data["fallback_type"] == "general"
        assert data["object"] == "next_fallback"

    @patch('litellm.proxy.proxy_server.user_api_key_auth')
    @patch('litellm.proxy.proxy_server.llm_router')
    @patch('litellm.proxy.proxy_server.general_settings')
    @patch('litellm.proxy.proxy_server.get_key_models')
    @patch('litellm.proxy.proxy_server.get_team_models')
    @patch('litellm.proxy.proxy_server.get_complete_model_list')
    def test_fallback_chain_progression(
        self, mock_get_complete_model_list, mock_get_team_models, 
        mock_get_key_models, mock_general_settings, mock_router, mock_auth
    ):
        """Test the complete fallback chain progression"""
        # Setup mocks
        mock_auth.return_value = self.mock_user_api_key_dict
        mock_router = self.create_mock_router()
        
        accessible_models = ["claude-4-sonnet", "bedrock-claude-sonnet-4", "google-claude-sonnet-4"]
        mock_get_key_models.return_value = []
        mock_get_team_models.return_value = []
        mock_get_complete_model_list.return_value = accessible_models
        mock_general_settings.get.return_value = False
        
        test_cases = [
            ("claude-4-sonnet", 200, "bedrock-claude-sonnet-4"),
            ("bedrock-claude-sonnet-4", 200, "google-claude-sonnet-4"),
            ("google-claude-sonnet-4", 404, None),  # No more fallbacks
        ]
        
        for model, expected_status, expected_fallback in test_cases:
            with patch('litellm.proxy.proxy_server.llm_router', mock_router):
                response = self.client.get(
                    f"{self.base_url}/{model}/next-fallback",
                    headers={"Authorization": self.test_token}
                )
            
            assert response.status_code == expected_status
            
            if expected_status == 200:
                data = response.json()
                assert data["current_model"] == model
                assert data["next_fallback"] == expected_fallback
            else:
                # 404 case
                data = response.json()
                assert "error" in data
                assert data["error"]["code"] == "no_fallback_available"

    @patch('litellm.proxy.proxy_server.user_api_key_auth')
    @patch('litellm.proxy.proxy_server.llm_router')
    @patch('litellm.proxy.proxy_server.general_settings')
    @patch('litellm.proxy.proxy_server.get_key_models')
    @patch('litellm.proxy.proxy_server.get_team_models')
    @patch('litellm.proxy.proxy_server.get_complete_model_list')
    def test_different_fallback_types(
        self, mock_get_complete_model_list, mock_get_team_models, 
        mock_get_key_models, mock_general_settings, mock_router, mock_auth
    ):
        """Test different fallback types via query parameter"""
        # Setup mocks
        mock_auth.return_value = self.mock_user_api_key_dict
        mock_router = self.create_mock_router()
        mock_router.context_window_fallbacks = [
            {"claude-4-sonnet": ["bedrock-claude-sonnet-4"]}
        ]
        
        accessible_models = ["claude-4-sonnet", "bedrock-claude-sonnet-4"]
        mock_get_key_models.return_value = []
        mock_get_team_models.return_value = []
        mock_get_complete_model_list.return_value = accessible_models
        mock_general_settings.get.return_value = False
        
        with patch('litellm.proxy.proxy_server.llm_router', mock_router):
            response = self.client.get(
                f"{self.base_url}/claude-4-sonnet/next-fallback?fallback_type=context_window",
                headers={"Authorization": self.test_token}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["fallback_type"] == "context_window"
        assert data["next_fallback"] == "bedrock-claude-sonnet-4"

    @patch('litellm.proxy.proxy_server.user_api_key_auth')
    @patch('litellm.proxy.proxy_server.llm_router')
    def test_invalid_fallback_type(self, mock_router, mock_auth):
        """Test invalid fallback_type parameter returns 400"""
        mock_auth.return_value = self.mock_user_api_key_dict
        mock_router = self.create_mock_router()
        
        with patch('litellm.proxy.proxy_server.llm_router', mock_router):
            response = self.client.get(
                f"{self.base_url}/claude-4-sonnet/next-fallback?fallback_type=invalid",
                headers={"Authorization": self.test_token}
            )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "invalid_fallback_type"
        assert "Invalid fallback_type" in data["error"]["message"]

    @patch('litellm.proxy.proxy_server.user_api_key_auth')
    @patch('litellm.proxy.proxy_server.llm_router', None)
    def test_no_router_configured(self, mock_auth):
        """Test error when no router is configured"""
        mock_auth.return_value = self.mock_user_api_key_dict
        
        response = self.client.get(
            f"{self.base_url}/claude-4-sonnet/next-fallback",
            headers={"Authorization": self.test_token}
        )
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "no_router_configured"

    @patch('litellm.proxy.proxy_server.user_api_key_auth')
    @patch('litellm.proxy.proxy_server.llm_router')
    @patch('litellm.proxy.proxy_server.general_settings')
    @patch('litellm.proxy.proxy_server.get_key_models')
    @patch('litellm.proxy.proxy_server.get_team_models')
    @patch('litellm.proxy.proxy_server.get_complete_model_list')
    def test_model_not_found(
        self, mock_get_complete_model_list, mock_get_team_models, 
        mock_get_key_models, mock_general_settings, mock_router, mock_auth
    ):
        """Test 404 when model is not accessible to user"""
        # Setup mocks
        mock_auth.return_value = self.mock_user_api_key_dict
        mock_router = self.create_mock_router()
        
        # Model not in accessible models list
        accessible_models = ["other-model"]
        mock_get_key_models.return_value = []
        mock_get_team_models.return_value = []
        mock_get_complete_model_list.return_value = accessible_models
        mock_general_settings.get.return_value = False
        
        with patch('litellm.proxy.proxy_server.llm_router', mock_router):
            response = self.client.get(
                f"{self.base_url}/claude-4-sonnet/next-fallback",
                headers={"Authorization": self.test_token}
            )
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "model_not_found"

    def test_missing_authorization_header(self):
        """Test that missing Authorization header is handled properly"""
        # Note: The actual auth behavior depends on the user_api_key_auth dependency
        # This test may need adjustment based on actual auth implementation
        response = self.client.get(f"{self.base_url}/claude-4-sonnet/next-fallback")
        
        # Expect some form of authentication error (status code may vary based on implementation)
        assert response.status_code in [401, 403, 422]

    def test_invalid_authorization_header(self):
        """Test that invalid Authorization header is handled properly"""
        response = self.client.get(
            f"{self.base_url}/claude-4-sonnet/next-fallback",
            headers={"Authorization": "Bearer invalid-token"}
        )
        
        # Expect some form of authentication error
        assert response.status_code in [401, 403, 422]

    @patch('litellm.proxy.proxy_server.user_api_key_auth')
    @patch('litellm.proxy.proxy_server.llm_router')
    @patch('litellm.proxy.proxy_server.general_settings')
    @patch('litellm.proxy.proxy_server.get_key_models')
    @patch('litellm.proxy.proxy_server.get_team_models')
    @patch('litellm.proxy.proxy_server.get_complete_model_list')
    def test_user_no_access_to_fallback_model(
        self, mock_get_complete_model_list, mock_get_team_models, 
        mock_get_key_models, mock_general_settings, mock_router, mock_auth
    ):
        """Test when user has access to primary model but not the fallback"""
        # Setup mocks
        mock_auth.return_value = self.mock_user_api_key_dict
        mock_router = self.create_mock_router()
        
        # User only has access to primary model, not fallback
        accessible_models = ["claude-4-sonnet"]  # Missing bedrock-claude-sonnet-4
        mock_get_key_models.return_value = []
        mock_get_team_models.return_value = []
        mock_get_complete_model_list.return_value = accessible_models
        mock_general_settings.get.return_value = False
        
        with patch('litellm.proxy.proxy_server.llm_router', mock_router):
            response = self.client.get(
                f"{self.base_url}/claude-4-sonnet/next-fallback",
                headers={"Authorization": self.test_token}
            )
        
        # Should try to find next accessible fallback or return 404
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "no_accessible_fallback"

    @patch('litellm.proxy.proxy_server.user_api_key_auth')
    @patch('litellm.proxy.proxy_server.llm_router')
    @patch('litellm.proxy.proxy_server.general_settings')
    @patch('litellm.proxy.proxy_server.get_key_models')
    @patch('litellm.proxy.proxy_server.get_team_models')
    @patch('litellm.proxy.proxy_server.get_complete_model_list')
    def test_skip_inaccessible_fallback_to_next_accessible(
        self, mock_get_complete_model_list, mock_get_team_models, 
        mock_get_key_models, mock_general_settings, mock_router, mock_auth
    ):
        """Test that inaccessible fallbacks are skipped to find next accessible one"""
        # Setup mocks
        mock_auth.return_value = self.mock_user_api_key_dict
        mock_router = self.create_mock_router()
        
        # User has access to primary and last fallback, but not the first fallback
        accessible_models = ["claude-4-sonnet", "google-claude-sonnet-4"]  # Missing bedrock-claude-sonnet-4
        mock_get_key_models.return_value = []
        mock_get_team_models.return_value = []
        mock_get_complete_model_list.return_value = accessible_models
        mock_general_settings.get.return_value = False
        
        with patch('litellm.proxy.proxy_server.llm_router', mock_router):
            response = self.client.get(
                f"{self.base_url}/claude-4-sonnet/next-fallback",
                headers={"Authorization": self.test_token}
            )
        
        assert response.status_code == 200
        data = response.json()
        # Should skip bedrock-claude-sonnet-4 and return google-claude-sonnet-4
        assert data["next_fallback"] == "google-claude-sonnet-4"

    @patch('litellm.proxy.proxy_server.user_api_key_auth')
    @patch('litellm.proxy.proxy_server.llm_router')
    @patch('litellm.proxy.proxy_server.general_settings')
    @patch('litellm.proxy.proxy_server.get_key_models')
    @patch('litellm.proxy.proxy_server.get_team_models')
    @patch('litellm.proxy.proxy_server.get_complete_model_list')
    def test_response_format_compliance(
        self, mock_get_complete_model_list, mock_get_team_models, 
        mock_get_key_models, mock_general_settings, mock_router, mock_auth
    ):
        """Test that response format matches the expected schema"""
        # Setup mocks
        mock_auth.return_value = self.mock_user_api_key_dict
        mock_router = self.create_mock_router()
        
        accessible_models = ["claude-4-sonnet", "bedrock-claude-sonnet-4"]
        mock_get_key_models.return_value = []
        mock_get_team_models.return_value = []
        mock_get_complete_model_list.return_value = accessible_models
        mock_general_settings.get.return_value = False
        
        with patch('litellm.proxy.proxy_server.llm_router', mock_router):
            response = self.client.get(
                f"{self.base_url}/claude-4-sonnet/next-fallback",
                headers={"Authorization": self.test_token}
            )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        required_fields = ["current_model", "next_fallback", "fallback_type", "object"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Check field types and values
        assert isinstance(data["current_model"], str)
        assert isinstance(data["next_fallback"], str)
        assert isinstance(data["fallback_type"], str)
        assert data["object"] == "next_fallback"
        assert data["fallback_type"] in ["general", "context_window", "content_policy"]

    @pytest.mark.parametrize("fallback_type", ["general", "context_window", "content_policy"])
    @patch('litellm.proxy.proxy_server.user_api_key_auth')
    @patch('litellm.proxy.proxy_server.llm_router')
    @patch('litellm.proxy.proxy_server.general_settings')
    @patch('litellm.proxy.proxy_server.get_key_models')
    @patch('litellm.proxy.proxy_server.get_team_models')
    @patch('litellm.proxy.proxy_server.get_complete_model_list')
    def test_all_valid_fallback_types(
        self, mock_get_complete_model_list, mock_get_team_models, 
        mock_get_key_models, mock_general_settings, mock_router, mock_auth, fallback_type
    ):
        """Test all valid fallback_type parameter values"""
        # Setup mocks
        mock_auth.return_value = self.mock_user_api_key_dict
        mock_router = self.create_mock_router()
        
        # Set up fallbacks for all types
        mock_router.context_window_fallbacks = [{"claude-4-sonnet": ["bedrock-claude-sonnet-4"]}]
        mock_router.content_policy_fallbacks = [{"claude-4-sonnet": ["google-claude-sonnet-4"]}]
        
        accessible_models = ["claude-4-sonnet", "bedrock-claude-sonnet-4", "google-claude-sonnet-4"]
        mock_get_key_models.return_value = []
        mock_get_team_models.return_value = []
        mock_get_complete_model_list.return_value = accessible_models
        mock_general_settings.get.return_value = False
        
        with patch('litellm.proxy.proxy_server.llm_router', mock_router):
            response = self.client.get(
                f"{self.base_url}/claude-4-sonnet/next-fallback?fallback_type={fallback_type}",
                headers={"Authorization": self.test_token}
            )
        
        # Should either return 200 with fallback or 404 if no fallback configured for this type
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert data["fallback_type"] == fallback_type