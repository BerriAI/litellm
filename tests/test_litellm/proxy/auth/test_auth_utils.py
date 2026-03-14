"""
Unit tests for auth_utils functions related to rate limiting and customer ID extraction.
"""

from unittest.mock import patch

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import (
    _get_customer_id_from_standard_headers,
    get_end_user_id_from_request_body,
    get_model_from_request,
    get_key_model_rpm_limit,
    get_key_model_tpm_limit,
)


class TestGetKeyModelRpmLimit:
    """Tests for get_key_model_rpm_limit function."""

    def test_returns_key_metadata_when_present(self):
        """Key metadata takes priority over team metadata."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            metadata={"model_rpm_limit": {"gpt-4": 100}},
            team_metadata={"model_rpm_limit": {"gpt-4": 50}},
        )
        result = get_key_model_rpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 100}

    def test_falls_back_to_team_metadata_when_key_has_other_metadata(self):
        """Should fall back to team metadata when key metadata exists but has no model_rpm_limit."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            metadata={
                "some_other_key": "value"
            },  # Has metadata, but not model_rpm_limit
            team_metadata={"model_rpm_limit": {"gpt-4": 50}},
        )
        result = get_key_model_rpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 50}

    def test_extracts_from_model_max_budget(self):
        """Should extract rpm_limit from model_max_budget when metadata is empty."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            model_max_budget={
                "gpt-4": {"rpm_limit": 100, "tpm_limit": 1000},
                "gpt-3.5-turbo": {"rpm_limit": 200},
            },
        )
        result = get_key_model_rpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 100, "gpt-3.5-turbo": 200}

    def test_skips_models_without_rpm_limit(self):
        """Should skip models that don't have rpm_limit in model_max_budget."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            model_max_budget={
                "gpt-4": {"rpm_limit": 100},
                "gpt-3.5-turbo": {"tpm_limit": 1000},  # No rpm_limit
            },
        )
        result = get_key_model_rpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 100}

    def test_returns_none_when_no_limits_configured(self):
        """Should return None when no rate limits are configured."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        result = get_key_model_rpm_limit(user_api_key_dict)
        assert result is None


class TestGetKeyModelTpmLimit:
    """Tests for get_key_model_tpm_limit function."""

    def test_returns_key_metadata_when_present(self):
        """Key metadata takes priority over team metadata."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            metadata={"model_tpm_limit": {"gpt-4": 10000}},
            team_metadata={"model_tpm_limit": {"gpt-4": 5000}},
        )
        result = get_key_model_tpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 10000}

    def test_falls_back_to_team_metadata_when_key_has_other_metadata(self):
        """Should fall back to team metadata when key metadata exists but has no model_tpm_limit."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            metadata={
                "some_other_key": "value"
            },  # Has metadata, but not model_tpm_limit
            team_metadata={"model_tpm_limit": {"gpt-4": 5000}},
        )
        result = get_key_model_tpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 5000}

    def test_extracts_from_model_max_budget(self):
        """Should extract tpm_limit from model_max_budget when metadata is empty."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            model_max_budget={
                "gpt-4": {"tpm_limit": 10000, "rpm_limit": 100},
                "gpt-3.5-turbo": {"tpm_limit": 20000},
            },
        )
        result = get_key_model_tpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 10000, "gpt-3.5-turbo": 20000}

    def test_skips_models_without_tpm_limit(self):
        """Should skip models that don't have tpm_limit in model_max_budget."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            model_max_budget={
                "gpt-4": {"tpm_limit": 10000},
                "gpt-3.5-turbo": {"rpm_limit": 100},  # No tpm_limit
            },
        )
        result = get_key_model_tpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 10000}

    def test_returns_none_when_no_limits_configured(self):
        """Should return None when no rate limits are configured."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        result = get_key_model_tpm_limit(user_api_key_dict)
        assert result is None

    def test_model_max_budget_priority_over_team(self):
        """model_max_budget should take priority over team_metadata."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            model_max_budget={"gpt-4": {"tpm_limit": 10000}},
            team_metadata={"model_tpm_limit": {"gpt-4": 5000}},
        )
        result = get_key_model_tpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 10000}


class TestGetCustomerIdFromStandardHeaders:
    """Tests for _get_customer_id_from_standard_headers helper function."""

    def test_should_return_customer_id_from_x_litellm_customer_id_header(self):
        """Should extract customer ID from x-litellm-customer-id header."""
        headers = {"x-litellm-customer-id": "customer-123"}
        result = _get_customer_id_from_standard_headers(request_headers=headers)
        assert result == "customer-123"

    def test_should_return_customer_id_from_x_litellm_end_user_id_header(self):
        """Should extract customer ID from x-litellm-end-user-id header."""
        headers = {"x-litellm-end-user-id": "end-user-456"}
        result = _get_customer_id_from_standard_headers(request_headers=headers)
        assert result == "end-user-456"

    def test_should_return_none_when_headers_is_none(self):
        """Should return None when headers is None."""
        result = _get_customer_id_from_standard_headers(request_headers=None)
        assert result is None

    def test_should_return_none_when_no_standard_headers_present(self):
        """Should return None when no standard customer ID headers are present."""
        headers = {"x-other-header": "some-value"}
        result = _get_customer_id_from_standard_headers(request_headers=headers)
        assert result is None


class TestGetEndUserIdFromRequestBodyWithStandardHeaders:
    """Tests for get_end_user_id_from_request_body with standard customer ID headers."""

    def test_should_prioritize_standard_header_over_body_user(self):
        """Standard customer ID header should take precedence over body user field."""
        headers = {"x-litellm-customer-id": "header-customer"}
        request_body = {"user": "body-user"}

        with patch("litellm.proxy.proxy_server.general_settings", {}):
            result = get_end_user_id_from_request_body(
                request_body=request_body, request_headers=headers
            )
        assert result == "header-customer"

    def test_should_fall_back_to_body_when_no_standard_header(self):
        """Should fall back to body user when no standard headers are present."""
        headers = {"x-other-header": "value"}
        request_body = {"user": "body-user"}

        with patch("litellm.proxy.proxy_server.general_settings", {}):
            result = get_end_user_id_from_request_body(
                request_body=request_body, request_headers=headers
            )
        assert result == "body-user"


def test_get_model_from_request_supports_google_model_names_with_slashes():
    assert (
        get_model_from_request(
            request_data={},
            route="/v1beta/models/bedrock/claude-sonnet-3.7:generateContent",
        )
        == "bedrock/claude-sonnet-3.7"
    )
    assert (
        get_model_from_request(
            request_data={},
            route="/models/hosted_vllm/gpt-oss-20b:generateContent",
        )
        == "hosted_vllm/gpt-oss-20b"
    )


def test_get_model_from_request_vertex_passthrough_still_works():
    route = "/vertex_ai/v1/projects/p/locations/l/publishers/google/models/gemini-1.5-pro:generateContent"
    assert get_model_from_request(request_data={}, route=route) == "gemini-1.5-pro"


def test_get_customer_user_header_returns_none_when_no_customer_role():
    from litellm.proxy.auth.auth_utils import get_customer_user_header_from_mapping

    mappings = [
        {"header_name": "X-OpenWebUI-User-Id", "litellm_user_role": "internal_user"}
    ]
    result = get_customer_user_header_from_mapping(mappings)
    assert result is None


def test_get_customer_user_header_returns_none_for_single_non_customer_mapping():
    from litellm.proxy.auth.auth_utils import get_customer_user_header_from_mapping

    mapping = {"header_name": "X-Only-Internal", "litellm_user_role": "internal_user"}
    result = get_customer_user_header_from_mapping(mapping)
    assert result is None

def test_get_customer_user_header_from_mapping_returns_customer_header():
    from litellm.proxy.auth.auth_utils import get_customer_user_header_from_mapping

    mappings = [
        {"header_name": "X-OpenWebUI-User-Id", "litellm_user_role": "internal_user"},
        {"header_name": "X-OpenWebUI-User-Email", "litellm_user_role": "customer"},
    ]
    result = get_customer_user_header_from_mapping(mappings)
    assert result == ["x-openwebui-user-email"]


def test_get_customer_user_header_returns_customers_header_when_multiple_exist():
    from litellm.proxy.auth.auth_utils import get_customer_user_header_from_mapping

    mappings = [
        {"header_name": "X-OpenWebUI-User-Id", "litellm_user_role": "internal_user"},
        {"header_name": "X-OpenWebUI-User-Email", "litellm_user_role": "customer"},
        {"header_name": "X-User-Id", "litellm_user_role": "customer"},
    ]
    result = get_customer_user_header_from_mapping(mappings)
    assert result == ['x-openwebui-user-email', 'x-user-id']
    
def test_get_customer_user_header_returns_sorted_customers_header_when_multiple_exist():
    from litellm.proxy.auth.auth_utils import get_customer_user_header_from_mapping

    mappings = [
        {"header_name": "X-OpenWebUI-User-Id", "litellm_user_role": "internal_user"},
        {"header_name": "X-User-Id", "litellm_user_role": "customer"},
        {"header_name": "X-OpenWebUI-User-Email", "litellm_user_role": "customer"},
    ]
    result = get_customer_user_header_from_mapping(mappings)
    assert result == ['x-user-id', 'x-openwebui-user-email']


####################################


def test_get_end_user_id_returns_id_from_user_header_mappings():
    from litellm.proxy.auth.auth_utils import get_end_user_id_from_request_body

    mappings = [
        {"header_name": "x-openwebui-user-id", "litellm_user_role": "internal_user"},
        {"header_name": "x-openwebui-user-email", "litellm_user_role": "customer"},
    ]
    general_settings = {"user_header_mappings": mappings}
    headers = {"x-openwebui-user-email": "1234"}

    with patch("litellm.proxy.auth.auth_utils._get_customer_id_from_standard_headers", return_value=None), \
         patch("litellm.proxy.proxy_server.general_settings", general_settings):
        result = get_end_user_id_from_request_body(request_body={}, request_headers=headers)

    assert result == "1234"


def test_get_end_user_id_returns_first_customer_header_when_multiple_mappings_exist():
    from litellm.proxy.auth.auth_utils import get_end_user_id_from_request_body

    mappings = [
        {"header_name": "x-openwebui-user-id", "litellm_user_role": "internal_user"},
        {"header_name": "x-user-id", "litellm_user_role": "customer"},
        {"header_name": "x-openwebui-user-email", "litellm_user_role": "customer"},
    ]
    general_settings = {"user_header_mappings": mappings}
    headers = {
        "x-user-id": "user-456",
        "x-openwebui-user-email": "user@example.com",
    }

    with patch("litellm.proxy.auth.auth_utils._get_customer_id_from_standard_headers", return_value=None), \
         patch("litellm.proxy.proxy_server.general_settings", general_settings):
        result = get_end_user_id_from_request_body(request_body={}, request_headers=headers)

    assert result == "user-456"


def test_get_end_user_id_returns_none_when_no_customer_role_in_mappings():
    from litellm.proxy.auth.auth_utils import get_end_user_id_from_request_body

    mappings = [
        {"header_name": "x-openwebui-user-id", "litellm_user_role": "internal_user"},
    ]
    general_settings = {"user_header_mappings": mappings}
    headers = {"x-openwebui-user-id": "user-789"}

    with patch("litellm.proxy.auth.auth_utils._get_customer_id_from_standard_headers", return_value=None), \
         patch("litellm.proxy.proxy_server.general_settings", general_settings):
        result = get_end_user_id_from_request_body(request_body={}, request_headers=headers)

    assert result is None


