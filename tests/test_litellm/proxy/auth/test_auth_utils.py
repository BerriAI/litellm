"""
Unit tests for auth_utils functions related to rate limiting and customer ID extraction.
"""

import base64
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import (
    _get_customer_id_from_standard_headers,
    abbreviate_api_key,
    check_complete_credentials,
    get_end_user_id_from_request_body,
    get_key_model_rpm_limit,
    get_key_model_tpm_limit,
    get_model_from_request,
    get_project_model_rpm_limit,
    get_project_model_tpm_limit,
    is_request_body_safe,
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

    def test_team_metadata_empty_rpm_dict_falls_through_to_deployment_default(self):
        """Explicitly empty team model_rpm_limit ({}) should be returned as-is, not fallen through."""
        # An empty dict is a valid team limit map (no per-model limits configured).
        # It should be returned directly rather than falling through to deployment defaults,
        # so a team with an empty map is treated as unconstrained at the team level.
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            team_metadata={"model_rpm_limit": {}},
        )
        result = get_key_model_rpm_limit(user_api_key_dict)
        assert result == {}


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

    def test_team_metadata_empty_tpm_dict_falls_through_to_deployment_default(self):
        """Explicitly empty team model_tpm_limit ({}) should be returned as-is, not fallen through."""
        # An empty dict is a valid team limit map (no per-model limits configured).
        # It should be returned directly rather than falling through to deployment defaults,
        # so a team with an empty map is treated as unconstrained at the team level.
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            team_metadata={"model_tpm_limit": {}},
        )
        result = get_key_model_tpm_limit(user_api_key_dict)
        assert result == {}

    def test_skips_deployments_with_malformed_limit_value(self):
        """Deployments with non-integer-parseable limit values are skipped without raising."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            {
                "model_name": "model1",
                "litellm_params": {"default_api_key_tpm_limit": "not-a-number"},
            },
            _make_deployment_dict("model1", tpm=500),
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_tpm_limit(user_api_key_dict, model_name="model1")
        # The malformed deployment is skipped; the valid one provides 500
        assert result == {"model1": 500}


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


def test_get_model_from_request_openai_deployment_route_still_works():
    assert (
        get_model_from_request(
            request_data={},
            route="/openai/deployments/my-azure-deployment/chat/completions",
        )
        == "my-azure-deployment"
    )


def test_get_model_from_request_includes_file_endpoint_header_model():
    assert (
        get_model_from_request(
            request_data={},
            route="/v1/files",
            request_headers={"X-LiteLLM-Model": "restricted-model"},
        )
        == "restricted-model"
    )


def test_get_model_from_request_ignores_routing_header_on_standard_llm_routes():
    assert (
        get_model_from_request(
            request_data={"model": "allowed-model"},
            route="/v1/chat/completions",
            request_headers={"x-litellm-model": "restricted-model"},
        )
        == "allowed-model"
    )


def test_get_model_from_request_authorizes_all_file_routing_model_sources():
    models = get_model_from_request(
        request_data={"model": "body-model"},
        route="/v1/files",
        request_headers={"x-litellm-model": "header-model"},
        request_query_params={"target_model_names": "query-model-a,query-model-b"},
    )
    assert isinstance(models, list)
    assert set(models) == {
        "body-model",
        "query-model-a",
        "query-model-b",
        "header-model",
    }


def test_get_model_from_request_extracts_simple_encoded_file_id_model():
    from litellm.proxy.openai_files_endpoints.common_utils import (
        encode_file_id_with_model,
    )

    file_id = encode_file_id_with_model(
        file_id="file-provider-id",
        model="restricted-model",
    )

    assert (
        get_model_from_request(
            request_data={"file_id": file_id},
            route="/v1/files/{file_id}",
        )
        == "restricted-model"
    )


def test_get_model_from_request_extracts_unified_file_id_models():
    raw_unified_file_id = (
        "litellm_proxy:application/octet-stream;unified_id,test-id;"
        "target_model_names,model-a,model-b;llm_output_file_id,file-provider-id"
    )
    encoded_unified_file_id = (
        base64.urlsafe_b64encode(raw_unified_file_id.encode()).decode().rstrip("=")
    )

    assert get_model_from_request(
        request_data={"file_id": encoded_unified_file_id},
        route="/v1/files/{file_id}",
    ) == ["model-a", "model-b"]


def test_get_model_from_request_extracts_eval_completion_model():
    assert (
        get_model_from_request(
            request_data={"completion": {"model": "judge-model"}},
            route="/v1/evals/{eval_id}/runs",
        )
        == "judge-model"
    )


def test_get_model_from_request_includes_fine_tuning_target_model_query():
    assert (
        get_model_from_request(
            request_data={},
            route="/v1/fine_tuning/jobs",
            request_query_params={"target_model_names": "fine-tune-model"},
        )
        == "fine-tune-model"
    )


def test_get_model_from_request_extracts_video_id_model():
    from litellm.types.videos.utils import encode_video_id_with_provider

    video_id = encode_video_id_with_provider(
        video_id="video-provider-id",
        provider="openai",
        model_id="video-model",
    )

    assert (
        get_model_from_request(
            request_data={"video_id": video_id},
            route="/v1/videos/{video_id}",
        )
        == "video-model"
    )


def test_get_model_from_request_only_runs_media_decoders_for_matching_fields():
    with (
        patch(
            "litellm.types.videos.utils.decode_video_id_with_provider",
            return_value={"model_id": "video-model"},
        ) as video_decoder,
        patch(
            "litellm.types.videos.utils.decode_character_id_with_provider",
            return_value={"model_id": "character-model"},
        ) as character_decoder,
    ):
        assert (
            get_model_from_request(
                request_data={"file_id": "file-provider-id"},
                route="/v1/files/{file_id}",
            )
            is None
        )
        video_decoder.assert_not_called()
        character_decoder.assert_not_called()

        assert (
            get_model_from_request(
                request_data={"video_id": "video-provider-id"},
                route="/v1/videos/{video_id}",
            )
            == "video-model"
        )
        video_decoder.assert_called_once_with("video-provider-id")
        character_decoder.assert_not_called()

        video_decoder.reset_mock()
        character_decoder.reset_mock()
        assert (
            get_model_from_request(
                request_data={"character_id": "character-provider-id"},
                route="/v1/videos/{character_id}",
            )
            == "character-model"
        )
        video_decoder.assert_not_called()
        character_decoder.assert_called_once_with("character-provider-id")


def test_get_model_from_request_handles_managed_id_decoder_failures():
    with (
        patch(
            "litellm.proxy.openai_files_endpoints.common_utils.decode_model_from_file_id",
            side_effect=Exception("decode failed"),
        ),
        patch(
            "litellm.llms.base_llm.managed_resources.utils.parse_unified_id",
            side_effect=Exception("parse failed"),
        ),
        patch(
            "litellm.types.videos.utils.decode_video_id_with_provider",
            side_effect=Exception("video decode failed"),
        ),
    ):
        assert (
            get_model_from_request(
                request_data={"file_id": "not-a-managed-resource-id"},
                route="/v1/files/{file_id}",
            )
            is None
        )
        assert (
            get_model_from_request(
                request_data={"video_id": "not-a-managed-resource-id"},
                route="/v1/videos/{video_id}",
            )
            is None
        )


def test_abbreviate_api_key():
    assert abbreviate_api_key("sk-test-1234") == "sk-...1234"


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


def test_get_customer_user_header_returns_customers_header_in_config_order_when_multiple_exist():
    from litellm.proxy.auth.auth_utils import get_customer_user_header_from_mapping

    mappings = [
        {"header_name": "X-OpenWebUI-User-Id", "litellm_user_role": "internal_user"},
        {"header_name": "X-OpenWebUI-User-Email", "litellm_user_role": "customer"},
        {"header_name": "X-User-Id", "litellm_user_role": "customer"},
    ]
    result = get_customer_user_header_from_mapping(mappings)
    assert result == ["x-openwebui-user-email", "x-user-id"]


def test_get_end_user_id_returns_id_from_user_header_mappings():
    from litellm.proxy.auth.auth_utils import get_end_user_id_from_request_body

    mappings = [
        {"header_name": "x-openwebui-user-id", "litellm_user_role": "internal_user"},
        {"header_name": "x-openwebui-user-email", "litellm_user_role": "customer"},
    ]
    general_settings = {"user_header_mappings": mappings}
    headers = {"x-openwebui-user-email": "1234"}

    with (
        patch(
            "litellm.proxy.auth.auth_utils._get_customer_id_from_standard_headers",
            return_value=None,
        ),
        patch("litellm.proxy.proxy_server.general_settings", general_settings),
    ):
        result = get_end_user_id_from_request_body(
            request_body={}, request_headers=headers
        )

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

    with (
        patch(
            "litellm.proxy.auth.auth_utils._get_customer_id_from_standard_headers",
            return_value=None,
        ),
        patch("litellm.proxy.proxy_server.general_settings", general_settings),
    ):
        result = get_end_user_id_from_request_body(
            request_body={}, request_headers=headers
        )

    assert result == "user-456"


def test_get_end_user_id_returns_none_when_no_customer_role_in_mappings():
    from litellm.proxy.auth.auth_utils import get_end_user_id_from_request_body

    mappings = [
        {"header_name": "x-openwebui-user-id", "litellm_user_role": "internal_user"},
    ]
    general_settings = {"user_header_mappings": mappings}
    headers = {"x-openwebui-user-id": "user-789"}

    with (
        patch(
            "litellm.proxy.auth.auth_utils._get_customer_id_from_standard_headers",
            return_value=None,
        ),
        patch("litellm.proxy.proxy_server.general_settings", general_settings),
    ):
        result = get_end_user_id_from_request_body(
            request_body={}, request_headers=headers
        )

    assert result is None


def test_get_end_user_id_falls_back_to_deprecated_user_header_name():
    from litellm.proxy.auth.auth_utils import get_end_user_id_from_request_body

    general_settings = {"user_header_name": "x-custom-user-id"}
    headers = {"x-custom-user-id": "user-legacy"}

    with (
        patch(
            "litellm.proxy.auth.auth_utils._get_customer_id_from_standard_headers",
            return_value=None,
        ),
        patch("litellm.proxy.proxy_server.general_settings", general_settings),
    ):
        result = get_end_user_id_from_request_body(
            request_body={}, request_headers=headers
        )

    assert result == "user-legacy"


def _make_deployment_dict(
    model_name: str, tpm: Optional[int] = None, rpm: Optional[int] = None
) -> dict:
    """Helper to build a minimal deployment dict as returned by router.get_model_list."""
    litellm_params: dict = {"model": model_name}
    if tpm is not None:
        litellm_params["default_api_key_tpm_limit"] = tpm
    if rpm is not None:
        litellm_params["default_api_key_rpm_limit"] = rpm
    return {"model_name": model_name, "litellm_params": litellm_params}


_ROUTER_PATCH = "litellm.proxy.proxy_server.llm_router"


class TestDeploymentDefaultRpmLimit:
    """Tests for deployment default_api_key_rpm_limit fallback in get_key_model_rpm_limit."""

    def test_returns_deployment_default_when_key_has_no_limits(self):
        """Case 2 from spec: key has no model-specific limits, falls back to deployment default."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _make_deployment_dict("model1", rpm=200)
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_rpm_limit(user_api_key_dict, model_name="model1")
        assert result == {"model1": 200}

    def test_key_model_limit_takes_priority_over_deployment_default(self):
        """Case 1 from spec: key model-specific limit wins over deployment default."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            metadata={"model_rpm_limit": {"model1": 10}},
        )
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _make_deployment_dict("model1", rpm=200)
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_rpm_limit(user_api_key_dict, model_name="model1")
        assert result == {"model1": 10}

    def test_returns_none_when_no_deployment_default_and_no_key_limits(self):
        """Returns None when neither the key nor the deployment has any rpm limit."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _make_deployment_dict("model1")  # no rpm default
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_rpm_limit(user_api_key_dict, model_name="model1")
        assert result is None

    def test_returns_none_without_model_name_even_when_deployment_has_default(self):
        """No model_name means deployment fallback is skipped."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _make_deployment_dict("model1", rpm=200)
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_rpm_limit(user_api_key_dict)
        assert result is None

    def test_returns_none_when_llm_router_is_none(self):
        """No router means deployment fallback returns None gracefully."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        with patch(_ROUTER_PATCH, None):
            result = get_key_model_rpm_limit(user_api_key_dict, model_name="model1")
        assert result is None

    def test_returns_minimum_across_multiple_deployments(self):
        """When multiple deployments share a model name, the minimum rpm limit is used."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _make_deployment_dict("model1", rpm=200),
            _make_deployment_dict("model1", rpm=50),
            _make_deployment_dict("model1", rpm=150),
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_rpm_limit(user_api_key_dict, model_name="model1")
        assert result == {"model1": 50}

    def test_ignores_deployments_without_default_when_others_have_it(self):
        """Deployments missing the field are skipped; min is taken over those that have it."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _make_deployment_dict("model1"),  # no rpm default
            _make_deployment_dict("model1", rpm=75),
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_rpm_limit(user_api_key_dict, model_name="model1")
        assert result == {"model1": 75}

    def test_skips_deployments_with_malformed_limit_value(self):
        """Deployments with non-integer-parseable limit values are skipped without raising."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            {
                "model_name": "model1",
                "litellm_params": {"default_api_key_rpm_limit": "not-a-number"},
            },
            _make_deployment_dict("model1", rpm=100),
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_rpm_limit(user_api_key_dict, model_name="model1")
        # The malformed deployment is skipped; the valid one provides 100
        assert result == {"model1": 100}


class TestDeploymentDefaultTpmLimit:
    """Tests for deployment default_api_key_tpm_limit fallback in get_key_model_tpm_limit."""

    def test_returns_deployment_default_when_key_has_no_limits(self):
        """Case 2 from spec: key has no model-specific limits, falls back to deployment default."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _make_deployment_dict("model1", tpm=100)
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_tpm_limit(user_api_key_dict, model_name="model1")
        assert result == {"model1": 100}

    def test_key_model_limit_takes_priority_over_deployment_default(self):
        """Case 1 from spec: key model-specific limit wins over deployment default."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            metadata={"model_tpm_limit": {"model1": 20}},
        )
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _make_deployment_dict("model1", tpm=100)
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_tpm_limit(user_api_key_dict, model_name="model1")
        assert result == {"model1": 20}

    def test_returns_none_when_no_deployment_default_and_no_key_limits(self):
        """Returns None when neither the key nor the deployment has any tpm limit."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _make_deployment_dict("model1")  # no tpm default
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_tpm_limit(user_api_key_dict, model_name="model1")
        assert result is None

    def test_returns_none_without_model_name_even_when_deployment_has_default(self):
        """No model_name means deployment fallback is skipped."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _make_deployment_dict("model1", tpm=100)
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_tpm_limit(user_api_key_dict)
        assert result is None

    def test_returns_none_when_llm_router_is_none(self):
        """No router means deployment fallback returns None gracefully."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        with patch(_ROUTER_PATCH, None):
            result = get_key_model_tpm_limit(user_api_key_dict, model_name="model1")
        assert result is None

    def test_returns_minimum_across_multiple_deployments(self):
        """When multiple deployments share a model name, the minimum tpm limit is used."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _make_deployment_dict("model1", tpm=1000),
            _make_deployment_dict("model1", tpm=300),
            _make_deployment_dict("model1", tpm=700),
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_tpm_limit(user_api_key_dict, model_name="model1")
        assert result == {"model1": 300}

    def test_ignores_deployments_without_default_when_others_have_it(self):
        """Deployments missing the field are skipped; min is taken over those that have it."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _make_deployment_dict("model1"),  # no tpm default
            _make_deployment_dict("model1", tpm=400),
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_tpm_limit(user_api_key_dict, model_name="model1")
        assert result == {"model1": 400}


class TestGetProjectModelRpmLimit:
    """Tests for get_project_model_rpm_limit function."""

    def test_returns_project_metadata_rpm_limit(self):
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            project_metadata={"model_rpm_limit": {"gpt-4": 200}},
        )
        result = get_project_model_rpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 200}

    def test_returns_none_when_no_project_metadata(self):
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        result = get_project_model_rpm_limit(user_api_key_dict)
        assert result is None

    def test_returns_none_when_project_metadata_missing_key(self):
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            project_metadata={"other_key": "value"},
        )
        result = get_project_model_rpm_limit(user_api_key_dict)
        assert result is None


class TestGetProjectModelTpmLimit:
    """Tests for get_project_model_tpm_limit function."""

    def test_returns_project_metadata_tpm_limit(self):
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            project_metadata={"model_tpm_limit": {"gpt-4": 50000}},
        )
        result = get_project_model_tpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 50000}

    def test_returns_none_when_no_project_metadata(self):
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        result = get_project_model_tpm_limit(user_api_key_dict)
        assert result is None

    def test_returns_none_when_project_metadata_missing_key(self):
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            project_metadata={"other_key": "value"},
        )
        result = get_project_model_tpm_limit(user_api_key_dict)
        assert result is None


class TestCheckCompleteCredentials:
    """Tests for the api_key validation in check_complete_credentials."""

    def test_returns_false_when_api_key_missing(self):
        result = check_complete_credentials({"model": "gpt-4"})
        assert result is False

    def test_returns_false_when_api_key_is_none(self):
        result = check_complete_credentials({"model": "gpt-4", "api_key": None})
        assert result is False

    def test_returns_false_when_api_key_is_empty_string(self):
        result = check_complete_credentials({"model": "gpt-4", "api_key": ""})
        assert result is False

    def test_returns_false_when_api_key_is_whitespace(self):
        result = check_complete_credentials({"model": "gpt-4", "api_key": "   "})
        assert result is False

    def test_returns_true_when_api_key_is_valid(self):
        result = check_complete_credentials({"model": "gpt-4", "api_key": "sk-valid"})
        assert result is True


class TestCheckCompleteCredentialsBlocksSSRF:
    """
    Even with credentials supplied, ``api_base`` / ``base_url`` must not
    point at private / internal / cloud-metadata addresses. Without this
    the gate accepts ``api_key=anything`` plus a malicious target and the
    proxy is used as an SSRF pivot.

    The check only runs when ``litellm.user_url_validation`` is True, so
    every test in this class flips the toggle. Tests stay mock-only — no
    real DNS is performed.
    """

    @pytest.fixture(autouse=True)
    def _enable_url_validation(self, monkeypatch):
        import litellm

        monkeypatch.setattr(litellm, "user_url_validation", True, raising=False)

    @pytest.mark.parametrize(
        "url_field",
        ["api_base", "base_url"],
    )
    @pytest.mark.parametrize(
        "blocked_url",
        [
            "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
            "http://metadata.google.internal/computeMetadata/v1/",
            "http://127.0.0.1:8080/admin",
            "http://10.0.0.1/",
            "http://192.168.1.1/",
        ],
    )
    def test_rejects_private_or_metadata_targets(self, url_field, blocked_url):
        from litellm.litellm_core_utils.url_utils import SSRFError

        with patch(
            "litellm.proxy.auth.auth_utils.validate_url",
            side_effect=SSRFError(f"blocked: {blocked_url}"),
        ):
            with pytest.raises(ValueError) as exc_info:
                check_complete_credentials(
                    {
                        "model": "gpt-4",
                        "api_key": "sk-some-clientside-key",
                        url_field: blocked_url,
                    }
                )
        assert url_field in str(exc_info.value)
        assert "SSRF" in str(exc_info.value)

    def test_allows_public_target_when_validate_url_passes(self):
        # ``validate_url`` is mocked so no real DNS is performed.
        with patch(
            "litellm.proxy.auth.auth_utils.validate_url",
            return_value=("https://api.openai.com/v1", "api.openai.com"),
        ):
            result = check_complete_credentials(
                {
                    "model": "gpt-4",
                    "api_key": "sk-some-clientside-key",
                    "api_base": "https://api.openai.com/v1",
                }
            )
        assert result is True

    def test_skips_url_validation_when_toggle_is_off(self, monkeypatch):
        # Admins who disable ``user_url_validation`` (default) should not
        # have requests rejected at the proxy boundary even if the URL
        # would fail the SSRF guard.
        import litellm

        monkeypatch.setattr(litellm, "user_url_validation", False, raising=False)
        with patch(
            "litellm.proxy.auth.auth_utils.validate_url",
        ) as mocked:
            result = check_complete_credentials(
                {
                    "model": "gpt-4",
                    "api_key": "sk-some-clientside-key",
                    "api_base": "http://127.0.0.1:8080/admin",
                }
            )
        assert result is True
        mocked.assert_not_called()


class TestGetDynamicLitellmParamsClearsAdminConfigOnBaseOverride:
    """
    When the caller redirects ``api_base`` / ``base_url`` to their own
    server, admin-set fields like ``OpenAI-Organization``, ``extra_body``,
    AWS / Vertex / Azure tokens, and per-deployment ``api_version`` must
    NOT flow through to that destination.
    """

    def test_clears_admin_organization_and_extra_body_on_base_override(self):
        from litellm.router_utils.clientside_credential_handler import (
            get_dynamic_litellm_params,
        )

        admin_params = {
            "model": "gpt-4",
            "api_key": "sk-admin-key",
            "api_base": "https://admin.upstream/v1",
            "organization": "org-admin-corp",
            "extra_body": {"x-admin-secret": "super-secret"},
            "api_version": "2026-04-01",
        }
        out = get_dynamic_litellm_params(
            litellm_params=dict(admin_params),
            request_kwargs={
                "api_key": "sk-attacker",
                "api_base": "https://attacker.example",
            },
        )
        assert out["api_base"] == "https://attacker.example"
        assert out["api_key"] == "sk-attacker"
        assert "organization" not in out
        assert "extra_body" not in out
        assert "api_version" not in out

    def test_clears_aws_and_vertex_secrets_on_base_override(self):
        from litellm.router_utils.clientside_credential_handler import (
            get_dynamic_litellm_params,
        )

        admin_params = {
            "model": "bedrock/claude-3",
            "aws_access_key_id": "AKIA-EXAMPLE",
            "aws_secret_access_key": "secret-example",
            "aws_session_token": "session-example",
            "vertex_credentials": '{"private_key":"-----BEGIN..."}',
            "vertex_project": "admin-gcp-project",
        }
        out = get_dynamic_litellm_params(
            litellm_params=dict(admin_params),
            request_kwargs={"base_url": "https://attacker.example"},
        )
        assert "aws_access_key_id" not in out
        assert "aws_secret_access_key" not in out
        assert "aws_session_token" not in out
        assert "vertex_credentials" not in out
        assert "vertex_project" not in out

    def test_caller_resupplied_value_overrides_admin_value_on_base_override(self):
        # When the caller redirects ``api_base`` and *also* supplies their
        # own value for one of the admin fields (e.g. ``organization``),
        # the caller's value must win — never the admin's. The naive
        # ``if field not in request_kwargs: pop`` shape lets a caller echo
        # the field name with any value (or empty string) to keep the
        # admin's value forwarded, which is the exfiltration vector this
        # test guards against.
        from litellm.router_utils.clientside_credential_handler import (
            get_dynamic_litellm_params,
        )

        out = get_dynamic_litellm_params(
            litellm_params={
                "api_base": "https://admin.upstream/v1",
                "organization": "org-admin",
                "extra_body": {"admin": "value"},
            },
            request_kwargs={
                "api_base": "https://attacker.example",
                "organization": "org-attacker",
                "extra_body": {"attacker": "value"},
            },
        )
        assert out["organization"] == "org-attacker"
        assert out["extra_body"] == {"attacker": "value"}

    def test_field_echo_does_not_preserve_admin_value(self):
        # Regression: a caller that echoes an admin-config field name with
        # an *empty* value (or any value) must not be able to keep the
        # admin's value in ``litellm_params``.
        from litellm.router_utils.clientside_credential_handler import (
            get_dynamic_litellm_params,
        )

        out = get_dynamic_litellm_params(
            litellm_params={
                "api_base": "https://admin.upstream/v1",
                "organization": "org-admin-secret",
                "extra_body": {"x-admin-only": "secret"},
            },
            request_kwargs={
                "api_base": "https://attacker.example",
                "organization": "",
                "extra_body": "",
            },
        )
        assert out["organization"] == ""
        assert out["extra_body"] == ""
        assert "org-admin-secret" not in str(out)

    def test_no_clearing_when_only_api_key_overridden(self):
        from litellm.router_utils.clientside_credential_handler import (
            get_dynamic_litellm_params,
        )

        # Caller only overrides api_key (BYOK pattern); admin's organization /
        # extra_body / region still apply because the destination is unchanged.
        out = get_dynamic_litellm_params(
            litellm_params={
                "api_base": "https://admin.upstream/v1",
                "organization": "org-admin",
                "api_version": "2026-04-01",
            },
            request_kwargs={"api_key": "sk-byok"},
        )
        assert out["organization"] == "org-admin"
        assert out["api_version"] == "2026-04-01"
        assert out["api_base"] == "https://admin.upstream/v1"


class TestIsRequestBodySafeBlocksEndpointTargetingFields:
    """
    ``is_request_body_safe`` rejects request-body fields that retarget the
    outbound request to a caller-controlled host. Beyond the original
    ``api_base`` / ``base_url``, the same protection must apply to:

    * ``aws_bedrock_runtime_endpoint`` — Bedrock endpoint redirect; an
      attacker-controlled value coerces the proxy to authenticate against
      their host with the admin's AWS creds.
    * ``langsmith_base_url`` — Langsmith callback host; attacker-controlled
      values exfiltrate the entire request payload (incl. message content)
      via the observability hook.
    * ``langfuse_host`` — same exfil vector via the Langfuse hook.
    """

    @pytest.fixture(autouse=True)
    def _disable_url_validation(self, monkeypatch):
        # The new banned-params entries should be rejected even when
        # ``user_url_validation`` is off — the gate isn't the URL guard,
        # it's the banned-params list.
        import litellm

        monkeypatch.setattr(litellm, "user_url_validation", False, raising=False)

    @pytest.mark.parametrize(
        "field",
        [
            "aws_bedrock_runtime_endpoint",
            "langsmith_base_url",
            "langfuse_host",
            "posthog_host",
            "braintrust_host",
            "slack_webhook_url",
            "s3_endpoint_url",
            "sagemaker_base_url",
            "deployment_url",
        ],
    )
    def test_endpoint_targeting_field_in_request_body_is_rejected(self, field):
        with pytest.raises(ValueError) as exc:
            is_request_body_safe(
                request_body={"model": "gpt-4", field: "https://attacker.example"},
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )
        # The function lists the offending param name in the error.
        assert field in str(exc.value)

    @pytest.mark.parametrize(
        "field",
        ["api_base", "base_url", "user_config", "langfuse_host", "slack_webhook_url"],
    )
    def test_api_key_does_not_bypass_blocklist(self, field):
        # Regression: the historical ``check_complete_credentials`` clause
        # made the entire blocklist a no-op for any caller that supplied
        # a non-empty ``api_key``. That bypass turned every missing entry
        # on the blocklist into an SSRF / credential-exfil hole. Verify
        # that supplying an api_key (alongside the banned param) does NOT
        # bypass the gate — it can only be opened by an admin opt-in.
        with pytest.raises(ValueError) as exc:
            is_request_body_safe(
                request_body={
                    "model": "gpt-4",
                    "api_key": "sk-anything",
                    field: "https://attacker.example",
                },
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )
        assert field in str(exc.value)

    def test_admin_opt_in_proxy_wide_still_allows(self):
        # ``general_settings.allow_client_side_credentials = True`` remains
        # the documented proxy-wide BYOK opt-in.
        assert (
            is_request_body_safe(
                request_body={"model": "gpt-4", "api_base": "https://my-byok.example"},
                general_settings={"allow_client_side_credentials": True},
                llm_router=None,
                model="gpt-4",
            )
            is True
        )


# ── is_request_body_safe nested-config recursion (VERIA-6) ────────────────────


class TestIsRequestBodySafeNestedConfig:
    """The Milvus vector store transformer unpacks
    ``litellm_embedding_config`` as ``**kwargs`` into ``litellm.embedding(...)``
    — same SSRF / credential-exfil surface as a top-level ``api_base`` in
    the request body. ``is_request_body_safe`` must recurse into this
    nested dict so a banned param can't be smuggled in via nesting."""

    def test_root_level_api_base_blocked_when_no_opt_in(self):
        """Sanity check: pre-existing root-level enforcement still works."""
        with pytest.raises(ValueError, match="api_base"):
            is_request_body_safe(
                request_body={"api_base": "https://attacker.example.com"},
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )

    def test_nested_api_base_in_embedding_config_blocked(self):
        """Smuggling ``api_base`` inside ``litellm_embedding_config`` is
        the VERIA-6 bypass — must be blocked by the recursive check."""
        with pytest.raises(ValueError, match="api_base"):
            is_request_body_safe(
                request_body={
                    "litellm_embedding_config": {
                        "api_base": "https://attacker.example.com",
                        "api_key": "leaked-key",
                    }
                },
                general_settings={},
                llm_router=None,
                model="milvus-store",
            )

    def test_nested_langfuse_host_in_embedding_config_blocked(self):
        """The recursion uses the *full* banned-param list, not a special
        subset — so any flag that's banned at the root is also banned
        when nested."""
        with pytest.raises(ValueError, match="langfuse_host"):
            is_request_body_safe(
                request_body={
                    "litellm_embedding_config": {
                        "langfuse_host": "https://attacker.example.com"
                    }
                },
                general_settings={},
                llm_router=None,
                model="milvus-store",
            )

    def test_nested_api_base_allowed_when_admin_opts_in(self):
        """Admins who explicitly enable client-side credential passthrough
        keep the existing escape hatch — same UX as for root-level."""
        assert (
            is_request_body_safe(
                request_body={
                    "litellm_embedding_config": {
                        "api_base": "https://my-azure.example.com"
                    }
                },
                general_settings={"allow_client_side_credentials": True},
                llm_router=None,
                model="milvus-store",
            )
            is True
        )

    def test_safe_nested_config_accepted(self):
        """A nested config without any banned params passes — there's no
        false-positive on legitimate ``api_version`` / model params."""
        assert (
            is_request_body_safe(
                request_body={
                    "litellm_embedding_config": {
                        "api_version": "2024-02-15-preview",
                    }
                },
                general_settings={},
                llm_router=None,
                model="milvus-store",
            )
            is True
        )

    def test_non_dict_nested_config_does_not_break_check(self):
        """A bogus type for ``litellm_embedding_config`` (string, list,
        None) must not crash the validator — it should just fall through."""
        assert (
            is_request_body_safe(
                request_body={"litellm_embedding_config": "not-a-dict"},
                general_settings={},
                llm_router=None,
                model="x",
            )
            is True
        )

    def test_deeply_nested_config_does_not_recurse(self):
        """Greptile P1: ``is_request_body_safe`` is iterative single-level —
        a deeply-nested ``litellm_embedding_config`` cannot exhaust the
        Python call stack to trigger a 500 ``RecursionError``. Build a
        body 1000 levels deep; the validator must complete in O(1)
        descent."""
        body = {"litellm_embedding_config": {}}
        cur = body["litellm_embedding_config"]
        for _ in range(1000):
            cur["litellm_embedding_config"] = {}
            cur = cur["litellm_embedding_config"]
        # Banned param at the deepest level shouldn't be reached — single
        # level only.
        cur["api_base"] = "https://attacker.example.com"

        # No exception raised: deeper levels aren't checked.
        assert (
            is_request_body_safe(
                request_body=body,
                general_settings={},
                llm_router=None,
                model="x",
            )
            is True
        )


# ── observability-callback ban (root + metadata) ───────────────────────────


class TestObservabilityCallbackBans:
    """The proxy must reject observability credentials, hosts, and project
    identifiers regardless of whether they arrive at the request body root,
    in ``metadata`` / ``litellm_metadata``, or in a JSON-string-encoded
    metadata blob (multipart/``extra_body`` path).

    The ban list is derived from
    ``litellm.litellm_core_utils.initialize_dynamic_callback_params._supported_callback_params``
    minus a small ``_SAFE_CLIENT_CALLBACK_PARAMS`` allow-list, plus
    ``_EXTRA_BANNED_OBSERVABILITY_PARAMS`` for fields integrations read but
    that are not yet in the canonical allow-list. The derivation keeps the
    proxy in sync as new integrations are added.
    """

    @pytest.fixture(autouse=True)
    def _disable_url_validation(self, monkeypatch):
        import litellm

        monkeypatch.setattr(litellm, "user_url_validation", False, raising=False)

    @pytest.mark.parametrize(
        "field",
        [
            "langfuse_public_key",
            "langfuse_secret",
            "langfuse_secret_key",
            "langsmith_api_key",
            "langsmith_project",
            "langsmith_tenant_id",
            "arize_api_key",
            "arize_space_key",
            "arize_space_id",
            "posthog_api_key",
            "posthog_api_url",
            "braintrust_api_key",
            "braintrust_project",
            "phoenix_project_name",
            "wandb_api_key",
            "weave_project_id",
            "gcs_bucket_name",
            "gcs_path_service_account",
            "humanloop_api_key",
            "lunary_public_key",
        ],
    )
    def test_observability_field_in_request_body_root_is_rejected(self, field):
        with pytest.raises(ValueError) as exc:
            is_request_body_safe(
                request_body={"model": "gpt-4", field: "attacker-value"},
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )
        assert field in str(exc.value)

    @pytest.mark.parametrize(
        "metadata_key",
        ["metadata", "litellm_metadata"],
    )
    @pytest.mark.parametrize(
        "field",
        [
            "langfuse_host",
            "langfuse_secret_key",
            "langsmith_api_key",
            "posthog_api_url",
            "braintrust_project",
            "phoenix_project_name",
        ],
    )
    def test_observability_field_in_metadata_dict_is_rejected(
        self, metadata_key, field
    ):
        # Verifies the metadata walk: a value smuggled inside ``metadata``
        # or ``litellm_metadata`` is just as dangerous as the same field
        # at the body root, and must hit the same gate.
        with pytest.raises(ValueError) as exc:
            is_request_body_safe(
                request_body={
                    "model": "gpt-4",
                    metadata_key: {field: "attacker-value"},
                },
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )
        assert field in str(exc.value)

    @pytest.mark.parametrize(
        "metadata_key",
        ["metadata", "litellm_metadata"],
    )
    def test_observability_field_in_json_string_metadata_is_rejected(
        self, metadata_key
    ):
        # Multipart/form-data and ``extra_body`` callers send metadata as a
        # JSON-encoded string. The bouncer parses it before applying the
        # banned-params check so the JSON-string path can't smuggle past
        # the ``isinstance(dict)`` guard.
        import json

        with pytest.raises(ValueError) as exc:
            is_request_body_safe(
                request_body={
                    "model": "gpt-4",
                    metadata_key: json.dumps(
                        {"langfuse_host": "https://attacker.example"}
                    ),
                },
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )
        assert "langfuse_host" in str(exc.value)

    def test_admin_opt_in_allows_metadata_credential_passthrough(self):
        # The opt-in gate covers the metadata path the same way it covers
        # the root path — operators running BYO observability with
        # clientside creds flip a single flag and both paths work.
        assert (
            is_request_body_safe(
                request_body={
                    "model": "gpt-4",
                    "metadata": {
                        "langfuse_host": "https://my-langfuse.example",
                        "langfuse_public_key": "pk-mine",
                        "langfuse_secret_key": "sk-mine",
                    },
                },
                general_settings={"allow_client_side_credentials": True},
                llm_router=None,
                model="gpt-4",
            )
            is True
        )

    def test_safe_per_request_observability_metadata_is_allowed(self):
        # Informational fields (sampling rate, prompt version) describe
        # the request being logged — they don't choose the destination or
        # credentials, so they must remain accepted from clients without
        # the opt-in flag.
        assert (
            is_request_body_safe(
                request_body={
                    "model": "gpt-4",
                    "metadata": {
                        "langfuse_prompt_version": "v2",
                        "langsmith_sampling_rate": 0.1,
                    },
                },
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )
            is True
        )


def test_model_level_allow_does_not_skip_subsequent_banned_params(monkeypatch):
    """Greptile P1: ``_check_banned_params`` previously ``return``-ed when a
    deployment's ``configurable_clientside_auth_params`` permitted one
    banned field, exiting before any later banned field in the same body
    was checked. The metadata walk this PR adds multiplies the surface
    where that bypass matters: a body pairing a model-level-allowed
    ``api_base`` with an observability credential like ``langfuse_host``
    must still reject on the second field, not silently pass."""
    from litellm.proxy.auth import auth_utils

    monkeypatch.setattr(
        auth_utils,
        "_allow_model_level_clientside_configurable_parameters",
        lambda model, param, request_body_value, llm_router: param == "api_base",
    )

    with pytest.raises(ValueError) as exc:
        is_request_body_safe(
            request_body={
                "model": "gpt-4",
                "api_base": "https://allowed-by-deployment.example",
                "langfuse_host": "https://attacker.example",
            },
            general_settings={},
            llm_router=None,
            model="gpt-4",
        )
    assert "langfuse_host" in str(exc.value)


def test_observability_ban_covers_canonical_supported_callback_params():
    """Guard test: every entry in the canonical
    ``_supported_callback_params`` allow-list must end up either banned by
    the proxy or explicitly safe-listed. New integrations added to that
    list are banned by default (the safe failure mode); flagging them as
    safe is an explicit decision recorded in
    ``_SAFE_CLIENT_CALLBACK_PARAMS``."""
    from litellm.litellm_core_utils.initialize_dynamic_callback_params import (
        _request_blocked_callback_params,
        _supported_callback_params,
    )
    from litellm.proxy.auth.auth_utils import (
        _BANNED_REQUEST_BODY_PARAMS,
        _SAFE_CLIENT_CALLBACK_PARAMS,
    )

    banned = set(_BANNED_REQUEST_BODY_PARAMS)
    for param in _supported_callback_params:
        assert param in banned or param in _SAFE_CLIENT_CALLBACK_PARAMS, (
            f"{param} is in _supported_callback_params but neither banned nor "
            f"safe-listed. Add it to _SAFE_CLIENT_CALLBACK_PARAMS if it is an "
            f"informational per-request field; otherwise the derivation will "
            f"ban it automatically."
        )
    for param in _request_blocked_callback_params:
        assert param in banned, (
            f"{param} is in _request_blocked_callback_params but is not banned "
            "at the proxy request-body boundary."
        )
