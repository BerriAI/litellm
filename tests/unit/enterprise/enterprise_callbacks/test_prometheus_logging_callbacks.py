

from unittest.mock import patch

import pytest
from prometheus_client import REGISTRY

import litellm
from litellm.types.utils import (
    StandardLoggingHiddenParams,
    StandardLoggingMetadata,
    StandardLoggingModelInformation,
    StandardLoggingPayload,
)

try:
    from litellm.integrations.prometheus import (
        PrometheusLogger,
        get_custom_labels_from_metadata,
    )
except Exception:
    PrometheusLogger = None


@pytest.fixture
def prometheus_logger() -> PrometheusLogger:
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)
    return PrometheusLogger()


@pytest.fixture
def known_model_router():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-5-mini",
                "litellm_params": {"model": "openai/gpt-5-mini", "api_key": "fake-key"},
            },
            {
                "model_name": "us/azure/openai/gpt-5-mini",
                "litellm_params": {"model": "openai/gpt-5-mini", "api_key": "fake-key"},
            },
        ]
    )
    with patch("litellm.proxy.proxy_server.llm_router", router, create=True):  # test-quality-ok: production reads proxy_server.llm_router lazily, no injection seam
        yield router


def create_standard_logging_payload() -> StandardLoggingPayload:
    return StandardLoggingPayload(
        id="test_id",
        call_type="completion",
        stream=False,
        response_cost=0.1,
        response_cost_failure_debug_info=None,
        status="success",
        total_tokens=30,
        prompt_tokens=20,
        completion_tokens=10,
        startTime=1234567890.0,
        endTime=1234567891.0,
        completionStartTime=1234567890.5,
        model_map_information=StandardLoggingModelInformation(
            model_map_key="gpt-5-mini", model_map_value=None
        ),
        model="gpt-5-mini",
        model_id="model-123",
        model_group="openai-gpt",
        custom_llm_provider="openai",
        api_base="https://api.openai.com",
        metadata=StandardLoggingMetadata(
            user_api_key_hash="test_hash",
            user_api_key_alias="test_alias",
            user_api_key_team_id="test_team",
            user_api_key_user_id="test_user",
            user_api_key_user_email="test@example.com",
            user_api_key_team_alias="test_team_alias",
            user_api_key_org_id=None,
            spend_logs_metadata=None,
            requester_ip_address="127.0.0.1",
            requester_metadata=None,
            user_api_key_end_user_id="test_end_user",
        ),
        cache_hit=False,
        cache_key=None,
        saved_cache_cost=0.0,
        request_tags=[],
        end_user=None,
        requester_ip_address="127.0.0.1",
        messages=[{"role": "user", "content": "Hello, world!"}],
        response={"choices": [{"message": {"content": "Hi there!"}}]},
        error_str=None,
        model_parameters={"stream": True},
        hidden_params=StandardLoggingHiddenParams(
            model_id="model-123",
            cache_key=None,
            api_base="https://api.openai.com",
            response_cost="0.1",
            additional_headers=None,
        ),
    )






































def test_get_custom_labels_from_metadata(monkeypatch):
    monkeypatch.setattr(
        "litellm.custom_prometheus_metadata_labels", ["metadata.foo", "metadata.bar"]
    )
    metadata = {"foo": "bar", "bar": "baz", "taz": "qux"}
    assert get_custom_labels_from_metadata(metadata) == {
        "metadata_foo": "bar",
        "metadata_bar": "baz",
    }


def test_get_custom_labels_from_metadata_tags(monkeypatch):
    monkeypatch.setattr("litellm.custom_prometheus_metadata_labels", [])
    metadata = {"foo": "bar", "bar": "baz", "taz": "qux"}
    assert get_custom_labels_from_metadata(metadata) == {}


def test_get_custom_labels_from_top_level_metadata(monkeypatch):
    """
    Test that get_custom_labels_from_metadata can extract fields from top-level metadata,
    such as requester_ip_address, not just from nested dictionaries like requester_metadata.
    """
    monkeypatch.setattr(
        "litellm.custom_prometheus_metadata_labels",
        ["requester_ip_address", "user_api_key_alias"],
    )
    # Simulate metadata structure with top-level fields
    metadata = {
        "requester_ip_address": "10.48.203.20",  # Top-level field
        "user_api_key_alias": "TestAlias",  # Top-level field
        "requester_metadata": {
            "nested_field": "nested_value"
        },  # Nested dict (excluded)
        "user_api_key_auth_metadata": {
            "another_nested": "value"
        },  # Nested dict (excluded)
    }
    result = get_custom_labels_from_metadata(metadata)
    assert result == {
        "requester_ip_address": "10.48.203.20",
        "user_api_key_alias": "TestAlias",
    }


def test_get_custom_labels_from_top_level_and_nested_metadata(monkeypatch):
    """
    Test that get_custom_labels_from_metadata can extract fields from both top-level
    and nested metadata (requester_metadata, user_api_key_auth_metadata).
    """
    monkeypatch.setattr(
        "litellm.custom_prometheus_metadata_labels",
        [
            "requester_ip_address",  # Top-level
            "metadata.foo",  # From requester_metadata
            "metadata.bar",  # From user_api_key_auth_metadata
        ],
    )
    # Simulate combined_metadata structure as it would appear after merging
    # This is what gets passed to get_custom_labels_from_metadata
    combined_metadata = {
        "requester_ip_address": "10.48.203.20",  # Top-level field
        "foo": "bar_value",  # From requester_metadata (spread)
        "bar": "baz_value",  # From user_api_key_auth_metadata (spread)
    }
    result = get_custom_labels_from_metadata(combined_metadata)
    assert result == {
        "requester_ip_address": "10.48.203.20",
        "metadata_foo": "bar_value",
        "metadata_bar": "baz_value",
    }


    # The test passes if labels() was called successfully, which means custom labels were handled correctly


def test_get_custom_labels_from_tags(monkeypatch):
    from litellm.integrations.prometheus import get_custom_labels_from_tags

    monkeypatch.setattr(
        "litellm.custom_prometheus_tags", ["prod", "test-env", "batch.job"]
    )
    tags = ["prod", "debug", "batch.job"]
    result = get_custom_labels_from_tags(tags)
    assert result == {
        "tag_prod": "true",
        "tag_test_env": "false",  # not in request tags
        "tag_batch_job": "true",  # dot replaced with underscore
    }


def test_get_custom_labels_from_tags_empty_config(monkeypatch):
    from litellm.integrations.prometheus import get_custom_labels_from_tags

    monkeypatch.setattr("litellm.custom_prometheus_tags", [])
    tags = ["prod", "debug"]
    result = get_custom_labels_from_tags(tags)
    assert result == {}


def test_get_custom_labels_from_tags_no_tags(monkeypatch):
    from litellm.integrations.prometheus import get_custom_labels_from_tags

    monkeypatch.setattr("litellm.custom_prometheus_tags", ["prod", "test"])
    tags = []
    result = get_custom_labels_from_tags(tags)
    assert result == {
        "tag_prod": "false",
        "tag_test": "false",
    }


def test_get_custom_labels_from_tags_wildcard_patterns(monkeypatch):
    """Test wildcard pattern matching for custom labels from tags."""
    from litellm.integrations.prometheus import get_custom_labels_from_tags

    # Configure tags with wildcard patterns
    monkeypatch.setattr(
        "litellm.custom_prometheus_tags",
        [
            "User-Agent: curl/*",
            "User-Agent: python-requests/*",
            "Environment: prod*",
            "Service: api-gateway*",
            "exact-match",
        ],
    )

    # Test tags that should match the wildcard patterns
    tags = [
        "User-Agent: curl/7.68.0",
        "User-Agent: python-requests/2.28.1",
        "Environment: production",
        "Service: api-gateway-v2",
        "exact-match",
        "other-tag",
    ]

    result = get_custom_labels_from_tags(tags)

    expected = {
        "tag_User_Agent__curl__": "true",  # matches "User-Agent: curl/*"
        "tag_User_Agent__python_requests__": "true",  # matches "User-Agent: python-requests/*"
        "tag_Environment__prod_": "true",  # matches "Environment: prod*"
        "tag_Service__api_gateway_": "true",  # matches "Service: api-gateway*"
        "tag_exact_match": "true",  # exact match
    }

    assert result == expected


def test_get_custom_labels_from_tags_wildcard_no_matches(monkeypatch):
    """Test wildcard patterns that don't match any tags."""
    from litellm.integrations.prometheus import get_custom_labels_from_tags

    # Configure tags with wildcard patterns
    monkeypatch.setattr(
        "litellm.custom_prometheus_tags",
        ["User-Agent: firefox/*", "Environment: dev*", "Service: web-app*"],
    )

    # Test tags that should NOT match the wildcard patterns
    tags = [
        "User-Agent: curl/7.68.0",  # doesn't match "User-Agent: firefox/*"
        "Environment: production",  # doesn't match "Environment: dev*"
        "Service: api-gateway-v2",  # doesn't match "Service: web-app*"
        "other-tag",
    ]

    result = get_custom_labels_from_tags(tags)

    expected = {
        "tag_User_Agent__firefox__": "false",  # no match for "User-Agent: firefox/*"
        "tag_Environment__dev_": "false",  # no match for "Environment: dev*"
        "tag_Service__web_app_": "false",  # no match for "Service: web-app*"
    }

    assert result == expected


def test_tag_matches_wildcard_configured_pattern():
    """Test the helper function for wildcard pattern matching."""
    from litellm.integrations.prometheus import _tag_matches_wildcard_configured_pattern

    # Test cases that should match
    assert (
        _tag_matches_wildcard_configured_pattern(
            tags=["User-Agent: curl/7.68.0", "prod", "other"],
            configured_tag="User-Agent: curl/*",
        )
        is True
    )

    assert (
        _tag_matches_wildcard_configured_pattern(
            tags=["User-Agent: python-requests/2.28.1", "test"],
            configured_tag="User-Agent: python-requests/*",
        )
        is True
    )

    assert (
        _tag_matches_wildcard_configured_pattern(
            tags=["Environment: production", "debug"],
            configured_tag="Environment: prod*",
        )
        is True
    )

    # Test exact match (no wildcard)
    assert (
        _tag_matches_wildcard_configured_pattern(
            tags=["prod", "test"], configured_tag="prod"
        )
        is True
    )

    # Test cases that should NOT match
    assert (
        _tag_matches_wildcard_configured_pattern(
            tags=["User-Agent: firefox/98.0", "prod"],
            configured_tag="User-Agent: curl/*",
        )
        is False
    )

    assert (
        _tag_matches_wildcard_configured_pattern(
            tags=["Environment: development", "test"],
            configured_tag="Environment: prod*",
        )
        is False
    )

    assert (
        _tag_matches_wildcard_configured_pattern(
            tags=["staging", "test"], configured_tag="prod"
        )
        is False
    )

    # Test with empty tags
    assert (
        _tag_matches_wildcard_configured_pattern(
            tags=[], configured_tag="User-Agent: curl/*"
        )
        is False
    )


















def test_get_exception_class_name(prometheus_logger):
    """
    Test that _get_exception_class_name correctly formats the exception class name
    """
    # Test case 1: Exception with llm_provider
    rate_limit_error = litellm.RateLimitError(
        message="Rate limit exceeded", llm_provider="openai", model="gpt-5-mini"
    )
    assert (
        prometheus_logger._get_exception_class_name(rate_limit_error)
        == "Openai.RateLimitError"
    )

    # Test case 2: Exception with empty llm_provider
    auth_error = litellm.AuthenticationError(
        message="Invalid API key", llm_provider="", model="gpt-5.5"
    )
    assert (
        prometheus_logger._get_exception_class_name(auth_error) == "AuthenticationError"
    )

    # Test case 3: Exception with None llm_provider
    context_window_error = litellm.ContextWindowExceededError(
        message="Context length exceeded", llm_provider=None, model="gpt-5.5"
    )
    assert (
        prometheus_logger._get_exception_class_name(context_window_error)
        == "ContextWindowExceededError"
    )




