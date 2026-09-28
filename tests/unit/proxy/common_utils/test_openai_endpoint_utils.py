import copy

import pytest

from litellm.proxy.common_utils.openai_endpoint_utils import (
    apply_openai_project_to_data,
    remove_sensitive_info_from_deployment,
)


@pytest.mark.parametrize(
    "model_config, expected_config",
    [
        # Test case 1: Empty litellm_params
        (
            {"model_name": "test-model", "litellm_params": {}},
            {"model_name": "test-model", "litellm_params": {}},
        ),
        # Test case 2: Full sensitive data removal, mixed secrets of azure, aws, gcp, and typical api_key
        (
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_key": "sk-sensitive-key-123",
                    "client_secret": "~v8Q4W:Zp9gJ-3sTqX5aB@LkR2mNfYdC",
                    "vertex_credentials": {"type": "service_account"},
                    "aws_access_key_id": "AKIA123456789",
                    "aws_secret_access_key": "secret-access-key",
                    "api_base": "https://api.openai.com/v1",
                    "temperature": 0.7,
                },
                "model_info": {"id": "test-id"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_base": "https://api.openai.com/v1",
                    "temperature": 0.7,
                },
                "model_info": {"id": "test-id"},
            },
        ),
        # Test case 3: Partial sensitive data, api_key
        (
            {
                "model_name": "claude-3",
                "litellm_params": {
                    "model": "anthropic/claude-3",
                    "api_key": "sk-anthropic-key",
                    "temperature": 0.5,
                },
            },
            {
                "model_name": "claude-3",
                "litellm_params": {"model": "anthropic/claude-3", "temperature": 0.5},
            },
        ),
        # Test case 4: No sensitive data
        (
            {
                "model_name": "local-model",
                "litellm_params": {
                    "model": "local/model",
                    "temperature": 0.8,
                    "max_tokens": 100,
                },
            },
            {
                "model_name": "local-model",
                "litellm_params": {
                    "model": "local/model",
                    "temperature": 0.8,
                    "max_tokens": 100,
                },
            },
        ),
    ],
)
def test_remove_sensitive_info_from_deployment(
    model_config: dict, expected_config: dict
):
    sanitized_config = remove_sensitive_info_from_deployment(model_config)
    assert sanitized_config == expected_config


def test_remove_sensitive_info_from_deployment_with_excluded_keys():
    """
    Test that excluded_keys prevents masking of specific keys (exact match).
    """
    base_config = {
        "model_name": "test-model",
        "litellm_params": {
            "model": "openai/gpt-4",
            "api_key": "sk-sensitive-key-123",
            "litellm_credentials_name": "my-credential-name",
            "access_token": "token-12345",
            "temperature": 0.7,
        },
    }

    # Without excluded_keys, access_token should be masked (contains "token")
    sanitized_config = remove_sensitive_info_from_deployment(copy.deepcopy(base_config))
    assert sanitized_config["litellm_params"]["access_token"] != "token-12345"
    assert "*" in sanitized_config["litellm_params"]["access_token"]

    # With excluded_keys, litellm_credentials_name should NOT be masked.
    # ``remove_sensitive_info_from_deployment`` mutates its input, so feed it
    # a fresh copy rather than the already-sanitized one.
    sanitized_config = remove_sensitive_info_from_deployment(
        copy.deepcopy(base_config), excluded_keys={"litellm_credentials_name"}
    )
    assert (
        sanitized_config["litellm_params"]["litellm_credentials_name"]
        == "my-credential-name"
    )

    # access_token should still be masked (not in excluded_keys)
    assert sanitized_config["litellm_params"]["access_token"] != "token-12345"
    assert "*" in sanitized_config["litellm_params"]["access_token"]

    # api_key should still be removed (popped) regardless of excluded_keys
    assert "api_key" not in sanitized_config["litellm_params"]


# --------------------------------------------------------------------------- #
# apply_openai_project_to_data: a caller-supplied OpenAI project selects which
# project the proxy's shared credential acts inside, so it only reaches the
# provider once the admin opted in with `forward_openai_project`.
# --------------------------------------------------------------------------- #

FORWARDED: dict = {"forward_openai_project": True}


class _FakeRequest:
    """Stand-in for the FastAPI request; the helper only reads `.headers`."""

    def __init__(self, headers: dict | None = None):
        self.headers = headers or {}


def test_openai_project_is_dropped_without_the_forwarding_opt_in():
    data = {"project": "proj_from_client"}

    apply_openai_project_to_data(
        data=data,
        request=_FakeRequest({"OpenAI-Project": "proj_from_header"}),
        general_settings={},
    )

    assert "project" not in data


def test_openai_project_is_dropped_when_general_settings_is_missing():
    data = {"project": "proj_from_client"}

    apply_openai_project_to_data(data=data, request=_FakeRequest())

    assert "project" not in data


def test_openai_project_header_is_forwarded_when_opted_in():
    data: dict = {}

    apply_openai_project_to_data(data=data, request=_FakeRequest({"OpenAI-Project": "proj_a"}), general_settings=FORWARDED)

    assert data["project"] == "proj_a"


def test_openai_project_header_wins_over_the_request_body_value():
    data = {"project": "proj_from_body"}

    apply_openai_project_to_data(
        data=data,
        request=_FakeRequest({"OpenAI-Project": "proj_from_header"}),
        general_settings=FORWARDED,
    )

    assert data["project"] == "proj_from_header"


def test_openai_project_body_value_is_forwarded_when_opted_in_without_header():
    data = {"project": "proj_from_body"}

    apply_openai_project_to_data(data=data, request=_FakeRequest(), general_settings=FORWARDED)

    assert data["project"] == "proj_from_body"


def test_openai_project_ignores_a_non_string_body_value():
    data = {"project": {"nested": "value"}}

    apply_openai_project_to_data(data=data, request=_FakeRequest(), general_settings=FORWARDED)

    assert "project" not in data


def test_openai_project_stays_absent_when_no_caller_supplied_one():
    data: dict = {"batch_id": "batch-1"}

    apply_openai_project_to_data(data=data, request=_FakeRequest(), general_settings=FORWARDED)

    assert "project" not in data
    assert data == {"batch_id": "batch-1"}
