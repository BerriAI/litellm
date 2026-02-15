import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import pytest
from fastapi import Request

from litellm.proxy.utils import _get_docs_url, _get_redoc_url

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import is_request_body_safe
from litellm.proxy.litellm_pre_call_utils import (
    _get_dynamic_logging_metadata,
    add_litellm_data_to_request,
)


@pytest.fixture
def mock_request(monkeypatch):
    mock_request = Mock(spec=Request)
    mock_request.query_params = {}  # Set mock query_params to an empty dictionary
    mock_request.headers = {"traceparent": "test_traceparent"}
    monkeypatch.setattr(
        "litellm.proxy.litellm_pre_call_utils.add_litellm_data_to_request", mock_request
    )
    return mock_request


@pytest.mark.parametrize("endpoint", ["/v1/threads", "/v1/thread/123"])
@pytest.mark.asyncio
async def test_add_litellm_data_to_request_thread_endpoint(endpoint, mock_request):
    mock_request.url.path = endpoint
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key", user_id="test_user_id", org_id="test_org_id"
    )
    proxy_config = Mock()

    data = {}
    await add_litellm_data_to_request(
        data, mock_request, user_api_key_dict, proxy_config
    )

    print("DATA: ", data)

    assert "litellm_metadata" in data
    assert "metadata" not in data


@pytest.mark.parametrize(
    "endpoint", ["/chat/completions", "/v1/completions", "/completions"]
)
@pytest.mark.asyncio
async def test_add_litellm_data_to_request_non_thread_endpoint(endpoint, mock_request):
    mock_request.url.path = endpoint
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key", user_id="test_user_id", org_id="test_org_id"
    )
    proxy_config = Mock()

    data = {}
    await add_litellm_data_to_request(
        data, mock_request, user_api_key_dict, proxy_config
    )

    print("DATA: ", data)

    assert "metadata" in data
    assert "litellm_metadata" not in data


# test adding traceparent


@pytest.mark.parametrize(
    "endpoint", ["/chat/completions", "/v1/completions", "/completions"]
)
@pytest.mark.asyncio
async def test_traceparent_not_added_by_default(endpoint, mock_request):
    """
    This tests that traceparent is not forwarded in the extra_headers

    We had an incident where bedrock calls were failing because traceparent was forwarded
    """
    from litellm.integrations.opentelemetry import OpenTelemetry

    otel_logger = OpenTelemetry()
    setattr(litellm.proxy.proxy_server, "open_telemetry_logger", otel_logger)

    mock_request.url.path = endpoint
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key", user_id="test_user_id", org_id="test_org_id"
    )
    proxy_config = Mock()

    data = {}
    await add_litellm_data_to_request(
        data, mock_request, user_api_key_dict, proxy_config
    )

    print("DATA: ", data)

    _extra_headers = data.get("extra_headers") or {}
    assert "traceparent" not in _extra_headers

    setattr(litellm.proxy.proxy_server, "open_telemetry_logger", None)


@pytest.mark.parametrize(
    "request_tags", [None, ["request_tag1", "request_tag2", "request_tag3"]]
)
@pytest.mark.parametrize(
    "request_sl_metadata", [None, {"request_key": "request_value"}]
)
@pytest.mark.parametrize("key_tags", [None, ["key_tag1", "key_tag2", "key_tag3"]])
@pytest.mark.parametrize("key_sl_metadata", [None, {"key_key": "key_value"}])
@pytest.mark.parametrize("team_tags", [None, ["team_tag1", "team_tag2", "team_tag3"]])
@pytest.mark.parametrize("team_sl_metadata", [None, {"team_key": "team_value"}])
@pytest.mark.asyncio
async def test_add_key_or_team_level_spend_logs_metadata_to_request(
    mock_request,
    request_tags,
    request_sl_metadata,
    team_tags,
    key_sl_metadata,
    team_sl_metadata,
    key_tags,
):
    ## COMPLETE LIST OF TAGS
    all_tags = []
    if request_tags is not None:
        print("Request Tags - {}".format(request_tags))
        all_tags.extend(request_tags)
    if key_tags is not None:
        print("Key Tags - {}".format(key_tags))
        all_tags.extend(key_tags)
    if team_tags is not None:
        print("Team Tags - {}".format(team_tags))
        all_tags.extend(team_tags)

    ## COMPLETE SPEND_LOGS METADATA
    all_sl_metadata = {}
    if request_sl_metadata is not None:
        all_sl_metadata.update(request_sl_metadata)
    if key_sl_metadata is not None:
        all_sl_metadata.update(key_sl_metadata)
    if team_sl_metadata is not None:
        all_sl_metadata.update(team_sl_metadata)

    print(f"team_sl_metadata: {team_sl_metadata}")
    mock_request.url.path = "/chat/completions"
    key_metadata = {
        "tags": key_tags,
        "spend_logs_metadata": key_sl_metadata,
    }
    team_metadata = {
        "tags": team_tags,
        "spend_logs_metadata": team_sl_metadata,
    }
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        user_id="test_user_id",
        org_id="test_org_id",
        metadata=key_metadata,
        team_metadata=team_metadata,
    )
    proxy_config = Mock()

    data = {"metadata": {}}
    if request_tags is not None:
        data["metadata"]["tags"] = request_tags
    if request_sl_metadata is not None:
        data["metadata"]["spend_logs_metadata"] = request_sl_metadata

    print(data)
    new_data = await add_litellm_data_to_request(
        data, mock_request, user_api_key_dict, proxy_config
    )

    print("New Data: {}".format(new_data))
    print("all_tags: {}".format(all_tags))
    assert "metadata" in new_data
    if len(all_tags) == 0:
        assert "tags" not in new_data["metadata"], "Expected=No tags. Got={}".format(
            new_data["metadata"]["tags"]
        )
    else:
        assert new_data["metadata"]["tags"] == all_tags, "Expected={}. Got={}".format(
            all_tags, new_data["metadata"].get("tags", None)
        )

    if len(all_sl_metadata.keys()) == 0:
        assert (
            "spend_logs_metadata" not in new_data["metadata"]
        ), "Expected=No spend logs metadata. Got={}".format(
            new_data["metadata"]["spend_logs_metadata"]
        )
    else:
        assert (
            new_data["metadata"]["spend_logs_metadata"] == all_sl_metadata
        ), "Expected={}. Got={}".format(
            all_sl_metadata, new_data["metadata"]["spend_logs_metadata"]
        )
    # assert (
    #     new_data["metadata"]["spend_logs_metadata"] == metadata["spend_logs_metadata"]
    # )


@pytest.mark.parametrize(
    "callback_vars",
    [
        {
            "langfuse_host": "https://us.cloud.langfuse.com",
            "langfuse_public_key": "pk-lf-9636b7a6-c066",
            "langfuse_secret_key": "sk-lf-7cc8b620",
        },
        {
            "langfuse_host": "os.environ/LANGFUSE_HOST_TEMP",
            "langfuse_public_key": "os.environ/LANGFUSE_PUBLIC_KEY_TEMP",
            "langfuse_secret_key": "os.environ/LANGFUSE_SECRET_KEY_TEMP",
        },
    ],
)
def test_dynamic_logging_metadata_key_and_team_metadata(callback_vars):
    os.environ["LANGFUSE_PUBLIC_KEY_TEMP"] = "pk-lf-9636b7a6-c066"
    os.environ["LANGFUSE_SECRET_KEY_TEMP"] = "sk-lf-7cc8b620"
    os.environ["LANGFUSE_HOST_TEMP"] = "https://us.cloud.langfuse.com"
    from litellm.proxy.proxy_server import ProxyConfig

    proxy_config = ProxyConfig()
    user_api_key_dict = UserAPIKeyAuth(
        token="sk-test-mock-token-789",
        key_name="sk-...63Fg",
        key_alias=None,
        spend=0.000111,
        max_budget=None,
        expires=None,
        models=[],
        aliases={},
        config={},
        user_id=None,
        team_id="ishaan-special-team_e02dd54f-f790-4755-9f93-73734f415898",
        max_parallel_requests=None,
        metadata={
            "logging": [
                {
                    "callback_name": "langfuse",
                    "callback_type": "success",
                    "callback_vars": callback_vars,
                }
            ]
        },
        tpm_limit=None,
        rpm_limit=None,
        budget_duration=None,
        budget_reset_at=None,
        allowed_cache_controls=[],
        permissions={},
        model_spend={},
        model_max_budget={},
        soft_budget_cooldown=False,
        litellm_budget_table=None,
        org_id=None,
        team_spend=0.000132,
        team_alias=None,
        team_tpm_limit=None,
        team_rpm_limit=None,
        team_max_budget=None,
        team_models=[],
        team_blocked=False,
        soft_budget=None,
        team_model_aliases=None,
        team_member_spend=None,
        team_member=None,
        team_metadata={},
        end_user_id=None,
        end_user_tpm_limit=None,
        end_user_rpm_limit=None,
        end_user_max_budget=None,
        last_refreshed_at=1726101560.967527,
        api_key="sk-test-mock-api-key-202",
        user_role=LitellmUserRoles.INTERNAL_USER,
        allowed_model_region=None,
        parent_otel_span=None,
        rpm_limit_per_model=None,
        tpm_limit_per_model=None,
    )
    callbacks = _get_dynamic_logging_metadata(
        user_api_key_dict=user_api_key_dict, proxy_config=proxy_config
    )

    assert callbacks is not None

    for var in callbacks.callback_vars.values():
        assert "os.environ" not in var


@pytest.mark.parametrize(
    "callback_vars",
    [
        {
            "turn_off_message_logging": True,
        },
        {
            "turn_off_message_logging": False,
        },
    ],
)
def test_dynamic_turn_off_message_logging(callback_vars):
    from litellm.proxy.proxy_server import ProxyConfig

    proxy_config = ProxyConfig()
    user_api_key_dict = UserAPIKeyAuth(
        token="sk-test-mock-token-789",
        key_name="sk-...63Fg",
        key_alias=None,
        spend=0.000111,
        max_budget=None,
        expires=None,
        models=[],
        aliases={},
        config={},
        user_id=None,
        team_id="ishaan-special-team_e02dd54f-f790-4755-9f93-73734f415898",
        max_parallel_requests=None,
        metadata={
            "logging": [
                {
                    "callback_name": "datadog",
                    "callback_vars": callback_vars,
                }
            ]
        },
        tpm_limit=None,
        rpm_limit=None,
        budget_duration=None,
        budget_reset_at=None,
        allowed_cache_controls=[],
        permissions={},
        model_spend={},
        model_max_budget={},
        soft_budget_cooldown=False,
        litellm_budget_table=None,
        org_id=None,
        team_spend=0.000132,
        team_alias=None,
        team_tpm_limit=None,
        team_rpm_limit=None,
        team_max_budget=None,
        team_models=[],
        team_blocked=False,
        soft_budget=None,
        team_model_aliases=None,
        team_member_spend=None,
        team_member=None,
        team_metadata={},
        end_user_id=None,
        end_user_tpm_limit=None,
        end_user_rpm_limit=None,
        end_user_max_budget=None,
        last_refreshed_at=1726101560.967527,
        api_key="sk-test-mock-api-key-202",
        user_role=LitellmUserRoles.INTERNAL_USER,
        allowed_model_region=None,
        parent_otel_span=None,
        rpm_limit_per_model=None,
        tpm_limit_per_model=None,
    )
    callbacks = _get_dynamic_logging_metadata(
        user_api_key_dict=user_api_key_dict, proxy_config=proxy_config
    )

    assert callbacks is not None
    assert (
        callbacks.callback_vars["turn_off_message_logging"]
        == callback_vars["turn_off_message_logging"]
    )


@pytest.mark.parametrize(
    "allow_client_side_credentials, expect_error", [(True, False), (False, True)]
)
def test_is_request_body_safe_global_enabled(
    allow_client_side_credentials, expect_error
):
    from litellm import Router

    error_raised = False

    llm_router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            }
        ]
    )
    try:
        is_request_body_safe(
            request_body={"api_base": "hello-world"},
            general_settings={
                "allow_client_side_credentials": allow_client_side_credentials
            },
            llm_router=llm_router,
            model="gpt-3.5-turbo",
        )
    except Exception as e:
        print(e)
        error_raised = True

    assert expect_error == error_raised


@pytest.mark.parametrize(
    "allow_client_side_credentials, expect_error", [(True, False), (False, True)]
)
def test_is_request_body_safe_model_enabled(
    allow_client_side_credentials, expect_error
):
    from litellm import Router

    error_raised = False

    llm_router = Router(
        model_list=[
            {
                "model_name": "fireworks_ai/*",
                "litellm_params": {
                    "model": "fireworks_ai/*",
                    "api_key": os.getenv("FIREWORKS_API_KEY"),
                    "configurable_clientside_auth_params": (
                        ["api_base"] if allow_client_side_credentials else []
                    ),
                },
            }
        ]
    )
    try:
        is_request_body_safe(
            request_body={"api_base": "hello-world"},
            general_settings={},
            llm_router=llm_router,
            model="fireworks_ai/my-new-model",
        )
    except Exception as e:
        print(e)
        error_raised = True

    assert expect_error == error_raised


def test_reading_openai_org_id_from_headers():
    from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup

    headers = {
        "OpenAI-Organization": "test_org_id",
    }
    org_id = LiteLLMProxyRequestSetup.get_openai_org_id_from_headers(headers)
    assert org_id == "test_org_id"


@pytest.mark.parametrize(
    "headers, general_settings, expected_data",
    [
        (
            {"X-OpenWebUI-User-Id": "ishaan3"},
            {"user_header_name": "X-OpenWebUI-User-Id"},
            "ishaan3",
        ),
        (
            {"x-openwebui-user-id": "ishaan3"},
            {"user_header_name": "X-OpenWebUI-User-Id"},
            "ishaan3",
        ),
        ({"X-OpenWebUI-User-Id": "ishaan3"}, {}, None),
        ({}, None, None),
    ],
)
def test_add_litellm_data_for_backend_llm_call(
    headers, general_settings, expected_data
):
    import json

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup

    UserAPIKeyAuth(api_key="test_api_key", user_id="test_user_id", org_id="test_org_id")

    data = LiteLLMProxyRequestSetup.get_user_from_headers(
        headers=headers,
        general_settings=general_settings,
    )

    assert json.dumps(data, sort_keys=True) == json.dumps(expected_data, sort_keys=True)


def test_foward_litellm_user_info_to_backend_llm_call():
    import json

    litellm.add_user_information_to_llm_headers = True

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key", user_id="test_user_id", org_id="test_org_id"
    )

    data = LiteLLMProxyRequestSetup.add_headers_to_llm_call(
        headers={},
        user_api_key_dict=user_api_key_dict,
    )

    expected_data = {
        "x-litellm-user_api_key_user_id": "test_user_id",
        "x-litellm-user_api_key_org_id": "test_org_id",
        "x-litellm-user_api_key_hash": "test_api_key",
        "x-litellm-user_api_key_spend": 0.0,
        "x-litellm-user_api_key_auth_metadata": {},
    }

    assert json.dumps(data, sort_keys=True) == json.dumps(expected_data, sort_keys=True)


def test_update_internal_user_params():
    from litellm.proxy._types import NewUserRequest
    from litellm.proxy.management_endpoints.internal_user_endpoints import (
        _update_internal_new_user_params,
    )

    litellm.default_internal_user_params = {
        "max_budget": 100,
        "budget_duration": "30d",
        "models": ["gpt-3.5-turbo"],
    }

    data = NewUserRequest(user_role="internal_user", user_email="krrish3@berri.ai")
    data_json = data.model_dump()
    updated_data_json = _update_internal_new_user_params(data_json, data)
    assert updated_data_json["models"] == litellm.default_internal_user_params["models"]
    assert (
        updated_data_json["max_budget"]
        == litellm.default_internal_user_params["max_budget"]
    )
    assert (
        updated_data_json["budget_duration"]
        == litellm.default_internal_user_params["budget_duration"]
    )


def test_update_internal_new_user_params_with_no_initial_role_set():
    from litellm.proxy._types import NewUserRequest
    from litellm.proxy.management_endpoints.internal_user_endpoints import (
        _update_internal_new_user_params,
    )

    litellm.default_internal_user_params = {
        "max_budget": 100,
        "budget_duration": "30d",
        "models": ["gpt-3.5-turbo"],
    }

    data = NewUserRequest(user_email="krrish3@berri.ai")
    data_json = data.model_dump()
    updated_data_json = _update_internal_new_user_params(data_json, data)
    assert updated_data_json["models"] == litellm.default_internal_user_params["models"]
    assert (
        updated_data_json["max_budget"]
        == litellm.default_internal_user_params["max_budget"]
    )
    assert (
        updated_data_json["budget_duration"]
        == litellm.default_internal_user_params["budget_duration"]
    )


def test_update_internal_new_user_params_with_user_defined_values():
    from litellm.proxy._types import NewUserRequest
    from litellm.proxy.management_endpoints.internal_user_endpoints import (
        _update_internal_new_user_params,
    )

    litellm.default_internal_user_params = {
        "max_budget": 100,
        "budget_duration": "30d",
        "models": ["gpt-3.5-turbo"],
        "user_role": "proxy_admin",
    }

    data = NewUserRequest(
        user_email="krrish3@berri.ai", max_budget=1000, budget_duration="1mo"
    )
    data_json = data.model_dump()
    updated_data_json = _update_internal_new_user_params(data_json, data)
    assert updated_data_json["user_email"] == "krrish3@berri.ai"
    assert updated_data_json["user_role"] == "proxy_admin"
    assert updated_data_json["max_budget"] == 1000
    assert updated_data_json["budget_duration"] == "1mo"


@pytest.mark.asyncio
async def test_proxy_config_update_from_db():
    from pydantic import BaseModel

    from litellm.proxy.proxy_server import ProxyConfig

    proxy_config = ProxyConfig()

    pc = AsyncMock()

    test_config = {
        "litellm_settings": {
            "callbacks": ["prometheus", "otel"],
        }
    }

    class ReturnValue(BaseModel):
        param_name: str
        param_value: dict

    with patch.object(
        pc,
        "get_generic_data",
        new=AsyncMock(
            return_value=ReturnValue(
                param_name="litellm_settings",
                param_value={
                    "success_callback": "langfuse",
                },
            )
        ),
    ):
        new_config = await proxy_config._update_config_from_db(
            prisma_client=pc,
            config=test_config,
            store_model_in_db=True,
        )

        assert new_config == {
            "litellm_settings": {
                "callbacks": ["prometheus", "otel"],
                "success_callback": "langfuse",
            }
        }


@pytest.mark.asyncio
async def test_prepare_key_update_data():
    from litellm.proxy._types import UpdateKeyRequest
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        prepare_key_update_data,
    )

    existing_key_row = MagicMock()
    data = UpdateKeyRequest(key="test_key", models=["gpt-4"], duration="120s")
    updated_data = await prepare_key_update_data(data, existing_key_row)
    assert "expires" in updated_data

    data = UpdateKeyRequest(key="test_key", metadata={})
    updated_data = await prepare_key_update_data(data, existing_key_row)
    assert updated_data["metadata"] == {}

    data = UpdateKeyRequest(key="test_key", metadata=None)
    updated_data = await prepare_key_update_data(data, existing_key_row)
    assert updated_data["metadata"] is None

    # Test duration "-1" sets expires to None (never expires)
    data = UpdateKeyRequest(key="test_key", duration="-1")
    updated_data = await prepare_key_update_data(data, existing_key_row)
    assert updated_data["expires"] is None


@pytest.mark.parametrize(
    "env_vars, expected_url",
    [
        ({}, "/redoc"),  # default case
        ({"REDOC_URL": "/custom-redoc"}, "/custom-redoc"),  # custom URL
        (
            {"REDOC_URL": "https://example.com/redoc"},
            "https://example.com/redoc",
        ),  # full URL
        ({"NO_REDOC": "True"}, None),  # Redoc disabled
    ],
)
def test_get_redoc_url(env_vars, expected_url):
    # Clear relevant environment variables
    for key in ["REDOC_URL", "NO_REDOC"]:
        os.environ.pop(key, None)

    # Set test environment variables
    for key, value in env_vars.items():
        os.environ[key] = value

    result = _get_redoc_url()
    assert result == expected_url


@pytest.mark.parametrize(
    "env_vars, expected_url",
    [
        ({}, "/"),  # default case
        ({"DOCS_URL": "/custom-docs"}, "/custom-docs"),  # custom URL
        (
            {"DOCS_URL": "https://example.com/docs"},
            "https://example.com/docs",
        ),  # full URL
        ({"NO_DOCS": "True"}, None),  # docs disabled
    ],
)
def test_get_docs_url(env_vars, expected_url):
    # Clear relevant environment variables
    for key in ["DOCS_URL", "NO_DOCS"]:
        os.environ.pop(key, None)

    # Set test environment variables
    for key, value in env_vars.items():
        os.environ[key] = value

    result = _get_docs_url()
    assert result == expected_url


@pytest.mark.parametrize(
    "request_tags, tags_to_add, expected_tags",
    [
        (None, None, []),  # both None
        (["tag1", "tag2"], None, ["tag1", "tag2"]),  # tags_to_add is None
        (None, ["tag3", "tag4"], ["tag3", "tag4"]),  # request_tags is None
        (
            ["tag1", "tag2"],
            ["tag3", "tag4"],
            ["tag1", "tag2", "tag3", "tag4"],
        ),  # both have unique tags
        (
            ["tag1", "tag2"],
            ["tag2", "tag3"],
            ["tag1", "tag2", "tag3"],
        ),  # overlapping tags
        ([], [], []),  # both empty lists
        ("not_a_list", ["tag1"], ["tag1"]),  # request_tags invalid type
        (["tag1"], "not_a_list", ["tag1"]),  # tags_to_add invalid type
        (
            ["tag1"],
            ["tag1", "tag2"],
            ["tag1", "tag2"],
        ),  # duplicate tags in inputs
    ],
)
def test_merge_tags(request_tags, tags_to_add, expected_tags):
    from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup

    result = LiteLLMProxyRequestSetup._merge_tags(
        request_tags=request_tags, tags_to_add=tags_to_add
    )

    assert isinstance(result, list)
    assert sorted(result) == sorted(expected_tags)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key_tags, request_tags, expected_tags",
    [
        # exact duplicates
        (["tag1", "tag2", "tag3"], ["tag1", "tag2", "tag3"], ["tag1", "tag2", "tag3"]),
        # partial duplicates
        (
            ["tag1", "tag2", "tag3"],
            ["tag2", "tag3", "tag4"],
            ["tag1", "tag2", "tag3", "tag4"],
        ),
        # duplicates within key tags
        (["tag1", "tag2"], ["tag3", "tag4"], ["tag1", "tag2", "tag3", "tag4"]),
        # duplicates within request tags
        (["tag1", "tag2"], ["tag2", "tag3", "tag4"], ["tag1", "tag2", "tag3", "tag4"]),
        # case sensitive duplicates
        (["Tag1", "TAG2"], ["tag1", "tag2"], ["Tag1", "TAG2", "tag1", "tag2"]),
    ],
)
async def test_add_litellm_data_to_request_duplicate_tags(
    key_tags, request_tags, expected_tags
):
    """
    Test to verify duplicate tags between request and key metadata are handled correctly


    Aggregation logic when checking spend can be impacted if duplicate tags are not handled correctly.

    User feedback:
    "If I register my key with tag1 and
    also pass the same tag1 when using the key
    then I see tag1 twice in the
    LiteLLM_SpendLogs table request_tags column. This can mess up aggregation logic"
    """
    mock_request = Mock(spec=Request)
    mock_request.url.path = "/chat/completions"
    mock_request.query_params = {}
    mock_request.headers = {}

    # Setup key with tags in metadata
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        user_id="test_user_id",
        org_id="test_org_id",
        metadata={"tags": key_tags},
    )

    # Setup request data with tags
    data = {"metadata": {"tags": request_tags}}

    # Process request
    proxy_config = Mock()
    result = await add_litellm_data_to_request(
        data=data,
        request=mock_request,
        user_api_key_dict=user_api_key_dict,
        proxy_config=proxy_config,
    )

    # Verify results
    assert "metadata" in result
    assert "tags" in result["metadata"]
    assert sorted(result["metadata"]["tags"]) == sorted(
        expected_tags
    ), f"Expected {expected_tags}, got {result['metadata']['tags']}"


@pytest.mark.parametrize(
    "general_settings, user_api_key_dict, request_body, expected_error",
    [
        (
            {"enforced_params": ["param1", "param2"]},
            UserAPIKeyAuth(
                api_key="test_api_key", user_id="test_user_id", org_id="test_org_id"
            ),
            {},
            True,
        ),
        (
            {"service_account_settings": {"enforced_params": ["user"]}},
            UserAPIKeyAuth(
                api_key="test_api_key", user_id="test_user_id", org_id="test_org_id"
            ),
            {},
            False,
        ),
        (
            {"service_account_settings": {"enforced_params": ["user"]}},
            UserAPIKeyAuth(
                api_key="test_api_key",
                user_id="test_user_id",
                org_id="test_org_id",
                metadata={"service_account_id": "test_service_account_id"},
            ),
            {},
            True,
        ),
        (
            {},
            UserAPIKeyAuth(
                api_key="test_api_key",
                metadata={"enforced_params": ["user"]},
            ),
            {},
            True,
        ),
        (
            {},
            UserAPIKeyAuth(
                api_key="test_api_key",
                metadata={"enforced_params": ["user"]},
            ),
            {"user": "test_user"},
            False,
        ),
        (
            {"enforced_params": ["user"]},
            UserAPIKeyAuth(
                api_key="test_api_key",
            ),
            {"user": "test_user"},
            False,
        ),
        (
            {"service_account_settings": {"enforced_params": ["user"]}},
            UserAPIKeyAuth(
                api_key="test_api_key",
                metadata={"service_account_id": "test_service_account_id"},
            ),
            {"user": "test_user"},
            False,
        ),
        (
            {"enforced_params": ["metadata.generation_name"]},
            UserAPIKeyAuth(
                api_key="test_api_key",
            ),
            {"metadata": {}},
            True,
        ),
        (
            {"enforced_params": ["metadata.generation_name"]},
            UserAPIKeyAuth(
                api_key="test_api_key",
            ),
            {"metadata": {"generation_name": "test_generation_name"}},
            False,
        ),
    ],
)
def test_enforced_params_check(
    general_settings, user_api_key_dict, request_body, expected_error
):
    from litellm.proxy.litellm_pre_call_utils import _enforced_params_check

    if expected_error:
        with pytest.raises(ValueError):
            _enforced_params_check(
                request_body=request_body,
                general_settings=general_settings,
                user_api_key_dict=user_api_key_dict,
                premium_user=True,
            )
    else:
        _enforced_params_check(
            request_body=request_body,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            premium_user=True,
        )


def test_get_key_models():
    from collections import defaultdict

    from litellm.proxy.auth.model_checks import get_key_models

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        user_id="test_user_id",
        org_id="test_org_id",
        models=["default"],
    )
    proxy_model_list = ["gpt-4o", "gpt-3.5-turbo"]
    model_access_groups = defaultdict(list)
    model_access_groups["default"].extend(["gpt-4o", "gpt-3.5-turbo"])
    model_access_groups["default"].extend(["gpt-4o-mini"])
    model_access_groups["team2"].extend(["gpt-3.5-turbo"])

    result = get_key_models(
        user_api_key_dict=user_api_key_dict,
        proxy_model_list=proxy_model_list,
        model_access_groups=model_access_groups,
    )
    assert result == ["gpt-4o", "gpt-3.5-turbo", "gpt-4o-mini"]


def test_get_team_models():
    from collections import defaultdict

    from litellm.proxy.auth.model_checks import get_team_models

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        user_id="test_user_id",
        org_id="test_org_id",
        models=[],
        team_models=["default"],
    )
    proxy_model_list = ["gpt-4o", "gpt-3.5-turbo"]
    model_access_groups = defaultdict(list)
    model_access_groups["default"].extend(["gpt-4o", "gpt-3.5-turbo"])
    model_access_groups["default"].extend(["gpt-4o-mini"])
    model_access_groups["team2"].extend(["gpt-3.5-turbo"])

    team_models = user_api_key_dict.team_models
    result = get_team_models(
        team_models=team_models,
        proxy_model_list=proxy_model_list,
        model_access_groups=model_access_groups,
    )
    assert result == ["gpt-4o", "gpt-3.5-turbo", "gpt-4o-mini"]


def test_update_config_fields():
    from litellm.proxy.proxy_server import ProxyConfig

    proxy_config = ProxyConfig()

    args = {
        "current_config": {
            "litellm_settings": {
                "default_team_settings": [
                    {
                        "team_id": "c91e32bb-0f2a-4aa1-86c4-307ca2e03ea3",
                        "success_callback": ["langfuse"],
                        "failure_callback": ["langfuse"],
                        "langfuse_public_key": "my-fake-key",
                        "langfuse_secret": "my-fake-secret",
                    }
                ]
            },
        },
        "param_name": "litellm_settings",
        "db_param_value": {
            "telemetry": False,
            "drop_params": True,
            "num_retries": 5,
            "request_timeout": 600,
            "success_callback": ["langfuse"],
            "default_team_settings": None,
            "context_window_fallbacks": [{"gpt-3.5-turbo": ["gpt-3.5-turbo-large"]}],
        },
    }
    updated_config = proxy_config._update_config_fields(**args)

    print("updated_config", updated_config)
    all_team_config = updated_config["litellm_settings"]["default_team_settings"]

    # check if team id config returned
    print("all_team_config", all_team_config)
    team_config = proxy_config._get_team_config(
        team_id="c91e32bb-0f2a-4aa1-86c4-307ca2e03ea3", all_teams_config=all_team_config
    )
    print("team_config", team_config)
    assert team_config["langfuse_public_key"] == "my-fake-key"
    assert team_config["langfuse_secret"] == "my-fake-secret"


def test_update_config_fields_default_internal_user_params(monkeypatch):
    from litellm.proxy.proxy_server import ProxyConfig

    proxy_config = ProxyConfig()

    monkeypatch.setattr(litellm, "default_internal_user_params", None)

    args = {
        "current_config": {},
        "param_name": "litellm_settings",
        "db_param_value": {
            "default_internal_user_params": {
                "user_role": "proxy_admin",
                "max_budget": 1000,
                "budget_duration": "1mo",
            },
        },
    }
    proxy_config._update_config_fields(**args)

    assert litellm.default_internal_user_params == {
        "user_role": "proxy_admin",
        "max_budget": 1000,
        "budget_duration": "1mo",
    }

    monkeypatch.setattr(
        litellm, "default_internal_user_params", None
    )  # reset to default


@pytest.mark.parametrize(
    "proxy_model_list,model_list,provider",
    [
        (
            ["openai/*"],
            [{"model_name": "openai/*", "litellm_params": {"model": "openai/*"}}],
            "openai",
        ),
        (
            ["bedrock/*"],
            [{"model_name": "bedrock/*", "litellm_params": {"model": "bedrock/*"}}],
            "bedrock",
        ),
        (
            ["anthropic/*"],
            [{"model_name": "anthropic/*", "litellm_params": {"model": "anthropic/*"}}],
            "anthropic",
        ),
        (
            ["cohere/*"],
            [{"model_name": "cohere/*", "litellm_params": {"model": "cohere/*"}}],
            "cohere",
        ),
    ],
)
def test_get_complete_model_list(proxy_model_list, model_list, provider):
    """
    Test that get_complete_model_list correctly expands model groups like 'openai/*' into individual models with provider prefixes
    """
    from litellm import Router
    from litellm.proxy.auth.model_checks import get_complete_model_list

    llm_router = Router(model_list=model_list)

    complete_list = get_complete_model_list(
        proxy_model_list=proxy_model_list,
        key_models=[],
        team_models=[],
        user_model=None,
        infer_model_from_keys=False,
        llm_router=llm_router,
    )

    # Check that we got a non-empty list back
    assert len(complete_list) > 0

    print("complete_list", json.dumps(complete_list, indent=4))

    for _model in complete_list:
        assert provider in _model


def test_team_callback_metadata_all_none_values():
    from litellm.proxy._types import TeamCallbackMetadata

    resp = TeamCallbackMetadata(
        success_callback=None,
        failure_callback=None,
        callback_vars=None,
    )

    assert resp.success_callback == []
    assert resp.failure_callback == []
    assert resp.callback_vars == {}


@pytest.mark.parametrize(
    "none_key",
    [
        "success_callback",
        "failure_callback",
        "callback_vars",
    ],
)
def test_team_callback_metadata_none_values(none_key):
    from litellm.proxy._types import TeamCallbackMetadata

    if none_key == "success_callback":
        args = {
            "success_callback": None,
            "failure_callback": ["test"],
            "callback_vars": None,
        }
    elif none_key == "failure_callback":
        args = {
            "success_callback": ["test"],
            "failure_callback": None,
            "callback_vars": None,
        }
    elif none_key == "callback_vars":
        args = {
            "success_callback": ["test"],
            "failure_callback": ["test"],
            "callback_vars": None,
        }

    resp = TeamCallbackMetadata(**args)

    assert none_key not in resp


def test_proxy_config_state_post_init_callback_call(monkeypatch):
    """
    Ensures team_id is still in config, after callback is called

    Addresses issue: https://github.com/BerriAI/litellm/issues/6787

    Where team_id was being popped from config, after callback was called

    Note: Environment variables are mocked to avoid validation errors
    in parallel execution where env vars may not be set.
    """
    from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
    from litellm.proxy.proxy_server import ProxyConfig

    # Mock environment variables to avoid Pydantic validation errors
    # when env vars are resolved to None in parallel execution
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test_public_key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test_secret_key")

    pc = ProxyConfig()

    pc.update_config_state(
        config={
            "litellm_settings": {
                "default_team_settings": [
                    {
                        "team_id": "test",
                        "success_callback": ["langfuse"],
                        "langfuse_public_key": "os.environ/LANGFUSE_PUBLIC_KEY",
                        "langfuse_secret": "os.environ/LANGFUSE_SECRET_KEY",
                    }
                ]
            }
        }
    )

    LiteLLMProxyRequestSetup.add_team_based_callbacks_from_config(
        team_id="test",
        proxy_config=pc,
    )

    config = pc.get_config_state()
    assert config["litellm_settings"]["default_team_settings"][0]["team_id"] == "test"


def test_proxy_config_state_get_config_state_error():
    """
    Ensures that get_config_state does not raise an error when the config is not a valid dictionary
    """
    import threading

    from litellm.proxy.proxy_server import ProxyConfig

    test_config = {
        "callback_list": [
            {
                "lock": threading.RLock(),  # This will cause the deep copy to fail
                "name": "test_callback",
            }
        ],
        "model_list": ["gpt-4", "claude-3"],
    }

    pc = ProxyConfig()
    pc.config = test_config
    config = pc.get_config_state()
    assert config == {}


@pytest.mark.parametrize(
    "associated_budget_table, expected_user_api_key_auth_key, expected_user_api_key_auth_value",
    [
        (
            {
                "litellm_budget_table_max_budget": None,
                "litellm_budget_table_tpm_limit": None,
                "litellm_budget_table_rpm_limit": 1,
                "litellm_budget_table_model_max_budget": None,
            },
            "rpm_limit",
            1,
        ),
        (
            {},
            None,
            None,
        ),
        (
            {
                "litellm_budget_table_max_budget": None,
                "litellm_budget_table_tpm_limit": None,
                "litellm_budget_table_rpm_limit": None,
                "litellm_budget_table_model_max_budget": {"gpt-4o": 100},
            },
            "model_max_budget",
            {"gpt-4o": 100},
        ),
    ],
)
def test_litellm_verification_token_view_response_with_budget_table(
    associated_budget_table,
    expected_user_api_key_auth_key,
    expected_user_api_key_auth_value,
):
    from litellm.proxy._types import LiteLLM_VerificationTokenView

    args: Dict[str, Any] = {
        "token": "sk-test-mock-token-303",
        "key_name": "sk-...if_g",
        "key_alias": None,
        "soft_budget_cooldown": False,
        "spend": 0.011441999999999997,
        "expires": None,
        "models": [],
        "aliases": {},
        "config": {},
        "user_id": None,
        "team_id": "test",
        "permissions": {},
        "max_parallel_requests": None,
        "metadata": {},
        "blocked": None,
        "tpm_limit": None,
        "rpm_limit": None,
        "max_budget": None,
        "budget_duration": None,
        "budget_reset_at": None,
        "allowed_cache_controls": [],
        "model_spend": {},
        "model_max_budget": {},
        "budget_id": "my-test-tier",
        "created_at": "2024-12-26T02:28:52.615+00:00",
        "updated_at": "2024-12-26T03:01:51.159+00:00",
        "team_spend": 0.012134999999999998,
        "team_max_budget": None,
        "team_tpm_limit": None,
        "team_rpm_limit": None,
        "team_models": [],
        "team_metadata": {},
        "team_blocked": False,
        "team_alias": None,
        "team_members_with_roles": [{"role": "admin", "user_id": "default_user_id"}],
        "team_member_spend": None,
        "team_model_aliases": None,
        "team_member": None,
        **associated_budget_table,
    }
    resp = LiteLLM_VerificationTokenView(**args)
    if expected_user_api_key_auth_key is not None:
        assert (
            getattr(resp, expected_user_api_key_auth_key)
            == expected_user_api_key_auth_value
        )


def test_litellm_verification_token_view_budget_does_not_override_key_model_max_budget():
    """
    When key has non-empty model_max_budget, budget's model_max_budget is NOT applied.
    Regression test for per-model budget: only apply budget's model_max_budget when key's is empty.
    """
    from litellm.proxy._types import LiteLLM_VerificationTokenView

    key_model_max_budget = {"gpt-4": {"max_budget": 50.0, "budget_duration": "1d"}}
    args = {
        "token": "sk-test-mock-token-303",
        "key_name": "sk-...if_g",
        "key_alias": None,
        "soft_budget_cooldown": False,
        "spend": 0.0,
        "expires": None,
        "models": [],
        "aliases": {},
        "config": {},
        "user_id": None,
        "team_id": "test",
        "permissions": {},
        "max_parallel_requests": None,
        "metadata": {},
        "blocked": None,
        "tpm_limit": None,
        "rpm_limit": None,
        "max_budget": None,
        "budget_duration": None,
        "budget_reset_at": None,
        "allowed_cache_controls": [],
        "model_spend": {},
        "model_max_budget": key_model_max_budget,
        "budget_id": "my-test-tier",
        "created_at": "2024-12-26T02:28:52.615+00:00",
        "updated_at": "2024-12-26T03:01:51.159+00:00",
        "team_spend": None,
        "team_max_budget": None,
        "team_tpm_limit": None,
        "team_rpm_limit": None,
        "team_models": [],
        "team_metadata": {},
        "team_blocked": False,
        "team_alias": None,
        "team_members_with_roles": [],
        "team_member_spend": None,
        "team_model_aliases": None,
        "team_member": None,
        "litellm_budget_table_model_max_budget": {
            "gpt-4o": {"max_budget": 100.0, "budget_duration": "1d"}
        },
    }
    resp = LiteLLM_VerificationTokenView(**args)
    assert resp.model_max_budget == key_model_max_budget


def test_is_allowed_to_make_key_request():
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _is_allowed_to_make_key_request,
    )

    assert (
        _is_allowed_to_make_key_request(
            user_api_key_dict=UserAPIKeyAuth(
                user_id="test_user_id", user_role=LitellmUserRoles.PROXY_ADMIN
            ),
            user_id="test_user_id",
            team_id="test_team_id",
        )
        is True
    )

    assert (
        _is_allowed_to_make_key_request(
            user_api_key_dict=UserAPIKeyAuth(
                user_id="test_user_id",
                user_role=LitellmUserRoles.INTERNAL_USER,
                team_id="litellm-dashboard",
            ),
            user_id="test_user_id",
            team_id="test_team_id",
        )
        is True
    )


def test_get_model_group_info():
    from litellm import Router
    from litellm.proxy.proxy_server import _get_model_group_info

    router = Router(
        model_list=[
            {
                "model_name": "openai/tts-1",
                "litellm_params": {
                    "model": "openai/tts-1",
                    "api_key": "sk-1234",
                },
            },
            {
                "model_name": "openai/gpt-3.5-turbo",
                "litellm_params": {
                    "model": "openai/gpt-3.5-turbo",
                    "api_key": "sk-1234",
                },
            },
        ]
    )
    model_list = _get_model_group_info(
        llm_router=router,
        all_models_str=["openai/tts-1", "openai/gpt-3.5-turbo"],
        model_group="openai/tts-1",
    )
    assert len(model_list) == 1


@pytest.fixture
def mock_team_data():
    return [
        {"team_id": "team1", "team_name": "Test Team 1"},
        {"team_id": "team2", "team_name": "Test Team 2"},
    ]


@pytest.fixture
def mock_key_data():
    return [
        {"token": "test_token_1", "key_name": "key1", "team_id": None, "spend": 0},
        {"token": "test_token_2", "key_name": "key2", "team_id": "team1", "spend": 100},
        {
            "token": "test_token_3",
            "key_name": "key3",
            "team_id": "litellm-dashboard",
            "spend": 50,
        },
    ]


class MockDb:
    def __init__(self, mock_team_data, mock_key_data):
        self.mock_team_data = mock_team_data
        self.mock_key_data = mock_key_data

    async def query_raw(self, query: str, *args):
        # Simulate the SQL query response
        filtered_keys = [
            k
            for k in self.mock_key_data
            if k["team_id"] != "litellm-dashboard" or k["team_id"] is None
        ]

        return [{"teams": self.mock_team_data, "keys": filtered_keys}]


class MockPrismaClientDB:
    def __init__(
        self,
        mock_team_data,
        mock_key_data,
    ):
        self.db = MockDb(mock_team_data, mock_key_data)


@pytest.mark.asyncio
async def test_get_user_info_for_proxy_admin(mock_team_data, mock_key_data):
    # Patch the prisma_client import
    from litellm.proxy._types import UserInfoResponse

    with patch(
        "litellm.proxy.proxy_server.prisma_client",
        MockPrismaClientDB(mock_team_data, mock_key_data),
    ):
        from litellm.proxy.management_endpoints.internal_user_endpoints import (
            _get_user_info_for_proxy_admin,
        )

        # Execute the function
        result = await _get_user_info_for_proxy_admin()

        # Verify the result structure
        assert isinstance(result, UserInfoResponse)
        assert len(result.keys) == 2


def test_custom_openid_response():
    from litellm.caching import DualCache
    from litellm.proxy._types import LiteLLM_JWTAuth
    from litellm.proxy.management_endpoints.ui_sso import (
        JWTHandler,
        generic_response_convertor,
    )

    jwt_handler = JWTHandler()
    jwt_handler.update_environment(
        prisma_client={},
        user_api_key_cache=DualCache(),
        litellm_jwtauth=LiteLLM_JWTAuth(
            team_ids_jwt_field="department",
        ),
    )
    response = {
        "sub": "3f196e06-7484-451e-be5a-ea6c6bb86c5b",
        "email_verified": True,
        "name": "Krish Dholakia",
        "preferred_username": "krrishd",
        "given_name": "Krish",
        "department": ["/test-group"],
        "family_name": "Dholakia",
        "email": "krrishdholakia@gmail.com",
    }

    resp = generic_response_convertor(
        response=response,
        jwt_handler=jwt_handler,
    )
    assert resp.team_ids == ["/test-group"]


def test_update_key_request_validation():
    """
    Ensures that the UpdateKeyRequest model validates the temp_budget_increase and temp_budget_expiry fields together
    """
    from litellm.proxy._types import UpdateKeyRequest

    with pytest.raises(Exception):
        UpdateKeyRequest(
            key="test_key",
            temp_budget_increase=100,
        )

    with pytest.raises(Exception):
        UpdateKeyRequest(
            key="test_key",
            temp_budget_expiry="2024-01-20T00:00:00Z",
        )

    UpdateKeyRequest(
        key="test_key",
        temp_budget_increase=100,
        temp_budget_expiry="2024-01-20T00:00:00Z",
    )


def test_get_temp_budget_increase():
    from datetime import datetime, timedelta

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import _get_temp_budget_increase

    expiry = datetime.now() + timedelta(days=1)
    expiry_in_isoformat = expiry.isoformat()

    valid_token = UserAPIKeyAuth(
        max_budget=100,
        spend=0,
        metadata={
            "temp_budget_increase": 100,
            "temp_budget_expiry": expiry_in_isoformat,
        },
    )
    assert _get_temp_budget_increase(valid_token) == 100


def test_update_key_budget_with_temp_budget_increase():
    from datetime import datetime, timedelta

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import (
        _update_key_budget_with_temp_budget_increase,
    )

    expiry = datetime.now() + timedelta(days=1)
    expiry_in_isoformat = expiry.isoformat()

    valid_token = UserAPIKeyAuth(
        max_budget=100,
        spend=0,
        metadata={
            "temp_budget_increase": 100,
            "temp_budget_expiry": expiry_in_isoformat,
        },
    )
    assert _update_key_budget_with_temp_budget_increase(valid_token).max_budget == 200


@pytest.mark.asyncio
async def test_health_check_not_called_when_disabled(monkeypatch):
    from litellm.proxy.proxy_server import ProxyStartupEvent

    # Mock environment variable
    monkeypatch.setenv("DISABLE_PRISMA_HEALTH_CHECK_ON_STARTUP", "true")

    # Create mock prisma client
    mock_prisma = MagicMock()
    mock_prisma.connect = AsyncMock()
    mock_prisma.health_check = AsyncMock()
    mock_prisma.check_view_exists = AsyncMock()
    mock_prisma._set_spend_logs_row_count_in_proxy_state = AsyncMock()
    # Mock the db attribute with start_token_refresh_task for RDS IAM token refresh
    mock_db = MagicMock()
    mock_db.start_token_refresh_task = AsyncMock()
    mock_prisma.db = mock_db
    # Mock PrismaClient constructor
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.PrismaClient", lambda **kwargs: mock_prisma
    )

    # Call the setup function
    await ProxyStartupEvent._setup_prisma_client(
        database_url="mock_url",
        proxy_logging_obj=MagicMock(),
        user_api_key_cache=MagicMock(),
    )

    # Verify health check wasn't called
    mock_prisma.health_check.assert_not_called()


@patch(
    "litellm.proxy.proxy_server.get_openapi_schema",
    return_value={
        "paths": {
            "/new/route": {"get": {"summary": "New"}},
        }
    },
)
def test_custom_openapi(mock_get_openapi_schema):
    from litellm.proxy.proxy_server import custom_openapi

    openapi_schema = custom_openapi()
    assert openapi_schema is not None


from litellm.proxy.utils import ProxyUpdateSpend


@pytest.mark.asyncio
async def test_end_user_transactions_reset():
    # Setup
    mock_client = MagicMock()
    end_user_list_transactions = {"1": 10.0}  # Bad log
    mock_client.db.tx = AsyncMock(side_effect=Exception("DB Error"))

    # Call function - should raise error
    with pytest.raises(Exception):
        await ProxyUpdateSpend.update_end_user_spend(
            n_retry_times=0,
            prisma_client=mock_client,
            proxy_logging_obj=MagicMock(),
            end_user_list_transactions=end_user_list_transactions,
        )


@pytest.mark.asyncio
async def test_spend_logs_cleanup_after_error():
    # Setup test data
    import asyncio

    mock_client = MagicMock()
    mock_client.spend_log_transactions = [
        {"id": 1, "amount": 10.0},
        {"id": 2, "amount": 20.0},
        {"id": 3, "amount": 30.0},
    ]
    # Add lock for spend_log_transactions (matches real PrismaClient)
    mock_client._spend_log_transactions_lock = asyncio.Lock()
    # Make the DB operation fail
    mock_client.db.litellm_spendlogs.create_many = AsyncMock(
        side_effect=Exception("DB Error")
    )

    original_logs = mock_client.spend_log_transactions.copy()

    # Call function - should raise error
    with pytest.raises(Exception):
        await ProxyUpdateSpend.update_spend_logs(
            n_retry_times=0,
            prisma_client=mock_client,
            db_writer_client=None,  # Test DB write path
            proxy_logging_obj=MagicMock(),
        )

    # Verify the first batch was removed from spend_log_transactions
    assert (
        mock_client.spend_log_transactions == original_logs[100:]
    ), "Should remove processed logs even after error"


def test_provider_specific_header():
    """Test that provider_specific_header is set correctly for Anthropic headers."""
    from litellm.proxy.litellm_pre_call_utils import (
        add_provider_specific_headers_to_request,
    )

    data = {
        "model": "gemini-1.5-flash",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Tell me a joke"}],
            }
        ],
        "stream": True,
        "proxy_server_request": {
            "url": "http://0.0.0.0:4000/v1/chat/completions",
            "method": "POST",
            "headers": {
                "content-type": "application/json",
                "anthropic-beta": "prompt-caching-2024-07-31",
                "user-agent": "PostmanRuntime/7.32.3",
                "accept": "*/*",
                "postman-token": "81cccd87-c91d-4b2f-b252-c0fe0ca82529",
                "host": "0.0.0.0:4000",
                "accept-encoding": "gzip, deflate, br",
                "connection": "keep-alive",
                "content-length": "240",
            },
            "body": {
                "model": "gemini-1.5-flash",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Tell me a joke"}],
                    }
                ],
                "stream": True,
            },
        },
    }

    headers = {
        "content-type": "application/json",
        "anthropic-beta": "prompt-caching-2024-07-31",
        "user-agent": "PostmanRuntime/7.32.3",
        "accept": "*/*",
        "postman-token": "81cccd87-c91d-4b2f-b252-c0fe0ca82529",
        "host": "0.0.0.0:4000",
        "accept-encoding": "gzip, deflate, br",
        "connection": "keep-alive",
        "content-length": "240",
    }

    add_provider_specific_headers_to_request(
        data=data,
        headers=headers,
    )
    # Verify multi-provider support: anthropic headers work across multiple providers
    assert data["provider_specific_header"] == {
        "custom_llm_provider": "anthropic,bedrock,vertex_ai",
        "extra_headers": {
            "anthropic-beta": "prompt-caching-2024-07-31",
        },
    }


def test_provider_specific_header_multi_provider():
    """Test that provider_specific_header supports multiple providers for Anthropic headers."""
    from litellm.proxy.litellm_pre_call_utils import (
        add_provider_specific_headers_to_request,
    )

    data = {
        "model": "gemini-1.5-flash",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Tell me a joke"}],
            }
        ],
        "stream": True,
        "proxy_server_request": {
            "url": "http://0.0.0.0:4000/v1/chat/completions",
            "method": "POST",
            "headers": {
                "content-type": "application/json",
                "anthropic-beta": "context-1m-2025-08-07",
                "anthropic-version": "2023-06-01",
                "user-agent": "PostmanRuntime/7.32.3",
                "accept": "*/*",
                "postman-token": "81cccd87-c91d-4b2f-b252-c0fe0ca82529",
                "host": "0.0.0.0:4000",
                "accept-encoding": "gzip, deflate, br",
                "connection": "keep-alive",
                "content-length": "240",
            },
            "body": {
                "model": "gemini-1.5-flash",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Tell me a joke"}],
                    }
                ],
                "stream": True,
            },
        },
    }

    headers = {
        "content-type": "application/json",
        "anthropic-beta": "context-1m-2025-08-07",
        "anthropic-version": "2023-06-01",
        "user-agent": "PostmanRuntime/7.32.3",
        "accept": "*/*",
        "postman-token": "81cccd87-c91d-4b2f-b252-c0fe0ca82529",
        "host": "0.0.0.0:4000",
        "accept-encoding": "gzip, deflate, br",
        "connection": "keep-alive",
        "content-length": "240",
    }

    add_provider_specific_headers_to_request(
        data=data,
        headers=headers,
    )

    # Verify that provider_specific_header contains comma-separated providers
    assert "provider_specific_header" in data
    assert (
        data["provider_specific_header"]["custom_llm_provider"]
        == "anthropic,bedrock,vertex_ai"
    )
    assert data["provider_specific_header"]["extra_headers"] == {
        "anthropic-beta": "context-1m-2025-08-07",
        "anthropic-version": "2023-06-01",
    }


# @pytest.mark.parametrize(
#     "custom_llm_provider, expected_result",
#     [
#         ("anthropic", {"anthropic-beta": "test"}),
#         ("bedrock", {"anthropic-beta": "test"}),
#         ("vertex_ai", {"anthropic-beta": "test"}),
#     ],
# )
# def test_provider_specific_header_in_request(custom_llm_provider, expected_result):
#     from litellm.types.utils import ProviderSpecificHeader
#     from litellm.llms.custom_httpx.http_handler import HTTPHandler
#     from unittest.mock import patch

#     litellm.set_verbose = True
#     client = HTTPHandler()
#     with patch.object(client, "post", return_value=MagicMock()) as mock_post:
#         try:
#             litellm.completion(
#                 model="anthropic/claude-3-5-sonnet-v2@20241022",
#                 messages=[{"role": "user", "content": "Hello world"}],
#                 provider_specific_header=ProviderSpecificHeader(
#                     custom_llm_provider="anthropic",
#                     extra_headers={"anthropic-beta": "test"},
#                 ),
#                 client=client,
#             )
#         except Exception as e:
#             print(f"Error: {e}")

#         mock_post.assert_called_once()
#         print(mock_post.call_args.kwargs["headers"])
#         assert "anthropic-beta" in mock_post.call_args.kwargs["headers"]


from litellm.proxy._types import LiteLLM_UserTable


@pytest.mark.parametrize(
    "wildcard_model, litellm_params, expected_models",
    [
        (
            "anthropic/*",
            {"model": "anthropic/*"},
            ["anthropic/claude-3-5-haiku-20241022", "anthropic/claude-3-opus-20240229"],
        ),
        (
            "vertex_ai/gemini-*",
            {"model": "vertex_ai/gemini-*"},
            ["vertex_ai/gemini-1.5-flash", "vertex_ai/gemini-1.5-pro"],
        ),
        (
            "foo/*",
            {"model": "openai/*"},
            ["foo/gpt-4o", "foo/gpt-4o-mini"],
        ),
    ],
)
def test_get_known_models_from_wildcard(
    wildcard_model, litellm_params, expected_models
):
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard
    from litellm.types.router import LiteLLM_Params

    wildcard_models = get_known_models_from_wildcard(
        wildcard_model=wildcard_model, litellm_params=LiteLLM_Params(**litellm_params)
    )
    # Check if all expected models are in the returned list
    print(f"wildcard_models: {wildcard_models}\n")
    for model in expected_models:
        if model not in wildcard_models:
            print(f"Missing expected model: {model}")

    assert all(model in wildcard_models for model in expected_models)


@pytest.mark.parametrize(
    "data, user_api_key_dict, expected_model",
    [
        # Test case 1: Model exists in team aliases
        (
            {"model": "gpt-4o"},
            UserAPIKeyAuth(
                api_key="test_key", team_model_aliases={"gpt-4o": "gpt-4o-team-1"}
            ),
            "gpt-4o-team-1",
        ),
        # Test case 2: Model doesn't exist in team aliases
        (
            {"model": "gpt-4o"},
            UserAPIKeyAuth(
                api_key="test_key", team_model_aliases={"claude-3": "claude-3-team-1"}
            ),
            "gpt-4o",
        ),
        # Test case 3: No team aliases defined
        (
            {"model": "gpt-4o"},
            UserAPIKeyAuth(api_key="test_key", team_model_aliases=None),
            "gpt-4o",
        ),
        # Test case 4: No model in request data
        (
            {"messages": []},
            UserAPIKeyAuth(
                api_key="test_key", team_model_aliases={"gpt-4o": "gpt-4o-team-1"}
            ),
            None,
        ),
    ],
)
def test_update_model_if_team_alias_exists(data, user_api_key_dict, expected_model):
    from litellm.proxy.litellm_pre_call_utils import _update_model_if_team_alias_exists

    # Make a copy of the input data to avoid modifying the test parameters
    test_data = data.copy()

    # Call the function
    _update_model_if_team_alias_exists(
        data=test_data, user_api_key_dict=user_api_key_dict
    )

    # Check if model was updated correctly
    assert test_data.get("model") == expected_model


@pytest.fixture
def mock_prisma_client():
    client = MagicMock()
    client.db = MagicMock()
    client.db.litellm_teamtable = AsyncMock()
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_id, user_info, user_role, mock_teams, expected_teams, should_query_db",
    [
        ("no_user_info", None, "proxy_admin", None, [], False),
        (
            "no_teams_found",
            LiteLLM_UserTable(
                teams=["team1", "team2"],
                user_id="user1",
                max_budget=100,
                spend=0,
                user_email="user1@example.com",
                user_role="proxy_admin",
            ),
            "proxy_admin",
            None,
            [],
            True,
        ),
        (
            "admin_user_with_teams",
            LiteLLM_UserTable(
                teams=["team1", "team2"],
                user_id="user1",
                max_budget=100,
                spend=0,
                user_email="user1@example.com",
                user_role="proxy_admin",
            ),
            "proxy_admin",
            [
                MagicMock(
                    model_dump=lambda: {
                        "team_id": "team1",
                        "members_with_roles": [{"role": "admin", "user_id": "user1"}],
                    }
                ),
                MagicMock(
                    model_dump=lambda: {
                        "team_id": "team2",
                        "members_with_roles": [
                            {"role": "admin", "user_id": "user1"},
                            {"role": "user", "user_id": "user2"},
                        ],
                    }
                ),
            ],
            ["team1", "team2"],
            True,
        ),
        (
            "non_admin_user",
            LiteLLM_UserTable(
                teams=["team1", "team2"],
                user_id="user1",
                max_budget=100,
                spend=0,
                user_email="user1@example.com",
                user_role="internal_user",
            ),
            "internal_user",
            [
                MagicMock(
                    model_dump=lambda: {"team_id": "team1", "members": ["user1"]}
                ),
                MagicMock(
                    model_dump=lambda: {
                        "team_id": "team2",
                        "members": ["user1", "user2"],
                    }
                ),
            ],
            [],
            True,
        ),
    ],
)
async def test_get_admin_team_ids(
    test_id: str,
    user_info: Optional[LiteLLM_UserTable],
    user_role: str,
    mock_teams: Optional[List[MagicMock]],
    expected_teams: List[str],
    should_query_db: bool,
    mock_prisma_client,
):
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        get_admin_team_ids,
    )

    # Setup
    mock_prisma_client.db.litellm_teamtable.find_many.return_value = mock_teams
    user_api_key_dict = UserAPIKeyAuth(
        user_role=user_role, user_id=user_info.user_id if user_info else None
    )

    # Execute
    result = await get_admin_team_ids(
        complete_user_info=user_info,
        user_api_key_dict=user_api_key_dict,
        prisma_client=mock_prisma_client,
    )

    # Assert
    assert result == expected_teams, f"Expected {expected_teams}, but got {result}"

    if should_query_db:
        mock_prisma_client.db.litellm_teamtable.find_many.assert_called_once_with(
            where={"team_id": {"in": user_info.teams}}
        )
    else:
        mock_prisma_client.db.litellm_teamtable.find_many.assert_not_called()


@pytest.mark.asyncio
async def test_post_call_failure_hook_auth_error_key_info_route():
    """
    Test that post_call_failure_hook does NOT call _handle_logging_proxy_only_error
    when we get an auth error from /key/info route (since it's not an LLM API route).
    """
    from unittest.mock import AsyncMock, patch

    from fastapi import HTTPException

    from litellm.caching.caching import DualCache
    from litellm.proxy._types import ProxyErrorTypes
    from litellm.proxy.utils import ProxyLogging

    # Setup
    cache = DualCache()
    proxy_logging = ProxyLogging(user_api_key_cache=cache)

    # Mock the _handle_logging_proxy_only_error method
    with patch.object(
        proxy_logging, "_handle_logging_proxy_only_error", new_callable=AsyncMock
    ) as mock_handle_logging:
        # Create an auth error (HTTPException)
        auth_error = HTTPException(
            status_code=401, detail="Authentication Error: invalid user key"
        )

        # Create request data for /key/info route
        request_data = {
            "route": "/key/info",
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test"}],
            "litellm_call_id": "test_call_id_123",
        }

        # Create user API key dict
        user_api_key_dict = UserAPIKeyAuth(
            api_key="test_key", user_id="test_user", token="test_token"
        )

        # Call post_call_failure_hook with auth error from /key/info route
        await proxy_logging.post_call_failure_hook(
            request_data=request_data,
            original_exception=auth_error,
            user_api_key_dict=user_api_key_dict,
            error_type=ProxyErrorTypes.auth_error,
            route="/key/info",
        )

        # Assert that _handle_logging_proxy_only_error was NOT called
        # because /key/info is not an LLM API route
        mock_handle_logging.assert_not_called()


@pytest.mark.asyncio
async def test_post_call_failure_hook_auth_error_llm_api_route():
    """
    Test that post_call_failure_hook DOES call _handle_logging_proxy_only_error
    when we get an auth error from /v1/chat/completions route (since it is an LLM API route).
    """
    from unittest.mock import AsyncMock, patch

    from fastapi import HTTPException

    from litellm.caching.caching import DualCache
    from litellm.proxy._types import ProxyErrorTypes
    from litellm.proxy.utils import ProxyLogging

    # Setup
    cache = DualCache()
    proxy_logging = ProxyLogging(user_api_key_cache=cache)

    # Mock the _handle_logging_proxy_only_error method
    with patch.object(
        proxy_logging, "_handle_logging_proxy_only_error", new_callable=AsyncMock
    ) as mock_handle_logging:
        # Create an auth error (HTTPException)
        auth_error = HTTPException(
            status_code=401, detail="Authentication Error: invalid user key"
        )

        # Create request data for /v1/chat/completions route
        request_data = {
            "route": "/v1/chat/completions",
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test"}],
            "litellm_call_id": "test_call_id_123",
        }

        # Create user API key dict
        user_api_key_dict = UserAPIKeyAuth(
            api_key="test_key",
            user_id="test_user",
            token="test_token",
            request_route="/v1/chat/completions",
        )

        # Call post_call_failure_hook with auth error from /v1/chat/completions route
        await proxy_logging.post_call_failure_hook(
            request_data=request_data,
            original_exception=auth_error,
            user_api_key_dict=user_api_key_dict,
            error_type=ProxyErrorTypes.auth_error,
            route="/v1/chat/completions",
        )

        # Assert that _handle_logging_proxy_only_error WAS called
        # because /v1/chat/completions is an LLM API route
        mock_handle_logging.assert_called_once()


@pytest.mark.asyncio
async def test_during_call_hook_parallel_execution():
    """
    Test that multiple guardrails in during_call_hook are executed in parallel.
    Verifies parallel execution by checking timing and execution order.
    """
    from litellm.caching.caching import DualCache
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.proxy.utils import ProxyLogging
    from litellm.types.guardrails import GuardrailEventHooks

    cache = DualCache()
    proxy_logging = ProxyLogging(user_api_key_cache=cache)
    execution_order = []

    class TestGuardrail(CustomGuardrail):
        def __init__(self, name):
            super().__init__(
                guardrail_name=name,
                event_hook=GuardrailEventHooks.during_call,
                default_on=True,
            )
            self.name = name

        async def async_moderation_hook(self, data, user_api_key_dict, call_type):
            execution_order.append(f"{self.name}_start")
            await asyncio.sleep(0.1)
            execution_order.append(f"{self.name}_end")
            return data

    original_callbacks = litellm.callbacks.copy() if litellm.callbacks else []

    try:
        litellm.callbacks = [TestGuardrail(f"g{i}") for i in range(3)]

        start_time = asyncio.get_event_loop().time()
        result = await proxy_logging.during_call_hook(
            data={"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]},
            user_api_key_dict=UserAPIKeyAuth(api_key="test_key", user_id="test_user"),
            call_type="completion",
        )
        execution_time = asyncio.get_event_loop().time() - start_time

        # Verify parallel execution: all start before any end
        first_end_idx = next(
            i for i, item in enumerate(execution_order) if "end" in item
        )
        starts_before_end = sum(
            1 for item in execution_order[:first_end_idx] if "start" in item
        )
        assert (
            starts_before_end == 3
        ), f"Expected 3 starts before first end, got {starts_before_end}"

        # Verify timing: parallel ~0.1s vs sequential ~0.3s
        assert (
            execution_time < 0.2
        ), f"Parallel execution took {execution_time}s, expected < 0.2s"
        assert result["model"] == "gpt-4"
    finally:
        litellm.callbacks = original_callbacks


@pytest.mark.asyncio
async def test_during_call_hook_parallel_execution_with_error():
    """
    Test that exceptions from guardrails are properly raised in parallel execution.
    """
    from litellm.caching.caching import DualCache
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.proxy.utils import ProxyLogging
    from litellm.types.guardrails import GuardrailEventHooks

    cache = DualCache()
    proxy_logging = ProxyLogging(user_api_key_cache=cache)

    class FailingGuardrail(CustomGuardrail):
        def __init__(self):
            super().__init__(
                guardrail_name="failing_guardrail",
                event_hook=GuardrailEventHooks.during_call,
                default_on=True,
            )

        async def async_moderation_hook(self, data, user_api_key_dict, call_type):
            raise ValueError("Guardrail violation detected!")

    original_callbacks = litellm.callbacks.copy() if litellm.callbacks else []

    try:
        litellm.callbacks = [FailingGuardrail()]

        with pytest.raises(ValueError) as exc_info:
            await proxy_logging.during_call_hook(
                data={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "test"}],
                },
                user_api_key_dict=UserAPIKeyAuth(
                    api_key="test_key", user_id="test_user"
                ),
                call_type="completion",
            )

        assert "Guardrail violation detected!" in str(exc_info.value)
    finally:
        litellm.callbacks = original_callbacks
