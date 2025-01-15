import asyncio
import os
import sys
from typing import Any, Dict
from unittest.mock import Mock
from litellm.proxy.utils import _get_redoc_url, _get_docs_url
import json
import pytest
from fastapi import Request

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from unittest.mock import MagicMock, patch, AsyncMock

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import is_request_body_safe
from litellm.proxy.litellm_pre_call_utils import (
    _get_dynamic_logging_metadata,
    add_litellm_data_to_request,
)
from litellm.types.utils import SupportedCacheControls


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
        token="6f8688eaff1d37555bb9e9a6390b6d7032b3ab2526ba0152da87128eab956432",
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
        api_key="7c305cc48fe72272700dc0d67dc691c2d1f2807490ef5eb2ee1d3a3ca86e12b1",
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
        token="6f8688eaff1d37555bb9e9a6390b6d7032b3ab2526ba0152da87128eab956432",
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
        api_key="7c305cc48fe72272700dc0d67dc691c2d1f2807490ef5eb2ee1d3a3ca86e12b1",
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
    "headers, expected_data",
    [
        ({"OpenAI-Organization": "test_org_id"}, {"organization": "test_org_id"}),
        ({"openai-organization": "test_org_id"}, {"organization": "test_org_id"}),
        ({}, {}),
        (
            {
                "OpenAI-Organization": "test_org_id",
                "Authorization": "Bearer test_token",
            },
            {
                "organization": "test_org_id",
            },
        ),
    ],
)
def test_add_litellm_data_for_backend_llm_call(headers, expected_data):
    import json
    from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
    from litellm.proxy._types import UserAPIKeyAuth

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key", user_id="test_user_id", org_id="test_org_id"
    )

    data = LiteLLMProxyRequestSetup.add_litellm_data_for_backend_llm_call(
        headers=headers,
        user_api_key_dict=user_api_key_dict,
        general_settings=None,
    )

    assert json.dumps(data, sort_keys=True) == json.dumps(expected_data, sort_keys=True)


def test_foward_litellm_user_info_to_backend_llm_call():
    import json

    litellm.add_user_information_to_llm_headers = True

    from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
    from litellm.proxy._types import UserAPIKeyAuth

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
    }

    assert json.dumps(data, sort_keys=True) == json.dumps(expected_data, sort_keys=True)


def test_update_internal_user_params():
    from litellm.proxy.management_endpoints.internal_user_endpoints import (
        _update_internal_new_user_params,
    )
    from litellm.proxy._types import NewUserRequest

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


@pytest.mark.asyncio
async def test_proxy_config_update_from_db():
    from litellm.proxy.proxy_server import ProxyConfig
    from pydantic import BaseModel

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


def test_prepare_key_update_data():
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        prepare_key_update_data,
    )
    from litellm.proxy._types import UpdateKeyRequest

    existing_key_row = MagicMock()
    data = UpdateKeyRequest(key="test_key", models=["gpt-4"], duration="120s")
    updated_data = prepare_key_update_data(data, existing_key_row)
    assert "expires" in updated_data

    data = UpdateKeyRequest(key="test_key", metadata={})
    updated_data = prepare_key_update_data(data, existing_key_row)
    assert updated_data["metadata"] == {}

    data = UpdateKeyRequest(key="test_key", metadata=None)
    updated_data = prepare_key_update_data(data, existing_key_row)
    assert updated_data["metadata"] is None


@pytest.mark.parametrize(
    "env_value, expected_url",
    [
        (None, "/redoc"),  # default case
        ("/custom-redoc", "/custom-redoc"),  # custom URL
        ("https://example.com/redoc", "https://example.com/redoc"),  # full URL
    ],
)
def test_get_redoc_url(env_value, expected_url):
    if env_value is not None:
        os.environ["REDOC_URL"] = env_value
    else:
        os.environ.pop("REDOC_URL", None)  # ensure env var is not set

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
    "general_settings, user_api_key_dict, expected_enforced_params",
    [
        (
            {"enforced_params": ["param1", "param2"]},
            UserAPIKeyAuth(
                api_key="test_api_key", user_id="test_user_id", org_id="test_org_id"
            ),
            ["param1", "param2"],
        ),
        (
            {"service_account_settings": {"enforced_params": ["param1", "param2"]}},
            UserAPIKeyAuth(
                api_key="test_api_key", user_id="test_user_id", org_id="test_org_id"
            ),
            ["param1", "param2"],
        ),
        (
            {"service_account_settings": {"enforced_params": ["param1", "param2"]}},
            UserAPIKeyAuth(
                api_key="test_api_key",
                metadata={"enforced_params": ["param3", "param4"]},
            ),
            ["param1", "param2", "param3", "param4"],
        ),
    ],
)
def test_get_enforced_params(
    general_settings, user_api_key_dict, expected_enforced_params
):
    from litellm.proxy.litellm_pre_call_utils import _get_enforced_params

    enforced_params = _get_enforced_params(general_settings, user_api_key_dict)
    assert enforced_params == expected_enforced_params


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
    from litellm.proxy.auth.model_checks import get_key_models
    from collections import defaultdict

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
    from litellm.proxy.auth.model_checks import get_team_models
    from collections import defaultdict

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

    result = get_team_models(
        user_api_key_dict=user_api_key_dict,
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
            "default_team_settings": [],
            "context_window_fallbacks": [{"gpt-3.5-turbo": ["gpt-3.5-turbo-large"]}],
        },
    }
    updated_config = proxy_config._update_config_fields(**args)

    all_team_config = updated_config["litellm_settings"]["default_team_settings"]

    # check if team id config returned
    team_config = proxy_config._get_team_config(
        team_id="c91e32bb-0f2a-4aa1-86c4-307ca2e03ea3", all_teams_config=all_team_config
    )
    assert team_config["langfuse_public_key"] == "my-fake-key"
    assert team_config["langfuse_secret"] == "my-fake-secret"


@pytest.mark.parametrize(
    "proxy_model_list,provider",
    [
        (["openai/*"], "openai"),
        (["bedrock/*"], "bedrock"),
        (["anthropic/*"], "anthropic"),
        (["cohere/*"], "cohere"),
    ],
)
def test_get_complete_model_list(proxy_model_list, provider):
    """
    Test that get_complete_model_list correctly expands model groups like 'openai/*' into individual models with provider prefixes
    """
    from litellm.proxy.auth.model_checks import get_complete_model_list

    complete_list = get_complete_model_list(
        proxy_model_list=proxy_model_list,
        key_models=[],
        team_models=[],
        user_model=None,
        infer_model_from_keys=False,
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


def test_proxy_config_state_post_init_callback_call():
    """
    Ensures team_id is still in config, after callback is called

    Addresses issue: https://github.com/BerriAI/litellm/issues/6787

    Where team_id was being popped from config, after callback was called
    """
    from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
    from litellm.proxy.proxy_server import ProxyConfig

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
    from litellm.proxy.proxy_server import ProxyConfig
    import threading

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
        "token": "78b627d4d14bc3acf5571ae9cb6834e661bc8794d1209318677387add7621ce1",
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


def test_is_allowed_to_create_key():
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _is_allowed_to_create_key,
    )

    assert (
        _is_allowed_to_create_key(
            user_api_key_dict=UserAPIKeyAuth(
                user_id="test_user_id", user_role=LitellmUserRoles.PROXY_ADMIN
            ),
            user_id="test_user_id",
            team_id="test_team_id",
        )
        is True
    )

    assert (
        _is_allowed_to_create_key(
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
    from litellm.proxy.proxy_server import _get_model_group_info
    from litellm import Router

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


import pytest
import asyncio
from unittest.mock import AsyncMock, patch
import json


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
    from litellm.proxy.management_endpoints.ui_sso import generic_response_convertor
    from litellm.proxy.management_endpoints.ui_sso import JWTHandler
    from litellm.proxy._types import LiteLLM_JWTAuth
    from litellm.caching import DualCache

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
    from litellm.proxy.auth.user_api_key_auth import _get_temp_budget_increase
    from litellm.proxy._types import UserAPIKeyAuth
    from datetime import datetime, timedelta

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
    from litellm.proxy.auth.user_api_key_auth import (
        _update_key_budget_with_temp_budget_increase,
    )
    from litellm.proxy._types import UserAPIKeyAuth
    from datetime import datetime, timedelta

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


from unittest.mock import MagicMock, AsyncMock


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
