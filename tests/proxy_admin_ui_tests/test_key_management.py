
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv

load_dotenv()

# this file is to test litellm/proxy

import logging

import pytest

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *
from litellm.proxy.management_endpoints.key_management_endpoints import (
    generate_key_fn,
)
from litellm.proxy.proxy_server import (
    LitellmUserRoles,
)
from litellm.proxy.utils import ProxyLogging

verbose_proxy_logger.setLevel(level=logging.DEBUG)


from litellm.caching.caching import DualCache
from litellm.proxy._types import (
    GenerateKeyRequest,
    UpdateKeyRequest,
    UserAPIKeyAuth,
)
from tests._master_key import MASTER_KEY

proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())




################ Unit Tests for testing regeneration of keys ###########




















def test_is_team_key():
    from litellm.proxy.management_endpoints.key_management_endpoints import _is_team_key

    assert _is_team_key(GenerateKeyRequest(team_id="test_team_id"))
    assert not _is_team_key(GenerateKeyRequest(user_id="test_user_id"))


def test_team_key_generation_team_member_check():
    from fastapi import HTTPException

    from litellm.proxy._types import LiteLLM_TeamTableCachedObj
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _team_key_generation_check,
    )

    litellm.key_generation_settings = {
        "team_key_generation": {"allowed_team_member_roles": ["admin"]}
    }

    team_table = LiteLLM_TeamTableCachedObj(
        team_id="test_team_id",
        team_alias="test_team_alias",
        members_with_roles=[Member(role="admin", user_id="test_user_id")],
    )

    assert _team_key_generation_check(
        team_table=team_table,
        user_api_key_dict=UserAPIKeyAuth(
            user_id="test_user_id",
            user_role=LitellmUserRoles.INTERNAL_USER,
            api_key=MASTER_KEY,
            team_member=Member(role="admin", user_id="test_user_id"),
        ),
        data=GenerateKeyRequest(),
        route=KeyManagementRoutes.KEY_GENERATE,
    )

    team_table = LiteLLM_TeamTableCachedObj(
        team_id="test_team_id",
        team_alias="test_team_alias",
        members_with_roles=[Member(role="user", user_id="test_user_id")],
    )

    with pytest.raises(HTTPException):
        _team_key_generation_check(
            team_table=team_table,
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.INTERNAL_USER,
                api_key=MASTER_KEY,
                user_id="test_user_id",
                team_member=Member(role="user", user_id="test_user_id"),
            ),
            data=GenerateKeyRequest(),
            route=KeyManagementRoutes.KEY_GENERATE,
        )


@pytest.mark.parametrize(
    "team_key_generation_settings, input_data, expected_result",
    [
        ({"required_params": ["tags"]}, GenerateKeyRequest(tags=["test_tags"]), True),
        ({}, GenerateKeyRequest(), True),
        (
            {"required_params": ["models"]},
            GenerateKeyRequest(tags=["test_tags"]),
            False,
        ),
    ],
)
@pytest.mark.parametrize("key_type", ["team_key", "personal_key"])
def test_key_generation_required_params_check(
    team_key_generation_settings, input_data, expected_result, key_type
):
    from fastapi import HTTPException

    from litellm.proxy._types import LiteLLM_TeamTableCachedObj
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _personal_key_generation_check,
        _team_key_generation_check,
    )
    from litellm.types.utils import (
        PersonalUIKeyGenerationConfig,
        StandardKeyGenerationConfig,
        TeamUIKeyGenerationConfig,
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        api_key=MASTER_KEY,
        user_id="test_user_id",
        team_id="test_team_id",
        team_member=None,
    )

    team_table = LiteLLM_TeamTableCachedObj(
        team_id="test_team_id",
        team_alias="test_team_alias",
        members_with_roles=[Member(role="admin", user_id="test_user_id")],
    )

    if key_type == "team_key":
        litellm.key_generation_settings = StandardKeyGenerationConfig(
            team_key_generation=TeamUIKeyGenerationConfig(
                **team_key_generation_settings
            )
        )
    elif key_type == "personal_key":
        litellm.key_generation_settings = StandardKeyGenerationConfig(
            personal_key_generation=PersonalUIKeyGenerationConfig(
                **team_key_generation_settings
            )
        )

    if expected_result:
        if key_type == "team_key":
            assert _team_key_generation_check(
                team_table=team_table,
                user_api_key_dict=user_api_key_dict,
                data=input_data,
                route=KeyManagementRoutes.KEY_GENERATE,
            )
        elif key_type == "personal_key":
            assert _personal_key_generation_check(
                user_api_key_dict=user_api_key_dict,
                data=input_data,
            )
    else:
        if key_type == "team_key":
            with pytest.raises(HTTPException):
                _team_key_generation_check(
                    team_table=team_table,
                    user_api_key_dict=user_api_key_dict,
                    data=input_data,
                    route=KeyManagementRoutes.KEY_GENERATE,
                )
        elif key_type == "personal_key":
            with pytest.raises(HTTPException):
                _personal_key_generation_check(user_api_key_dict, input_data)


def test_personal_key_generation_check():
    from fastapi import HTTPException

    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _personal_key_generation_check,
    )

    litellm.key_generation_settings = {
        "personal_key_generation": {"allowed_user_roles": ["proxy_admin"]}
    }

    assert _personal_key_generation_check(
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN, api_key=MASTER_KEY, user_id="admin"
        ),
        data=GenerateKeyRequest(),
    )

    with pytest.raises(HTTPException):
        _personal_key_generation_check(
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.INTERNAL_USER,
                api_key=MASTER_KEY,
                user_id="admin",
            ),
            data=GenerateKeyRequest(),
        )


@pytest.mark.parametrize(
    "update_request_data, non_default_values, existing_metadata, expected_result",
    [
        (
            {"metadata": {"test": "new"}},
            {"metadata": {"test": "new"}},
            {"test": "test"},
            {"metadata": {"test": "new"}},
        ),
        (
            {"tags": ["new_tag"]},
            {},
            {"tags": ["old_tag"]},
            {"metadata": {"tags": ["new_tag"]}},
        ),
        (
            {"enforced_params": ["metadata.tags"]},
            {},
            {"tags": ["old_tag"]},
            {"metadata": {"tags": ["old_tag"], "enforced_params": ["metadata.tags"]}},
        ),
        (
            {"disable_global_guardrails": True},
            {},
            {},
            {"metadata": {"disable_global_guardrails": True}},
        ),
        (
            {"disable_global_guardrails": False},
            {},
            {"disable_global_guardrails": True},
            {"metadata": {"disable_global_guardrails": False}},
        ),
    ],
)
def test_prepare_metadata_fields(
    update_request_data, non_default_values, existing_metadata, expected_result
):
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        prepare_metadata_fields,
    )

    args = {
        "data": UpdateKeyRequest(
            key="sk-1qGQUJJTcljeaPfzgWRrXQ", **update_request_data
        ),
        "non_default_values": non_default_values,
        "existing_metadata": existing_metadata,
    }

    updated_non_default_values = prepare_metadata_fields(**args)
    assert updated_non_default_values == expected_result











@pytest.mark.asyncio
@patch("litellm.proxy.management_endpoints.key_management_endpoints.get_team_object")
async def test_key_generate_always_db_team(mock_get_team_object):

    setattr(litellm.proxy.proxy_server, "prisma_client", MagicMock())
    mock_get_team_object.return_value = None
    try:
        await generate_key_fn(
            data=GenerateKeyRequest(team_id="1234"),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key=MASTER_KEY,
                user_id="admin",
            ),
        )
    except Exception as e:
        print(f"Error: {e}")

    mock_get_team_object.assert_called_once()
    assert mock_get_team_object.call_args.kwargs["check_db_only"] == True
