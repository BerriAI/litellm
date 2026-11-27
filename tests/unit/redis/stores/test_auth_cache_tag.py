"""Python tag wire fixtures shared with the Rust auth-store and Redis tests."""

import json
from typing import Final
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from litellm.models.tag import LiteLLM_TagTable
from litellm.proxy.auth.auth_checks import read_tag_objects_batch
from litellm.proxy.common_utils.cache_pydantic_utils import CacheCodec
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.redis.stores.auth_cache import AuthObject

PLAIN_ENTRY: Final = (
    '{"tag_name": "plain", "description": null, "models": [], "model_info": null,'
    ' "spend": 0.0, "budget_id": null, "litellm_budget_table": null,'
    ' "created_at": "2026-11-19T10:00:00Z", "created_by": null, "updated_at": "2026-11-19T10:00:00Z"}'
)
LIMITED_ENTRY: Final = (
    '{"tag_name": "limited", "description": "Team tag", "models": ["claude"],'
    ' "model_info": {"deployment-1": "claude"}, "spend": 1.5, "budget_id": "bare",'
    ' "litellm_budget_table": {"budget_id": "bare", "soft_budget": null, "max_budget": 10.5,'
    ' "max_parallel_requests": null, "tpm_limit": 1000, "rpm_limit": 3, "tpd_limit": null,'
    ' "model_max_budget": null, "model_tpm_limit": null, "model_rpm_limit": null, "budget_duration": null,'
    ' "allowed_models": [], "temp_budget_increase": null, "temp_budget_expiry": null},'
    ' "created_at": "2026-11-19T10:00:00.123000Z", "created_by": "creator",'
    ' "updated_at": "2026-11-19T10:00:00Z"}'
)


@pytest.mark.parametrize(
    ("tag_name", "row", "entry"),
    [
        (
            "plain",
            '{"tag_name":"plain","models":null,"spend":null,"created_at":"2026-11-19T10:00:00+00:00",'
            '"updated_at":"2026-11-19T10:00:00+00:00"}',
            PLAIN_ENTRY,
        ),
        (
            "limited",
            '{"tag_name":"limited","description":"Team tag","models":["claude"],"model_info":{"deployment-1":"claude"},'
            '"spend":1.5,"budget_id":"bare","litellm_budget_table":{"budget_id":"bare","max_budget":10.5,'
            '"rpm_limit":3,"tpm_limit":1000,"allowed_models":[]},'
            '"created_at":"2026-11-19T10:00:00.123+00:00","created_by":"creator",'
            '"updated_at":"2026-11-19T10:00:00+00:00"}',
            LIMITED_ENTRY,
        ),
    ],
    ids=["no_budget", "linked_budget"],
)
@pytest.mark.asyncio
async def test_tag_reader_accepts_the_bytes_the_python_model_and_rust_store_write(
    tag_name: str, row: str, entry: str
) -> None:
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(
        AuthObject.tag(tag_name), LiteLLM_TagTable.model_validate_json(row), model_type=LiteLLM_TagTable
    )
    assert json.dumps(cache.get_local(AuthObject.tag(tag_name))) == entry

    found: Final = await read_tag_objects_batch([tag_name], MagicMock(), cache)
    assert not isinstance(found, Exception)

    assert json.dumps(CacheCodec.serialize(found[tag_name], model_type=LiteLLM_TagTable)) == entry


def test_the_tag_model_refuses_a_model_info_array() -> None:
    with pytest.raises(ValidationError):
        LiteLLM_TagTable.model_validate_json('{"tag_name":"malformed","model_info":[]}')
