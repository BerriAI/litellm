"""
Behavior tests for the cache analytics queries against a real Postgres. The info-route
exclusion and the Unknown grouping live in SQL, so these tests are the ones that exercise
them; the endpoint wiring is unit-tested in
tests/test_litellm/proxy/analytics_endpoints/test_analytics_endpoints.py.
"""

import json
import uuid
from datetime import datetime
from typing import Final

import pytest

from litellm.proxy.analytics_endpoints.cache_activity import (
    ERROR_BREAKDOWN_SQL,
    GROUPS_SQL,
    INFO_ROUTES_JSON,
    KEY_ALIAS_OPTIONS_SQL,
    MODEL_OPTIONS_SQL,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")

DAY: Final = datetime(2001, 3, 7)
AT_NOON: Final = DAY.replace(hour=12)
RUN: Final = uuid.uuid4()
INFERENCE_KEY: Final = f"ca-inference-{RUN}"
INFO_ONLY_KEY: Final = f"ca-info-only-{RUN}"
INFERENCE_ALIAS: Final = f"alias-inference-{RUN}"
INFO_ONLY_ALIAS: Final = f"alias-info-only-{RUN}"
INFERENCE_MODEL: Final = f"gpt-5.4-mini-{RUN}"
INFO_ONLY_MODEL: Final = f"ghost-model-{RUN}"
NO_FILTER: Final = "[]"


async def _spend_log(db, api_key: str, call_type: str, status: str, model: str = "", error_code: str = "") -> None:
    metadata: Final = {"error_information": {"error_code": error_code, "error_class": "ProxyException"}}
    await db.execute_raw(
        'INSERT INTO "LiteLLM_SpendLogs" ("request_id", "call_type", "api_key", "startTime", "endTime", "model", '
        '"status", "metadata") VALUES ($1, $2, $3, $4::timestamp, $4::timestamp, $5, $6, $7::jsonb)',
        str(uuid.uuid4()),
        call_type,
        api_key,
        AT_NOON,
        model,
        status,
        json.dumps(metadata if status == "failure" else {}),
    )


@pytest.fixture(scope="module", autouse=True)
async def seeded(db):
    for token, alias in ((INFERENCE_KEY, INFERENCE_ALIAS), (INFO_ONLY_KEY, INFO_ONLY_ALIAS)):
        await db.execute_raw(
            'INSERT INTO "LiteLLM_VerificationToken" ("token", "key_alias") VALUES ($1, $2)', token, alias
        )
    await _spend_log(db, INFERENCE_KEY, "acompletion", "success", model=INFERENCE_MODEL)
    await _spend_log(db, INFERENCE_KEY, "acompletion", "failure", model=INFERENCE_MODEL, error_code="429")
    await _spend_log(db, INFERENCE_KEY, "", "failure", error_code="401")
    await _spend_log(db, INFERENCE_KEY, "/model/info", "failure", error_code="401")
    await _spend_log(db, INFO_ONLY_KEY, "/v1/models", "failure", model=INFO_ONLY_MODEL, error_code="401")
    await _spend_log(db, INFO_ONLY_KEY, "/key/info", "success")
    yield
    keys: Final = [INFERENCE_KEY, INFO_ONLY_KEY]
    await db.execute_raw('DELETE FROM "LiteLLM_SpendLogs" WHERE "api_key" = ANY($1::text[])', keys)
    await db.execute_raw('DELETE FROM "LiteLLM_VerificationToken" WHERE "token" = ANY($1::text[])', keys)


async def _groups(db, key_aliases: list[str]) -> dict[str, dict]:
    rows: Final = await db.query_raw(GROUPS_SQL, DAY, DAY, json.dumps(key_aliases), NO_FILTER, INFO_ROUTES_JSON)
    return {row["call_type"]: row for row in rows}


async def test_groups_drop_info_routes_and_keep_unknown_for_rows_without_an_endpoint(db):
    groups: Final = await _groups(db, [INFERENCE_ALIAS])
    assert set(groups) == {"acompletion", "Unknown"}
    assert (groups["acompletion"]["api_requests"], groups["acompletion"]["failed_requests"]) == (1, 1)
    assert (groups["Unknown"]["api_requests"], groups["Unknown"]["failed_requests"]) == (0, 1)


async def test_key_with_only_info_route_traffic_has_no_groups(db):
    assert await _groups(db, [INFO_ONLY_ALIAS]) == {}


async def test_error_breakdown_drops_info_routes(db):
    rows: Final = await db.query_raw(
        ERROR_BREAKDOWN_SQL, DAY, DAY, json.dumps([INFERENCE_ALIAS, INFO_ONLY_ALIAS]), NO_FILTER, INFO_ROUTES_JSON
    )
    assert {(row["call_type"], row["error_code"], row["count"]) for row in rows} == {
        ("acompletion", "429", 1),
        ("Unknown", "401", 1),
    }


async def test_filter_options_only_offer_values_that_return_analytics(db):
    key_alias_rows: Final = await db.query_raw(KEY_ALIAS_OPTIONS_SQL, DAY, DAY, INFO_ROUTES_JSON)
    model_rows: Final = await db.query_raw(MODEL_OPTIONS_SQL, DAY, DAY, INFO_ROUTES_JSON)
    key_aliases: Final = {row["key_alias"] for row in key_alias_rows}
    models: Final = {row["model"] for row in model_rows}
    assert INFERENCE_ALIAS in key_aliases and INFO_ONLY_ALIAS not in key_aliases
    assert INFERENCE_MODEL in models and INFO_ONLY_MODEL not in models
