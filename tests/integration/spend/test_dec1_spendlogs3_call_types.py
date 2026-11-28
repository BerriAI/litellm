"""Contract: the spend log row of every non chat inference route, ahead of the Spend Logs V2 port.

Rounds 1 and 2 pin chat, streaming, embeddings, text completions, `/v1/messages`, `/v1/responses`, pass-through,
MCP, batches and realtime. The Rust callback subscribes to `attempt_ended` for every route, so the media and
rerank routes must keep the same row too. This file runs one table of routes against a scripted provider
(`media_upstream`): OpenAI speech (`aspeech`), transcription (`atranscription`), moderation (`amoderation`),
image generation (`aimage_generation`) and Cohere v2 rerank (`arerank`).

For each route it pins: one success row with the route's `call_type` and the caller's model group; the
`request_id` rule (the provider's response id when the answer has one, else the litellm call id,
`get_spend_logs_id`, litellm/proxy/spend_tracking/spend_tracking_utils.py:408); the key, team and owner columns;
the provider column; one ledger delivery; the daily `endpoint` from `ROUTE_ENDPOINT_MAPPING`
(litellm/proxy/route_llm_request.py:156); the model usage counter; and a failure row plus a request error counter
when the provider answers 500. Spend is deliberately not asserted: image prices are lost today (#52103) and the
speech and rerank units are priced per character and per query, outside the shared 0.001 / 0.002 token prices.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually, string_value
from tests.integration._support.wire import Wire, wire_server
from tests.integration.spend.dec1_spend_support import LANDING_SECONDS, failing, hashed, rows_of_key
from tests.integration.spend.dec1_spendlogs2_support import CALL_ID_HEADER, failure_rows
from tests.integration.spend.dec1_spendlogs3_support import (
    ROUTES,
    Route,
    counters,
    daily_table_rows,
    media_upstream,
    model_usage_reaches,
    request_error_rows,
)

pytestmark: Final = pytest.mark.shares_worker_state

ROUTE_IDS: Final = [route.name for route in ROUTES]


@contextmanager
def _broken_upstream() -> Iterator[Wire]:
    with wire_server(failing(500)) as wire:
        yield wire


def _served(route: Route, gateway: Gateway, model: str, key: str) -> str:
    """Send the route's request, assert it was served, and return the `x-litellm-call-id` it was answered with."""
    response: Final = route.send(gateway, model, key)
    assert response.status_code == 200, f"{route.name}: {response.status_code} {response.text}"
    return response.headers[CALL_ID_HEADER]


@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
def test_each_route_writes_one_success_row_with_its_call_type(gateway: Gateway, route: Route) -> None:
    """One served call writes exactly one row for the key with the route's `call_type` (`aspeech`,
    `atranscription`, `amoderation`, `aimage_generation`, `arerank`), `status` `success` and the public model
    name as `model_group`. A port that only subscribes the chat surfaces loses these rows."""
    with media_upstream() as wire, gateway.scenario() as scenario:
        model: Final = route.model(scenario, wire)
        key: Final = scenario.key(models=[model])
        _served(route, gateway, model, key)
        row: Final = rows_of_key(gateway, key, 1)[0]
        assert (row["call_type"], row["status"], row["model_group"]) == (route.call_type, "success", model), row


@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
def test_each_route_row_is_filed_under_the_provider_id_or_the_call_id(gateway: Gateway, route: Route) -> None:
    """Moderation and rerank answers carry an `id` (`modr-...`, `dec1-sl3-rerank-...`), so their row's
    `request_id` starts with it; speech, transcription and image answers have none, so the row falls back to the
    litellm call id, which equals the `x-litellm-call-id` response header (`get_spend_logs_id` candidates:
    response id, StandardLoggingPayload id, litellm call id, spend_tracking_utils.py:415). UNVERIFIED: the
    StandardLoggingPayload id of an id-less answer is the call id."""
    with media_upstream() as wire, gateway.scenario() as scenario:
        model: Final = route.model(scenario, wire)
        key: Final = scenario.key(models=[model])
        call_id: Final = _served(route, gateway, model, key)
        request_id: Final = string_value(rows_of_key(gateway, key, 1)[0]["request_id"])
        if route.provider_id_prefix is None:
            assert request_id == call_id, (request_id, call_id)
        else:
            assert request_id.startswith(route.provider_id_prefix), request_id


@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
def test_each_route_row_keeps_the_callers_team_and_owner(gateway: Gateway, route: Route) -> None:
    """A key of a team, owned by a user: the row has `api_key` equal to the key's sha256 hash, `team_id` the
    team and `user` the owner, whatever the route (`get_logging_payload`, spend_tracking_utils.py:1031-1035)."""
    with media_upstream() as wire, gateway.scenario() as scenario:
        model: Final = route.model(scenario, wire)
        team: Final = scenario.team()
        owner: Final = scenario.user()
        key: Final = scenario.key(models=[model], team_id=team, user_id=owner)
        _served(route, gateway, model, key)
        row: Final = rows_of_key(gateway, key, 1)[0]
        assert (row["api_key"], row["team_id"], row["user"]) == (hashed(key), team, owner), row


@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
def test_each_route_row_names_the_deployment_provider(gateway: Gateway, route: Route) -> None:
    """The row's `custom_llm_provider` is the provider prefix of the deployment (`openai` for the OpenAI media
    routes, `cohere` for rerank). UNVERIFIED: the rerank call logs `cohere` rather than `cohere_chat`."""
    with media_upstream() as wire, gateway.scenario() as scenario:
        model: Final = route.model(scenario, wire)
        key: Final = scenario.key(models=[model])
        _served(route, gateway, model, key)
        row: Final = rows_of_key(gateway, key, 1)[0]
        assert row["custom_llm_provider"] == route.upstream_model.split("/")[0], row


@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
def test_each_route_row_gets_one_delivery_that_ends_applied(gateway: Gateway, route: Route) -> None:
    """The route's row has exactly one `LiteLLM_SpendLogDelivery` with its `request_id`, and that delivery
    reaches `applied`: the ledger worker projects media and rerank charges like chat ones
    (`spend_log_insert.py`, `flush_accounting_deliveries`, litellm/accounting/worker.py:48)."""
    with media_upstream() as wire, gateway.scenario() as scenario:
        model: Final = route.model(scenario, wire)
        key: Final = scenario.key(models=[model])
        _served(route, gateway, model, key)
        request_id: Final = string_value(rows_of_key(gateway, key, 1)[0]["request_id"])
        states: Final = eventually(
            lambda: gateway.rows(
                'SELECT state::text AS state FROM "LiteLLM_SpendLogDelivery" WHERE request_id = %s', (request_id,)
            ),
            lambda found: [row["state"] for row in found] == ["applied"],
            seconds=LANDING_SECONDS,
        )
        assert len(states) == 1, states


@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
def test_each_route_daily_user_row_uses_the_mapped_endpoint(gateway: Gateway, route: Route) -> None:
    """The key's `LiteLLM_DailyUserSpend` row for the call has `endpoint` from `ROUTE_ENDPOINT_MAPPING`
    (`/audio/speech`, `/audio/transcriptions`, `/moderations`, `/image/generations`, `/rerank`; note the singular
    `/image/` for images), `model_group` the public name, `api_requests` 1 and `successful_requests` 1
    (projection.py:178)."""
    with media_upstream() as wire, gateway.scenario() as scenario:
        model: Final = route.model(scenario, wire)
        key: Final = scenario.key(models=[model])
        _served(route, gateway, model, key)
        daily: Final = daily_table_rows(gateway, "LiteLLM_DailyUserSpend", "api_key", hashed(key), 1)[0]
        assert (daily["endpoint"], daily["model_group"]) == (route.endpoint, model), daily
        assert counters(daily, "api_requests", "successful_requests", "failed_requests") == (1, 1, 0), daily


@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
def test_each_route_counts_one_request_in_model_usage(gateway: Gateway, route: Route) -> None:
    """The route's model group gets a `LiteLLM_DailyModelUsage` count of one successful request (projection.py:
    263); the model usage card covers every inference route, not only chat."""
    with media_upstream() as wire, gateway.scenario() as scenario:
        model: Final = route.model(scenario, wire)
        key: Final = scenario.key(models=[model])
        _served(route, gateway, model, key)
        rows: Final = model_usage_reaches(gateway, model, 1)
        assert [counters(row, "request_count", "successful_requests") for row in rows] == [(1, 1)], rows


@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
def test_each_route_provider_failure_writes_one_failure_row_with_its_call_type(gateway: Gateway, route: Route) -> None:
    """The provider answers every call with 500. The caller gets an error, and exactly one `failure` row lands for
    the key with the route's `call_type` and `spend` 0 (a 500 retried and failed again writes one row, round 1).
    UNVERIFIED: the status the caller sees is 500 on every route, so only "not 200" is asserted."""
    with _broken_upstream() as wire, gateway.scenario() as scenario:
        model: Final = route.model(scenario, wire)
        key: Final = scenario.key(models=[model])
        assert route.send(gateway, model, key).status_code != 200
        row: Final = failure_rows(gateway, key, 1)[0]
        assert (row["call_type"], float(str(row["spend"]))) == (route.call_type, 0.0), row


@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
def test_each_route_provider_failure_counts_a_500_request_error(gateway: Gateway, route: Route) -> None:
    """The same 500 is counted once in `LiteLLM_DailyRequestErrors` for the key, the model group and status 500
    (request_error_tracking.py:96): the error rollup reads every failure row, whatever its route."""
    with _broken_upstream() as wire, gateway.scenario() as scenario:
        model: Final = route.model(scenario, wire)
        key: Final = scenario.key(models=[model])
        assert route.send(gateway, model, key).status_code != 200
        row: Final = request_error_rows(gateway, key, 1)[0]
        assert (row["model_group"], row["status_code"], row["failed_requests"]) == (model, 500, 1), row
