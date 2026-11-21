"""Health probes in the spend ledger and in alerting: who pays for them and whether they page anyone.

Every probe is logged like a request, but against the internal health service account instead of a caller:
`update_model_params_with_health_check_tracking_information`
(litellm/litellm_core_utils/health_check_helpers.py:180) attaches a `UserAPIKeyAuth` whose `api_key`,
`team_id`, `key_alias` and `team_alias` are all `litellm-internal-health-check`, and the request tag of the
same name. Since #50987 (Nov 11) the probe is logged with the mode's call type (`aembedding` for an
embedding probe) and the deployment's `model_id`, so it is priced like the deployment. The spend log UI can
drop these rows with `exclude_internal_health_checks=true`
(litellm/proxy/spend_tracking/spend_management_endpoints.py:3003). What must never happen is that a
caller's key or team is charged for a probe it triggered.

Probes never go through the proxy's request hooks, so a failing probe posts no `llm_exceptions` alert, and
because the probe's call params carry no deployment id the Slack outage counter (keyed by `model_info.id`,
slack_alerting.py:1754) never sees it either: health checks do not page on-call. Spend logging and callbacks
are being ported (CustomLoggerV2); these are the outside effects to keep.

One owned proxy with background checks every 2 seconds, `alerting: ["slack"]` pointed at the scripted
upstream's `/slack` path, outage thresholds cut to 2 and 3, and deployments with a custom price so a probe
costs a visible amount.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import eventually, held, list_value, object_value
from tests.integration.routing.dec1_healthchk_support import (
    HEALTH_SERVICE_ACCOUNT,
    Rig,
    chat,
    deployment,
    health,
    health_rig,
    report,
)

pytestmark: Final = [pytest.mark.timeout(400), pytest.mark.shares_worker_state]

RUN: Final = uuid.uuid4().hex[:6]
CONVERGENCE_SECONDS: Final = 90
PRICED_GROUP: Final = f"spend-chat-{RUN}"
EMBED_GROUP: Final = f"spend-embed-{RUN}"
FAILING_GROUP: Final = f"spend-failing-{RUN}"
PRICED_MODEL: Final = f"ok-spend-chat-{RUN}"
EMBED_MODEL: Final = f"ok-spend-embed-{RUN}"
FAILING_MODEL: Final = f"status-500-spend-{RUN}"
PRICE: Final[dict[str, JsonValue]] = {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002}


def _models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        deployment(PRICED_GROUP, PRICED_MODEL, api_base, **PRICE),
        deployment(EMBED_GROUP, EMBED_MODEL, api_base, model_info={"mode": "embedding"}, input_cost_per_token=0.001),
        deployment(FAILING_GROUP, FAILING_MODEL, api_base),
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with health_rig(
        tmp_path_factory.mktemp("dec1-healthchk-spend"),
        _models,
        general_settings={
            "background_health_checks": True,
            "health_check_interval": 2,
            "alerting": ["slack"],
            "alert_types": ["llm_exceptions", "outage_alerts"],
            "alerting_args": {"minor_outage_alert_threshold": 2, "major_outage_alert_threshold": 3},
        },
        environment=lambda url: {"SLACK_WEBHOOK_URL": f"{url}/slack"},
    ) as value:
        yield value


def _probe_rows(rig: Rig, deployment_id: str) -> list[dict[str, JsonValue]]:
    return rig.gateway.rows(
        'SELECT api_key, team_id, call_type, model_id, request_tags, spend FROM "LiteLLM_SpendLogs" '
        "WHERE model_id = %s AND team_id = %s",
        (deployment_id, HEALTH_SERVICE_ACCOUNT),
    )


def _slack_texts(rig: Rig) -> tuple[str, ...]:
    return tuple(str(json.loads(seen.raw).get("text", "")) for seen in rig.upstream.on_path("/slack"))


def test_a_probe_is_logged_against_the_internal_health_account(rig: Rig) -> None:
    """The background loop probes `spend-chat-<run>` (deployment `hc-ok-spend-chat-<run>`).

    Within 90 seconds a `LiteLLM_SpendLogs` row must exist with that `model_id`, `api_key` and `team_id` both
    `litellm-internal-health-check` (the name is stored as is, it is not a secret,
    spend_tracking_utils.py:158), and `request_tags` containing `litellm-internal-health-check`.
    UNVERIFIED that `api_key` is stored unhashed; the exclusion filter accepts both forms.
    """
    rows: Final = eventually(
        lambda: _probe_rows(rig, f"hc-{PRICED_MODEL}"), lambda found: len(found) >= 1, seconds=CONVERGENCE_SECONDS
    )
    assert rows[0]["api_key"] == HEALTH_SERVICE_ACCOUNT
    tags: Final = rows[0]["request_tags"]
    assert HEALTH_SERVICE_ACCOUNT in (tags if isinstance(tags, list) else json.loads(str(tags)))


def test_an_embedding_probe_is_logged_with_the_embedding_call_type(rig: Rig) -> None:
    """The loop probes `spend-embed-<run>`, a deployment with `model_info.mode: embedding`.

    Its health account spend row must have `call_type` `aembedding` and `model_id` `hc-ok-spend-embed-<run>`
    (#50987: before it the probe was logged as `acompletion` and could not be priced as an embedding).
    """
    rows: Final = eventually(
        lambda: _probe_rows(rig, f"hc-{EMBED_MODEL}"), lambda found: len(found) >= 1, seconds=CONVERGENCE_SECONDS
    )
    assert rows[0]["call_type"] == "aembedding"


def test_a_probe_of_a_custom_priced_deployment_is_priced_like_the_deployment(rig: Rig) -> None:
    """`spend-chat-<run>` sets `input_cost_per_token: 0.001` and `output_cost_per_token: 0.002`; the upstream
    reports 4 prompt and 2 completion tokens.

    Its probe's spend row must have `spend` 0.008 (4 * 0.001 + 2 * 0.002): the probe is visible in spend, on
    the health account. UNVERIFIED that the deployment's custom price reaches the probe's logging object.
    """
    rows: Final = eventually(
        lambda: _probe_rows(rig, f"hc-{PRICED_MODEL}"), lambda found: len(found) >= 1, seconds=CONVERGENCE_SECONDS
    )
    assert float(str(rows[0]["spend"])) == pytest.approx(4 * 0.001 + 2 * 0.002)


def test_a_key_that_calls_health_is_never_charged_for_the_probe(rig: Rig) -> None:
    """A fresh key allowed `spend-chat-<run>` calls `GET /health?model=spend-chat-<run>` three times (in
    background mode the answer comes from the cache, so it triggers no probe of its own).

    After the probes of the period are logged, the key's `spend` in `LiteLLM_VerificationToken` must still be
    0 and no `LiteLLM_SpendLogs` row may carry the key's hashed token.
    """
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(models=[PRICED_GROUP])
        for _ in range(3):
            assert health(rig.gateway, key=key, model=PRICED_GROUP).status_code in (200, 503)
        eventually(
            lambda: _probe_rows(rig, f"hc-{PRICED_MODEL}"), lambda found: len(found) >= 2, seconds=CONVERGENCE_SECONDS
        )
        token: Final = object_value(rig.gateway.get("/key/info", params={"key": key})["info"])
        assert float(str(token.get("spend") or 0)) == 0.0
        hashed: Final = hashlib.sha256(key.encode()).hexdigest()
        assert rig.gateway.rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE api_key = %s', (hashed,)) == []


def test_the_health_account_is_never_created_as_a_team(rig: Rig) -> None:
    """Probes are logged with `team_id: litellm-internal-health-check`.

    After probe rows exist, `LiteLLM_TeamTable` must have no row with that `team_id`: the spend writer must
    not upsert a team for the service account. UNVERIFIED: read from the team spend update path, which only
    updates existing rows.
    """
    eventually(
        lambda: _probe_rows(rig, f"hc-{PRICED_MODEL}"), lambda found: len(found) >= 1, seconds=CONVERGENCE_SECONDS
    )
    assert (
        rig.gateway.rows('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id = %s', (HEALTH_SERVICE_ACCOUNT,)) == []
    )


def test_the_spend_log_ui_can_exclude_internal_health_checks(rig: Rig) -> None:
    """After probe rows for `hc-ok-spend-chat-<run>` exist, the master key calls `GET /spend/logs/ui` for the
    last hour with `page_size=1000`, once plainly and once with `exclude_internal_health_checks=true`.

    The plain answer must include rows whose `api_key` is `litellm-internal-health-check`; the excluding
    answer must include none. UNVERIFIED: the `data` list name and the `YYYY-MM-DD HH:MM:SS` date format.
    """
    eventually(
        lambda: _probe_rows(rig, f"hc-{PRICED_MODEL}"), lambda found: len(found) >= 1, seconds=CONVERGENCE_SECONDS
    )
    now: Final = datetime.now(UTC)
    window: Final = {
        "start_date": (now - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": (now + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
        "page_size": "1000",
    }

    def health_keys(extra: dict[str, str]) -> list[JsonValue]:
        response = rig.gateway.request("GET", "/spend/logs/ui", params={**window, **extra})
        assert response.status_code == 200, response.text
        return [object_value(row).get("api_key") for row in list_value(report(response)["data"])]

    assert HEALTH_SERVICE_ACCOUNT in health_keys({})
    assert HEALTH_SERVICE_ACCOUNT not in health_keys({"exclude_internal_health_checks": "true"})


def test_failing_background_probes_post_no_slack_alert(rig: Rig) -> None:
    """`spend-failing-<run>` answers 500 to every probe, and the outage thresholds are 2 (minor) and 3 (major).

    After at least five failed probes, and for another 10 seconds, the Slack webhook must have received no post
    that names `status-500-spend-<run>`: probes skip the proxy's failure hook (no `llm_exceptions`) and carry no
    deployment id for the outage counter. Read from code; a port that routes probes through the normal request
    pipeline would start paging on every unhealthy deployment every interval.
    """
    eventually(lambda: len(rig.upstream.probes(FAILING_MODEL)), lambda count: count >= 5, seconds=CONVERGENCE_SECONDS)
    assert (
        held(
            lambda: [text for text in _slack_texts(rig) if FAILING_MODEL in text],
            lambda found: found == [],
            holding=10,
            seconds=30,
        )
        == []
    )


def test_a_client_call_to_the_same_failing_deployment_does_post_an_alert(rig: Rig) -> None:
    """Control for the previous test: a client chat request for `spend-failing-<run>` returns the provider's 500.

    Within 60 seconds the Slack webhook must receive a post whose text contains `llm_exceptions`, so the silence
    above is about probes, not about a broken webhook.
    """
    response: Final = chat(rig.gateway, FAILING_GROUP)
    assert response.status_code == 500, response.text
    eventually(
        lambda: [text for text in _slack_texts(rig) if "llm_exceptions" in text],
        lambda found: len(found) >= 1,
        seconds=60,
    )
