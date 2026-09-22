"""One priced request must land the same response_cost on every spend surface.

A customer reconciles the bill from whichever surface they look at: the spend
log row, the key's and the team's rolled-up spend on /key/info and /team/info,
the usage page's Export Usage Data CSV (the dashboard serializes the
/user/daily/activity/aggregated rows it already holds; there is no server-side
CSV endpoint), and the litellm_spend_metric counter Prometheus scrapes. Each is
written by a different writer (the spend log insert, the key and team rollups in
db_spend_update_writer, the daily spend tables, the Prometheus success callback),
so one of them can drift without the others noticing: the cause of the
key-versus-log mismatch in LIT-3620 and the export-versus-console mismatch in
LIT-5045. The deployment carries its own per-token rates, so the expected cost
is computed from the returned usage rather than read off any one surface, and
every surface is held to that number.

/metrics is per pod, so every replica the stack exports (PROXY_REPLICA_URLS) is
scraped directly and the samples merged; a stack that exports only its balancer
is scraped there until the pod that served the call answers. The request itself
is sent once.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from itertools import groupby
from math import isclose
from types import MappingProxyType
from typing import Final

import pytest
from e2e_config import provider_edge_base, unique_marker
from e2e_http import ProbeResult
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, KeyGenerateBody, LiteLLMParamsBody, TeamNewBody
from prometheus_client.parser import text_string_to_metric_families
from proxy_client import Converged, await_converged
from spend_e2e_client import SpendClient, unwrap
from spend_reconciliation import INPUT_RATE, OUTPUT_RATE

pytestmark = pytest.mark.e2e

SPEND_METRIC: Final = "litellm_spend_metric_total"
KEY_HASH_LABEL: Final = "hashed_api_key"
TEAM_LABEL: Final = "team"

SeriesLabels = tuple[tuple[str, str], ...]


def _spend_series_for_key(scrapes: Mapping[str, ProbeResult], token: str) -> Mapping[SeriesLabels, float]:
    samples: Final = sorted(
        (tuple(sorted(sample.labels.items())), sample.value)
        for scrape in scrapes.values()
        if scrape.status_code == 200
        for family in text_string_to_metric_families(scrape.body)
        for sample in family.samples
        if sample.name == SPEND_METRIC and sample.labels.get(KEY_HASH_LABEL) == token
    )
    return MappingProxyType(
        {labels: sum(value for _, value in group) for labels, group in groupby(samples, key=lambda sample: sample[0])}
    )


def _poll_spend_series_for_key(client: SpendClient, token: str) -> Mapping[SeriesLabels, float]:
    outcome: Final = await_converged(
        client.scrape_metrics,
        converged=lambda scrapes: bool(_spend_series_for_key(scrapes, token)),
        timeout=client.proxy.poll_timeout,
        interval=client.proxy.poll_interval,
        now=time.monotonic,
        sleep=time.sleep,
    )
    scrapes: Final = outcome.result if isinstance(outcome, Converged) else outcome.last_result
    assert _spend_series_for_key(scrapes, token), (
        f"{SPEND_METRIC} never exposed a series for {KEY_HASH_LABEL}={token} on any replica; "
        f"last scrape status per replica: {({replica: scrape.status_code for replica, scrape in scrapes.items()})}"
    )
    return _spend_series_for_key(scrapes, token)


def _same_spend(actual: float | None, expected: float) -> bool:
    return actual is not None and isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-9)


class TestSpendSurfaceConsistency:
    @pytest.mark.replayable
    @pytest.mark.covers("quota_management.spend_tracking.surface_consistency.matches_every_surface")
    def test_one_request_lands_the_same_spend_on_every_surface(
        self, client: SpendClient, resources: ResourceManager
    ) -> None:
        started: Final = datetime.now(timezone.utc)
        marker: Final = unique_marker()
        base: Final = provider_edge_base("openai")
        model: Final = f"e2e-spend-surfaces-{marker}"
        model_id: Final = client.proxy.create_model(
            model,
            LiteLLMParamsBody(
                model="openai/gpt-5.6-luna",
                api_key="os.environ/OPENAI_API_KEY",
                api_base=None if base is None else f"{base}/v1",
                input_cost_per_token=INPUT_RATE,
                output_cost_per_token=OUTPUT_RATE,
            ),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        team_id: Final = client.proxy.create_team(TeamNewBody(team_alias=f"e2e-spend-surfaces-{marker}"))
        resources.defer(lambda: client.proxy.delete_team(team_id))
        record: Final = client.generate_key_record(
            KeyGenerateBody(team_id=team_id, models=[model], key_alias=f"e2e-spend-surfaces-{marker}")
        )
        resources.defer(lambda: client.proxy.delete_key(record.key))
        assert record.token, "/key/generate answered without the key's token hash"
        token: Final = record.token

        response: Final = unwrap(
            client.proxy.chat(
                record.key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content=f"Reply with one word. {marker}")],
                    max_completion_tokens=128,
                ),
            )
        )
        usage: Final = response.usage
        assert response.id, "successful response must have an ID"
        assert usage is not None and usage.prompt_tokens and usage.completion_tokens, f"no billable usage: {usage}"
        expected: Final = usage.prompt_tokens * INPUT_RATE + usage.completion_tokens * OUTPUT_RATE

        rows: Final = client.proxy.poll_logs_for_request_id(response.id)
        assert len(rows) == 1, f"expected one spend row for {response.id}, saw {len(rows)}: {rows}"
        row: Final = rows[0]
        assert row.api_key == token, f"spend row keyed by {row.api_key}, not the key's token hash {token}"
        assert row.team_id == team_id, f"spend row attributed to team {row.team_id}, not {team_id}"
        assert row.status == "success", f"spend row status {row.status}"

        key_spend: Final = client.poll_key_spend(record.key, minimum=expected * 0.999999)
        team_spend: Final = client.poll_team_spend(team_id, minimum=expected * 0.999999)
        export_row: Final = client.poll_usage_export_row_for_key(
            token, start=started - timedelta(days=1), end=datetime.now(timezone.utc), min_requests=1
        )
        assert export_row is not None, f"/user/daily/activity/aggregated never listed key {token} under api_keys"
        series: Final = _poll_spend_series_for_key(client, token)
        off_team: Final = tuple(labels for labels in series if dict(labels).get(TEAM_LABEL) != team_id)
        assert not off_team, f"{SPEND_METRIC} series for the key carry a team other than {team_id}: {off_team}"

        observed: Final = MappingProxyType(
            {
                "/spend/logs row": row.spend,
                "/key/info spend": key_spend,
                "/team/info spend": team_spend,
                "usage export row (/user/daily/activity/aggregated)": export_row.metrics.spend,
                SPEND_METRIC: sum(series.values()),
            }
        )
        drifted: Final = tuple(surface for surface, spend in observed.items() if not _same_spend(spend, expected))
        assert not drifted, (
            f"response_cost {expected} (usage {usage.prompt_tokens}x{INPUT_RATE} + "
            f"{usage.completion_tokens}x{OUTPUT_RATE}) drifted on {drifted}; every surface: {dict(observed)}"
        )
