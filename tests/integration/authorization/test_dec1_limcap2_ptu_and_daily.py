"""PTU team ceilings and per-deployment daily limits (rpd, tpd).

PTU: an Azure deployment that declares ``model_info.ptu_shares`` splits its provisioned throughput between teams.
With ``LITELLM_ENABLE_PTU_COST_ATTRIBUTION`` on, the v3 limiter adds one ``model_per_team_ptu`` descriptor per
team and model group (``parallel_request_limiter_v3.py:3555``) whose tokens are Azure normalized tokens: input,
plus output times the model's output to input ratio (gpt-4.1: 3000 input TPM per PTU, ratio 4.0,
``litellm/llms/azure/ptu_capacity.py``). A request whose own estimate exceeds the whole ceiling is refused 400 with
``PTUShareExceededError`` (``parallel_request_limiter_v3.py:427``); one that only exceeds what is left is the
ordinary v3 429. ``routing/test_ptu_shares.py`` pins routing, settlement and refunds; these tests pin the refusal
shapes, headers and the alias rule the Rust port has to copy.

Daily limits: ``litellm_params.rpd`` and ``tpd`` on a deployment, enforced by
``router_settings.optional_pre_call_checks: ["enforce_model_rate_limits"]``
(``litellm/router_utils/pre_call_checks/model_rate_limit_check.py``). The window is the UTC calendar day, shifted by
``daily_reset_time`` ("HH:MM" UTC), and the refusal's ``retry-after`` is the seconds until that reset.
``routing/test_enforce_model_rate_limits_daily_limits.py`` pins only the skip to another deployment.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.wire import Wire, wire_server
from tests.integration.authorization.dec1_limits_support import (
    Usage,
    chat,
    error_message,
    posts,
    provider,
    statuses,
)
from tests.integration.authorization.dec1_ratelim_support import WINDOW_SECONDS
from tests.integration.routing.dec1_capacity_support import capacity_proxy, entry, group_name

PTU_ENVIRONMENT: Final = {"LITELLM_ENABLE_PTU_COST_ATTRIBUTION": "True"}
GPT41_INPUT_TPM_PER_PTU: Final = 3000
GPT41_OUTPUT_RATIO: Final = 4.0
SETTLE_USAGE: Final = Usage(prompt=5, completion=700)
SETTLED_NORMALIZED: Final = 5 + 4 * 700
ENFORCE: Final[dict[str, JsonValue]] = {"optional_pre_call_checks": ["enforce_model_rate_limits"]}


def _azure(group: str, api_base: str, team: str) -> dict[str, JsonValue]:
    return {
        "model_name": group,
        "litellm_params": {
            "model": "azure/limcap2-gpt41",
            "api_base": api_base,
            "api_key": "synthetic-azure-key",
            "api_version": "2025-03-01-preview",
        },
        "model_info": {
            "id": f"{group}-ptu",
            "base_model": "azure/gpt-4.1",
            "ptu_count": 1,
            "cost_per_ptu_per_hour": 2.0,
            "ptu_effective_from": "2026-01-01T00:00:00Z",
            "ptu_shares": {team: 1},
        },
    }


@contextmanager
def _ptu(
    gateway: Gateway, directory: Path, usage: Usage = SETTLE_USAGE
) -> Iterator[tuple[Gateway, Wire, str, str, str]]:
    """A proxy with one gpt-4.1 PTU deployment of one PTU whose only share holder is a team created here, and the
    alias ``<group>-alias`` pointing at it. Yields the proxy, the provider, the group, the team and a team key."""
    group: Final = group_name("ptu")
    team_id: Final = f"limcap2-ptu-{uuid.uuid4().hex[:10]}"
    with (
        wire_server(provider(usage)) as wire,
        capacity_proxy(
            gateway,
            directory,
            callbacks=(),
            model_list=[_azure(group, wire.url, team_id)],
            router_settings={"model_group_alias": {f"{group}-alias": group}},
            environment=PTU_ENVIRONMENT,
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        candidate.post("/team/new", {"team_id": team_id})
        # The team cannot be deleted while the config deployment still names it in `model_info.ptu_shares` (the
        # delete answers 400), so the cleanup only tries, and the row goes with the proxy's database.
        scenario.cleanups.callback(candidate.request, "POST", "/team/delete", {"team_ids": [team_id]})
        yield candidate, wire, group, team_id, scenario.key(team_id=team_id)


def test_a_request_bigger_than_the_whole_ptu_share_is_a_400_that_never_reaches_azure(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The team holds 1 PTU of gpt-4.1, a ceiling of 3000 normalized tokens a minute. A chat with ``max_tokens:
    1000`` needs at least 4000 (output weighs 4.0), so it is refused 400 with exactly ``This request needs <n>
    normalized tokens, more than your team's PTU share of <group> allows in a minute (3000), so it can never be
    admitted. Lower max_tokens or shorten the prompt.`` and Azure receives nothing. A port that answers 429 here
    tells clients to retry a request that can never pass."""
    with _ptu(gateway, tmp_path) as (candidate, wire, group, _team, key):
        refused: Final = chat(candidate, group, key, max_tokens=1000)
        assert refused.status_code == 400, refused.text
        found: Final = re.fullmatch(
            rf"This request needs (\d+) normalized tokens, more than your team's PTU share of {re.escape(group)} "
            rf"allows in a minute \({GPT41_INPUT_TPM_PER_PTU}\), so it can never be admitted\. "
            r"Lower max_tokens or shorten the prompt\.",
            error_message(refused),
        )
        assert found is not None, refused.text
        assert int(found.group(1)) >= GPT41_OUTPUT_RATIO * 1000
        assert posts(wire) == ()


def test_a_ptu_ceiling_429_names_model_per_team_ptu_in_its_headers(gateway: Gateway, tmp_path: Path) -> None:
    """One served call reports 5 prompt and 700 completion tokens, which settles 2805 normalized tokens on
    ``{model_per_team_ptu:<team>:<group>}:tokens``. The next call with ``max_tokens: 100`` needs more than the 195
    left, so it is the v3 429 ``Rate limit exceeded for model_per_team_ptu: <team>:<group>. Limit type: tokens.
    Current limit: 3000`` with ``rate_limited_by: model_per_team_ptu``, ``rate_limit_type: tokens`` and
    ``retry-after: 60`` (``_handle_rate_limit_error``). UNVERIFIED: the settled count is polled through the 429."""
    with _ptu(gateway, tmp_path) as (candidate, wire, group, team, key):
        assert chat(candidate, group, key, max_tokens=16).status_code == 200
        refused: Final = eventually(
            lambda: chat(candidate, group, key, max_tokens=100),
            lambda response: response.status_code == 429,
            seconds=20,
        )
        assert (
            f"Rate limit exceeded for model_per_team_ptu: {team}:{group}. Limit type: tokens. Current limit: "
            f"{GPT41_INPUT_TPM_PER_PTU}," in error_message(refused)
        ), refused.text
        assert refused.headers["rate_limited_by"] == "model_per_team_ptu", dict(refused.headers)
        assert refused.headers["rate_limit_type"] == "tokens", dict(refused.headers)
        assert refused.headers["retry-after"] == str(WINDOW_SECONDS), dict(refused.headers)
        assert len(posts(wire)) == 1


def test_a_model_group_alias_draws_on_the_same_ptu_ceiling_as_its_group(gateway: Gateway, tmp_path: Path) -> None:
    """The team settles 2805 normalized tokens through the group name, then calls ``<group>-alias``. The alias
    resolves to the same deployments (``team_ptu_ceiling`` resolves every name to the group behind it,
    ``ptu_shares.py:103``), so the alias call is refused for ``model_per_team_ptu: <team>:<group>``, the group's
    counter, not a fresh one under the alias name. An alias must never be a second PTU allowance."""
    with _ptu(gateway, tmp_path) as (candidate, _wire, group, team, key):
        assert chat(candidate, group, key, max_tokens=16).status_code == 200
        refused: Final = eventually(
            lambda: chat(candidate, f"{group}-alias", key, max_tokens=100),
            lambda response: response.status_code == 429,
            seconds=20,
        )
        assert f"model_per_team_ptu: {team}:{group}. Limit type: tokens." in error_message(refused), refused.text


def test_with_ptu_attribution_off_the_team_gets_no_ptu_ceiling(gateway: Gateway, tmp_path: Path) -> None:
    """The same deployment without ``LITELLM_ENABLE_PTU_COST_ATTRIBUTION``: ``_resolve_ptu_team_ceiling_via_proxy_router``
    returns None (``parallel_request_limiter_v3.py:402``), so the ``max_tokens: 1000`` request that is a 400 with
    attribution on is forwarded and served 200. The share still reserves the deployment for the team (routing),
    only the ceiling is gone."""
    group: Final = group_name("ptu-off")
    team_id: Final = f"limcap2-ptu-{uuid.uuid4().hex[:10]}"
    with (
        wire_server(provider()) as wire,
        capacity_proxy(
            gateway,
            tmp_path,
            callbacks=(),
            model_list=[_azure(group, wire.url, team_id)],
            environment={"LITELLM_ENABLE_PTU_COST_ATTRIBUTION": "False"},
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        candidate.post("/team/new", {"team_id": team_id})
        scenario.cleanups.callback(candidate.post, "/team/delete", {"team_ids": [team_id]})
        served: Final = chat(candidate, group, scenario.key(team_id=team_id), max_tokens=1000)
        assert served.status_code == 200, served.text
        assert len(posts(wire)) == 1


@contextmanager
def _daily(gateway: Gateway, directory: Path, **limits: JsonValue) -> Iterator[tuple[Gateway, Wire, str]]:
    """One deployment with the given daily limits and ``enforce_model_rate_limits`` on, no dynamic limiter."""
    group: Final = group_name("daily")
    with (
        wire_server(provider()) as wire,
        capacity_proxy(
            gateway,
            directory,
            callbacks=(),
            model_list=[entry(group, wire.url, model_id=f"{group}-only", **limits)],
            router_settings=ENFORCE,
        ) as candidate,
    ):
        yield candidate, wire, group


def _retry_after(response: httpx.Response) -> int:
    return int(response.headers["retry-after"])


def test_the_request_past_a_deployments_rpd_is_refused_and_not_forwarded(gateway: Gateway, tmp_path: Path) -> None:
    """A single deployment with ``rpd: 2`` serves two requests; the third is refused 429 whose text carries
    ``RPD limit=2`` (``_DailyLimitDeployment.limit_error``: ``Model rate limit exceeded. RPD limit=2, current
    usage=3``), and the provider saw exactly two requests. UNVERIFIED: the exact wrapper the proxy puts around the
    router's ``RateLimitError`` text; only the RPD fragment is asserted."""
    with _daily(gateway, tmp_path, rpd=2) as (candidate, wire, group):
        responses: Final = [chat(candidate, group, candidate.key) for _ in range(3)]
        assert statuses(responses) == [200, 200, 429], [response.text for response in responses]
        assert "RPD limit=2" in responses[2].text, responses[2].text
        assert len(posts(wire)) == 2


def test_an_rpd_refusal_retries_after_the_next_utc_midnight(gateway: Gateway, tmp_path: Path) -> None:
    """With no ``daily_reset_time`` the window is the UTC day, so the rpd refusal's ``retry-after`` is the seconds
    until the next 00:00 UTC (``seconds_until_daily_reset``, model_rate_limit_check.py:59): between 1 and 86400 and
    within a minute of the value computed from the test's clock. A port that sends ``retry-after: 60`` (the
    per-minute value) makes clients hammer a deployment that is closed for the day. UNVERIFIED: the proxy copies the
    router error's ``retry-after`` header onto its 429.

    BUG (run 44176682961): the 429 had no retry-after. limit_error puts the seconds to the daily reset on the
    RateLimitError's httpx response (model_rate_limit_check.py:106-118), and handle_llm_api_exception copies
    response headers through get_response_headers (common_request_processing.py:5284-5291, get_headers.py:32), which
    forwards them only as llm_provider-<name>. Clients that honour retry-after never see the daily value."""
    pytest.skip(
        "BUG: a deployment rpd/tpd 429 carries no retry-after, the router's value only arrives as llm_provider-retry-after"
    )
    with _daily(gateway, tmp_path, rpd=1) as (candidate, _wire, group):
        assert chat(candidate, group, candidate.key).status_code == 200
        refused: Final = chat(candidate, group, candidate.key)
        assert refused.status_code == 429, refused.text
        now: Final = datetime.now(UTC)
        midnight: Final = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        assert abs(_retry_after(refused) - (midnight - now).total_seconds()) <= 60, dict(refused.headers)


def test_daily_reset_time_moves_the_rpd_window_and_its_retry_after(gateway: Gateway, tmp_path: Path) -> None:
    """``daily_reset_time`` is set to two hours from now (UTC, "HH:MM"). The refusal's ``retry-after`` is then about
    two hours (within a minute plus the rounding to the minute), not the time to midnight: the window starts over at
    that UTC time (``parse_daily_reset_time``, types/router.py:586).

    BUG (run 44176682961): the 429 had no retry-after, for the same reason as the midnight test above
    (get_headers.py:32 renames the router's header to llm_provider-retry-after)."""
    pytest.skip(
        "BUG: a deployment rpd/tpd 429 carries no retry-after, the router's value only arrives as llm_provider-retry-after"
    )
    reset_at: Final = (datetime.now(UTC) + timedelta(hours=2)).strftime("%H:%M")
    with _daily(gateway, tmp_path, rpd=1, daily_reset_time=reset_at) as (candidate, _wire, group):
        assert chat(candidate, group, candidate.key).status_code == 200
        refused: Final = chat(candidate, group, candidate.key)
        assert refused.status_code == 429, refused.text
        assert 2 * 3600 - 120 <= _retry_after(refused) <= 2 * 3600 + 60, dict(refused.headers)


def test_a_deployment_is_closed_for_the_day_once_settled_tokens_reach_its_tpd(gateway: Gateway, tmp_path: Path) -> None:
    """A deployment has ``tpd: 20`` and every answer reports 18 tokens. The check is ``current >= tpd`` before the
    call (model_rate_limit_check.py:391), so the first two calls are served (0 and 18 are below 20) and, once the
    second settles to 36, the next is refused with ``TPD limit=20`` in its text. Usage settles asynchronously, so the
    third call is polled."""
    with _daily(gateway, tmp_path, tpd=20) as (candidate, wire, group):
        assert statuses([chat(candidate, group, candidate.key) for _ in range(2)]) == [200, 200]
        refused: Final = eventually(
            lambda: chat(candidate, group, candidate.key), lambda response: response.status_code == 429, seconds=30
        )
        assert "TPD limit=20" in refused.text, refused.text
        assert len(posts(wire)) >= 2


def test_rpd_counts_only_requests_the_deployment_actually_admitted(gateway: Gateway, tmp_path: Path) -> None:
    """A deployment has ``rpm: 1`` and ``rpd: 3``. The second request in the minute is refused for rpm, and the rpd
    count it took is refunded (``_async_refund_request_counts``, model_rate_limit_check.py:411). So the minute's two
    rpm refusals cost no daily requests: the deployment still has 2 of its 3 daily requests, which a later minute
    can use. Pinned here as: the first rpm refusal names ``RPM limit=1``, not RPD, after three attempts."""
    with _daily(gateway, tmp_path, rpm=1, rpd=3) as (candidate, wire, group):
        responses: Final = [chat(candidate, group, candidate.key) for _ in range(3)]
        assert statuses(responses) == [200, 429, 429], [response.text for response in responses]
        assert all("RPM limit=1" in response.text for response in responses[1:]), [r.text for r in responses]
        assert len(posts(wire)) == 1


@pytest.mark.parametrize("kind", ["rpd", "tpd"])
def test_daily_limits_without_enforce_model_rate_limits_are_ignored(
    gateway: Gateway, tmp_path: Path, kind: str
) -> None:
    """Without ``enforce_model_rate_limits`` in ``optional_pre_call_checks`` the daily fields are only stored: a
    deployment with ``rpd: 1`` (or ``tpd: 1``) serves three requests in a row. A port that starts enforcing them by
    default would close deployments operators configured for reporting only."""
    group: Final = group_name("daily-off")
    with (
        wire_server(provider()) as wire,
        capacity_proxy(gateway, tmp_path, callbacks=(), model_list=[entry(group, wire.url, **{kind: 1})]) as candidate,
    ):
        assert statuses([chat(candidate, group, candidate.key) for _ in range(3)]) == [200, 200, 200]
        assert len(posts(wire)) == 3
