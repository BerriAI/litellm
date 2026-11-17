"""Model capacity (``dynamic_rate_limiter_v3``) on the paths the first capacity wave left open.

Names: the pre-call hook looks up ``registry().group(data["model"])`` with the name the caller sent
(dynamic_rate_limiter_v3.py:1688). For a router ``model_group_alias`` that lookup returns the target group's info
(``_core_get_model_group_info``, litellm/router_rust/surface/model_info.py:294), but the model-wide counter is
keyed by the requested name (``_create_model_tracking_descriptor`` uses ``value=model``,
dynamic_rate_limiter_v3.py:1175), and the proxy leaves ``data["model"]`` as the alias until the router rewrites it
(litellm_pre_call_utils.py:3787). So an alias is not a full bypass (it is held to the group's rpm under its own name)
but it is a second window: the alias and the group together can use twice the capacity. Borrower shares already
resolve the alias (``_borrowing_shares`` uses ``rate_limited_model``), which shows the intent. The intended
behaviour is written and skipped as BUG; the current per-name behaviour is pinned beside it.

Settlement: the success callback adds the reported usage to ``{model_saturation_check:<group>}:tokens``
(dynamic_rate_limiter_v3.py:1785). ``async_post_call_failure_hook`` gives back a borrower's held tokens when a later
hook refuses after admission (dynamic_rate_limiter_v3.py:1909), and every settlement of a hold is window guarded
(``_window_guarded``, dynamic_rate_limiter_v3.py:303), so a hold taken in one window never moves the next window.

Every test owns its proxy through ``capacity_proxy`` from ``dec1_capacity_support`` and its provider.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_limcap2_support import (
    STREAM_INPUT_TOKENS,
    STREAM_OUTPUT_TOKENS,
    BatchProvider,
    anthropic_stream_provider,
    blocking_guardrail,
    messages_stream,
    submit_batch,
)
from tests.integration.authorization.dec1_limits_support import complete, counter_key, provider, redis_integer
from tests.integration.routing.dec1_capacity_support import (
    anthropic_entry,
    anthropic_messages,
    capacity_message,
    capacity_proxy,
    chat,
    entry,
    error_message,
    group_name,
    model_counter,
    posts,
    settled,
    shared_redis,
    statuses,
)

PROVIDER_TOTAL: Final = 18


def test_an_alias_and_its_group_share_one_capacity_window(gateway: Gateway, tmp_path: Path) -> None:
    """Intended: a group has ``rpm: 2`` and ``router_settings.model_group_alias`` maps ``<group>-alias`` to it. One
    request by the group name and one by the alias use the whole window, so a third by either name is refused with
    ``Model capacity reached for ...``. Today the alias has its own ``{model_saturation_check:<alias>}`` counter with
    the group's limit, so the pair can use 4 requests a minute (see module docstring for path:line)."""
    pytest.skip("BUG: a model_group_alias counts model capacity in its own window, so alias plus group get 2x rpm")
    group: Final = group_name("alias")
    with (
        wire_server(provider()) as wire,
        capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(group, wire.url, rpm=2)],
            router_settings={"model_group_alias": {f"{group}-alias": group}},
        ) as candidate,
    ):
        served: Final = [chat(candidate, group, candidate.key), chat(candidate, f"{group}-alias", candidate.key)]
        assert statuses(served) == [200, 200]
        refused: Final = chat(candidate, f"{group}-alias", candidate.key)
        assert refused.status_code == 429, refused.text
        assert "Model capacity reached for " in error_message(refused)
        assert len(posts(wire)) == 2


def test_an_alias_is_held_to_its_groups_rpm_under_its_own_name_today(gateway: Gateway, tmp_path: Path) -> None:
    """Current behaviour beside the BUG above: three requests by ``<group>-alias`` to a group with ``rpm: 2``. The
    third is refused with exactly ``Model capacity reached for <group>-alias. Priority: None, Rate limit type:
    requests, Model TPM: not configured, Model RPM: 2, Remaining: 0`` (the alias name, the group's rpm), and the
    shared Redis holds ``{model_saturation_check:<group>-alias}:requests`` = 2 while the group's own counter is
    untouched. A port that fixes the split changes the name in this text and the counter key on purpose."""
    group: Final = group_name("alias-now")
    alias: Final = f"{group}-alias"
    with (
        wire_server(provider()) as wire,
        capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(group, wire.url, rpm=2)],
            router_settings={"model_group_alias": {alias: group}},
            shared=True,
        ) as candidate,
        shared_redis() as redis,
    ):
        responses: Final = [chat(candidate, alias, candidate.key) for _ in range(3)]
        assert statuses(responses) == [200, 200, 429], [response.text for response in responses]
        assert error_message(responses[2]) == capacity_message(alias, None, "requests", rpm=2)
        assert redis_integer(redis, model_counter(alias, "requests")) == 2
        assert redis_integer(redis, model_counter(group, "requests")) is None
        assert len(posts(wire)) == 2


def test_a_name_served_by_a_wildcard_deployment_is_held_to_that_deployments_rpm(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A deployment named ``<prefix>/*`` with ``rpm: 2`` serves ``<prefix>/alpha`` through the wildcard route. The
    group lookup builds the group's info from the wildcard deployment (``_get_wildcard_deployments``), so three
    requests to ``<prefix>/alpha`` are 200, 200 and a 429 ``Model capacity reached for <prefix>/alpha.`` with ``Model
    RPM: 2``. UNVERIFIED: that the wildcard deployment's ``rpm`` survives into the merged group info."""
    prefix: Final = group_name("wild")
    with (
        wire_server(provider()) as wire,
        capacity_proxy(gateway, tmp_path, model_list=[entry(f"{prefix}/*", wire.url, rpm=2)]) as candidate,
    ):
        responses: Final = [chat(candidate, f"{prefix}/alpha", candidate.key) for _ in range(3)]
        assert statuses(responses) == [200, 200, 429], [response.text for response in responses]
        assert error_message(responses[2]) == capacity_message(f"{prefix}/alpha", None, "requests", rpm=2)


def test_two_names_under_one_wildcard_deployment_share_its_capacity(gateway: Gateway, tmp_path: Path) -> None:
    """Intended: ``<prefix>/alpha`` and ``<prefix>/beta`` are both served by the one ``<prefix>/*`` deployment with
    ``rpm: 2``, so after one request to each the third (to either) is refused for capacity. Today each concrete name
    is its own ``model_saturation_check`` window (value is the requested name), so a caller can multiply the
    deployment's capacity by inventing names under the wildcard."""
    pytest.skip("BUG: every name matched by a wildcard deployment gets its own model capacity window")
    prefix: Final = group_name("wild-two")
    with (
        wire_server(provider()) as wire,
        capacity_proxy(gateway, tmp_path, model_list=[entry(f"{prefix}/*", wire.url, rpm=2)]) as candidate,
    ):
        assert statuses(
            [chat(candidate, f"{prefix}/alpha", candidate.key), chat(candidate, f"{prefix}/beta", candidate.key)]
        ) == [200, 200]
        assert chat(candidate, f"{prefix}/gamma", candidate.key).status_code == 429
        assert len(posts(wire)) == 2


def test_a_name_the_router_does_not_know_is_not_counted_against_any_capacity(gateway: Gateway, tmp_path: Path) -> None:
    """The pre-call hook allows a request whose model has no group info (``No model group info for <m>, allowing
    request``, dynamic_rate_limiter_v3.py:1690) and writes no counter. A request for an unknown name is answered by the
    router's own error (not a capacity 429), and Redis has no ``{model_saturation_check:<name>}:requests``."""
    group: Final = group_name("unknown")
    with (
        wire_server(provider()) as wire,
        capacity_proxy(gateway, tmp_path, model_list=[entry(group, wire.url, rpm=1)], shared=True) as candidate,
        shared_redis() as redis,
    ):
        missing: Final = f"{group}-missing"
        response: Final = chat(candidate, missing, candidate.key)
        assert response.status_code != 429, response.text
        assert "Model capacity reached" not in response.text
        assert redis_integer(redis, model_counter(missing, "requests")) is None


def test_a_response_cache_hit_still_takes_a_capacity_request_and_settles_its_tokens(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The suite config caches responses. The same prompt is sent twice to a group with ``rpm: 10`` and the provider
    receives it once (the second is a cache hit). The admission check runs before the cache lookup, so
    ``{model_saturation_check:<group>}:requests`` is 2, and the hit's usage (18) is settled too, so the tokens counter
    reaches 36. UNVERIFIED: that the success callback for a cache hit carries the cached usage into
    ``async_log_success_event``; ``routing/test_ptu_shares.py`` pins the same for the PTU ceiling."""
    group: Final = group_name("cache")
    text: Final = f"limcap2 cache {group}"
    with (
        wire_server(provider()) as wire,
        capacity_proxy(gateway, tmp_path, model_list=[entry(group, wire.url, rpm=10)], shared=True) as candidate,
        shared_redis() as redis,
    ):
        assert statuses([chat(candidate, group, candidate.key, text=text) for _ in range(2)]) == [200, 200]
        assert len(posts(wire)) == 1
        assert redis_integer(redis, model_counter(group, "requests")) == 2
        assert settled(redis, model_counter(group, "tokens"), 2 * PROVIDER_TOTAL) == 2 * PROVIDER_TOTAL


def test_a_streamed_messages_call_settles_input_plus_output_tokens_on_the_model_window(
    gateway: Gateway, tmp_path: Path
) -> None:
    """An Anthropic group with ``tpm: 1000``. A ``/v1/messages`` call with ``stream: true`` is answered by a stream
    whose ``message_start`` reports 5 input and ``message_delta`` 3 output tokens. After the stream ends,
    ``{model_saturation_check:<group>}:tokens`` holds 8: the streamed usage is assembled before settlement. A port
    that settles from ``message_start`` alone would write 5."""
    group: Final = group_name("msg-stream")
    with (
        wire_server(anthropic_stream_provider()) as wire,
        capacity_proxy(
            gateway, tmp_path, model_list=[anthropic_entry(group, wire.url, tpm=1000)], shared=True
        ) as candidate,
        shared_redis() as redis,
    ):
        response: Final = messages_stream(candidate, group, candidate.key)
        assert response.status_code == 200, response.text
        assert "event: message_stop" in response.text
        total: Final = STREAM_INPUT_TOKENS + STREAM_OUTPUT_TOKENS
        assert settled(redis, model_counter(group, "tokens"), total) == total


def test_a_messages_stream_over_capacity_is_refused_before_any_event(gateway: Gateway, tmp_path: Path) -> None:
    """An Anthropic group with ``rpm: 1``. The first streamed ``/v1/messages`` call is served; the second is refused
    429 as a whole response, not as an ``error`` event inside a 200 stream, with ``Model capacity reached for
    <group>.`` in the text and no ``event:`` line. The provider receives one request."""
    group: Final = group_name("msg-cap")
    with (
        wire_server(anthropic_stream_provider()) as wire,
        capacity_proxy(gateway, tmp_path, model_list=[anthropic_entry(group, wire.url, rpm=1)]) as candidate,
    ):
        assert messages_stream(candidate, group, candidate.key).status_code == 200
        refused: Final = messages_stream(candidate, group, candidate.key)
        assert refused.status_code == 429, refused.text
        assert f"Model capacity reached for {group}." in refused.text, refused.text
        assert "event:" not in refused.text
        assert len(posts(wire)) == 1


def test_a_plain_messages_call_settles_its_usage_on_the_model_window(gateway: Gateway, tmp_path: Path) -> None:
    """Same group, no streaming: the Anthropic answer reports 11 input and 7 output tokens (the limits provider), and
    ``{model_saturation_check:<group>}:tokens`` settles to 18."""
    group: Final = group_name("msg-plain")
    with (
        wire_server(provider()) as wire,
        capacity_proxy(
            gateway, tmp_path, model_list=[anthropic_entry(group, wire.url, tpm=1000)], shared=True
        ) as candidate,
        shared_redis() as redis,
    ):
        assert anthropic_messages(candidate, group, candidate.key).status_code == 200
        assert settled(redis, model_counter(group, "tokens"), PROVIDER_TOTAL) == PROVIDER_TOTAL


def test_text_completions_count_against_model_capacity(gateway: Gateway, tmp_path: Path) -> None:
    """``/v1/completions`` goes through the same pre-call hook: with ``rpm: 1`` the second text completion is refused
    with exactly ``Model capacity reached for <group>. Priority: None, Rate limit type: requests, Model TPM: not
    configured, Model RPM: 1, Remaining: 0``."""
    group: Final = group_name("completions")
    with (
        wire_server(provider()) as wire,
        capacity_proxy(gateway, tmp_path, model_list=[entry(group, wire.url, rpm=1)]) as candidate,
    ):
        assert complete(candidate, group, candidate.key).status_code == 200
        refused: Final = complete(candidate, group, candidate.key)
        assert refused.status_code == 429, refused.text
        assert error_message(refused) == capacity_message(group, None, "requests", rpm=1)


def test_a_batch_create_naming_a_capacity_limited_model_takes_one_capacity_request(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A group with ``rpm: 1`` is the target of ``POST /v1/batches`` (body ``model: <group>``). The dynamic limiter's
    hook runs for any call type whose data has ``model``, so the first create takes the one request and the second is
    refused 429 ``Model capacity reached for <group>.`` before the provider gets a second ``POST /batches``; the batch
    is one request however many rows it has. UNVERIFIED: that the file upload (``model`` form field) is not itself
    counted; if it is, the first create already fails and this pins that instead."""
    group: Final = group_name("batch")
    with (
        wire_server(BatchProvider()) as wire,
        capacity_proxy(gateway, tmp_path, model_list=[entry(group, wire.url, rpm=1)]) as candidate,
    ):
        assert submit_batch(candidate, group, candidate.key, rows=5).status_code == 200
        refused: Final = submit_batch(candidate, group, candidate.key, rows=1)
        assert refused.status_code == 429, refused.text
        assert f"Model capacity reached for {group}." in refused.text, refused.text


def test_a_hold_released_by_a_later_hook_refusal_returns_the_teams_tokens(gateway: Gateway, tmp_path: Path) -> None:
    """Teams borrow and the model is saturated (``saturation_threshold: 0.0``), so a team with ``model_tpm_limit:
    {<m>: 1000}`` holds its estimated tokens at admission on ``{model_per_team_saturated:<team>:<m>}:tokens``. The
    team's key also runs a ``pre_call`` content filter that blocks the prompt's word, so a later hook refuses the
    admitted request. ``async_post_call_failure_hook`` gives the hold back (#51367), so the counter ends at 0 (or is
    absent) instead of keeping the estimate for the rest of the window, and the provider is never called.
    UNVERIFIED: that the guardrail runs after the dynamic limiter in the pre-call walk; if it runs first nothing is
    held and the counter is absent, which this assertion also accepts."""
    group: Final = group_name("refund")
    with (
        wire_server(provider()) as wire,
        capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(group, wire.url, tpm=100000)],
            priority_settings={
                "saturation_threshold": 0.0,
                "model_limit_borrowing_config": {"borrow_idle_capacity": ["team"]},
            },
            shared=True,
        ) as candidate,
        candidate.scenario() as scenario,
        shared_redis() as redis,
    ):
        guard: Final = blocking_guardrail(candidate, scenario, "forbiddenword")
        team: Final = scenario.team(metadata={"model_tpm_limit": {group: 1000}})
        key: Final = scenario.key(team_id=team, guardrails=[guard])
        refused: Final = chat(candidate, group, key, text="limcap2 forbiddenword probe")
        assert refused.status_code == 400, refused.text
        share: Final = counter_key("model_per_team_saturated", f"{team}:{group}", "tokens")
        assert eventually(lambda: redis_integer(redis, share), lambda value: value in (None, 0), seconds=20) in (
            None,
            0,
        )
        assert posts(wire) == ()


def test_a_hold_settled_after_its_window_ended_does_not_touch_the_next_window(gateway: Gateway, tmp_path: Path) -> None:
    """Teams borrow on a saturated model and a team with ``model_tpm_limit: {<m>: 1000}`` sends one request that the
    provider holds open. The test waits (polling Redis, no sleep) until that hold's window has expired from
    ``{model_per_team_saturated:<team>:<m>}:tokens``, then releases the answer. Settlement is window guarded
    (``WindowGuardedIncrement``), so neither the 18 used nor the refund of the estimate lands in the new window: a
    fresh request afterwards leaves the counter at exactly its own 18. A port without the guard writes 18 plus or
    minus the stale hold. Takes up to one window (60 s)."""
    group: Final = group_name("window")
    release: Final = threading.Event()
    with (
        wire_server(provider(hold=release)) as wire,
        capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(group, wire.url, tpm=100000)],
            priority_settings={
                "saturation_threshold": 0.0,
                "model_limit_borrowing_config": {"borrow_idle_capacity": ["team"]},
            },
            shared=True,
        ) as candidate,
        candidate.scenario() as scenario,
        shared_redis() as redis,
        ThreadPoolExecutor(1) as pool,
    ):
        team: Final = scenario.team(metadata={"model_tpm_limit": {group: 1000}})
        key: Final = scenario.key(team_id=team)
        share: Final = counter_key("model_per_team_saturated", f"{team}:{group}", "tokens")
        held: Final = pool.submit(chat, candidate, group, key)
        try:
            eventually(lambda: redis_integer(redis, share), lambda value: (value or 0) > 0, seconds=20)
            eventually(lambda: redis.exists(share), lambda present: present == 0, seconds=90)
        finally:
            release.set()
        assert held.result().status_code == 200
        release_free: Final = chat(candidate, group, key)
        assert release_free.status_code == 200, release_free.text
        assert settled(redis, share, PROVIDER_TOTAL, seconds=30) == PROVIDER_TOTAL
        assert len(posts(wire)) == 2


def test_a_borrowers_fallback_is_held_to_its_share_on_the_fallback_model(gateway: Gateway, tmp_path: Path) -> None:
    """Two groups: ``primary`` with ``rpm: 1`` and ``backup`` with ``rpm: 1000``, ``fallbacks: [{primary: [backup]}]``.
    Teams borrow on a saturated model, and the team has ``model_rpm_limit: {primary: 100, backup: 1}``. Request 1 is
    served by primary. Request 2 finds primary at capacity, falls back, and ``async_fallback_target_allowed`` takes
    the team's one backup request (#51082 for borrowers). Request 3 has no room on primary and the team's backup
    share is spent, so it is refused 429. The backup provider receives exactly one request. UNVERIFIED: the shape of
    the final 429 after a refused fallback walk."""
    primary: Final = group_name("fb-primary")
    backup: Final = group_name("fb-backup")
    with (
        wire_server(provider()) as primary_wire,
        wire_server(provider()) as backup_wire,
        capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(primary, primary_wire.url, rpm=1), entry(backup, backup_wire.url, rpm=1000)],
            priority_settings={
                "saturation_threshold": 0.0,
                "model_limit_borrowing_config": {"borrow_idle_capacity": ["team"]},
            },
            router_settings={"fallbacks": [{primary: [backup]}]},
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        team: Final = scenario.team(metadata={"model_rpm_limit": {primary: 100, backup: 1}})
        key: Final = scenario.key(team_id=team)
        responses: Final = [chat(candidate, primary, key) for _ in range(3)]
        assert statuses(responses) == [200, 200, 429], [response.text for response in responses]
        assert (len(posts(primary_wire)), len(posts(backup_wire))) == (1, 1)


def test_the_legacy_v1_dynamic_rate_limiter_is_still_selectable_and_refuses_at_zero_available_rpm(
    gateway: Gateway, tmp_path: Path
) -> None:
    """``litellm_settings.callbacks: ["dynamic_rate_limiter"]`` still loads the v1 handler
    (``custom_logger_registry.py:111``, ``litellm/proxy/hooks/dynamic_rate_limiter.py``). With a group of ``rpm: 1``
    the first request is served; once the router's usage shows the minute used, a request is refused 429 with a text
    of the v1 form ``Key=<hashed key> over available RPM=0. Model RPM=0, Active keys=<n>``. Pinned so the Rust
    migration removes v1 on purpose (it is not a v3 alias). UNVERIFIED: that ``llm_router.health().usage`` reports
    the served request within the minute; the refusal is polled."""
    group: Final = group_name("v1")
    with (
        wire_server(provider()) as wire,
        capacity_proxy(
            gateway, tmp_path, callbacks=("dynamic_rate_limiter",), model_list=[entry(group, wire.url, rpm=1)]
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        key: Final = scenario.key()
        assert chat(candidate, group, key).status_code == 200
        refused: Final = eventually(
            lambda: chat(candidate, group, key), lambda response: response.status_code == 429, seconds=30
        )
        assert "over available RPM=0. Model RPM=0, Active keys=" in error_message(refused), refused.text
        assert "Model capacity reached" not in refused.text
