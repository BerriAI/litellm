"""Settlement corner cases of the v3 limiter, the headers it merges, and the management validation in front of it.

The contract:

- Settlement (``_build_success_event_pipeline_operations``, parallel_request_limiter_v3.py:5094-5247) skips calls
  stamped with ``metadata.internal_call_origin`` (other than ``autorouter_compaction``). A caller cannot use that
  exemption: ``internal_call_origin`` is a proxy-owned metadata field stripped from client requests
  (``_UNTRUSTED_METADATA_CONTROL_FIELDS``, litellm/proxy/litellm_pre_call_utils.py:587).
- When usage has no input/output split (prompt and completion both 0) the window is charged ``total_tokens`` under
  every ``token_rate_limit_type`` (``_aggregate_only_total_tokens``, parallel_request_limiter_v3.py:4578-4604).
- A rerank reserves nothing on a key's token window (no ``messages``, ``prompt`` or ``input`` for its call type) and
  its usage is not a modeled response usage, so the key token window is never charged for it; it still takes one
  request of the request window.
- Response headers are merged by descriptor word and limit kind only
  (``_merge_ratelimit_statuses_into_additional_headers``, parallel_request_limiter_v3.py:4750-4764), so two limited
  tags on one request write the same ``x-ratelimit-tag-*`` names and the last tag's numbers win.
- ``/team/new`` only accepts ``rpm_limit_type`` / ``tpm_limit_type`` of ``guaranteed_throughput`` or
  ``best_effort_throughput`` and non-negative limits (litellm/proxy/_types.py:1995-2021), and a deployment's
  ``daily_reset_time`` must be ``HH:MM`` in UTC (litellm/types/router.py:586-593 and 709-714).

Earlier rounds listed these as "Not covered" or "Suspected bugs" (test_dec1_ratelim_.md, test_dec1_limcap2_.md).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_limcap3_support import (
    TOTAL_ONLY_TOKENS,
    key_tokens_settle_to,
    rerank,
    rerank_model,
    rerank_provider,
    total_only_provider,
)
from tests.integration.authorization.dec1_limits_support import (
    DEFAULT_USAGE,
    assert_limited,
    chat,
    counter_key,
    hashed,
    limits_proxy,
    posts,
    provider,
    redis_integer,
    scope_headers,
    shared_redis,
)
from tests.integration.authorization.dec1_ratelim_support import chat_with, tag, tagged_chat


def test_a_client_supplied_internal_call_origin_does_not_exempt_the_request_from_settlement(gateway: Gateway) -> None:
    """A key with tpm_limit=10000 sends a chat request with ``metadata: {internal_call_origin:
    autorouter_classifier}``. The field is stripped as proxy-owned (litellm_pre_call_utils.py:587), so settlement is
    not skipped (parallel_request_limiter_v3.py:5107-5111) and ``{api_key:<hash>}:tokens`` settles to the reported
    11 + 7 = 18. A port that trusts the client's field turns token limits off for anyone who sets it."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=10000)
        served: Final = chat_with(
            gateway, model, key, metadata={"internal_call_origin": "autorouter_classifier"}, max_tokens=40
        )
        assert served.status_code == 200, served.text
        assert key_tokens_settle_to(key, DEFAULT_USAGE.total) == DEFAULT_USAGE.total


def test_a_client_supplied_internal_call_origin_never_reaches_the_provider_metadata(gateway: Gateway) -> None:
    """Same request; the body the provider receives must not carry ``internal_call_origin`` anywhere in a
    ``metadata`` object, which shows the strip ran before any callback could read it. UNVERIFIED: that the OpenAI
    transformation forwards no ``metadata`` at all, in which case this holds trivially."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=10000)
        served: Final = chat_with(gateway, model, key, metadata={"internal_call_origin": "autorouter_classifier"})
        assert served.status_code == 200, served.text
        (forwarded,) = posts(wire)
        assert "internal_call_origin" not in str(forwarded.get("metadata", {}))


def test_usage_with_only_a_total_is_charged_in_full_under_output_token_rate_limit_type(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A proxy with ``general_settings.token_rate_limit_type: output`` serves a key with tpm_limit=10000 from a provider
    whose usage is ``{prompt_tokens: 0, completion_tokens: 0, total_tokens: 25}``. Output is 0, so the window is
    charged the aggregate 25 instead (``_aggregate_only_total_tokens``, parallel_request_limiter_v3.py:4578-4604).
    UNVERIFIED: that ``Usage`` keeps ``total_tokens: 25`` when both parts are 0."""
    with (
        wire_server(total_only_provider()) as wire,
        limits_proxy(gateway, tmp_path, general_settings={"token_rate_limit_type": "output"}, shared=True) as owned,
        owned.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=10000)
        assert chat(owned, model, key, max_tokens=40).status_code == 200
        assert key_tokens_settle_to(key, TOTAL_ONLY_TOKENS) == TOTAL_ONLY_TOKENS


def test_a_rerank_takes_one_request_of_the_keys_request_window(gateway: Gateway) -> None:
    """A key with rpm_limit=1 calls ``/v1/rerank`` twice on a Cohere deployment. The first is served and the second is
    the ``api_key`` requests 429 with limit 1; the provider receives one rerank call. Rerank runs the proxy pre-call
    hook like every generation route."""
    with wire_server(rerank_provider()) as wire, gateway.scenario() as scenario:
        model: Final = rerank_model(scenario, wire)
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert rerank(gateway, model, key).status_code == 200
        assert_limited(rerank(gateway, model, key), "api_key", hashed(key), "requests", 1)
        assert len(wire.drain()) == 1


def test_a_rerank_is_never_charged_to_the_keys_token_window(gateway: Gateway) -> None:
    """A key with tpm_limit=1000 sends a rerank whose provider answer reports ``meta.tokens.input_tokens: 30``, then a
    chat request with ``max_tokens: 20``. The rerank reserved nothing (its call type reads no ``messages``, ``prompt``
    or ``input``, parallel_request_limiter_v3.py:1365-1380) and its ``RerankResponse`` is not a modeled usage, so
    settlement charges 0 (parallel_request_limiter_v3.py:5120-5138); the token counter ends at the chat's 18 alone.
    Pinned as current behaviour: ``_resolve_rerank_token_usage`` exists but only feeds project ITPM/OTPM and PTU."""
    with wire_server(rerank_provider()) as wire, gateway.scenario() as scenario:
        model: Final = rerank_model(scenario, wire)
        chat_model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model, chat_model], tpm_limit=1000)
        reranked: Final = rerank(gateway, model, key)
        assert reranked.status_code == 200, reranked.text
        assert chat(gateway, chat_model, key, max_tokens=20).status_code == 200
        assert key_tokens_settle_to(key, DEFAULT_USAGE.total) == DEFAULT_USAGE.total


def test_a_rerank_settles_its_meta_input_tokens_on_a_projects_input_window(gateway: Gateway) -> None:
    """A project has ``model_itpm_limit: {<rerank model>: 100000}`` and its key reranks once; the provider reports
    ``meta.tokens.input_tokens: 30``. ``{model_per_project_itpm:<project>:<model>}:tokens`` must settle to 30
    (``_resolve_rerank_token_usage`` reads ``meta.tokens`` first, parallel_request_limiter_v3.py:4767-4782).
    UNVERIFIED: that the project input window is reserved and reconciled for the rerank call type."""
    with wire_server(rerank_provider()) as wire, gateway.scenario() as scenario, shared_redis() as redis:
        model: Final = rerank_model(scenario, wire)
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, models=[model], model_itpm_limit={model: 100000})
        key: Final = scenario.key(team_id=team, project_id=project)
        assert rerank(gateway, model, key).status_code == 200
        name: Final = counter_key("model_per_project_itpm", f"{project}:{model}", "tokens")
        assert eventually(lambda: redis_integer(redis, name), lambda tokens: tokens == 30, seconds=20) == 30


def test_a_rerank_without_meta_tokens_settles_billed_total_tokens_on_a_projects_input_window(gateway: Gateway) -> None:
    """Same project, but the rerank answer's ``meta`` carries only ``billed_units: {total_tokens: 24}``. The input
    window must settle to 24, the fallback ``_resolve_rerank_token_usage`` takes when ``meta.tokens`` is absent
    (parallel_request_limiter_v3.py:4777-4781). UNVERIFIED as above."""
    meta: Final = {"billed_units": {"total_tokens": 24}}
    with wire_server(rerank_provider(meta)) as wire, gateway.scenario() as scenario, shared_redis() as redis:
        model: Final = rerank_model(scenario, wire)
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, models=[model], model_itpm_limit={model: 100000})
        key: Final = scenario.key(team_id=team, project_id=project)
        assert rerank(gateway, model, key).status_code == 200
        name: Final = counter_key("model_per_project_itpm", f"{project}:{model}", "tokens")
        assert eventually(lambda: redis_integer(redis, name), lambda tokens: tokens == 24, seconds=20) == 24


def test_two_limited_tags_on_one_request_report_only_the_last_tags_numbers(gateway: Gateway) -> None:
    """Tag A has rpm_limit=5 and tag B rpm_limit=2; a request carries ``metadata.tags: [A, B]``. Both windows are
    charged, but the response has a single ``x-ratelimit-tag-limit-requests`` / ``remaining-requests`` pair and it is
    B's: ``2`` and ``1`` (headers are keyed by descriptor word only, parallel_request_limiter_v3.py:4760-4763, and the
    tag descriptors keep request order, :3258-3267). Pinned as current behaviour."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        loose: Final = tag(gateway, scenario, rpm_limit=5)
        tight: Final = tag(gateway, scenario, rpm_limit=2)
        key: Final = scenario.key(models=[model])
        served: Final = tagged_chat(gateway, model, key, metadata_tags=(loose, tight))
        assert served.status_code == 200, served.text
        assert scope_headers(served, "tag") == {"limit-requests": "2", "remaining-requests": "1"}


def test_two_limited_tags_report_the_tightest_tag_whatever_their_order(gateway: Gateway) -> None:
    """Tags listed tight first: ``[B (rpm 2), A (rpm 5)]``. A caller reading the tag headers should see the tag closest
    to refusing (limit 2, remaining 1); today it sees A's 5 and 4 because the last status wins."""
    pytest.skip(
        "BUG: _merge_ratelimit_statuses_into_additional_headers keys x-ratelimit-tag-* by descriptor word only, so "
        "with several limited tags the last tag's numbers overwrite the tighter one's"
    )
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        tight: Final = tag(gateway, scenario, rpm_limit=2)
        loose: Final = tag(gateway, scenario, rpm_limit=5)
        key: Final = scenario.key(models=[model])
        served: Final = tagged_chat(gateway, model, key, metadata_tags=(tight, loose))
        assert scope_headers(served, "tag") == {"limit-requests": "2", "remaining-requests": "1"}


def test_new_team_refuses_a_dynamic_rpm_limit_type(gateway: Gateway) -> None:
    """``POST /team/new`` with ``rpm_limit_type: dynamic`` is refused with 422 because ``NewTeamRequest`` only allows
    ``guaranteed_throughput`` or ``best_effort_throughput`` (litellm/proxy/_types.py:2017), although
    ``/key/generate`` documents and accepts ``dynamic``. No team with that alias exists afterwards."""
    alias: Final = f"limcap3-team-{uuid.uuid4().hex[:10]}"
    refused: Final = gateway.request("POST", "/team/new", {"team_alias": alias, "rpm_limit_type": "dynamic"})
    assert refused.status_code == 422, refused.text
    assert "rpm_limit_type" in refused.text
    listed: Final = gateway.request("GET", "/team/list")
    assert alias not in listed.text


def test_new_team_refuses_a_dynamic_tpm_limit_type(gateway: Gateway) -> None:
    """``POST /team/new`` with ``tpm_limit_type: dynamic`` is refused with 422 for the same Literal
    (litellm/proxy/_types.py:2020), and the error names the field."""
    refused: Final = gateway.request(
        "POST", "/team/new", {"team_alias": f"limcap3-team-{uuid.uuid4().hex[:10]}", "tpm_limit_type": "dynamic"}
    )
    assert refused.status_code == 422, refused.text
    assert "tpm_limit_type" in refused.text


def test_a_key_accepts_a_dynamic_rpm_limit_type_and_is_served_while_the_model_is_healthy(gateway: Gateway) -> None:
    """``/key/generate`` with ``rpm_limit: 1`` and ``rpm_limit_type: dynamic`` is accepted (the field is moved into key
    metadata, litellm/proxy/_types.py:4803). With no recent failures the dynamic rpm is not enforced
    (``_should_enforce_rate_limit``, parallel_request_limiter_v3.py:3107-3126), so two requests are both served. The
    asymmetry with ``/team/new`` is the point of the pair."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1, rpm_limit_type="dynamic")
        first, second = (chat(gateway, model, key) for _ in range(2))
        assert (first.status_code, second.status_code) == (200, 200), second.text
        assert len(posts(wire)) == 2


@pytest.mark.parametrize("field", ("rpm_limit", "tpm_limit"))
def test_new_team_refuses_a_negative_limit(gateway: Gateway, field: str) -> None:
    """``POST /team/new`` with ``<field>: -1`` is refused with 422 (``Field(ge=0)``, litellm/proxy/_types.py:1995-1996),
    so a negative limit can never reach the limiter as a team descriptor."""
    refused: Final = gateway.request(
        "POST", "/team/new", {"team_alias": f"limcap3-team-{uuid.uuid4().hex[:10]}", field: -1}
    )
    assert refused.status_code == 422, refused.text
    assert field in refused.text


@pytest.mark.parametrize("value", ("9:30", "0930", "noon"))
def test_a_deployment_with_a_malformed_daily_reset_time_is_refused_at_model_new(gateway: Gateway, value: str) -> None:
    """``POST /model/new`` with ``litellm_params.daily_reset_time`` of ``9:30`` (no zero pad), ``0930`` (no colon) or
    ``noon`` must be refused and name ``daily_reset_time must be HH:MM in UTC``
    (``parse_daily_reset_time``, litellm/types/router.py:586-593, run by the validator at :709-714). An accepted value
    would make ``reset_offset()`` raise inside the deployment check later, which fails open. UNVERIFIED: the status is
    assumed to be FastAPI's 422 for a body validator error."""
    refused: Final = gateway.request(
        "POST",
        "/model/new",
        {
            "model_name": f"limcap3-daily-{uuid.uuid4().hex[:10]}",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "synthetic-provider-key",
                "rpd": 10,
                "daily_reset_time": value,
            },
        },
    )
    assert refused.status_code == 422, refused.text
    assert "daily_reset_time must be HH:MM in UTC" in refused.text


def test_a_daily_reset_time_with_the_right_shape_but_no_real_time_is_refused_with_pythons_message(
    gateway: Gateway,
) -> None:
    """``daily_reset_time: 25:00`` passes the ``\\d{2}:\\d{2}`` shape check, so ``datetime.time.fromisoformat`` raises
    instead (litellm/types/router.py:592). The deployment must still be refused with 422, but the text is Python's
    ``hour must be in 0..23`` (what ``fromisoformat`` raises for an out-of-range hour on Python 3.12+) rather than
    the ``HH:MM in UTC`` hint. Pinned as current behaviour."""
    refused: Final = gateway.request(
        "POST",
        "/model/new",
        {
            "model_name": f"limcap3-daily-{uuid.uuid4().hex[:10]}",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "synthetic-provider-key",
                "rpd": 10,
                "daily_reset_time": "25:00",
            },
        },
    )
    assert refused.status_code == 422, refused.text
    assert "Value error, hour must be in 0..23" in refused.text
    assert "daily_reset_time must be HH:MM in UTC" not in refused.text
