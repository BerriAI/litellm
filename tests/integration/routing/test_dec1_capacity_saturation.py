"""Model capacity: priority pools, the saturation threshold, borrowing below it and refusal above it.

With ``callbacks: ["dynamic_rate_limiter_v3"]`` a model group's ``rpm``/``tpm`` is one window shared by every caller.
``priority_reservation`` maps a priority name to a share of that window. A caller's priority is ``metadata.priority``
of its team, else of its key. A caller with no priority, or with a priority the reservation does not list, counts
in one shared ``default_pool`` sized by ``priority_reservation_settings.default_priority`` (default 0.25).

Before every request the limiter reads the model's saturation, the model-wide counter divided by the model's
``rpm`` (or settled tokens divided by ``tpm``, whichever is higher). Below ``saturation_threshold`` (default 0.5)
only the model-wide window is enforced, so a priority may borrow past its share, and its pool counter is still
tracked. At or above the threshold the request must also fit its pool, and the refusal is a 429 reading
``Priority-based rate limit exceeded. Model: <m>, Priority: <p>, Rate limit type: <t>, Model TPM: <tpm>, Model
RPM: <rpm>, Remaining: <n>, Model saturation: <x.x%>`` with ``retry-after: 60``, ``rate_limit_type``,
``x-litellm-priority`` and ``x-litellm-saturation: <x.xx%>`` headers. Weights that add up to more than 1 are
normalized, and a weight may be written as a plain share, a percent, or an absolute ``rpm``/``tpm`` value.

#52081 (merged 2026-11-16): below saturation a pool whose token counter is full keeps counting and reporting its
requests. A Rust rewrite that tracks pools with one all-or-nothing call silently stops counting there.

``authorization/test_dec1_limits_dynamic_priority.py`` already pins the model-wide capacity refusal, the priority
lookup and one borrowing case. This file pins the threshold and everything that happens at and after it.

Read from ``litellm/proxy/hooks/dynamic_rate_limiter_v3.py`` (``_check_model_saturation``,
``_get_priority_allocation``, ``_normalize_priority_weights``, ``_check_rate_limits``, ``_track_priority_usage``),
``litellm/proxy/hooks/rate_limiter_utils.py`` (``convert_priority_to_percent``) and ``litellm/types/utils.py``
(``PriorityReservationSettings``).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Final

from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually, object_value
from tests.integration._support.wire import Wire, wire_server
from tests.integration.routing.dec1_capacity_support import (
    MODEL_SCOPE,
    PRIORITY_SCOPE,
    WINDOW_SECONDS,
    capacity_message,
    capacity_proxy,
    chat,
    entry,
    error_message,
    error_of,
    group_name,
    posts,
    priority_message,
    provider,
    scope_headers,
    statuses,
)


@contextmanager
def _rig(
    gateway: Gateway,
    directory: Path,
    *,
    rpm: int | None = None,
    tpm: int | None = None,
    reservation: Mapping[str, JsonValue] | None = None,
    settings: Mapping[str, JsonValue] | None = None,
) -> Iterator[tuple[Gateway, Wire, str]]:
    group: Final = group_name("sat")
    limits: Final[dict[str, JsonValue]] = {
        **({"rpm": rpm} if rpm is not None else {}),
        **({"tpm": tpm} if tpm is not None else {}),
    }
    with (
        wire_server(provider()) as wire,
        capacity_proxy(
            gateway,
            directory,
            model_list=[entry(group, wire.url, **limits)],
            priority_reservation=reservation,
            priority_settings=settings,
        ) as candidate,
    ):
        yield candidate, wire, group


def test_a_priority_past_its_share_is_refused_once_the_model_reaches_half_its_rpm(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A model has ``rpm: 10`` and ``priority_reservation: {"low": 0.2}`` (2 requests). A ``low`` key sends six
    requests: the first five are served because the model is below 50% before each one, the sixth sees 5 of 10
    (50%) and is refused with ``Priority-based rate limit exceeded. Model: <m>, Priority: low, Rate limit type:
    requests, Model TPM: not configured, Model RPM: 10, Remaining: 0, Model saturation: 50.0%``. The provider
    sees exactly five calls."""
    with _rig(gateway, tmp_path, rpm=10, reservation={"low": 0.2}) as (candidate, wire, group):
        with candidate.scenario() as scenario:
            key: Final = scenario.key(metadata={"priority": "low"})
            responses: Final = [chat(candidate, group, key) for _ in range(6)]
            assert statuses(responses) == [200] * 5 + [429], [response.text for response in responses]
            assert error_message(responses[5]) == priority_message(group, "low", "requests", "50.0%", rpm=10)
            assert len(posts(wire)) == 5


def test_the_priority_refusal_carries_retry_after_type_priority_and_saturation_headers(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The saturated model refuses a ``low`` key over its share. The 429 must carry ``retry-after: 60``,
    ``rate_limit_type: requests``, ``x-litellm-priority: low`` and ``x-litellm-saturation: 50.00%`` (two
    decimals, unlike the one decimal in the message)."""
    with _rig(gateway, tmp_path, rpm=10, reservation={"low": 0.2}) as (candidate, _wire, group):
        with candidate.scenario() as scenario:
            key: Final = scenario.key(metadata={"priority": "low"})
            assert statuses([chat(candidate, group, key) for _ in range(5)]) == [200] * 5
            refused: Final = chat(candidate, group, key)
            assert refused.status_code == 429, refused.text
            assert {
                name: refused.headers.get(name)
                for name in ("retry-after", "rate_limit_type", "x-litellm-priority", "x-litellm-saturation")
            } == {
                "retry-after": str(WINDOW_SECONDS),
                "rate_limit_type": "requests",
                "x-litellm-priority": "low",
                "x-litellm-saturation": "50.00%",
            }


def test_the_priority_refusal_body_is_openai_shaped_and_keeps_the_structured_detail(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The saturated model refuses a ``low`` key. The body must be ``{"error": {"message": <text>, "type":
    "throttling_error", "param": null, "code": "429", "provider_specific_fields": {"error": <text>}}}``, the same
    shape the model-wide capacity refusal has, so clients parsing either keep working."""
    with _rig(gateway, tmp_path, rpm=10, reservation={"low": 0.2}) as (candidate, _wire, group):
        with candidate.scenario() as scenario:
            key: Final = scenario.key(metadata={"priority": "low"})
            assert statuses([chat(candidate, group, key) for _ in range(5)]) == [200] * 5
            refused: Final = chat(candidate, group, key)
            text: Final = priority_message(group, "low", "requests", "50.0%", rpm=10)
            assert refused.json() == {
                "error": {
                    "message": text,
                    "type": "throttling_error",
                    "param": None,
                    "code": "429",
                    "provider_specific_fields": {"error": text},
                }
            }, refused.text


def test_at_saturation_a_priority_with_room_in_its_own_share_is_still_served(gateway: Gateway, tmp_path: Path) -> None:
    """``priority_reservation: {"low": 0.2, "high": 0.8}`` on ``rpm: 10``. A ``low`` key saturates the model with
    five requests and its sixth is refused, but a ``high`` key, whose pool of 8 is untouched, must be served on
    the same saturated model."""
    with _rig(gateway, tmp_path, rpm=10, reservation={"low": 0.2, "high": 0.8}) as (candidate, _wire, group):
        with candidate.scenario() as scenario:
            low: Final = scenario.key(metadata={"priority": "low"})
            high: Final = scenario.key(metadata={"priority": "high"})
            assert statuses([chat(candidate, group, low) for _ in range(5)]) == [200] * 5
            assert chat(candidate, group, low).status_code == 429
            served: Final = chat(candidate, group, high)
            assert served.status_code == 200, served.text
            assert served.headers["x-litellm-priority"] == "high"


def test_a_refused_priority_request_uses_no_model_capacity(gateway: Gateway, tmp_path: Path) -> None:
    """The check is all-or-nothing: when the ``low`` pool refuses, the model-wide counter is not incremented. After
    five served ``low`` requests and one refused, a ``high`` request must report
    ``x-ratelimit-model_saturation_check-remaining-requests: 4`` (10 - 6), not 3."""
    with _rig(gateway, tmp_path, rpm=10, reservation={"low": 0.2, "high": 0.8}) as (candidate, _wire, group):
        with candidate.scenario() as scenario:
            low: Final = scenario.key(metadata={"priority": "low"})
            high: Final = scenario.key(metadata={"priority": "high"})
            assert statuses([chat(candidate, group, low) for _ in range(6)]) == [200] * 5 + [429]
            served: Final = chat(candidate, group, high)
            assert served.status_code == 200, served.text
            assert scope_headers(served, MODEL_SCOPE) == {"limit-requests": "10", "remaining-requests": "4"}


def test_at_saturation_the_served_response_reports_the_enforced_priority_pool(gateway: Gateway, tmp_path: Path) -> None:
    """With ``saturation_threshold: 0.0`` every request is in strict mode. A ``high`` key of ``{"high": 0.8}`` on
    ``rpm: 10`` is served and its response reports ``x-ratelimit-priority_model-limit-requests: 8`` and
    ``-remaining-requests: 7`` from the enforced pool, next to the model-wide ``10`` / ``9``."""
    with (
        _rig(gateway, tmp_path, rpm=10, reservation={"high": 0.8}, settings={"saturation_threshold": 0.0}) as (
            candidate,
            _wire,
            group,
        ),
        candidate.scenario() as scenario,
    ):
        served: Final = chat(candidate, group, scenario.key(metadata={"priority": "high"}))
        assert served.status_code == 200, served.text
        assert scope_headers(served, PRIORITY_SCOPE) == {"limit-requests": "8", "remaining-requests": "7"}
        assert scope_headers(served, MODEL_SCOPE) == {"limit-requests": "10", "remaining-requests": "9"}


def test_a_custom_saturation_threshold_moves_the_point_where_shares_are_enforced(
    gateway: Gateway, tmp_path: Path
) -> None:
    """``saturation_threshold: 0.8`` on ``rpm: 10`` with ``{"low": 0.2}``. A ``low`` key is served eight times (the
    model is below 80% before each) and its ninth request is refused with ``Model saturation: 80.0%``."""
    with (
        _rig(gateway, tmp_path, rpm=10, reservation={"low": 0.2}, settings={"saturation_threshold": 0.8}) as (
            candidate,
            _wire,
            group,
        ),
        candidate.scenario() as scenario,
    ):
        key: Final = scenario.key(metadata={"priority": "low"})
        responses: Final = [chat(candidate, group, key) for _ in range(9)]
        assert statuses(responses) == [200] * 8 + [429], [response.text for response in responses]
        assert error_message(responses[8]) == priority_message(group, "low", "requests", "80.0%", rpm=10)
        assert responses[8].headers["x-litellm-saturation"] == "80.00%"


def test_a_zero_saturation_threshold_enforces_shares_from_the_first_request(gateway: Gateway, tmp_path: Path) -> None:
    """``saturation_threshold: 0.0`` on ``rpm: 10`` with ``{"low": 0.2}``. A ``low`` key gets exactly its two
    requests and the third is refused with ``Model saturation: 20.0%``: there is no borrowing at all."""
    with (
        _rig(gateway, tmp_path, rpm=10, reservation={"low": 0.2}, settings={"saturation_threshold": 0.0}) as (
            candidate,
            _wire,
            group,
        ),
        candidate.scenario() as scenario,
    ):
        key: Final = scenario.key(metadata={"priority": "low"})
        responses: Final = [chat(candidate, group, key) for _ in range(3)]
        assert statuses(responses) == [200, 200, 429], [response.text for response in responses]
        assert error_message(responses[2]) == priority_message(group, "low", "requests", "20.0%", rpm=10)


def test_a_threshold_above_one_never_enforces_shares_and_only_total_capacity_refuses(
    gateway: Gateway, tmp_path: Path
) -> None:
    """``saturation_threshold: 1.5`` can never be reached because the model-wide window stops at 100%. A ``low`` key
    of ``{"low": 0.25}`` on ``rpm: 4`` is served four times and the fifth refusal is the model capacity text
    ``Model capacity reached for <m>. Priority: low, ...``, not the priority-based one."""
    with (
        _rig(gateway, tmp_path, rpm=4, reservation={"low": 0.25}, settings={"saturation_threshold": 1.5}) as (
            candidate,
            _wire,
            group,
        ),
        candidate.scenario() as scenario,
    ):
        key: Final = scenario.key(metadata={"priority": "low"})
        responses: Final = [chat(candidate, group, key) for _ in range(5)]
        assert statuses(responses) == [200] * 4 + [429], [response.text for response in responses]
        assert error_message(responses[4]) == capacity_message(group, "low", "requests", rpm=4)
        assert "x-litellm-saturation" not in responses[4].headers


def test_keys_without_a_priority_share_one_default_pool_sized_by_default_priority(
    gateway: Gateway, tmp_path: Path
) -> None:
    """``{"premium": 0.5}`` on ``rpm: 40``. Two unrelated keys without a priority each send one request. Both count
    in one ``default_pool`` of ``int(40 * 0.25) = 10`` requests (``default_priority`` defaults to 0.25), so the
    first reports ``x-ratelimit-priority_model-remaining-requests: 9`` and the second ``8``, both with limit 10."""
    with _rig(gateway, tmp_path, rpm=40, reservation={"premium": 0.5}) as (candidate, _wire, group):
        with candidate.scenario() as scenario:
            first: Final = chat(candidate, group, scenario.key())
            second: Final = chat(candidate, group, scenario.key())
            assert statuses([first, second]) == [200, 200], (first.text, second.text)
            assert scope_headers(first, PRIORITY_SCOPE) == {"limit-requests": "10", "remaining-requests": "9"}
            assert scope_headers(second, PRIORITY_SCOPE) == {"limit-requests": "10", "remaining-requests": "8"}
            assert second.headers["x-litellm-priority"] == "default"


def test_a_priority_missing_from_the_reservation_counts_in_the_default_pool_under_its_own_name(
    gateway: Gateway, tmp_path: Path
) -> None:
    """``{"premium": 0.5}`` on ``rpm: 40``. A key with ``priority: gold`` (not reserved) and a key with no priority
    share the default pool: the ``gold`` response reports ``x-litellm-priority: gold`` with priority pool remaining
    9 of 10, and the next request from the plain key reports 8 of 10."""
    with _rig(gateway, tmp_path, rpm=40, reservation={"premium": 0.5}) as (candidate, _wire, group):
        with candidate.scenario() as scenario:
            gold: Final = chat(candidate, group, scenario.key(metadata={"priority": "gold"}))
            plain: Final = chat(candidate, group, scenario.key())
            assert statuses([gold, plain]) == [200, 200]
            assert gold.headers["x-litellm-priority"] == "gold"
            assert scope_headers(gold, PRIORITY_SCOPE) == {"limit-requests": "10", "remaining-requests": "9"}
            assert scope_headers(plain, PRIORITY_SCOPE) == {"limit-requests": "10", "remaining-requests": "8"}


def test_the_default_priority_setting_resizes_the_default_pool(gateway: Gateway, tmp_path: Path) -> None:
    """``priority_reservation_settings.default_priority: 0.1`` with ``{"premium": 0.5}`` on ``rpm: 40``. A key without
    a priority must report ``x-ratelimit-priority_model-limit-requests: 4`` (``int(40 * 0.1)``)."""
    with (
        _rig(gateway, tmp_path, rpm=40, reservation={"premium": 0.5}, settings={"default_priority": 0.1}) as (
            candidate,
            _wire,
            group,
        ),
        candidate.scenario() as scenario,
    ):
        served: Final = chat(candidate, group, scenario.key())
        assert served.status_code == 200, served.text
        assert scope_headers(served, PRIORITY_SCOPE) == {"limit-requests": "4", "remaining-requests": "3"}


def test_the_default_pool_is_refused_at_saturation_and_names_no_priority(gateway: Gateway, tmp_path: Path) -> None:
    """``{"premium": 0.5}`` on ``rpm: 8``, so the default pool is ``int(8 * 0.25) = 2``. A key without a priority is
    served four times (below 50%) and its fifth request, at 4 of 8, is refused with ``Priority: None`` in the
    message and ``x-litellm-priority: default`` in the headers."""
    with _rig(gateway, tmp_path, rpm=8, reservation={"premium": 0.5}) as (candidate, _wire, group):
        with candidate.scenario() as scenario:
            key: Final = scenario.key()
            responses: Final = [chat(candidate, group, key) for _ in range(5)]
            assert statuses(responses) == [200] * 4 + [429], [response.text for response in responses]
            assert error_message(responses[4]) == priority_message(group, None, "requests", "50.0%", rpm=8)
            assert responses[4].headers["x-litellm-priority"] == "default"


def test_over_allocated_reservations_are_normalized_to_the_model_capacity(gateway: Gateway, tmp_path: Path) -> None:
    """``{"a": 3, "b": 1}`` adds up to 4, so the shares are normalized to 0.75 and 0.25. On ``rpm: 40`` an ``a``
    key must report a priority pool limit of 30 and a ``b`` key a limit of 10."""
    with _rig(gateway, tmp_path, rpm=40, reservation={"a": 3, "b": 1}) as (candidate, _wire, group):
        with candidate.scenario() as scenario:
            first: Final = chat(candidate, group, scenario.key(metadata={"priority": "a"}))
            second: Final = chat(candidate, group, scenario.key(metadata={"priority": "b"}))
            assert statuses([first, second]) == [200, 200]
            assert scope_headers(first, PRIORITY_SCOPE)["limit-requests"] == "30"
            assert scope_headers(second, PRIORITY_SCOPE)["limit-requests"] == "10"


def test_reservations_that_add_up_to_less_than_one_are_not_scaled_up(gateway: Gateway, tmp_path: Path) -> None:
    """``{"a": 0.25, "b": 0.25}`` adds up to 0.5 and is used as given: on ``rpm: 40`` an ``a`` key reports a pool
    limit of 10, not 20. A rewrite that always normalizes would double every pool."""
    with _rig(gateway, tmp_path, rpm=40, reservation={"a": 0.25, "b": 0.25}) as (candidate, _wire, group):
        with candidate.scenario() as scenario:
            served: Final = chat(candidate, group, scenario.key(metadata={"priority": "a"}))
            assert served.status_code == 200, served.text
            assert scope_headers(served, PRIORITY_SCOPE)["limit-requests"] == "10"


def test_a_reservation_given_in_rpm_is_converted_to_a_share_of_the_model_rpm(gateway: Gateway, tmp_path: Path) -> None:
    """``{"fixed": {"type": "rpm", "value": 10}}`` on ``rpm: 40`` is a 0.25 share, so a ``fixed`` key reports a
    priority pool limit of 10 requests."""
    with (
        _rig(gateway, tmp_path, rpm=40, reservation={"fixed": {"type": "rpm", "value": 10}}) as (
            candidate,
            _wire,
            group,
        ),
        candidate.scenario() as scenario,
    ):
        served: Final = chat(candidate, group, scenario.key(metadata={"priority": "fixed"}))
        assert served.status_code == 200, served.text
        assert scope_headers(served, PRIORITY_SCOPE) == {"limit-requests": "10", "remaining-requests": "9"}


def test_a_reservation_given_in_tpm_sizes_both_the_request_and_token_pools(gateway: Gateway, tmp_path: Path) -> None:
    """``{"fixed": {"type": "tpm", "value": 250}}`` on a model with ``tpm: 1000`` and ``rpm: 40`` is a 0.25 share of
    both, so the first response reports ``limit-requests: 10`` and ``limit-tokens: 250`` for the priority pool.
    The token side has 250 remaining because admission counts no tokens, only settlement does."""
    with (
        _rig(gateway, tmp_path, rpm=40, tpm=1000, reservation={"fixed": {"type": "tpm", "value": 250}}) as (
            candidate,
            _wire,
            group,
        ),
        candidate.scenario() as scenario,
    ):
        served: Final = chat(candidate, group, scenario.key(metadata={"priority": "fixed"}))
        assert served.status_code == 200, served.text
        assert scope_headers(served, PRIORITY_SCOPE) == {
            "limit-requests": "10",
            "remaining-requests": "9",
            "limit-tokens": "250",
            "remaining-tokens": "250",
        }


def test_a_reservation_written_as_a_percent_object_is_the_plain_share(gateway: Gateway, tmp_path: Path) -> None:
    """``{"half": {"type": "percent", "value": 0.5}}`` on ``rpm: 40`` behaves like ``0.5``: a ``half`` key reports a
    priority pool limit of 20."""
    with (
        _rig(gateway, tmp_path, rpm=40, reservation={"half": {"type": "percent", "value": 0.5}}) as (
            candidate,
            _wire,
            group,
        ),
        candidate.scenario() as scenario,
    ):
        served: Final = chat(candidate, group, scenario.key(metadata={"priority": "half"}))
        assert served.status_code == 200, served.text
        assert scope_headers(served, PRIORITY_SCOPE)["limit-requests"] == "20"


def test_without_a_priority_reservation_no_priority_pool_is_reported(gateway: Gateway, tmp_path: Path) -> None:
    """The dynamic limiter is on but ``priority_reservation`` is not set. A key with ``priority: premium`` is served
    on ``rpm: 10`` with the model-wide headers and ``x-litellm-priority: premium`` but no
    ``x-ratelimit-priority_model-*`` header at all, and is never refused below the model's capacity."""
    with _rig(gateway, tmp_path, rpm=10, settings={"saturation_threshold": 0.0}) as (candidate, _wire, group):
        with candidate.scenario() as scenario:
            key: Final = scenario.key(metadata={"priority": "premium"})
            responses: Final = [chat(candidate, group, key) for _ in range(10)]
            assert statuses(responses) == [200] * 10, [response.text for response in responses]
            assert all(scope_headers(response, PRIORITY_SCOPE) == {} for response in responses)
            assert responses[0].headers["x-litellm-priority"] == "premium"


def test_a_share_that_rounds_down_to_zero_requests_is_refused_as_soon_as_shares_are_enforced(
    gateway: Gateway, tmp_path: Path
) -> None:
    """``{"tiny": 0.05}`` on ``rpm: 10`` is ``int(0.5) = 0`` requests. With ``saturation_threshold: 0.0`` the very
    first ``tiny`` request is refused with ``Remaining: 0, Model saturation: 0.0%`` and the provider is never
    called. A rewrite that rounds up or floors at one would serve it."""
    with (
        _rig(gateway, tmp_path, rpm=10, reservation={"tiny": 0.05}, settings={"saturation_threshold": 0.0}) as (
            candidate,
            wire,
            group,
        ),
        candidate.scenario() as scenario,
    ):
        refused: Final = chat(candidate, group, scenario.key(metadata={"priority": "tiny"}))
        assert refused.status_code == 429, refused.text
        assert error_message(refused) == priority_message(group, "tiny", "requests", "0.0%", rpm=10)
        assert refused.headers["x-litellm-saturation"] == "0.00%"
        assert posts(wire) == ()


def test_every_key_of_a_team_with_a_priority_counts_in_the_teams_pool(gateway: Gateway, tmp_path: Path) -> None:
    """A team has ``metadata.priority: batch`` and ``{"batch": 0.25}`` is set on ``rpm: 40``. Two keys of the team
    without their own priority send one request each: the first reports priority pool remaining 9 of 10 and the
    second 8 of 10, and both report ``x-litellm-priority: batch``."""
    with _rig(gateway, tmp_path, rpm=40, reservation={"batch": 0.25}) as (candidate, _wire, group):
        with candidate.scenario() as scenario:
            team: Final = scenario.team(metadata={"priority": "batch"})
            first: Final = chat(candidate, group, scenario.key(team_id=team))
            second: Final = chat(candidate, group, scenario.key(team_id=team))
            assert statuses([first, second]) == [200, 200]
            assert [response.headers["x-litellm-priority"] for response in (first, second)] == ["batch", "batch"]
            assert scope_headers(first, PRIORITY_SCOPE) == {"limit-requests": "10", "remaining-requests": "9"}
            assert scope_headers(second, PRIORITY_SCOPE) == {"limit-requests": "10", "remaining-requests": "8"}


def test_a_full_priority_request_counter_keeps_being_served_below_saturation_and_reports_zero(
    gateway: Gateway, tmp_path: Path
) -> None:
    """``{"low": 0.02}`` on ``rpm: 100`` is a pool of 2 requests. A ``low`` key sends four requests, all served
    because the model is at most 3% saturated. The third and fourth responses report the pool as
    ``limit-requests: 2`` / ``remaining-requests: 0`` while the model-wide remaining keeps counting down to 97
    and 96."""
    with _rig(gateway, tmp_path, rpm=100, reservation={"low": 0.02}) as (candidate, _wire, group):
        with candidate.scenario() as scenario:
            key: Final = scenario.key(metadata={"priority": "low"})
            responses: Final = [chat(candidate, group, key) for _ in range(4)]
            assert statuses(responses) == [200] * 4, [response.text for response in responses]
            assert [scope_headers(response, PRIORITY_SCOPE) for response in responses[2:]] == [
                {"limit-requests": "2", "remaining-requests": "0"}
            ] * 2
            assert [scope_headers(response, MODEL_SCOPE)["remaining-requests"] for response in responses[2:]] == [
                "97",
                "96",
            ]


def test_a_pool_whose_tokens_ran_out_keeps_counting_requests_on_chat_completions(
    gateway: Gateway, tmp_path: Path
) -> None:
    """#52081 on ``/v1/chat/completions``: ``{"premium": 0.25}`` on ``tpm: 200, rpm: 400`` gives the pool 50 tokens
    and 100 requests. Each answer settles 18 tokens, so after three settled answers the pool's token counter is
    full while the model is only 27% saturated. The next two responses must report ``remaining-tokens: 0`` and
    ``remaining-requests`` still falling by one per request, with ``limit-requests: 100``. Before the fix the whole
    tracking call was dropped once tokens were full, freezing the request count."""
    with _rig(gateway, tmp_path, rpm=400, tpm=200, reservation={"premium": 0.25}) as (candidate, wire, group):
        with candidate.scenario() as scenario:
            key: Final = scenario.key(metadata={"priority": "premium"})
            drained: Final = eventually(
                lambda: chat(candidate, group, key),
                lambda response: scope_headers(response, PRIORITY_SCOPE).get("remaining-tokens") == "0",
                seconds=30,
            )
            after: Final = chat(candidate, group, key)
            sent: Final = len(posts(wire))
            assert (drained.status_code, after.status_code) == (200, 200), (drained.text, after.text)
            assert scope_headers(after, PRIORITY_SCOPE) == {
                "limit-requests": "100",
                "remaining-requests": str(100 - sent),
                "limit-tokens": "50",
                "remaining-tokens": "0",
            }
            assert scope_headers(drained, PRIORITY_SCOPE)["remaining-requests"] == str(100 - sent + 1)


def test_a_default_pool_whose_tokens_ran_out_keeps_counting_requests(gateway: Gateway, tmp_path: Path) -> None:
    """#52081 for the shared default pool: ``{"premium": 0.5}`` on ``tpm: 200, rpm: 400`` gives keys without a
    priority a pool of 50 tokens and 100 requests. Once settled answers fill its tokens, two different keys without
    a priority are still served and the pool's ``remaining-requests`` falls by one between their responses."""
    with _rig(gateway, tmp_path, rpm=400, tpm=200, reservation={"premium": 0.5}) as (candidate, _wire, group):
        with candidate.scenario() as scenario:
            first_key: Final = scenario.key()
            eventually(
                lambda: chat(candidate, group, first_key),
                lambda response: scope_headers(response, PRIORITY_SCOPE).get("remaining-tokens") == "0",
                seconds=30,
            )
            first: Final = chat(candidate, group, first_key)
            second: Final = chat(candidate, group, scenario.key())
            assert statuses([first, second]) == [200, 200]
            assert (
                int(scope_headers(first, PRIORITY_SCOPE)["remaining-requests"])
                - int(scope_headers(second, PRIORITY_SCOPE)["remaining-requests"])
                == 1
            )
            assert scope_headers(second, PRIORITY_SCOPE)["remaining-tokens"] == "0"


def test_a_tpm_model_saturated_by_settled_tokens_refuses_a_pool_on_its_tokens(gateway: Gateway, tmp_path: Path) -> None:
    """A model has only ``tpm: 100`` and ``{"low": 0.1}`` (a pool of 10 tokens). Each answer settles 18 tokens. A
    ``low`` key is served until the settled tokens put the model at or above 50%, and then refused with ``Rate
    limit type: tokens``, ``Model TPM: 100, Model RPM: not configured`` and ``rate_limit_type: tokens``. The
    saturation in the message is the settled tokens over 100, so it is a multiple of 18 percent (54.0% when
    settlement keeps up). UNVERIFIED: the exact request at which it turns depends on settlement having landed,
    which is asynchronous, so only the shape and the multiple are pinned."""
    with _rig(gateway, tmp_path, tpm=100, reservation={"low": 0.1}) as (candidate, _wire, group):
        with candidate.scenario() as scenario:
            key: Final = scenario.key(metadata={"priority": "low"})
            refused: Final = eventually(
                lambda: chat(candidate, group, key), lambda response: response.status_code == 429, seconds=30
            )
            message: Final = error_message(refused)
            prefix: Final = (
                f"Priority-based rate limit exceeded. Model: {group}, Priority: low, Rate limit type: tokens, "
                "Model TPM: 100, Model RPM: not configured, Remaining: 0, Model saturation: "
            )
            assert message.startswith(prefix), message
            assert message.removeprefix(prefix) in ("54.0%", "72.0%", "90.0%"), message
            assert refused.headers["rate_limit_type"] == "tokens"


def test_the_model_wide_window_still_refuses_a_saturated_model_at_full_capacity(
    gateway: Gateway, tmp_path: Path
) -> None:
    """``{"low": 0.5, "high": 0.5}`` on ``rpm: 4`` with ``saturation_threshold: 0.0``. Two ``low`` and two ``high``
    requests are served. A fifth ``high`` request finds both its pool and the model full; the model-wide check is
    evaluated first, so the refusal is the capacity text ``Model capacity reached ... Priority: high``."""
    with (
        _rig(
            gateway, tmp_path, rpm=4, reservation={"low": 0.5, "high": 0.5}, settings={"saturation_threshold": 0.0}
        ) as (candidate, _wire, group),
        candidate.scenario() as scenario,
    ):
        low: Final = scenario.key(metadata={"priority": "low"})
        high: Final = scenario.key(metadata={"priority": "high"})
        served: Final = [chat(candidate, group, key) for key in (low, low, high, high)]
        assert statuses(served) == [200] * 4, [response.text for response in served]
        refused: Final = chat(candidate, group, high)
        assert refused.status_code == 429, refused.text
        assert error_message(refused) == capacity_message(group, "high", "requests", rpm=4)
        assert object_value(error_of(refused)["provider_specific_fields"]) == {"error": error_message(refused)}
