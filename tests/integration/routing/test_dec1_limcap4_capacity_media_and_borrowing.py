"""Model capacity on image routes, and a borrowing team's held tokens when the provider fails.

Images. ``async_pre_call_hook`` of the dynamic rate limiter v3 (``litellm/proxy/hooks/dynamic_rate_limiter_v3.py``,
Rust port in draft PR #51540) admits any request whose body names a known model group, whatever the route
(:1680). ``/v1/images/generations`` and the multipart ``/v1/images/edits`` therefore take one request of the group's
model-wide window ``{model_saturation_check:<group>}:requests`` and are refused with ``Model capacity reached for
<group>. Priority: None, Rate limit type: requests, Model TPM: not configured, Model RPM: <n>, Remaining: 0`` when it
is full. Settlement charges ``total_tokens`` of the response usage and writes nothing when it is 0 (:1838), so an
image answer without usage leaves the token window untouched.

Borrowing. With ``priority_reservation_settings.model_limit_borrowing_config.borrow_idle_capacity: ["team"]`` a team
with ``model_tpm_limit`` for the model holds its estimated tokens at admission: on
``{model_per_team_saturated:<team>:<group>}:tokens`` once the model is saturated, and on
``{model_per_team_max_borrow_multiple:<team>:<group>}:tokens`` while idle when the team caps its borrowing
(``_holding_shares``, :260). When the provider fails, ``async_log_failure_event`` (:1931) settles the hold to the
usage the failed attempt recorded, which is none, so the whole hold comes back; ``async_post_call_failure_hook``
(:1909) is idempotent with it. ``test_dec1_limcap2_capacity_edges.py`` pins only the hook-refusal path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from tests.integration._support.client import Gateway
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_limcap4_support import (
    IMAGE_MODEL,
    PROVIDER_TOTAL,
    FlakyProvider,
    counter,
    counter_settles,
    edit_image,
    generate_image,
    image_posts,
    image_provider,
)
from tests.integration.routing.dec1_capacity_support import (
    MODEL_SCOPE,
    capacity_message,
    capacity_proxy,
    chat,
    entry,
    error_message,
    group_name,
    statuses,
)

TEAM_SHARE: Final = "model_per_team_saturated"
TEAM_CAP: Final = "model_per_team_max_borrow_multiple"
BORROWING: Final = {"model_limit_borrowing_config": {"borrow_idle_capacity": ["team"]}}


def test_an_image_generation_takes_one_request_of_model_capacity(gateway: Gateway, tmp_path: Path) -> None:
    """An image group has ``rpm: 1``. The first ``/v1/images/generations`` is 200; the second is refused 429 with
    exactly ``Model capacity reached for <group>. Priority: None, Rate limit type: requests, Model TPM: not
    configured, Model RPM: 1, Remaining: 0`` and the provider received one image request."""
    group: Final = group_name("imagegen")
    with (
        wire_server(image_provider()) as wire,
        capacity_proxy(gateway, tmp_path, model_list=[entry(group, wire.url, rpm=1, model=IMAGE_MODEL)]) as candidate,
    ):
        assert generate_image(candidate, group, candidate.key).status_code == 200
        refused: Final = generate_image(candidate, group, candidate.key)
        assert refused.status_code == 429, refused.text
        # The image generation route renders the limiter error with the exception class prefix (CI run 44176682961);
        # the image edit and chat routes do not.
        assert error_message(refused) == "litellm.RateLimitError: " + capacity_message(group, None, "requests", rpm=1)
        assert image_posts(wire) == 1


def test_an_image_generation_writes_one_request_and_no_tokens_to_the_model_window(
    gateway: Gateway, tmp_path: Path
) -> None:
    """An image group with ``rpm: 10`` serves one generation whose answer has no usage. Redis holds 1 on
    ``{model_saturation_check:<group>}:requests`` and nothing on ``{model_saturation_check:<group>}:tokens``, because
    settlement skips a response that used 0 tokens (:1838). UNVERIFIED: that ``_usage_of`` reads no tokens from an
    ``ImageResponse`` without ``usage``."""
    group: Final = group_name("imagecount")
    with (
        wire_server(image_provider()) as wire,
        capacity_proxy(
            gateway, tmp_path, model_list=[entry(group, wire.url, rpm=10, model=IMAGE_MODEL)], shared=True
        ) as candidate,
    ):
        assert generate_image(candidate, group, candidate.key).status_code == 200
        assert counter_settles(MODEL_SCOPE, group, "requests", 1) == 1
        assert counter(MODEL_SCOPE, group, "tokens") is None


def test_an_image_edit_takes_one_request_of_model_capacity(gateway: Gateway, tmp_path: Path) -> None:
    """An image group has ``rpm: 1``. The first multipart ``/v1/images/edits`` upload is 200; the second is refused
    429 with the same model capacity text, and the provider received one edit. The limiter reads ``model`` from the
    parsed form fields like it does from a JSON body."""
    group: Final = group_name("imageedit")
    with (
        wire_server(image_provider()) as wire,
        capacity_proxy(gateway, tmp_path, model_list=[entry(group, wire.url, rpm=1, model=IMAGE_MODEL)]) as candidate,
    ):
        assert edit_image(candidate, group, candidate.key).status_code == 200
        refused: Final = edit_image(candidate, group, candidate.key)
        assert refused.status_code == 429, refused.text
        assert error_message(refused) == capacity_message(group, None, "requests", rpm=1)
        assert image_posts(wire) == 1


def test_image_edits_and_generations_share_one_group_window(gateway: Gateway, tmp_path: Path) -> None:
    """An image group has ``rpm: 2``. One generation and one edit are both 200, and a third request on either route
    is refused with ``Model RPM: 2``: the window is the group's, not the route's."""
    group: Final = group_name("imageshare")
    with (
        wire_server(image_provider()) as wire,
        capacity_proxy(gateway, tmp_path, model_list=[entry(group, wire.url, rpm=2, model=IMAGE_MODEL)]) as candidate,
    ):
        assert generate_image(candidate, group, candidate.key).status_code == 200
        assert edit_image(candidate, group, candidate.key).status_code == 200
        refused: Final = generate_image(candidate, group, candidate.key)
        assert refused.status_code == 429, refused.text
        # The image generation route renders the limiter error with the exception class prefix (CI run 44176682961);
        # the image edit and chat routes do not.
        assert error_message(refused) == "litellm.RateLimitError: " + capacity_message(group, None, "requests", rpm=2)


def test_an_image_generation_by_a_priority_key_counts_in_its_priority_pool(gateway: Gateway, tmp_path: Path) -> None:
    """``priority_reservation: {high: 0.5}`` and a key with ``metadata.priority: high``. One image generation leaves
    1 on ``{priority_model:<group>:high}:requests`` next to the model-wide counter: priority pools are tracked from the
    first request on image routes too (``_create_priority_based_descriptors``, :707)."""
    group: Final = group_name("imageprio")
    with (
        wire_server(image_provider()) as wire,
        capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(group, wire.url, rpm=10, model=IMAGE_MODEL)],
            priority_reservation={"high": 0.5},
            shared=True,
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        key: Final = scenario.key(metadata={"priority": "high"})
        assert generate_image(candidate, group, key).status_code == 200
        assert counter_settles("priority_model", f"{group}:high", "requests", 1) == 1
        assert counter(MODEL_SCOPE, group, "requests") == 1


def test_a_provider_failure_returns_a_saturated_teams_held_tokens(gateway: Gateway, tmp_path: Path) -> None:
    """Teams borrow and ``saturation_threshold: 0.0`` keeps the model saturated, so a team with ``model_tpm_limit:
    {<group>: 1000}`` holds its estimate on ``{model_per_team_saturated:<team>:<group>}:tokens`` at admission. The
    provider answers 500. The caller gets the error, and the counter ends at 0 or absent: the failure event settled
    the hold to the failed attempt's usage, none."""
    group: Final = group_name("failsat")
    with (
        wire_server(FlakyProvider(failures=-1)) as wire,
        capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(group, wire.url, tpm=100000)],
            priority_settings={"saturation_threshold": 0.0, **BORROWING},
            shared=True,
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        team: Final = scenario.team(metadata={"model_tpm_limit": {group: 1000}})
        key: Final = scenario.key(team_id=team)
        failed: Final = chat(candidate, group, key, max_tokens=40)
        assert failed.status_code >= 500, failed.text
        assert counter_settles(TEAM_SHARE, f"{team}:{group}", "tokens", 0) in (None, 0)


def test_an_idle_teams_borrow_cap_hold_is_returned_on_a_provider_failure(gateway: Gateway, tmp_path: Path) -> None:
    """Teams borrow and the model stays idle (threshold 0.5, ``tpm: 100000``). A team with ``model_tpm_limit:
    {<group>: 1000}`` and ``model_limit_max_borrow_multiple: 2`` holds its estimate on the cap counter
    ``{model_per_team_max_borrow_multiple:<team>:<group>}:tokens`` while idle. After a provider 500 that counter ends
    at 0 or absent (``_failed_attempt_settlement``, :1968, settles the cap hold too)."""
    group: Final = group_name("failcap")
    with (
        wire_server(FlakyProvider(failures=-1)) as wire,
        capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(group, wire.url, tpm=100000)],
            priority_settings={"saturation_threshold": 0.5, **BORROWING},
            shared=True,
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        team: Final = scenario.team(metadata={"model_tpm_limit": {group: 1000}, "model_limit_max_borrow_multiple": 2})
        key: Final = scenario.key(team_id=team)
        failed: Final = chat(candidate, group, key, max_tokens=40)
        assert failed.status_code >= 500, failed.text
        assert counter_settles(TEAM_CAP, f"{team}:{group}", "tokens", 0) in (None, 0)


def test_repeated_provider_failures_never_exhaust_a_saturated_teams_share(gateway: Gateway, tmp_path: Path) -> None:
    """A saturated team's share is ``model_tpm_limit: {<group>: 60}`` and each request holds about 50 tokens (a 10
    token prompt and ``max_tokens: 40``). Three requests the provider fails all come back as provider errors, never
    as a 429 for ``model_per_team_saturated``: each failure returned its hold before the next admission. A port that
    keeps holds after a failure refuses the second request."""
    group: Final = group_name("failrepeat")
    with (
        wire_server(FlakyProvider(failures=-1)) as wire,
        capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(group, wire.url, tpm=100000)],
            priority_settings={"saturation_threshold": 0.0, **BORROWING},
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        team: Final = scenario.team(metadata={"model_tpm_limit": {group: 60}})
        key: Final = scenario.key(team_id=team)
        answers: Final = [chat(candidate, group, key, max_tokens=40) for _ in range(3)]
        assert all(answer.status_code >= 500 for answer in answers), statuses(answers)


def test_a_saturated_teams_share_settles_to_the_answering_retry_usage(gateway: Gateway, tmp_path: Path) -> None:
    """Same saturated team on a proxy with ``num_retries: 1``; the provider fails the first attempt and answers the
    retry. ``{model_per_team_saturated:<team>:<group>}:tokens`` settles to 18: the failed attempt returned the hold and
    the answering attempt charged its usage once. UNVERIFIED: that ``async_log_failure_event`` fires for the failed
    attempt inside the router's retry loop; if it does not, the success settlement alone also lands on 18."""
    group: Final = group_name("retryshare")
    with (
        wire_server(FlakyProvider(failures=1)) as wire,
        capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(group, wire.url, tpm=100000)],
            priority_settings={"saturation_threshold": 0.0, **BORROWING},
            router_settings={"num_retries": 1},
            shared=True,
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        team: Final = scenario.team(metadata={"model_tpm_limit": {group: 1000}})
        key: Final = scenario.key(team_id=team)
        assert chat(candidate, group, key, max_tokens=40).status_code == 200
        assert counter_settles(TEAM_SHARE, f"{team}:{group}", "tokens", PROVIDER_TOTAL) == PROVIDER_TOTAL
