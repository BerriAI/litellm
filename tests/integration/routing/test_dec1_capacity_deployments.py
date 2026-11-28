"""Model capacity of a model group with several deployments, and deployment rpm/tpm caps in the router.

The dynamic limiter sizes a model group's window from the router's merged group info: the group's ``rpm`` and
``tpm`` are the sums of the values each deployment sets in ``litellm_params`` or ``model_info``, and a deployment
that sets none adds nothing (``litellm/rust/crates/router-core/src/group_info.rs``, ``model_group_info``). That
window is checked once per request in the proxy's pre-call hook, before the router picks a deployment, so a
capacity refusal reaches no deployment, and every deployment's traffic counts in the one group window.

Separately, the router can treat a deployment's own ``rpm``/``tpm`` as a cap. With
``router_settings.optional_pre_call_checks: ["enforce_model_rate_limits"]`` the picked deployment is checked
(``litellm/router_utils/pre_call_checks/model_rate_limit_check.py``) and a full one fails the attempt like a
provider 429, so with retries the next attempt lands on another deployment of the group and without room anywhere
the caller gets a 429. ``authorization/test_dec1_limits_router.py`` pins the single deployment refusals and
usage-based routing; this file pins groups of several deployments and how the two layers stack.

Read from ``litellm/proxy/hooks/dynamic_rate_limiter_v3.py`` (``_create_model_tracking_descriptor``,
``async_pre_call_hook``), ``group_info.rs`` and ``litellm/router_rust/steps/selection.py``
(``routing_pre_call_checks``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.wire import wire_server
from tests.integration.routing.dec1_capacity_support import (
    MODEL_SCOPE,
    capacity_message,
    capacity_proxy,
    chat,
    entry,
    error_message,
    group_name,
    posts,
    provider,
    scope_headers,
    statuses,
)

ENFORCE: Final = {"optional_pre_call_checks": ["enforce_model_rate_limits"]}


def test_a_groups_capacity_is_the_sum_of_its_deployments_rpm(gateway: Gateway, tmp_path: Path) -> None:
    """A group has two deployments with ``rpm: 1`` and ``rpm: 2``. The first response reports
    ``x-ratelimit-model_saturation_check-limit-requests: 3``, three requests are served and the fourth is refused
    with ``Model capacity reached for <group>. Priority: None, ..., Model RPM: 3, Remaining: 0``."""
    group: Final = group_name("sum")
    with (
        wire_server(provider()) as first,
        wire_server(provider()) as second,
        capacity_proxy(
            gateway, tmp_path, model_list=[entry(group, first.url, rpm=1), entry(group, second.url, rpm=2)]
        ) as candidate,
    ):
        responses: Final = [chat(candidate, group, candidate.key) for _ in range(4)]
        assert statuses(responses) == [200, 200, 200, 429], [response.text for response in responses]
        assert scope_headers(responses[0], MODEL_SCOPE) == {"limit-requests": "3", "remaining-requests": "2"}
        assert error_message(responses[3]) == capacity_message(group, None, "requests", rpm=3)
        assert len(posts(first)) + len(posts(second)) == 3


def test_a_deployment_without_rpm_adds_nothing_to_the_group_capacity(gateway: Gateway, tmp_path: Path) -> None:
    """A group has one deployment with ``rpm: 2`` and one without any limit. The group's window is 2, not
    unlimited: the third request is refused with ``Model RPM: 2`` even though the second deployment has no cap of
    its own."""
    group: Final = group_name("partial")
    with (
        wire_server(provider()) as limited,
        wire_server(provider()) as open_ended,
        capacity_proxy(
            gateway, tmp_path, model_list=[entry(group, limited.url, rpm=2), entry(group, open_ended.url)]
        ) as candidate,
    ):
        responses: Final = [chat(candidate, group, candidate.key) for _ in range(3)]
        assert statuses(responses) == [200, 200, 429], [response.text for response in responses]
        assert error_message(responses[2]) == capacity_message(group, None, "requests", rpm=2)


def test_a_groups_tpm_is_the_sum_of_its_deployments_tpm(gateway: Gateway, tmp_path: Path) -> None:
    """A group has deployments with ``tpm: 100`` and ``tpm: 50``. The first response reports
    ``x-ratelimit-model_saturation_check-limit-tokens: 150`` and ``-remaining-tokens: 150``: admission reserves no
    tokens, only settlement adds them."""
    group: Final = group_name("tpm")
    with (
        wire_server(provider()) as first,
        wire_server(provider()) as second,
        capacity_proxy(
            gateway, tmp_path, model_list=[entry(group, first.url, tpm=100), entry(group, second.url, tpm=50)]
        ) as candidate,
    ):
        served: Final = chat(candidate, group, candidate.key)
        assert served.status_code == 200, served.text
        assert scope_headers(served, MODEL_SCOPE) == {"limit-tokens": "150", "remaining-tokens": "150"}


def test_rpm_on_one_deployment_and_tpm_on_another_give_the_group_both_windows(gateway: Gateway, tmp_path: Path) -> None:
    """One deployment sets only ``rpm: 5`` and the other only ``tpm: 500``. The group carries both, so the first
    response reports model-wide ``limit-requests: 5``, ``remaining-requests: 4``, ``limit-tokens: 500`` and
    ``remaining-tokens: 500``."""
    group: Final = group_name("mixed")
    with (
        wire_server(provider()) as first,
        wire_server(provider()) as second,
        capacity_proxy(
            gateway, tmp_path, model_list=[entry(group, first.url, rpm=5), entry(group, second.url, tpm=500)]
        ) as candidate,
    ):
        served: Final = chat(candidate, group, candidate.key)
        assert served.status_code == 200, served.text
        assert scope_headers(served, MODEL_SCOPE) == {
            "limit-requests": "5",
            "remaining-requests": "4",
            "limit-tokens": "500",
            "remaining-tokens": "500",
        }


def test_rpm_set_in_model_info_counts_toward_the_group_capacity(gateway: Gateway, tmp_path: Path) -> None:
    """A deployment sets ``rpm: 3`` under ``model_info`` instead of ``litellm_params``. The merge reads either place,
    so the group's window is 3: three requests are served and the fourth is refused with ``Model RPM: 3``."""
    group: Final = group_name("info")
    with (
        wire_server(provider()) as wire,
        capacity_proxy(gateway, tmp_path, model_list=[entry(group, wire.url, model_info={"rpm": 3})]) as candidate,
    ):
        responses: Final = [chat(candidate, group, candidate.key) for _ in range(4)]
        assert statuses(responses) == [200, 200, 200, 429], [response.text for response in responses]
        assert error_message(responses[3]) == capacity_message(group, None, "requests", rpm=3)


def test_a_capacity_refusal_reaches_no_deployment_of_the_group(gateway: Gateway, tmp_path: Path) -> None:
    """A group of two deployments with ``rpm: 1`` each serves two requests. Three more requests are refused and
    neither provider receives anything after the first two calls."""
    group: Final = group_name("noreach")
    with (
        wire_server(provider()) as first,
        wire_server(provider()) as second,
        capacity_proxy(
            gateway, tmp_path, model_list=[entry(group, first.url, rpm=1), entry(group, second.url, rpm=1)]
        ) as candidate,
    ):
        assert statuses([chat(candidate, group, candidate.key) for _ in range(2)]) == [200, 200]
        served: Final = len(posts(first)) + len(posts(second))
        refused: Final = [chat(candidate, group, candidate.key) for _ in range(3)]
        assert statuses(refused) == [429, 429, 429], [response.text for response in refused]
        assert (served, len(posts(first)) + len(posts(second))) == (2, 0)


def test_every_deployments_traffic_counts_down_the_one_group_window(gateway: Gateway, tmp_path: Path) -> None:
    """A group has two deployments with ``rpm: 2`` each. Four requests are served by whichever deployment the
    router picks, and their model-wide ``remaining-requests`` reads 3, 2, 1, 0: one window for the group, not one
    per deployment."""
    group: Final = group_name("onewindow")
    with (
        wire_server(provider()) as first,
        wire_server(provider()) as second,
        capacity_proxy(
            gateway, tmp_path, model_list=[entry(group, first.url, rpm=2), entry(group, second.url, rpm=2)]
        ) as candidate,
    ):
        responses: Final = [chat(candidate, group, candidate.key) for _ in range(4)]
        assert statuses(responses) == [200] * 4
        assert [scope_headers(response, MODEL_SCOPE)["remaining-requests"] for response in responses] == [
            "3",
            "2",
            "1",
            "0",
        ]


def test_a_deployment_added_at_runtime_raises_the_groups_capacity(gateway: Gateway, tmp_path: Path) -> None:
    """A group starts with one deployment of ``rpm: 2``. A second deployment of the same group with ``rpm: 3`` is
    added through ``POST /model/new``. Once the proxy picked it up, a response reports model-wide
    ``limit-requests: 5``: the window is recomputed from the live group, not frozen at boot."""
    group: Final = group_name("grow")
    with wire_server(provider()) as first, wire_server(provider()) as second:
        with capacity_proxy(gateway, tmp_path, model_list=[entry(group, first.url, rpm=2)]) as candidate:
            assert scope_headers(chat(candidate, group, candidate.key), MODEL_SCOPE)["limit-requests"] == "2"
            with candidate.scenario() as scenario:
                scenario.model(name=group, api_base=f"{second.url}/v1", rpm=3)
                grown: Final = eventually(
                    lambda: chat(candidate, group, candidate.key),
                    lambda response: scope_headers(response, MODEL_SCOPE).get("limit-requests") == "5",
                    seconds=60,
                )
                assert grown.status_code == 200, grown.text


def test_enforce_model_rate_limits_moves_traffic_off_a_full_deployment_to_the_next(
    gateway: Gateway, tmp_path: Path
) -> None:
    """Without the dynamic limiter, ``enforce_model_rate_limits`` is on and ``num_retries: 1``. The ``order: 1``
    deployment has ``rpm: 2`` and the ``order: 2`` one has no cap. Four requests are all served: the first two by
    the ``order: 1`` deployment and the last two by the other, because the full deployment fails its attempt and
    the retry picks the next one. Modeled on ``test_enforce_model_rate_limits_daily_limits.py``, which pins the
    same skip for ``rpd``."""
    group: Final = group_name("skip")
    with (
        wire_server(provider()) as capped,
        wire_server(provider()) as spare,
        capacity_proxy(
            gateway,
            tmp_path,
            callbacks=(),
            model_list=[entry(group, capped.url, rpm=2, order=1), entry(group, spare.url, order=2)],
            router_settings={**ENFORCE, "num_retries": 1},
        ) as candidate,
    ):
        responses: Final = [chat(candidate, group, candidate.key) for _ in range(4)]
        assert statuses(responses) == [200] * 4, [response.text for response in responses]
        assert (len(posts(capped)), len(posts(spare))) == (2, 2)


def test_enforce_model_rate_limits_answers_429_when_every_deployment_is_full(gateway: Gateway, tmp_path: Path) -> None:
    """Without the dynamic limiter, ``enforce_model_rate_limits`` is on and both deployments of a group have
    ``rpm: 1``. After two served requests a third must be refused with HTTP 429 and reach neither provider.
    UNVERIFIED: the message depends on whether the router reports the deployment's ``Model rate limit exceeded``
    error or a ``No deployments available`` cooldown answer, so only the status and the provider silence are
    pinned."""
    group: Final = group_name("allfull")
    with (
        wire_server(provider()) as first,
        wire_server(provider()) as second,
        capacity_proxy(
            gateway,
            tmp_path,
            callbacks=(),
            model_list=[entry(group, first.url, rpm=1, order=1), entry(group, second.url, rpm=1, order=2)],
            router_settings={**ENFORCE, "num_retries": 1},
        ) as candidate,
    ):
        assert statuses([chat(candidate, group, candidate.key) for _ in range(2)]) == [200, 200]
        assert (len(posts(first)), len(posts(second))) == (1, 1)
        refused: Final = chat(candidate, group, candidate.key)
        assert refused.status_code == 429, refused.text
        assert (len(posts(first)), len(posts(second))) == (0, 0)


def test_the_group_window_refuses_before_any_deployment_cap_is_checked(gateway: Gateway, tmp_path: Path) -> None:
    """Both layers are on: the dynamic limiter and ``enforce_model_rate_limits``, with two deployments of ``rpm: 1``.
    After two served requests the third is refused by the group window with the dynamic limiter's ``Model
    capacity reached for <group>`` text (``Model RPM: 2``), because the proxy pre-call hook runs before the router
    picks a deployment."""
    group: Final = group_name("stack")
    with (
        wire_server(provider()) as first,
        wire_server(provider()) as second,
        capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(group, first.url, rpm=1, order=1), entry(group, second.url, rpm=1, order=2)],
            router_settings={**ENFORCE, "num_retries": 1},
        ) as candidate,
    ):
        responses: Final = [chat(candidate, group, candidate.key) for _ in range(3)]
        assert statuses(responses) == [200, 200, 429], [response.text for response in responses]
        assert error_message(responses[2]) == capacity_message(group, None, "requests", rpm=2)


def test_a_deployment_tpm_cap_moves_traffic_once_its_settled_tokens_fill_it(gateway: Gateway, tmp_path: Path) -> None:
    """Without the dynamic limiter, ``enforce_model_rate_limits`` is on and ``num_retries: 1``. The ``order: 1``
    deployment has ``tpm: 30`` and every answer uses 18 tokens. After two answers settle on it (36 tokens) the
    router must send further requests to the ``order: 2`` deployment, and every request is served.
    UNVERIFIED: the deployment's tpm counter is settled asynchronously, so the switch is polled for."""
    group: Final = group_name("tpmskip")
    with (
        wire_server(provider()) as capped,
        wire_server(provider()) as spare,
        capacity_proxy(
            gateway,
            tmp_path,
            callbacks=(),
            model_list=[entry(group, capped.url, tpm=30, order=1), entry(group, spare.url, order=2)],
            router_settings={**ENFORCE, "num_retries": 1},
        ) as candidate,
    ):
        assert statuses([chat(candidate, group, candidate.key) for _ in range(2)]) == [200, 200]
        moved: Final = eventually(
            lambda: (chat(candidate, group, candidate.key), len(posts(spare))),
            lambda outcome: outcome[0].status_code == 200 and outcome[1] == 1,
            seconds=30,
        )
        assert moved[0].status_code == 200, moved[0].text
