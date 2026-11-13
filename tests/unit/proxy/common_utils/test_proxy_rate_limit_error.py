from typing import Final

import pytest

from litellm.proxy.common_utils.proxy_rate_limit_error import (
    SATURATION_FALLBACK_DESCRIPTOR_KEY,
    ProxyRateLimitError,
    RateLimitFallback,
    arm_saturation_fallback_walk,
    disarm_saturation_fallback_walk,
    fallback_on_saturation_enabled,
    fallback_on_saturation_requested,
    is_saturation_fallback,
    mark_saturation_fallbacks_exhausted,
    rate_limit_fallback_from_request_data,
    rate_limit_fallback_headers,
    saturation_fallbacks_exhausted,
)

STAMP: Final = {"original_model_group": "opus", "descriptor_key": "model_per_end_user"}


@pytest.mark.parametrize("bucket", ("metadata", "litellm_metadata"))
def test_rate_limit_fallback_is_read_from_either_metadata_bucket(bucket: str) -> None:
    fallback: Final = rate_limit_fallback_from_request_data({"model": "luna", bucket: {"rate_limit_fallback": STAMP}})

    assert fallback is not None
    assert fallback == RateLimitFallback(original_model_group="opus", descriptor_key="model_per_end_user")
    assert fallback.reason == "rate_limit:model_per_end_user"


@pytest.mark.parametrize(
    "request_data",
    (
        {"model": "luna"},
        {"model": "luna", "metadata": {}},
        {"model": "luna", "metadata": {"rate_limit_fallback": "opus"}},
        {"model": "luna", "metadata": {"rate_limit_fallback": {"original_model_group": "opus"}}},
    ),
)
def test_a_request_without_a_complete_stamp_has_no_rate_limit_fallback(request_data: dict[str, object]) -> None:
    assert rate_limit_fallback_from_request_data(request_data) is None


@pytest.mark.parametrize(("router_hops", "expected"), ((None, "1"), (0, "1"), ("0", "1"), (2, "3"), ("junk", "1")))
def test_rate_limit_fallback_headers_count_the_hop_on_top_of_the_router(router_hops: object, expected: str) -> None:
    headers: Final = rate_limit_fallback_headers(
        RateLimitFallback(original_model_group="opus", descriptor_key="model_per_key"), router_hops
    )

    assert headers == {
        "x-litellm-attempted-fallbacks": expected,
        "x-litellm-fallback-reason": "rate_limit:model_per_key",
    }


@pytest.mark.parametrize(
    ("attempts", "router_hops", "expected"), ((2, None, "2"), (2, 0, "2"), (3, 1, "4"), (0, 0, "1"))
)
def test_rate_limit_fallback_headers_count_every_candidate_a_limit_rejected(
    attempts: int, router_hops: object, expected: str
) -> None:
    headers: Final = rate_limit_fallback_headers(
        RateLimitFallback(original_model_group="opus", descriptor_key="model_per_key", attempts=attempts), router_hops
    )

    assert headers["x-litellm-attempted-fallbacks"] == expected


@pytest.mark.parametrize(
    ("router_settings", "expected"),
    (
        ({"fallback_on_saturation": True}, True),
        ({"fallback_on_saturation": False}, False),
        ({"fallback_on_saturation": "yes"}, False),
        ({"fallbacks": [{"luna": ["opus"]}]}, False),
        (None, False),
        ("fallback_on_saturation", False),
    ),
)
def test_fallback_on_saturation_is_enabled_only_by_a_true_router_setting(
    router_settings: object, expected: bool
) -> None:
    assert fallback_on_saturation_enabled(router_settings) is expected


def test_a_request_asks_for_saturation_fallbacks_only_while_the_walk_is_armed_and_until_it_has_run() -> None:
    request: Final = {"model": "luna", "router_settings_override": {"fallback_on_saturation": True}}
    assert not fallback_on_saturation_requested(request)

    walk: Final = arm_saturation_fallback_walk()
    try:
        assert fallback_on_saturation_requested(request)
        assert not saturation_fallbacks_exhausted()
        mark_saturation_fallbacks_exhausted()
        assert saturation_fallbacks_exhausted()
        assert not fallback_on_saturation_requested(request)
    finally:
        disarm_saturation_fallback_walk(walk)
    assert not saturation_fallbacks_exhausted()
    assert not fallback_on_saturation_requested(request)


def test_client_metadata_never_arms_skips_or_exhausts_the_walk() -> None:
    forged: Final = {
        "model": "luna",
        "router_settings_override": {"fallback_on_saturation": True},
        "metadata": {"saturation_fallback_walk": True, "saturation_fallbacks_exhausted": True},
        "litellm_metadata": {"saturation_fallback_walk": True, "saturation_fallbacks_exhausted": True},
    }
    assert not fallback_on_saturation_requested(forged)
    assert not saturation_fallbacks_exhausted()
    walk: Final = arm_saturation_fallback_walk()
    try:
        assert fallback_on_saturation_requested(forged)
    finally:
        disarm_saturation_fallback_walk(walk)


def test_a_request_without_the_setting_never_asks_for_saturation_fallbacks() -> None:
    assert not fallback_on_saturation_requested({"model": "luna"})
    assert not fallback_on_saturation_requested({"model": "luna", "router_settings_override": {"fallbacks": []}})


def test_only_the_saturation_descriptor_marks_a_fallback_walk_the_model_can_still_serve() -> None:
    assert is_saturation_fallback(ProxyRateLimitError(detail="x", descriptor_key=SATURATION_FALLBACK_DESCRIPTOR_KEY))
    assert not is_saturation_fallback(ProxyRateLimitError(detail="x", descriptor_key="model_per_team"))
    assert not is_saturation_fallback(ProxyRateLimitError(detail="x"))
