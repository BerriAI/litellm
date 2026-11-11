from typing import Final

import pytest

from litellm.proxy.common_utils.proxy_rate_limit_error import (
    RateLimitFallback,
    rate_limit_fallback_from_request_data,
    rate_limit_fallback_headers,
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
