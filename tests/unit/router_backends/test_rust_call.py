from __future__ import annotations

from collections.abc import Generator, Mapping
from typing import Final

import httpx
import pytest

import litellm
from litellm.router_backends.python_router import PythonRouter
from litellm.router_backends.rust_call import Attempt, RoutedCall, classify, exception_classes
from litellm.types.router import RouterRateLimitError

MODEL_LIST: Final = (
    {"model_name": "g", "litellm_params": {"model": "openai/a", "api_key": "k"}, "model_info": {"id": "a"}},
    {"model_name": "h", "litellm_params": {"model": "openai/b", "api_key": "k"}, "model_info": {"id": "b"}},
)


@pytest.fixture
def normalizer() -> Generator[PythonRouter]:
    router: Final = PythonRouter(model_list=[dict(entry) for entry in MODEL_LIST], fallbacks=[{"g": ["h"]}])
    router.discard()
    yield router


def _call(normalizer: PythonRouter, metadata: dict[str, object]) -> RoutedCall:  # mutable-ok: the request's own bucket
    call: Final = RoutedCall(
        normalizer,  # pyright: ignore[reportArgumentType]  # PythonRouter provides the AttemptRouter members
        {"model": "g", "messages": [], "metadata": metadata, "num_retries": 3, "fallbacks": []},
        "metadata",
        lambda name: getattr(normalizer, name),
    )
    call.start("g")
    return call


def _rate_limit(headers: Mapping[str, str] | None = None) -> litellm.RateLimitError:
    response: Final = httpx.Response(
        429, headers=dict(headers or {}), request=httpx.Request("POST", "https://example.invalid")
    )
    return litellm.RateLimitError(message="slow down", llm_provider="openai", model="a", response=response)


@pytest.mark.parametrize(
    ("error", "expected"),
    (
        (
            litellm.ContextWindowExceededError(message="m", model="a", llm_provider="openai"),
            ("ContextWindowExceeded", "BadRequest"),
        ),
        (
            litellm.ContentPolicyViolationError(message="m", model="a", llm_provider="openai"),
            ("ContentPolicyViolation", "BadRequest"),
        ),
        (_rate_limit(), ("RateLimit",)),
        (litellm.Timeout(message="m", model="a", llm_provider="openai"), ("Timeout",)),
        (ValueError("not litellm"), ()),
    ),
)
def test_exception_classes_follow_the_mro_most_specific_first(error: BaseException, expected: tuple[str, ...]) -> None:
    assert exception_classes(error) == expected


def test_classify_reports_the_retry_after_each_consumer_reads() -> None:
    error: Final = _rate_limit({"retry-after": "7"})

    classified: Final = classify(error, {}, callbacks_ran=True)

    assert classified["status_code"] == 429
    assert classified["sleep_retry_after"] == 7
    assert classified["cooldown_retry_after"] == 7
    assert classified["callbacks_ran"] is True
    assert classified["exact_litellm_type"] is True
    assert classified["message"] == str(error)


def test_classify_without_headers_reports_no_retry_after() -> None:
    classified: Final = classify(ValueError("x"), {}, callbacks_ran=False)

    assert classified["sleep_retry_after"] == -1
    assert classified["cooldown_retry_after"] is None
    assert classified["status_code"] is None
    assert classified["exact_litellm_type"] is False


def test_a_fallback_hop_opens_a_stamped_copy_of_its_parent_bucket(normalizer: PythonRouter) -> None:
    metadata: Final[dict[str, object]] = {"model_group": "g", "trace": "kept"}  # mutable-ok: the request's bucket
    call: Final = _call(normalizer, metadata)
    error: Final = _rate_limit()

    call.failure(
        error,
        [
            {"op": "log_retry", "bucket": 0, "model": "g", "error": error},
            {
                "op": "open_bucket",
                "id": 1,
                "copy_of": 0,
                "original_model_group": "g",
                "model_group": "h",
                "attempted_fallbacks": 1,
                "max_fallbacks": 5,
            },
        ],
    )

    hop: Final = call._buckets[1]  # pyright: ignore[reportPrivateUsage]  # the hop bucket is the behavior under test
    assert hop is not metadata
    assert hop["trace"] == "kept"
    assert (hop["model_group"], hop["attempted_fallbacks"], hop["original_model_group"]) == ("h", 1, "g")
    assert metadata["model_group"] == "g"
    assert metadata["request_retry_count"] == 1
    breadcrumbs: Final = metadata["previous_models"]
    assert isinstance(breadcrumbs, tuple)
    assert [crumb["model_group"] for crumb in breadcrumbs] == ["g"]


def test_one_rejection_id_raises_one_exception_object(normalizer: PythonRouter) -> None:
    call: Final = _call(normalizer, {})
    rejection: Final = {
        "rejection_id": 1,
        "kind": "no_deployments_available",
        "model": "g",
        "cooldown_time": 5,
        "cooldown_list": ["a"],
        "model_ids": ["a"],
        "enable_pre_call_checks": False,
    }

    first: Final = call.failure(rejection, [])
    second: Final = call.failure(dict(rejection), [])

    assert first is second
    assert isinstance(first, RouterRateLimitError)
    assert first.all_deployments_in_cooldown is True
    assert first.cooldown_time == 5


def test_no_healthy_deployments_rejection_is_the_routers_bad_request(normalizer: PythonRouter) -> None:
    error: Final = _call(normalizer, {}).failure(
        {"rejection_id": 2, "kind": "no_healthy_deployments", "model": "nope"}, []
    )

    assert isinstance(error, litellm.BadRequestError)
    assert "You passed in model=nope." in str(error)


@pytest.mark.parametrize("expose", (True, False))
def test_fallback_outcome_explains_the_failed_fallback_when_exposed(
    normalizer: PythonRouter, monkeypatch: pytest.MonkeyPatch, expose: bool
) -> None:
    monkeypatch.setattr(litellm, "expose_router_debug_in_errors", expose)
    primary: Final = _rate_limit()
    fallback: Final = litellm.InternalServerError(message="backend exploded", model="b", llm_provider="openai")
    before: Final = primary.message

    raised: Final = _call(normalizer, {}).failure(
        primary,
        [{"op": "fallback_outcome", "error": primary, "model_group": "g", "attempted": ["h"], "last": fallback}],
    )

    assert raised is primary
    message: Final = str(getattr(raised, "message", ""))
    assert (message != before) is expose
    assert ("Fallback to h also failed" in message) is expose
    assert ("backend exploded" in message) is expose


def test_exhausted_retries_stamp_the_raised_error(normalizer: PythonRouter) -> None:
    error: Final = _rate_limit()

    raised: Final = _call(normalizer, {}).failure(
        error, [{"op": "stamp_retries", "error": error, "max_retries": 3, "num_retries": 3}]
    )

    assert (getattr(raised, "max_retries", None), getattr(raised, "num_retries", None)) == (3, 3)


def test_router_only_kwargs_never_reach_the_attempt(normalizer: PythonRouter) -> None:
    call: Final = _call(normalizer, {})

    kwargs: Final = call._prepare(  # pyright: ignore[reportPrivateUsage]  # the attempt kwargs are the behavior under test
        call_attempt(bucket=0)
    )

    assert "num_retries" not in kwargs
    assert "fallbacks" not in kwargs
    assert "model" not in kwargs
    assert kwargs["metadata"] is call._buckets[0]  # pyright: ignore[reportPrivateUsage]  # identity is the contract
    assert (kwargs["metadata"]["attempted_retries"], kwargs["metadata"]["max_retries"]) == (1, 2)


def call_attempt(bucket: int) -> Attempt:
    return Attempt(
        deployment_id="a",
        model_group="g",
        bucket=bucket,
        fallback_depth=0,
        model_group_size=1,
        attempted_retries=1,
        max_retries=2,
        ops=(),
    )
