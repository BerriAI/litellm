from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from typing import Final, Protocol, TypeVar

from pydantic import TypeAdapter

from litellm.constants import MAX_IMAGE_FAN_OUT_REQUESTS
from litellm.exceptions import BadRequestError
from litellm.types.utils import ImageResponse, ImageUsage, Usage

_T = TypeVar("_T")
_REQUEST_COUNT: Final = TypeAdapter(int)
_USAGE: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])


class _Price(Protocol):
    @property
    def cost(self) -> float | None: ...


class _SpendLog(Protocol):
    def uncached_price(self, result: object) -> _Price: ...

    def record_partial_usage_for_failure(self, usage: Usage, response_cost: float) -> None: ...


def image_request_count(requested_images: object, *, model: str, llm_provider: str) -> int:
    """How many single-image upstream requests serve a caller's `n` for a provider that returns one image per call."""
    if requested_images is None:
        return 1
    request_count: Final = max(_REQUEST_COUNT.validate_python(requested_images), 1)
    if request_count > MAX_IMAGE_FAN_OUT_REQUESTS:
        raise BadRequestError(
            message=f"n={request_count} is above the maximum of {MAX_IMAGE_FAN_OUT_REQUESTS} images per request.",
            model=model,
            llm_provider=llm_provider,
        )
    return request_count


def _sum_token_counts(left: object, right: object) -> object:
    unsummed: Final = right if left is None else left
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return _sum_usage(_USAGE.validate_python(left), _USAGE.validate_python(right))
    if isinstance(left, int) and isinstance(right, int):
        return left + right
    return unsummed


def _sum_usage(left: Mapping[str, object], right: Mapping[str, object]) -> Mapping[str, object]:
    return {name: _sum_token_counts(left.get(name), right.get(name)) for name in {**left, **right}}


def _merge_usage(usages: Sequence[ImageUsage]) -> ImageUsage | None:
    if not usages:
        return None
    return ImageUsage.model_validate(
        reduce(_sum_usage, (_USAGE.validate_python(usage.model_dump()) for usage in usages))
    )


def merge_image_responses(responses: Sequence[ImageResponse]) -> ImageResponse:
    first: Final = responses[0]
    if len(responses) == 1:
        return first
    merged: Final = first.model_copy(
        update={
            "data": [image for response in responses for image in response.data or []],  # comprehension-ok: flatten
            "usage": _merge_usage([response.usage for response in responses if response.usage is not None]),
        }
    )
    merged.hidden_params = {**first.hidden_params}
    return merged


def _outcome(send: Callable[[], _T]) -> _T | BaseException:
    try:
        return send()
    except Exception as e:
        return e


def run_concurrently(request_count: int, send: Callable[[], _T]) -> tuple[_T | BaseException, ...]:
    """Every request runs to completion, so a failed one cannot hide what the others were billed."""
    if request_count == 1:
        return (_outcome(send),)
    with ThreadPoolExecutor(max_workers=request_count) as pool:
        futures: Final = tuple(pool.submit(_outcome, send) for _ in range(request_count))
        return tuple(future.result() for future in futures)


def split_outcomes(outcomes: Sequence[_T | BaseException]) -> tuple[tuple[_T, ...], Exception | None]:
    failure: Final = next((outcome for outcome in outcomes if isinstance(outcome, BaseException)), None)
    if failure is not None and not isinstance(failure, Exception):
        raise failure
    return tuple(outcome for outcome in outcomes if not isinstance(outcome, BaseException)), failure


def _summed_cost(logging_obj: _SpendLog, responses: Sequence[ImageResponse]) -> float | None:
    """Each upstream call is priced on its own usage, so summed tokens never cross a pricing tier no call reached."""
    costs: Final = tuple(logging_obj.uncached_price(response).cost for response in responses)
    return None if None in costs else sum(cost for cost in costs if cost is not None)


def merge_billed_image_responses(logging_obj: _SpendLog, responses: Sequence[ImageResponse]) -> ImageResponse:
    merged: Final = merge_image_responses(responses)
    if len(responses) == 1:
        return merged
    cost: Final = _summed_cost(logging_obj, responses)
    if cost is not None:
        merged.hidden_params = {**merged.hidden_params, "response_cost": cost}
    return merged


def record_partial_image_spend(logging_obj: _SpendLog, responses: Sequence[ImageResponse]) -> None:
    """Bill the images the provider did return when another request of the same fan-out failed."""
    if not responses:
        return
    usage: Final = merge_image_responses(responses).usage
    logging_obj.record_partial_usage_for_failure(
        Usage(
            prompt_tokens=usage.input_tokens if usage else 0,
            completion_tokens=usage.output_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        ),
        _summed_cost(logging_obj, responses) or 0.0,
    )
