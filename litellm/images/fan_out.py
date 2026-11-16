from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from typing import Final, TypeVar

from pydantic import TypeAdapter

from litellm.types.utils import ImageResponse, ImageUsage

_T = TypeVar("_T")
_REQUEST_COUNT: Final = TypeAdapter(int)
_USAGE: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])


def image_request_count(requested_images: object) -> int:
    """How many single-image upstream requests serve a caller's `n` for a provider that returns one image per call."""
    return 1 if requested_images is None else max(_REQUEST_COUNT.validate_python(requested_images), 1)


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
    return first.model_copy(
        update={
            "data": [image for response in responses for image in response.data or []],  # comprehension-ok: flatten
            "usage": _merge_usage([response.usage for response in responses if response.usage is not None]),
        }
    )


def run_concurrently(request_count: int, send: Callable[[], _T]) -> tuple[_T, ...]:
    if request_count == 1:
        return (send(),)
    with ThreadPoolExecutor(max_workers=request_count) as pool:
        futures: Final = tuple(pool.submit(send) for _ in range(request_count))
        return tuple(future.result() for future in futures)
