from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Generic, Literal, TypeAlias, TypeVar

from pydantic import TypeAdapter, ValidationError
from pydantic_core import ErrorDetails
from typing_extensions import assert_never

import litellm
from litellm.types.vector_stores import VectorStoreSearchQueryImage

_QUERY_IMAGE: Final = TypeAdapter(VectorStoreSearchQueryImage)
_Query: Final = TypeVar("_Query", bound=str | Sequence[str])


@dataclass(frozen=True, slots=True)
class TextSearchQuery(Generic[_Query]):
    query: _Query
    image: None = None


@dataclass(frozen=True, slots=True)
class ImageSearchQuery:
    image: VectorStoreSearchQueryImage
    query: Literal[""] = ""


@dataclass(frozen=True, slots=True)
class QueryAndImageBothSent:
    pass


@dataclass(frozen=True, slots=True)
class NoQuerySent:
    pass


@dataclass(frozen=True, slots=True)
class InvalidQueryImage:
    problems: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QueryImageUnsupported:
    custom_llm_provider: str


SearchQuery: TypeAlias = TextSearchQuery[_Query] | ImageSearchQuery
SearchQueryError: TypeAlias = QueryAndImageBothSent | NoQuerySent | InvalidQueryImage | QueryImageUnsupported


def _as_object_list(value: object) -> Sequence[object] | None:
    return value if isinstance(value, list) else None


def _has_text(query: object) -> bool:
    if isinstance(query, str):
        return bool(query.strip())
    items: Final = _as_object_list(query)
    if items is None:
        return True
    return any(not isinstance(item, str) or item.strip() for item in items)


def _problem(detail: ErrorDetails) -> str:
    location: Final = ".".join(("query_image", *(str(part) for part in detail["loc"])))
    return f"{location}: {detail['msg']}"


def _validated_query_image(query_image: object) -> ImageSearchQuery | InvalidQueryImage:
    try:
        return ImageSearchQuery(image=_QUERY_IMAGE.validate_python(query_image))
    except ValidationError as error:
        return InvalidQueryImage(
            problems=tuple(_problem(detail) for detail in error.errors(include_url=False, include_input=False))
        )


def resolve_search_query(
    query: _Query | None,
    query_image: object,
    custom_llm_provider: str,
    supports_query_image: bool,
    drop_params: bool,
) -> SearchQuery[_Query] | SearchQueryError:
    if query_image is None:
        return NoQuerySent() if query is None else TextSearchQuery(query=query)
    text_query: Final = query if query is not None and _has_text(query) else None
    if not supports_query_image and not drop_params:
        return QueryImageUnsupported(custom_llm_provider=custom_llm_provider)
    if not supports_query_image:
        return NoQuerySent() if text_query is None else TextSearchQuery(query=text_query)
    if text_query is not None:
        return QueryAndImageBothSent()
    return _validated_query_image(query_image)


def _error_message(error: SearchQueryError) -> str:
    match error:
        case QueryAndImageBothSent():
            return "Send either query or query_image, not both. Leave query empty when you send query_image."
        case NoQuerySent():
            return "query or query_image is required."
        case InvalidQueryImage(problems=problems):
            return (
                f"Invalid query_image: {'; '.join(problems)}. Expected "
                '{"type": "base64", "media_type": "image/jpeg" | "image/png" | "image/gif" | "image/webp", '
                '"data": "<base64>"}.'
            )
        case QueryImageUnsupported(custom_llm_provider=provider):
            return (
                f"{provider} vector stores do not support query_image. To drop it, set `litellm.drop_params=True` "
                "or for proxy:\n\n`litellm_settings:\n drop_params: true`\n"
            )
        case _:
            assert_never(error)


def searched_query(
    query: _Query | None,
    query_image: object,
    custom_llm_provider: str,
    supports_query_image: bool,
    drop_params: bool,
) -> SearchQuery[_Query]:
    resolved: Final = resolve_search_query(
        query=query,
        query_image=query_image,
        custom_llm_provider=custom_llm_provider,
        supports_query_image=supports_query_image,
        drop_params=drop_params,
    )
    match resolved:
        case TextSearchQuery() | ImageSearchQuery():
            return resolved
        case QueryAndImageBothSent() | NoQuerySent() | InvalidQueryImage() | QueryImageUnsupported():
            raise litellm.BadRequestError(
                message=_error_message(resolved), model=None, llm_provider=custom_llm_provider
            )
        case _:
            assert_never(resolved)
