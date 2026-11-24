from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Final, cast  # noqa: TID251  # narrows untyped request dicts at the session boundary

import httpx
from pydantic import TypeAdapter

from litellm.exceptions import AuthenticationError, UnsupportedParamsError
from litellm.litellm_core_utils.core_helpers import (
    as_object_list,
    as_str_mapping,
    drops_unsupported_params,
    loads_json_object,
)
from litellm.litellm_core_utils.prompt_templates.factory import EMPTY_TEXT_PLACEHOLDER
from litellm.litellm_core_utils.url_utils import ensure_api_route
from litellm.llms.anthropic.messages.native.transformation import (
    AnthropicMessagesConfig,
)
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse
from litellm.types.router import GenericLiteLLMParams

from ..authenticator import Authenticator
from ..common_utils import (
    DEFAULT_GITHUB_COPILOT_API_BASE,
    GetAccessTokenError,
    get_copilot_default_headers,
    pin_session_authorization,
)
from ..per_user_auth import require_github_copilot_user_session
from ..usage import copilot_cache_counts

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_MESSAGES_PROXY_API_VERSION: Final = "2026-06-01"
_REQUEST_FIELDS: Final = TypeAdapter(dict[str, object])


def _blocks(content: object) -> tuple[Mapping[str, object], ...]:
    return tuple(block for item in as_object_list(content) or () if (block := as_str_mapping(item)) is not None)


def _documents(content: object) -> Iterator[Mapping[str, object]]:
    for block in _blocks(content):
        match block:
            case {"type": "document"}:
                yield block
            case {"type": "tool_result", "content": nested}:
                yield from (inner for inner in _blocks(nested) if inner.get("type") == "document")
            case _:
                pass


def _text_document_data(block: Mapping[str, object]) -> str | None:
    if block.get("type") != "document":
        return None
    match as_str_mapping(block.get("source")):
        case {"type": "text", "data": str(data)}:
            return data
        case _:
            return None


def _shifts_a_document_index(documents: Sequence[Mapping[str, object]]) -> bool:
    converted: Final = tuple(_text_document_data(document) is not None for document in documents)
    first: Final = next((index for index, is_converted in enumerate(converted) if is_converted), len(converted))
    return not all(converted[first:])


@dataclass(frozen=True, slots=True)
class _DroppedBreakpoint:
    cache_control: object


def _text_document_content(content: object, drop_citations: bool) -> object:
    items: Final = as_object_list(content)
    if items is None:
        return content
    return _kept_or_placeholder(
        items, tuple(chain.from_iterable(_content_blocks(item, drop_citations) for item in items))
    )


def _kept_or_placeholder(items: Sequence[object], stream: Sequence[object]) -> Sequence[object]:
    blocks, orphan = _with_breakpoints(stream)
    if blocks or not items:
        return list(blocks)
    return [_with_cache_control({"type": "text", "text": EMPTY_TEXT_PLACEHOLDER}, orphan)]


def _with_breakpoints(stream: Sequence[object]) -> tuple[tuple[object, ...], object | None]:
    kept: Final = tuple(index for index, item in enumerate(stream) if not isinstance(item, _DroppedBreakpoint))
    if not kept:
        return (), _last_breakpoint(stream)
    return (
        tuple(
            _with_cache_control(stream[index], _last_breakpoint(stream[start:end]))
            for index, start, end in zip(kept, (0, *kept[1:]), (*kept[1:], len(stream)))
        ),
        None,
    )


def _last_breakpoint(items: Sequence[object]) -> object | None:
    breakpoints: Final = tuple(item.cache_control for item in items if isinstance(item, _DroppedBreakpoint))
    return breakpoints[-1] if breakpoints else None


def _with_cache_control(item: object, cache_control: object | None) -> object:
    block: Final = as_str_mapping(item)
    if cache_control is None or block is None or "cache_control" in block:
        return item
    return {**block, "cache_control": cache_control}


def _content_blocks(item: object, drop_citations: bool) -> tuple[object, ...]:
    block: Final = as_str_mapping(item)
    nested: Final = (
        as_object_list(block.get("content")) if block is not None and block.get("type") == "tool_result" else None
    )
    if block is None or nested is None:
        return _document_blocks(item, drop_citations)
    converted: Final = tuple(chain.from_iterable(_document_blocks(inner, drop_citations) for inner in nested))
    return ({**block, "content": _kept_or_placeholder(nested, converted)},)


def _document_blocks(item: object, drop_citations: bool) -> tuple[object, ...]:
    block: Final = as_str_mapping(item)
    if block is None or block.get("type") != "document":
        return (item,)
    data: Final = _text_document_data(block)
    if data is None:
        uncited: Final = drop_citations and "citations" in block
        return ({key: value for key, value in block.items() if key != "citations"},) if uncited else (item,)
    metadata: Final = tuple(
        f"{key.capitalize()}: {value}"
        for key in ("title", "context")
        if isinstance(value := block.get(key), str) and value
    )
    text: Final = "\n\n".join(part for part in (*metadata, data) if part)
    if not text.strip():
        return (_DroppedBreakpoint(block["cache_control"]),) if "cache_control" in block else ()
    return (
        {
            "type": "text",
            "text": text,
            **({"cache_control": block["cache_control"]} if "cache_control" in block else {}),
        },
    )


class GithubCopilotAnthropicMessagesConfig(AnthropicMessagesConfig):
    """
    GitHub Copilot implementation of Anthropic messages API.
    Routes requests to Copilot's /v1/messages endpoint with appropriate authentication and headers.
    """

    def __init__(self) -> None:
        super().__init__()
        self.authenticator = Authenticator()

    @property
    def custom_llm_provider(self) -> str | None:
        return "github_copilot"

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: Sequence[Mapping[str, object]],
        anthropic_messages_optional_request_params: dict[str, object],  # mutable-ok: base pops max_tokens in place
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: the base transformer takes the caller's headers dict
    ) -> dict[str, object]:  # mutable-ok: the base request transformer requires a JSON dict
        documents: Final = tuple(chain.from_iterable(_documents(message.get("content")) for message in messages))
        has_text_documents: Final = any(_text_document_data(document) is not None for document in documents)
        cites_documents: Final = has_text_documents and any(
            (as_str_mapping(document.get("citations")) or {}).get("enabled") is True for document in documents
        )
        if cites_documents and not drops_unsupported_params(litellm_params.drop_params):
            raise UnsupportedParamsError(
                message="GitHub Copilot text documents cannot preserve structured citations when converted to text. "
                "Disable document citations, set drop_params to drop them, "
                "or use a provider that supports text document sources.",
                model=model,
                llm_provider="github_copilot",
            )
        drop_citations: Final = cites_documents and _shifts_a_document_index(documents)
        normalized: Final = [
            {**message, "content": _text_document_content(message["content"], drop_citations)}
            if has_text_documents and message.get("role") == "user" and "content" in message
            else dict(message)
            for message in messages
        ]
        return _REQUEST_FIELDS.validate_python(
            super().transform_anthropic_messages_request(  # pyright: ignore[reportUnknownMemberType]  # the base transformer uses legacy untyped containers
                model=model,
                messages=normalized,
                anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
                litellm_params=litellm_params,
                headers=headers,
            )
        )

    def transform_anthropic_messages_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> AnthropicMessagesResponse:
        response: Final = super().transform_anthropic_messages_response(model, raw_response, logging_obj)
        usage: Final = response.get("usage")
        raw_payload: Final = loads_json_object(raw_response.content)
        if usage is not None and raw_payload is not None:
            read, write = copilot_cache_counts(usage, raw_payload.get("copilot_usage"))
            if read is not None:
                usage["cache_read_input_tokens"] = read
            if write is not None:
                usage["cache_creation_input_tokens"] = write
        return response

    def handles_web_search_natively(self) -> bool:
        """
        Copilot's /v1/messages endpoint does not execute ``web_search`` tools, so
        the interception handler must short-circuit web-search-only requests
        instead of routing them here.
        """
        return False

    def should_filter_anthropic_beta_headers(self) -> bool:
        """
        Copilot's /v1/messages is a native Anthropic Messages passthrough, so
        ``anthropic-beta`` values injected by ``_update_headers_with_anthropic_beta``
        (context_management, structured outputs, ...) must reach the upstream
        verbatim. The default provider-scoped filter would drop them because
        github_copilot has no entry in ``anthropic_beta_headers_config.json``.
        """
        return False

    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: Sequence[object],
        optional_params: dict[str, object],
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict, str | None]:
        """
        Validate environment for GitHub Copilot and add Copilot-specific headers.

        The caller-supplied ``api_base`` is intentionally ignored. Routing this
        request anywhere other than the authenticated Copilot endpoint would
        leak the Copilot bearer token to a caller-controlled URL.
        """
        # Always use the Copilot endpoint resolved from the authenticated
        # session, never the caller-supplied api_base. rstrip so a
        # tenant-specific base with a trailing slash does not yield a
        # double-slash URL once "/v1/messages" is appended downstream.
        user_session: Final = require_github_copilot_user_session(
            cast("dict[str, object]", litellm_params)  # cast-ok: litellm_params arrives as an untyped request dict
        )
        dynamic_api_base: Final = (
            user_session.api_base
            if user_session is not None
            else (self.authenticator.get_api_base() or DEFAULT_GITHUB_COPILOT_API_BASE)
        ).rstrip("/")
        try:
            dynamic_api_key: Final = (
                user_session.token if user_session is not None else self.authenticator.get_api_key()
            )
        except GetAccessTokenError as e:
            raise AuthenticationError(
                model=model,
                llm_provider="github_copilot",
                message=str(e),
            )

        # Merge Copilot headers with provided headers
        copilot_headers: Final = get_copilot_default_headers(dynamic_api_key)
        for key, value in copilot_headers.items():
            if key not in headers:
                headers[key] = value
        if user_session is not None:
            pin_session_authorization(
                cast("dict[str, str]", headers),  # cast-ok: headers is a str-valued request dict at runtime
                user_session.token,
                user_session.integration_id,
            )

        headers["openai-intent"] = "messages-proxy"
        headers["x-interaction-type"] = "messages-proxy"
        headers["x-github-api-version"] = _MESSAGES_PROXY_API_VERSION

        if "anthropic-version" not in headers:
            headers["anthropic-version"] = "2023-06-01"

        headers = self._update_headers_with_anthropic_beta(
            headers, optional_params, custom_llm_provider="github_copilot", messages=messages
        )

        return headers, dynamic_api_base

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict[str, object],
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        """
        Return the complete URL for GitHub Copilot /v1/messages endpoint.

        ``api_base`` here is the value already resolved by
        ``validate_anthropic_messages_environment`` (the authenticated Copilot
        host), not the raw caller-supplied base — that one is discarded there to
        avoid leaking the Copilot bearer token to a caller-controlled URL. We
        reuse it to avoid a second authenticator read, falling back to a fresh
        resolution only if it was not provided.
        """
        user_session: Final = require_github_copilot_user_session(
            cast("dict[str, object]", litellm_params)  # cast-ok: litellm_params arrives as an untyped request dict
        )
        resolved: Final = (
            (user_session.api_base if user_session is not None else None)
            or api_base
            or self.authenticator.get_api_base()
            or DEFAULT_GITHUB_COPILOT_API_BASE
        )
        return ensure_api_route(resolved, "v1/messages")
