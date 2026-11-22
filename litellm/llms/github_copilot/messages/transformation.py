from collections.abc import Iterator, Mapping, Sequence
from itertools import chain
from typing import Final, cast  # noqa: TID251  # narrows untyped request dicts at the session boundary

from pydantic import TypeAdapter

from litellm.exceptions import AuthenticationError, UnsupportedParamsError
from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping
from litellm.litellm_core_utils.url_utils import ensure_api_route
from litellm.llms.anthropic.messages.native.transformation import (
    AnthropicMessagesConfig,
)
from litellm.types.router import GenericLiteLLMParams

from ..authenticator import Authenticator
from ..common_utils import (
    DEFAULT_GITHUB_COPILOT_API_BASE,
    GetAccessTokenError,
    get_copilot_default_headers,
    pin_session_authorization,
)
from ..per_user_auth import require_github_copilot_user_session

_MESSAGES_PROXY_API_VERSION: Final = "2026-06-01"
_REQUEST_FIELDS: Final = TypeAdapter(dict[str, object])


def _documents(content: object) -> Iterator[Mapping[str, object]]:
    for item in as_object_list(content) or ():
        match as_str_mapping(item):
            case {"type": "document"} as document:
                yield document
            case {"type": "tool_result", "content": nested}:
                yield from _documents(nested)
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


def _text_document_content(content: object) -> object:
    items: Final = as_object_list(content)
    return content if items is None else [_text_document_block(item) for item in items]


def _text_document_block(item: object) -> object:
    block: Final = as_str_mapping(item)
    if block is None:
        return item
    if block.get("type") == "tool_result" and "content" in block:
        return {**block, "content": _text_document_content(block["content"])}
    data: Final = _text_document_data(block)
    if data is None:
        return item
    metadata: Final = tuple(
        f"{key.capitalize()}: {value}"
        for key in ("title", "context")
        if isinstance(value := block.get(key), str) and value
    )
    return {
        "type": "text",
        "text": "\n\n".join((*metadata, data)),
        **({"cache_control": block["cache_control"]} if "cache_control" in block else {}),
    }


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
        anthropic_messages_optional_request_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> dict[str, object]:  # mutable-ok: the base request transformer requires a JSON dict
        documents: Final = tuple(chain.from_iterable(_documents(message.get("content")) for message in messages))
        has_text_documents: Final = any(_text_document_data(document) is not None for document in documents)
        if has_text_documents and any(
            (as_str_mapping(document.get("citations")) or {}).get("enabled") is True for document in documents
        ):
            raise UnsupportedParamsError(
                message="GitHub Copilot text documents cannot preserve structured citations when converted to text. "
                "Disable document citations or use a provider that supports text document sources.",
                model=model,
                llm_provider="github_copilot",
            )
        normalized: Final = [
            {**message, "content": _text_document_content(message["content"])}
            if has_text_documents and message.get("role") == "user" and "content" in message
            else dict(message)
            for message in messages
        ]
        return _REQUEST_FIELDS.validate_python(
            super().transform_anthropic_messages_request(  # pyright: ignore[reportUnknownMemberType]  # the base transformer uses legacy untyped containers
                model=model,
                messages=normalized,
                anthropic_messages_optional_request_params=dict(anthropic_messages_optional_request_params),
                litellm_params=litellm_params,
                headers=dict(headers),
            )
        )

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
        optional_params: dict,
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
        optional_params: dict,
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
