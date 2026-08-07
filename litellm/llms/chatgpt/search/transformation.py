from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Annotated, Final, Protocol
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from litellm.exceptions import AuthenticationError
from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig
from litellm.llms.openai.common_utils import OpenAIError
from litellm.types.llms.openai import AllMessageValues

from ..authenticator import Authenticator
from ..common_utils import GetAccessTokenError, ensure_chatgpt_session_id, get_chatgpt_default_headers

_FORWARDED_REQUEST_HEADERS: Final = frozenset({"originator", "x-codex-turn-metadata"})
_FORWARDED_RESPONSE_HEADERS: Final = frozenset(
    {
        "content-length",
        "content-type",
        "openai-processing-ms",
        "retry-after",
        "x-request-id",
    }
)
_FORWARDED_RESPONSE_HEADER_PREFIXES: Final = ("x-codex-", "x-litellm-", "x-ratelimit-")
_EMPTY_MAPPING: Final[Mapping[str, str]] = MappingProxyType({})


class ChatGPTAuthenticator(Protocol):
    def get_access_token(self) -> str: ...

    def get_account_id(self) -> str | None: ...

    def get_api_base(self) -> str: ...


class ChatGPTSearchRequest(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, strict=True)

    id: str | None = None
    model: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _ProxyServerRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    headers: Mapping[str, str] | None = None


def _parse_proxy_server_request(value: object) -> _ProxyServerRequest:
    try:
        return _ProxyServerRequest.model_validate(value if value is not None else _EMPTY_MAPPING)
    except ValidationError:
        return _ProxyServerRequest()


class ChatGPTSearchPassthroughConfig(BasePassthroughConfig):
    def __init__(self, authenticator: ChatGPTAuthenticator | None = None) -> None:
        self.authenticator: Final = authenticator or Authenticator()

    def is_streaming_request(self, endpoint: str, request_data: Mapping[str, object]) -> bool:
        return False

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        endpoint: str,
        request_query_params: Mapping[str, str | Sequence[str]] | None,
        litellm_params: Mapping[str, object],
    ) -> tuple[httpx.URL, str]:
        if endpoint.strip("/") != "alpha/search":
            raise ValueError(f"Unsupported ChatGPT passthrough endpoint: {endpoint}")
        base_target_url: Final = api_base or self.authenticator.get_api_base()
        base_url: Final = httpx.URL(f"{base_target_url.rstrip('/')}/{endpoint.lstrip('/')}")
        url: Final = (
            base_url.copy_with(query=urlencode(request_query_params, doseq=True).encode("ascii"))
            if request_query_params
            else base_url
        )
        return url, base_target_url

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str:
        return api_base or Authenticator().get_api_base()

    @staticmethod
    def get_base_model(model: str) -> str | None:
        return model

    def get_models(
        self, api_key: str | None = None, api_base: str | None = None
    ) -> list[str]:  # mutable-ok: BaseLLMModelInfo requires a list result
        return []  # mutable-ok: BaseLLMModelInfo requires a list result

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: passthrough transport requires mutable headers
        try:
            access_token: Final = self.authenticator.get_access_token()
        except GetAccessTokenError as exc:
            raise AuthenticationError(
                model=model,
                llm_provider="chatgpt",
                message=str(exc),
            ) from exc
        account_id: Final = self.authenticator.get_account_id()
        session_id: Final = ensure_chatgpt_session_id(litellm_params)
        return {  # mutable-ok: passthrough transport requires mutable headers
            **get_chatgpt_default_headers(access_token, account_id, session_id),
            **headers,
            "accept": "application/json",
        }

    def sign_request(
        self,
        headers: Mapping[str, str],
        litellm_params: Mapping[str, object],
        request_data: Mapping[str, object] | None,
        api_base: str,
        model: str | None = None,
    ) -> tuple[dict[str, str], bytes | None]:  # mutable-ok: passthrough transport requires mutable headers
        proxy_request: Final = _parse_proxy_server_request(litellm_params.get("proxy_server_request"))
        forwarded_headers: Final = MappingProxyType(
            {
                name.lower(): value
                for name, value in (proxy_request.headers or _EMPTY_MAPPING).items()
                if name.lower() in _FORWARDED_REQUEST_HEADERS
            }
        )
        session_id: Final = request_data.get("id") if request_data is not None else None
        session_headers: Final = (
            MappingProxyType({"session_id": session_id}) if isinstance(session_id, str) else _EMPTY_MAPPING
        )
        return {**headers, **forwarded_headers, **session_headers}, None  # mutable-ok: transport requires a dict

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | httpx.Headers,  # mutable-ok: BasePassthroughConfig fixes this parameter type
    ) -> OpenAIError:
        return OpenAIError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )


def is_chatgpt_search_response_header(name: str) -> bool:
    normalized_name: Final = name.lower()
    return normalized_name in _FORWARDED_RESPONSE_HEADERS or normalized_name.startswith(
        _FORWARDED_RESPONSE_HEADER_PREFIXES
    )
