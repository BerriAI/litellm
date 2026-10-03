import base64
import binascii
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, NoReturn
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from typing_extensions import Self

from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,  # pyright: ignore[reportUnknownVariableType]  # native logging decorator has an untyped signature
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

_API_BASE: Final = "https://api.ismalicious.com"
_MAX_BODY_BYTES: Final = 1024 * 1024
_REFUSAL: Final = "IsMalicious could not allow the inspected MCP content"


class _Response(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    verdict: Literal["allow", "warn", "block"]
    latency_ms: int = Field(ge=0, le=2**63 - 1)


class _Link(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    url: str
    entity: str
    verdict: Literal["clean", "suspicious", "malicious", "unknown"]
    sources: int = Field(ge=0)


class _Span(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    family: str

    @model_validator(mode="after")
    def ordered_interval(self) -> Self:
        if self.end < self.start:
            raise ValueError("Invalid injection span interval")
        return self


class _Injection(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    score: float = Field(ge=0, le=1)
    families: list[str]
    spans: list[_Span]


class _Scan(_Response):
    injection: _Injection
    links: list[_Link]
    links_truncated: bool
    mode: Literal["fast", "thorough"]
    source: _Link | None = None
    sanitized_content: str | None = None


class _Url(_Response):
    url: str
    entity: str
    sources: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class _Failure:
    blocked_content: bool = False


def _verdict(response: httpx.Response, url: str | None) -> _Failure | None:
    try:
        response.raise_for_status()
        parsed: Final = (_Scan if url is None else _Url).model_validate_json(response.content)
    except (httpx.HTTPError, ValidationError, ValueError):
        return _Failure()
    if parsed.verdict != "allow":
        return _Failure(blocked_content=True)
    if isinstance(parsed, _Scan) and parsed.links_truncated:
        return _Failure()
    if isinstance(parsed, _Url) and parsed.url != url:
        return _Failure()
    return None


def _encoded_body(texts: list[str]) -> bytes | _Failure:
    try:
        content: Final = json.dumps(texts, ensure_ascii=False, separators=(",", ":"))
        body: Final = json.dumps({"content": content, "mode": "fast"}, ensure_ascii=False).encode("utf-8")
    except (UnicodeError, ValueError):
        return _Failure()
    return body if len(body) <= _MAX_BODY_BYTES else _Failure()


def _url_argument(text: str) -> str | None | _Failure:
    if not text.startswith(("http://", "https://")):
        return None
    try:
        parsed: Final = urlsplit(text)
    except ValueError:
        return _Failure()
    if not parsed.hostname or parsed.username or parsed.password or any(ord(char) < 33 for char in text):
        return _Failure()
    return text


class IsMaliciousGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        guardrail_name: str | None = None,
        event_hook: GuardrailEventHooks | list[GuardrailEventHooks] | None = None,
        default_on: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        resolved_key: Final = api_key
        if not resolved_key:
            raise ValueError("IsMalicious requires a Base64 API key and secret pair")
        try:
            decoded_key: Final = base64.b64decode(resolved_key, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeError):
            raise ValueError("IsMalicious requires a Base64 API key and secret pair") from None
        if ":" not in decoded_key or not all(decoded_key.split(":", 1)):
            raise ValueError("IsMalicious requires a Base64 API key and secret pair")
        if api_base is not None and api_base.rstrip("/") != _API_BASE:
            raise ValueError("IsMalicious credentials may only be sent to its HTTPS API")
        modes: Final = event_hook if isinstance(event_hook, list) else [event_hook]
        if not modes or any(mode not in self.get_supported_event_hooks() for mode in modes):
            raise ValueError("IsMalicious requires pre_mcp_call or post_mcp_call")
        self._headers = {"X-API-KEY": resolved_key, "Content-Type": "application/json"}
        self._transport = transport
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # base guardrail accepts untyped kwargs
            guardrail_name=guardrail_name,
            supported_event_hooks=self.get_supported_event_hooks(),
            event_hook=event_hook,
            default_on=default_on,
        )

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [GuardrailEventHooks.pre_mcp_call, GuardrailEventHooks.post_mcp_call]

    @staticmethod
    def get_config_model() -> type["GuardrailConfigModel[BaseModel]"] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.ismalicious import IsMaliciousConfigModel

        return IsMaliciousConfigModel

    def _raise_failure(self, failure: _Failure) -> NoReturn:
        raise GuardrailRaisedException(
            guardrail_name=self.guardrail_name,
            message=_REFUSAL,
            blocked_content=failure.blocked_content,
        )

    async def _inspect(
        self, client: httpx.AsyncClient, *, body: bytes | None = None, url: str | None = None
    ) -> _Failure | None:
        try:
            response: Final = (
                await client.get("/gate/url", params={"u": url})
                if url is not None
                else await client.post("/gate/scan", content=body)
            )
        except httpx.HTTPError:
            return _Failure()
        return _verdict(response, url)

    async def _inspect_url_argument(self, client: httpx.AsyncClient, text: str) -> None:
        url: Final = _url_argument(text)
        if isinstance(url, _Failure):
            self._raise_failure(url)
        if isinstance(url, str):
            failure: Final = await self._inspect(client, url=url)
            if failure is not None:
                self._raise_failure(failure)

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> GenericGuardrailAPIInputs:
        texts: Final = inputs.get("texts", [])
        if inputs.get("images") or not texts:
            self._raise_failure(_Failure())
        body: Final = _encoded_body(texts)
        if isinstance(body, _Failure):
            self._raise_failure(body)
        async with httpx.AsyncClient(
            base_url=_API_BASE,
            headers=self._headers,
            transport=self._transport,
            timeout=15,
            follow_redirects=False,
            verify=True,
            trust_env=False,
        ) as client:
            if input_type == "request":
                for text in texts:
                    await self._inspect_url_argument(client, text)
            scan_failure: Final = await self._inspect(client, body=body)
            if scan_failure is not None:
                self._raise_failure(scan_failure)
        return inputs
