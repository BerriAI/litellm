from collections.abc import Mapping
from enum import Enum, auto
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from httpx import URL, QueryParams, Response
from pydantic import TypeAdapter

from litellm.constants import REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
from litellm.llms.openai.realtime.handler import OpenAIRealtime
from litellm.llms.openai.realtime.http_transformation import OpenAIRealtimeHTTPConfig
from litellm.types.realtime import RealtimeQueryParams
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import get_model_info

from .authenticator import Authenticator
from .common_utils import without_oauth_identity_headers
from .responses.transformation import ChatGPTResponsesAPIConfig

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection


class CallAccounting(Enum):
    SUPERVISED = auto()


def accounts_for_call_usage(params: GenericLiteLLMParams) -> bool:
    return getattr(params, "chatgpt_call_accounting", None) is not CallAccounting.SUPERVISED


def configured_realtime_headers(headers: Mapping[str, object] | None) -> Mapping[str, str]:
    validated: Final = TypeAdapter(Mapping[str, str]).validate_python(
        without_oauth_identity_headers(headers or MappingProxyType({}))
    )
    return MappingProxyType({key.lower(): value for key, value in validated.items()})


def configured_realtime_query(params: GenericLiteLLMParams) -> Mapping[str, str | tuple[str, ...]]:
    inbound: Final = TypeAdapter(Mapping[str, str]).validate_python(
        getattr(params, "chatgpt_realtime_client_query", None) or MappingProxyType({})
    )
    configured: Final = TypeAdapter(
        Mapping[str, str | int | float | bool | None | tuple[str | int | float | bool | None, ...]]
    ).validate_python(getattr(params, "extra_query", None) or MappingProxyType({}))
    merged: Final = QueryParams(
        tuple((key, value) for key, value in inbound.items() if key in ("intent", "architecture"))
    ).merge(configured)
    return MappingProxyType(
        {key: merged[key] if len(merged.get_list(key)) == 1 else tuple(merged.get_list(key)) for key in merged}
    )


def realtime_call_headers(params: GenericLiteLLMParams) -> dict[str, str]:  # mutable-ok: HTTP handler header contract
    inbound: Final = TypeAdapter(Mapping[str, str]).validate_python(
        getattr(params, "chatgpt_realtime_client_headers", None) or MappingProxyType({})
    )
    configured: Final = TypeAdapter(Mapping[str, object]).validate_python(
        getattr(params, "extra_headers", None) or MappingProxyType({})
    )
    return {  # mutable-ok: HTTP handler header contract
        **MappingProxyType(
            {
                key.lower(): value
                for key, value in inbound.items()
                if key.lower() in ("openai-alpha", "openai-beta", "x-session-id", "x-oai-attestation")
            }
        ),
        **configured_realtime_headers(configured),
    }


def realtime_headers(
    params: GenericLiteLLMParams, headers: Mapping[str, str], extra_headers: Mapping[str, object] | None = None
) -> dict[str, str]:  # mutable-ok: HTTP handler header contract
    forwarded: Final = MappingProxyType(
        {
            key.lower(): value
            for key, value in headers.items()
            if key.lower() in ("openai-alpha", "openai-beta", "x-session-id", "x-oai-attestation")
        }
    )
    return {  # mutable-ok: HTTP handler updates headers
        **ChatGPTResponsesAPIConfig().validate_environment(
            headers={},  # mutable-ok: Responses adapter header contract
            model="",
            litellm_params=params,
        ),
        **forwarded,
        **configured_realtime_headers(extra_headers),
    }


def realtime_endpoint(model: str) -> str:
    try:
        model_info: Final = get_model_info(model, custom_llm_provider="chatgpt")
    except Exception:  # noqa: BLE001  # get_model_info raises bare Exception for unmapped models
        return "realtime"
    return "live" if "/v1/live" in (model_info.get("supported_endpoints") or ()) else "realtime"


class ChatGPTRealtime(OpenAIRealtime):
    async def open_call_connection(self, model: str, api_base: str) -> "ClientConnection":
        import websockets

        url: Final = self._construct_url(api_base, RealtimeQueryParams(model=model))
        return await websockets.connect(
            url,
            additional_headers=self._profile_headers,
            max_size=REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES,
            ssl=self._get_ssl_config(url),
            open_timeout=20,
        )

    async def close_call(self, connection: "ClientConnection", model: str, api_base: str) -> None:
        from websockets.exceptions import ConnectionClosed

        if realtime_endpoint(model) == "live":
            try:
                await connection.send('{"type":"session.close"}')
                return
            except (ConnectionClosed, OSError):
                await self.hangup_call(api_base)
                return
        await self.hangup_call(api_base)

    async def hangup_call(self, api_base: str) -> None:
        from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
        from litellm.types.utils import LlmProviders

        base: Final = URL(api_base)
        url: Final = base.copy_with(
            scheme="https" if base.scheme in ("https", "wss") else "http",
            path=f"{base.path.rstrip('/')}/realtime/calls/{self._call_id}/hangup",
            params=tuple(
                (key, value)
                for key, value in QueryParams(self._extra_query).multi_items()
                if key not in ("model", "call_id")
            ),
        )
        client: Final = get_async_httpx_client(llm_provider=LlmProviders.CHATGPT)
        response: Final = await client.post(str(url), headers=self._profile_headers, data=b"", timeout=10)
        response.raise_for_status()

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str:
        return api_base or Authenticator.get_api_base(default_base="https://api.openai.com/v1")

    def __init__(
        self,
        params: GenericLiteLLMParams,
        headers: Mapping[str, str],
        extra_headers: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        self._profile_headers = realtime_headers(params, headers, extra_headers)
        self._call_id = TypeAdapter(str | None).validate_python(getattr(params, "chatgpt_realtime_call_id", None))
        self._extra_query = configured_realtime_query(params)
        self._account_usage = accounts_for_call_usage(params)

    def _get_default_api_base(self) -> str:
        return self.get_api_base()

    def _resolve_api_key(self, api_key: str | None) -> str:
        return "chatgpt-oauth"

    def _accounts_for_call_usage(self) -> bool:
        return self._account_usage

    def _get_additional_headers(
        self, api_key: str, *, openai_beta_realtime: bool = False
    ) -> dict[str, str]:  # mutable-ok: HTTP handler header contract
        return {  # mutable-ok: HTTP handler updates headers
            **(MappingProxyType({"OpenAI-Beta": "realtime=v1"}) if openai_beta_realtime else MappingProxyType({})),
            **self._profile_headers,
        }

    def _construct_url(self, api_base: str, query_params: RealtimeQueryParams) -> str:
        base: Final = URL(api_base)
        endpoint: Final = realtime_endpoint(query_params.get("model", ""))
        if self._call_id:
            gateway_query: Final = tuple(
                (key, value)
                for key, value in QueryParams(self._extra_query).multi_items()
                if key not in ("model", "call_id")
            )
            return str(
                base.copy_with(
                    scheme="wss" if base.scheme in ("https", "wss") else "ws",
                    path=f"{base.path.rstrip('/')}/{endpoint}/{self._call_id}"
                    if endpoint == "live"
                    else f"{base.path.rstrip('/')}/realtime",
                    params=gateway_query + (() if endpoint == "live" else (("call_id", self._call_id),)),
                )
            )
        return str(
            base.copy_with(
                scheme="wss" if base.scheme in ("https", "wss") else "ws",
                path=f"{base.path.rstrip('/')}/{endpoint}",
                params=QueryParams(TypeAdapter(Mapping[str, str | None]).validate_python(query_params)).merge(
                    tuple(
                        (key, value)
                        for key, value in QueryParams(self._extra_query).multi_items()
                        if key not in ("model", "call_id")
                    )
                ),
            )
        )


class ChatGPTRealtimeHTTPConfig(OpenAIRealtimeHTTPConfig):
    realtime_calls_json: Final = True

    def __init__(self, params: GenericLiteLLMParams, use_codex_backend: bool = True) -> None:
        self._params = params
        self._use_codex_backend = use_codex_backend

    def get_api_base(
        self,
        api_base: str | None,
        **kwargs: object,  # kwargs-ok: provider interface accepts optional credentials
    ) -> str:
        return api_base or (Authenticator.get_api_base() if self._use_codex_backend else ChatGPTRealtime.get_api_base())

    def resolve_api_base(self, api_base: str | None, dynamic_api_base: str | None) -> str:
        return self.get_api_base(api_base)

    def get_realtime_calls_extra_headers(
        self, headers: dict[str, object] | None
    ) -> dict[str, object]:  # mutable-ok: shared HTTP handler accepts a mutable header dictionary
        return {**realtime_call_headers(self._params)}  # mutable-ok: shared HTTP header contract

    def get_api_key(
        self,
        api_key: str | None,
        **kwargs: object,  # kwargs-ok: provider interface accepts optional credentials
    ) -> str:
        return "chatgpt-oauth"

    def get_realtime_calls_url(self, api_base: str | None, model: str, api_version: str | None = None) -> str:
        query: Final = configured_realtime_query(self._params)
        return str(URL(f"{self.get_api_base(api_base).rstrip('/')}/realtime/calls", params=query))

    def transform_realtime_calls_response(
        self, response: Response, model: str, model_id: str | None, headers: Mapping[str, object] | None
    ) -> Response:
        response.extensions["chatgpt_realtime"] = (
            MappingProxyType(  # rebind-ok: HTTPX response extensions carry provider routing metadata
                {
                    "model": model,
                    "model_id": model_id,
                    "api_base": ChatGPTRealtime.get_api_base(self._params.api_base),
                    "extra_headers": configured_realtime_headers(headers),
                    "extra_query": configured_realtime_query(self._params),
                }
            )
        )
        return response

    def get_realtime_calls_headers(
        self, ephemeral_key: str
    ) -> dict[str, str]:  # mutable-ok: HTTP handler header contract
        return realtime_headers(self._params, MappingProxyType({}))

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: HTTP handler header contract
        return {  # mutable-ok: HTTP handler updates headers
            **realtime_headers(self._params, headers),
            "Content-Type": "application/json",
        }

    def get_complete_url(self, api_base: str | None, model: str, api_version: str | None = None) -> str:
        return f"{self.get_api_base(api_base).rstrip('/')}/realtime/client_secrets"

    def get_transcription_session_url(
        self,
        api_base: str | None,
        model: str,
        api_version: str | None = None,
    ) -> str:
        return f"{self.get_api_base(api_base).rstrip('/')}/realtime/transcription_sessions"
