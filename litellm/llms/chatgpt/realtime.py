from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from httpx import URL
from pydantic import TypeAdapter

from litellm.llms.openai.realtime.handler import OpenAIRealtime
from litellm.llms.openai.realtime.http_transformation import OpenAIRealtimeHTTPConfig
from litellm.types.realtime import RealtimeQueryParams
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import get_model_info

from .common_utils import CHATGPT_API_BASE
from .responses.transformation import ChatGPTResponsesAPIConfig


def realtime_headers(
    params: GenericLiteLLMParams, headers: Mapping[str, str]
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
    }


def realtime_endpoint(model: str) -> str:
    try:
        model_info: Final = get_model_info(model, custom_llm_provider="chatgpt")
    except Exception:  # noqa: BLE001  # get_model_info raises bare Exception for unmapped models
        return "realtime"
    return "live" if "/v1/live" in (model_info.get("supported_endpoints") or ()) else "realtime"


class ChatGPTRealtime(OpenAIRealtime):
    def __init__(self, params: GenericLiteLLMParams, headers: Mapping[str, str]) -> None:
        super().__init__()
        self._profile_headers = realtime_headers(params, headers)
        self._call_id = TypeAdapter(str | None).validate_python(getattr(params, "chatgpt_realtime_call_id", None))

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
            return str(
                base.copy_with(
                    scheme="wss" if base.scheme in ("https", "wss") else "ws",
                    path=f"{base.path.rstrip('/')}/{endpoint}/{self._call_id}"
                    if endpoint == "live"
                    else f"{base.path.rstrip('/')}/realtime",
                    params=() if endpoint == "live" else (("call_id", self._call_id),),
                )
            )
        return str(
            base.copy_with(
                scheme="wss" if base.scheme in ("https", "wss") else "ws",
                path=f"{base.path.rstrip('/')}/{endpoint}",
                params=query_params,
            )
        )


class ChatGPTRealtimeHTTPConfig(OpenAIRealtimeHTTPConfig):
    realtime_calls_json: Final = True

    def __init__(self, params: GenericLiteLLMParams) -> None:
        self._params = params

    def get_api_base(
        self,
        api_base: str | None,
        **kwargs: object,  # kwargs-ok: provider interface accepts optional credentials
    ) -> str:
        return api_base or CHATGPT_API_BASE

    def get_api_key(
        self,
        api_key: str | None,
        **kwargs: object,  # kwargs-ok: provider interface accepts optional credentials
    ) -> str:
        return "chatgpt-oauth"

    def get_realtime_calls_url(self, api_base: str | None, model: str, api_version: str | None = None) -> str:
        query: Final = TypeAdapter(Mapping[str, str]).validate_python(
            getattr(self._params, "extra_query", None) or MappingProxyType({})
        )
        return str(URL(f"{self.get_api_base(api_base).rstrip('/')}/realtime/calls", params=query))

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
        return "https://api.openai.com/v1/realtime/client_secrets"

    def get_transcription_session_url(
        self,
        api_base: str | None,
        model: str,
        api_version: str | None = None,
    ) -> str:
        return "https://api.openai.com/v1/realtime/transcription_sessions"
