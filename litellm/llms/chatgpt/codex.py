from collections.abc import Mapping
from typing import Final
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, Field
from typing_extensions import ReadOnly, TypedDict

from litellm.types.realtime import RealtimeQueryParams, RealtimeSessionConfig


class CodexRealtimeOffer(BaseModel):
    sdp: str = Field(min_length=1)
    session: RealtimeSessionConfig


class CodexRealtimeCall(BaseModel):
    call_id: str = Field(pattern=r"^rtc_[A-Za-z0-9_-]+$")
    model: str
    alias: str
    api_base: str | None = None
    owner: str
    expires_at: float


class ChatGPTCallRouting(BaseModel):
    model: str
    api_base: str | None = None


class CodexSidebandRequest(TypedDict):
    api_base: ReadOnly[str | None]
    model: ReadOnly[str]
    chatgpt_realtime_call_id: ReadOnly[str]
    query_params: ReadOnly[RealtimeQueryParams]


def build_call_request(
    offer: CodexRealtimeOffer, query: Mapping[str, str], headers: Mapping[str, str]
) -> dict[str, object]:  # mutable-ok: proxy processor enriches the request dictionary
    return {  # mutable-ok: proxy processor enriches the request dictionary
        "model": offer.session.model,
        "sdp_body": offer.sdp.encode(),
        "session": offer.session.model_dump(exclude_none=True),
        "openai_ephemeral_key": "",
        "extra_query": {  # mutable-ok: router request parameters
            key: value for key, value in query.items() if key in ("intent", "architecture")
        },
        "extra_headers": {  # mutable-ok: router request headers
            key: value
            for key, value in headers.items()
            if key in ("openai-alpha", "openai-beta", "x-session-id", "x-oai-attestation")
        },
    }


def parse_call_response(response: httpx.Response, alias: str, owner: str, expires_at: float) -> CodexRealtimeCall:
    routing_data: Final = response.extensions.get("chatgpt_realtime")
    if not routing_data:
        raise ValueError("Direct call signaling requires a ChatGPT deployment")
    routing: Final = ChatGPTCallRouting.model_validate(routing_data)
    call_id: Final = urlsplit(response.headers.get("location", "")).path.rstrip("/").rsplit("/", 1)[-1]
    return CodexRealtimeCall(
        call_id=call_id,
        model=routing.model,
        alias=alias,
        owner=owner,
        expires_at=expires_at,
        api_base=routing.api_base,
    )


def build_sideband_request(call: CodexRealtimeCall) -> CodexSidebandRequest:
    return CodexSidebandRequest(
        api_base=call.api_base,
        model=f"chatgpt/{call.model}",
        chatgpt_realtime_call_id=call.call_id,
        query_params=RealtimeQueryParams(model=call.model),
    )
