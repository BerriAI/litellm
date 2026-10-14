"""Splunk HTTP Event Collector sink: one HEC event per audit record, the OCSF object as the event body, newline
separated in one POST per batch (https://docs.splunk.com/Documentation/Splunk/latest/Data/FormateventsforHTTPEventCollector)."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.proxy.control_plane.audit.ocsf import AuditRecord
from litellm.proxy.control_plane.audit.pipeline import Batch, Delivered, DeliveryResult, Undelivered
from litellm.proxy.control_plane.audit.transport import Poster
from litellm.proxy.control_plane.settings import SplunkHecSettings

EVENT_PATH: Final = "/services/collector/event"


class HecReply(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    text: str = ""
    code: int = -1


_REPLY: Final = TypeAdapter(HecReply)


def hec_event(record: AuditRecord, settings: SplunkHecSettings) -> Mapping[str, object]:
    """``sourcetype`` is ``ocsf:<class>`` so a CIM or OCSF add-on can pick the class without parsing the body."""
    index: Final = {"index": settings.index} if settings.index is not None else {}
    return {
        "time": record.event.at.timestamp(),
        "source": settings.source,
        "sourcetype": f"ocsf:{record.event.kind}",
        **index,
        "event": record.body,
    }


def hec_payload(batch: Batch, settings: SplunkHecSettings) -> bytes:
    return "\n".join(
        json.dumps(hec_event(record, settings), separators=(",", ":"), default=str) for record in batch
    ).encode()


def _reply(response: httpx.Response) -> HecReply:
    """HEC reports its own result in the body, ``code`` 0 is the only success, so a 200 without it is not one."""
    try:
        return _REPLY.validate_json(response.content)
    except ValidationError:
        return HecReply(text="no HEC reply body")


def _reply_text(response: httpx.Response) -> str:
    reply: Final = _reply(response)
    return f"{reply.text} (code {reply.code})"


@dataclass(frozen=True, slots=True)
class SplunkHecSink:
    settings: SplunkHecSettings
    post: Poster
    name: str = "splunk_hec"

    async def deliver(self, batch: Batch) -> DeliveryResult:
        url: Final = f"{self.settings.url.rstrip('/')}{EVENT_PATH}"
        headers: Final = {
            "Authorization": f"Splunk {self.settings.token.get_secret_value()}",
            "Content-Type": "application/json",
        }
        try:
            response: Final = await self.post.post(
                url, hec_payload(batch, self.settings), headers, self.settings.timeout_seconds
            )
        except httpx.HTTPError as error:
            return Undelivered(f"{type(error).__name__} posting to splunk hec")
        if response.status_code == 200 and _reply(response).code == 0:
            return Delivered(len(batch))
        return Undelivered(f"HTTP {response.status_code} from splunk hec: {_reply_text(response)}")

    async def close(self) -> None:
        return None
