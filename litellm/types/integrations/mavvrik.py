"""
Mavvrik integration types for LiteLLM.

Data flow:
  LiteLLM_DailyUserSpend rows
    → CSV via MavvrikTransformer
    → gzip compress
    → GET signed URL from Mavvrik API (x-api-key auth)
    → POST to initiate resumable upload session
    → PUT session URI to upload gzip payload
"""

from typing import Optional


class MavvrikSignedUrlResponse:
    """Parsed response from the Mavvrik signed URL endpoint.

    GET {api_endpoint}/metrics/agent/ai/{connection_id}/upload-url
        ?name={date_str}&type=metrics
    Response: { "url": "https://storage.googleapis.com/..." }
    """

    __slots__ = ("url",)

    def __init__(self, url: str) -> None:
        self.url = url


class MavvrikSettings:
    """In-memory representation of decrypted Mavvrik settings loaded from LiteLLM_Config."""

    __slots__ = (
        "api_key",
        "api_endpoint",
        "connection_id",
        "marker",
    )

    def __init__(
        self,
        api_key: str,
        api_endpoint: str,
        connection_id: str,
        marker: Optional[str] = None,
    ) -> None:
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.connection_id = connection_id
        # ISO-8601 UTC timestamp of the last successfully uploaded interval.
        # None means no upload has occurred yet — first run will query all history.
        self.marker = marker
