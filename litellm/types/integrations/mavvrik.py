"""
Mavvrik integration types for LiteLLM.

Data flow:
  LiteLLM_DailyUserSpend rows
    → MavvrikRecord (NDJSON lines)
    → gzip compress
    → GET signed URL from Mavvrik API (x-api-key auth)
    → POST signed URL to initiate resumable GCS upload
    → PUT session URI to upload gzip payload
"""

from typing import Any, Dict, Optional


class MavvrikRecord(Dict[str, Any]):
    """A single cost record in Mavvrik's NDJSON format.

    One record is written per LiteLLM_DailyUserSpend row.
    Multiple records are newline-joined and gzip-compressed
    before being uploaded to GCS via a Mavvrik-issued signed URL.

    Fields:
        metricname:          Always "litellm_cost" (identifies the data source)
        timestamp:           ISO-8601 UTC datetime string (from spend row date)
        value:               Cost in USD (float)
        tags:                JSON-encoded string of all dimension metadata

    Tags sub-fields:
        model:               LLM model name (e.g. "gpt-4o")
        provider:            LLM provider (e.g. "openai", "anthropic")
        model_group:         Model family/group
        api_key:             Hashed API key
        api_key_alias:       Human-readable key alias
        team_id:             Team identifier
        team_alias:          Human-readable team name
        user_id:             User identifier (nullable)
        prompt_tokens:       Input tokens used
        completion_tokens:   Output tokens used
        total_tokens:        Total tokens (prompt + completion)
        api_requests:        Total API requests
        successful_requests: Successful API requests
        failed_requests:     Failed API requests
        spend:               Cost in USD (also top-level as value)
    """

    pass


# Type alias for function signatures
MavvrikRecordDict = Dict[str, Any]


class MavvrikSignedUrlResponse:
    """Parsed response from the Mavvrik signed URL endpoint.

    GET {api_url}/{tenant}/k8s/agent/{instance_id}/upload-url
        ?name={interval}&provider=gcp&type=metrics
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
        "tenant",
        "instance_id",
        "timezone",
        "marker",
    )

    def __init__(
        self,
        api_key: str,
        api_endpoint: str,
        tenant: str,
        instance_id: str,
        timezone: str = "UTC",
        marker: Optional[str] = None,
    ) -> None:
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.tenant = tenant
        self.instance_id = instance_id
        self.timezone = timezone
        # ISO-8601 UTC timestamp of the last successfully uploaded interval.
        # None means no upload has occurred yet — first run will query all history.
        self.marker = marker
