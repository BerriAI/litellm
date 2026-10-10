from dataclasses import dataclass, field
from typing import Final

from pydantic import Field

from litellm.types.integrations.custom_logger import StandardCustomLoggerInitParams

RETRYABLE_INGEST_STATUS_CODES: Final = frozenset({408, 429, 500, 502, 503, 504})

TOKEN_REFRESH_LEEWAY_SECONDS: Final = 60


class ZerobusInitParams(StandardCustomLoggerInitParams):
    """
    Params for initializing a Databricks Zerobus logger on litellm.

    Every connection field falls back to its ``ZEROBUS_*`` environment variable, which is
    what the proxy UI writes. ``table_name`` is the fully qualified ``catalog.schema.table``.
    """

    workspace_url: str | None = None
    server_endpoint: str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    table_name: str | None = None
    batch_size: int = Field(default=100, gt=0)
    flush_interval: int = Field(default=10, gt=0)


@dataclass(frozen=True, slots=True)
class ZerobusConnection:
    """Everything needed to mint a token for one table and post rows to it."""

    workspace_url: str
    workspace_id: str
    server_endpoint: str
    client_id: str
    client_secret: str = field(repr=False)
    table_name: str


@dataclass(frozen=True, slots=True)
class ZerobusAccessToken:
    value: str = field(repr=False)
    expires_at: float


@dataclass(frozen=True, slots=True)
class ZerobusIngestFailure:
    """Why a batch could not be written, and whether a later attempt could still succeed."""

    detail: str
    retryable: bool
