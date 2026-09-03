from dataclasses import dataclass
from typing import Final

from pydantic import Field

from litellm.types.integrations.custom_logger import StandardCustomLoggerInitParams

RETRYABLE_UPLOAD_STATUS_CODES: Final = frozenset({429, 500, 502, 503, 504})

DEFAULT_API_URL: Final = "https://api.pointfive.co/query"


class PointFiveInitParams(StandardCustomLoggerInitParams):
    """
    Params for initializing a PointFive logger on litellm.

    Defaults trade freshness for fewer, larger uploads: every flush becomes one object, so
    the interval is minutes rather than seconds. ``max_batch_bytes`` bounds how much a
    single object may hold, which matters most when message logging is left on, since an
    unredacted payload is orders of magnitude larger than a redacted one.
    """

    api_key: str | None = None
    api_url: str | None = None
    batch_size: int = Field(default=10_000, gt=0)
    flush_interval: int = Field(default=300, gt=0)
    max_batch_bytes: int = Field(default=8 * 1024 * 1024, gt=0)
    max_upload_retries: int = Field(default=3, ge=1)


@dataclass(frozen=True, slots=True)
class PointFiveUploadTarget:
    """A single-use presigned destination for one batch, issued by the PointFive API."""

    upload_url: str
    object_key: str


@dataclass(frozen=True, slots=True)
class PointFiveUploadFailure:
    """Why a batch could not be uploaded, and whether a later attempt could still succeed."""

    detail: str
    retryable: bool
