"""Alice WonderFence guardrail configuration models."""

from typing import Optional

from pydantic import Field

from .base import GuardrailConfigModel


class WonderFenceGuardrailConfigModel(GuardrailConfigModel):
    """Configuration parameters for the Alice WonderFence guardrail.

    Per-request ``api_key`` and ``app_id`` are read from request / API-key /
    team metadata using these keys: ``alice_wonderfence_api_key``,
    ``alice_wonderfence_app_id``. ``api_id`` has no default. ``api_key`` falls
    back to the value below or the ``ALICE_API_KEY`` env var.
    """

    api_key: Optional[str] = Field(
        default=None,
        description="Default API key for WonderFence (overridable per request via metadata.alice_wonderfence_api_key). Env: ALICE_API_KEY.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="Override for WonderFence API base URL.",
    )
    api_timeout: Optional[float] = Field(
        default=20.0,
        description="Timeout in seconds for API calls.",
    )
    platform: Optional[str] = Field(
        default=None,
        description="Cloud platform (e.g., aws, azure, databricks).",
    )
    fail_open: Optional[bool] = Field(
        default=False,
        description="When True, proceed with the request/response if WonderFence is unreachable. BLOCK actions are always enforced. Default: False (fail closed).",
    )
    block_message: Optional[str] = Field(
        default="Content violates our policies and has been blocked",
        description="User-facing error message returned when content is blocked.",
    )
    debug: Optional[bool] = Field(
        default=False,
        description="Set guardrail logger to DEBUG level.",
    )
    max_cached_clients: Optional[int] = Field(
        default=10,
        description="Max SDK clients cached per guardrail (LRU, keyed by api_key). Env: ALICE_MAX_CACHED_CLIENTS.",
    )
    connection_pool_limit: Optional[int] = Field(
        default=None,
        description="Max connections per SDK client HTTP pool. Env: ALICE_CONNECTION_POOL_LIMIT.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Alice WonderFence Guardrail"
