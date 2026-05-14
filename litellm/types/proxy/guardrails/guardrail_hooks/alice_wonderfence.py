"""Alice WonderFence guardrail configuration models."""

from typing import Optional

from pydantic import Field

from .base import GuardrailConfigModel


class WonderFenceGuardrailConfigModel(GuardrailConfigModel):
    """Configuration parameters for the Alice WonderFence guardrail.

    Resolution order for ``api_key`` and ``app_id`` (highest first):
    API-key metadata → team metadata → request metadata (only when
    ``allow_request_metadata_override`` is True) → ``api_key`` falls back to
    the configured default below or the ``ALICE_API_KEY`` env var; ``app_id``
    has no default.

    By default, request-body metadata is ignored so a caller cannot bypass
    an admin-pinned WonderFence ``app_id`` / ``api_key`` on their virtual
    key. Enable ``allow_request_metadata_override`` for trusted-gateway
    deployments that legitimately need request-level overrides.
    """

    api_key: Optional[str] = Field(
        default=None,
        description="Default API key for WonderFence. Overridable via API-key / team metadata, or via request metadata (alice_wonderfence_api_key) only when allow_request_metadata_override is True. Env: ALICE_API_KEY.",
    )
    allow_request_metadata_override: Optional[bool] = Field(
        default=False,
        description="When True, allow alice_wonderfence_api_key and alice_wonderfence_app_id in request metadata as a last-resort source (after API-key and team metadata). Default False so caller-controlled request fields cannot bypass admin-pinned credentials.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="Override for WonderFence API base URL.",
    )
    api_timeout: Optional[float] = Field(
        default=10.0,
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
