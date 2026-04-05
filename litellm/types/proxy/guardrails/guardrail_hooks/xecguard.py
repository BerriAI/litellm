from typing import Dict, List, Literal, Optional

from pydantic import Field

from .base import GuardrailConfigModel


class XecGuardConfigModel(GuardrailConfigModel):
    """
    Config for the CyCraft XecGuard guardrail.

    Supports input/response scanning and optional RAG context-grounding verification.
    """

    api_key: Optional[str] = Field(
        default=None,
        description="XecGuard service token. Env: XECGUARD_SERVICE_TOKEN.",
    )

    api_base: Optional[str] = Field(
        default=None,
        description="XecGuard API base URL. Env: XECGUARD_API_BASE. Default: https://api-xecguard.cycraft.ai",
    )

    model: Optional[str] = Field(
        default="xecguard_v2",
        description="XecGuard model name. Default: xecguard_v2.",
    )

    policy_names: Optional[List[str]] = Field(
        default=None,
        description="List of XecGuard policy names to apply. Uses default policies if not specified.",
    )

    grounding_enabled: Optional[bool] = Field(
        default=False,
        description="Enable RAG context-grounding verification on post_call.",
    )

    grounding_strictness: Optional[Literal["BALANCED", "STRICT"]] = Field(
        default="BALANCED",
        description="Grounding strictness level: BALANCED or STRICT.",
    )

    grounding_documents: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Default grounding documents for RAG context-grounding. Each item has document_id and context.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "XecGuard"


class XecGuardUIConfigModel(GuardrailConfigModel):
    """
    UI-only config model for XecGuard — excludes Context Grounding fields.

    Context Grounding settings (grounding_enabled, grounding_strictness) are
    intentionally hidden from the web UI but remain fully functional at the
    API level via XecGuardConfigModel.
    """

    api_key: Optional[str] = Field(
        default=None,
        description="XecGuard service token. Env: XECGUARD_SERVICE_TOKEN.",
    )

    api_base: Optional[str] = Field(
        default=None,
        description="XecGuard API base URL. Env: XECGUARD_API_BASE. Default: https://api-xecguard.cycraft.ai",
    )

    model: Optional[str] = Field(
        default="xecguard_v2",
        description="XecGuard model name. Default: xecguard_v2.",
    )

    policy_names: Optional[List[str]] = Field(
        default=None,
        description="List of XecGuard policy names to apply. Uses default policies if not specified.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "XecGuard"
