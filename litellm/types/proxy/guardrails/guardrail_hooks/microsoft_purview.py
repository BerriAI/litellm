from typing import List, Optional

from pydantic import Field

from .base import GuardrailConfigModel


class MicrosoftPurviewDLPConfigModel(GuardrailConfigModel):
    """Configuration parameters for Microsoft Purview DLP guardrail.

    Note: ``api_key`` and ``api_base`` are inherited from
    ``BaseLitellmParams`` and used as the OAuth2 client secret and the
    Purview scan endpoint respectively, so they are not re-declared here.
    """

    tenant_id: Optional[str] = Field(
        default=None,
        description="Azure AD tenant ID for OAuth2 client-credentials authentication",
    )
    client_id: Optional[str] = Field(
        default=None,
        description=(
            "Azure AD application (client) ID. If omitted, falls back to "
            "the PURVIEW_CLIENT_ID environment variable."
        ),
    )
    block_on_violation: Optional[bool] = Field(
        default=True,
        description=(
            "If True (default), raise HTTP 400 when a DLP policy violation is "
            "detected. If False, log the violation only — do not raise. "
            "API/network errors follow the same flag."
        ),
    )
    sensitive_info_types: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional list of sensitive information type names to scan for. "
            "When None, the Purview DLP policy default types are used."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Microsoft Purview DLP"
