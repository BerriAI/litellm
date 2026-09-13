from pydantic import BaseModel, Field


class IBMGuardrailsBaseConfigModel(BaseModel):
    """Base configuration parameters for IBM Guardrails"""

    auth_token: str | None = Field(
        default=None,
        description="Authorization bearer token for IBM Guardrails API. Reads from IBM_GUARDRAILS_AUTH_TOKEN env var if None.",
    )

    base_url: str | None = Field(
        default=None,
        description="Base URL for the IBM Guardrails server",
    )

    detector_id: str | None = Field(
        default=None,
        description="Name of the detector inside the server (e.g., 'jailbreak-detector')",
    )

    is_detector_server: bool | None = Field(
        default=True,
        description="Boolean flag to determine if calling a detector server (True) or the FMS Orchestrator (False). Defaults to True.",
    )

    verify_ssl: bool | None = Field(
        default=True,
        description="Whether to verify SSL certificates. Defaults to True.",
    )
