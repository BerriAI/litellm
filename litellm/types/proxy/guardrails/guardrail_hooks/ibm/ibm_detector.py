from typing import Any

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ..base import GuardrailConfigModel
from .base import IBMGuardrailsBaseConfigModel

# TypedDicts for IBM Detector API Request/Response Structure


class IBMDetectorRequestBodyDetectorServer(TypedDict):
    """Request body for calling IBM Detector Server directly"""

    contents: list[str]
    detector_params: dict[str, Any]


class IBMDetectorRequestBodyOrchestrator(TypedDict):
    """Request body for calling IBM Detector via FMS Guardrails Orchestrator"""

    content: str
    detectors: dict[str, dict[str, Any]]


class IBMDetectorDetection(TypedDict, total=False):
    """Individual detection from IBM Detector"""

    start: int
    end: int
    text: str
    detection: str
    detection_type: str
    score: float
    evidences: list[Any]
    metadata: dict[str, Any]
    detector_id: str | None  # Only present in orchestrator response


class IBMDetectorResponseDetectorServer(TypedDict):
    """Response from IBM Detector Server (returns list of lists)"""

    detections: list[list[IBMDetectorDetection]]


class IBMDetectorResponseOrchestrator(TypedDict):
    """Response from IBM FMS Guardrails Orchestrator"""

    detections: list[IBMDetectorDetection]


# Pydantic Config Models


class IBMDetectorOptionalParams(BaseModel):
    """Optional parameters for IBM Detector guardrail"""

    detector_params: dict[str, Any] | None = Field(
        default_factory=lambda: {},
        description="Dictionary of arguments to pass to the detector.",
    )

    extra_headers: dict[str, Any] | None = Field(
        default_factory=lambda: {},
        description="Dictionary of extra headers to pass to the detector.",
    )

    score_threshold: float | None = Field(
        default=None,
        description="Minimum score threshold to consider a detection as a violation (0.0 to 1.0). If set, detections below this threshold will be ignored.",
    )

    block_on_detection: bool | None = Field(
        default=True,
        description="Whether to block requests when detections are found. Defaults to True.",
    )


class IBMDetectorGuardrailConfigModel(
    IBMGuardrailsBaseConfigModel,
    GuardrailConfigModel[IBMDetectorOptionalParams],
):
    """Configuration model for IBM Detector guardrail"""

    optional_params: IBMDetectorOptionalParams | None = Field(
        default_factory=IBMDetectorOptionalParams,
        description="Optional parameters for the IBM Detector guardrail",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "IBM Guardrails Detector"
