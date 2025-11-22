from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ..base import GuardrailConfigModel
from .base import IBMGuardrailsBaseConfigModel

# TypedDicts for IBM Detector API Request/Response Structure


class IBMDetectorRequestBodyDetectorServer(TypedDict):
    """Request body for calling IBM Detector Server directly"""

    contents: List[str]
    detector_params: Dict[str, Any]


class IBMDetectorRequestBodyOrchestrator(TypedDict):
    """Request body for calling IBM Detector via FMS Guardrails Orchestrator"""

    content: str
    detectors: Dict[str, Dict[str, Any]]


class IBMDetectorDetection(TypedDict, total=False):
    """Individual detection from IBM Detector"""

    start: int
    end: int
    text: str
    detection: str
    detection_type: str
    score: float
    evidences: List[Any]
    metadata: Dict[str, Any]
    detector_id: Optional[str]  # Only present in orchestrator response


class IBMDetectorResponseDetectorServer(TypedDict):
    """Response from IBM Detector Server (returns list of lists)"""

    detections: List[List[IBMDetectorDetection]]


class IBMDetectorResponseOrchestrator(TypedDict):
    """Response from IBM FMS Guardrails Orchestrator"""

    detections: List[IBMDetectorDetection]


# Pydantic Config Models


class IBMDetectorOptionalParams(BaseModel):
    """Optional parameters for IBM Detector guardrail"""

    detector_params: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {},
        description="Dictionary of arguments to pass to the detector.",
    )

    extra_headers: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {},
        description="Dictionary of extra headers to pass to the detector.",
    )

    score_threshold: Optional[float] = Field(
        default=None,
        description="Minimum score threshold to consider a detection as a violation (0.0 to 1.0). If set, detections below this threshold will be ignored.",
    )

    block_on_detection: Optional[bool] = Field(
        default=True,
        description="Whether to block requests when detections are found. Defaults to True.",
    )


class IBMDetectorGuardrailConfigModel(
    IBMGuardrailsBaseConfigModel,
    GuardrailConfigModel[IBMDetectorOptionalParams],
):
    """Configuration model for IBM Detector guardrail"""

    optional_params: Optional[IBMDetectorOptionalParams] = Field(
        default_factory=IBMDetectorOptionalParams,
        description="Optional parameters for the IBM Detector guardrail",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "IBM Guardrails Detector"
