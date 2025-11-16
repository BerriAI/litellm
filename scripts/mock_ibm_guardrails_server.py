"""
Mock FastAPI server for IBM FMS Guardrails Orchestrator Detector API.

This server implements the Detector API endpoints for testing purposes.
Based on: https://foundation-model-stack.github.io/fms-guardrails-orchestrator/

Usage:
    python scripts/mock_ibm_guardrails_server.py

The server will run on http://localhost:8001 by default.
"""

import uuid
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

app = FastAPI(
    title="IBM FMS Guardrails Orchestrator Mock",
    description="Mock server for testing IBM Guardrails Detector API",
    version="1.0.0",
)


# Request Models
class DetectorParams(BaseModel):
    """Parameters specific to the detector."""

    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    custom_param: Optional[str] = None


class TextDetectionRequest(BaseModel):
    """Request model for text detection."""

    contents: List[str] = Field(..., description="Text content to analyze")
    detector_params: Optional[DetectorParams] = None


class TextGenerationDetectionRequest(BaseModel):
    """Request model for text generation detection."""

    detector_id: str = Field(..., description="ID of the detector to use")
    prompt: str = Field(..., description="Input prompt")
    generated_text: str = Field(..., description="Generated text to analyze")
    detector_params: Optional[DetectorParams] = None


class ContextDetectionRequest(BaseModel):
    """Request model for detection with context."""

    detector_id: str = Field(..., description="ID of the detector to use")
    content: str = Field(..., description="Text content to analyze")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    detector_params: Optional[DetectorParams] = None


# Response Models
class Detection(BaseModel):
    """Individual detection result."""

    detection_type: str = Field(..., description="Type of detection")
    detection: bool = Field(..., description="Whether content was detected as harmful")
    score: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    start: Optional[int] = Field(None, description="Start position in text")
    end: Optional[int] = Field(None, description="End position in text")
    text: Optional[str] = Field(None, description="Detected text segment")
    evidence: Optional[List[str]] = Field(None, description="Supporting evidence")


class DetectionResponse(BaseModel):
    """Response model for detection results."""

    detections: List[Detection] = Field(..., description="List of detections")
    detection_id: str = Field(..., description="Unique ID for this detection request")


# Mock detector configurations
MOCK_DETECTORS = {
    "hate": {
        "name": "Hate Speech Detector",
        "triggers": ["hate", "offensive", "discriminatory", "slur"],
        "default_score": 0.85,
    },
    "pii": {
        "name": "PII Detector",
        "triggers": ["email", "ssn", "credit card", "phone number", "address"],
        "default_score": 0.92,
    },
    "toxicity": {
        "name": "Toxicity Detector",
        "triggers": ["toxic", "abusive", "profanity", "insult"],
        "default_score": 0.78,
    },
    "jailbreak": {
        "name": "Jailbreak Detector",
        "triggers": ["ignore instructions", "override", "bypass", "jailbreak"],
        "default_score": 0.88,
    },
    "prompt_injection": {
        "name": "Prompt Injection Detector",
        "triggers": ["ignore previous", "new instructions", "system prompt"],
        "default_score": 0.90,
    },
}


def simulate_detection(
    detector_id: str, content: str, detector_params: Optional[DetectorParams] = None
) -> List[Detection]:
    """
    Simulate detection logic based on detector type and content.

    Args:
        detector_id: ID of the detector to simulate
        content: Text content to analyze
        detector_params: Optional detector parameters

    Returns:
        List of Detection objects
    """
    detections = []
    content_lower = " ".join(c for c in content).lower()

    # Get detector config
    detector_config = MOCK_DETECTORS.get(detector_id)
    if not detector_config:
        # Unknown detector - return no detections
        return detections

    # Check for triggers in content
    for trigger in detector_config["triggers"]:
        if trigger in content_lower:
            # Calculate score (use threshold if provided, otherwise default)
            base_score = detector_config["default_score"]
            threshold = (
                detector_params.threshold
                if detector_params and detector_params.threshold
                else None
            )

            # Adjust score slightly based on content length (longer content = slightly lower confidence)
            score_adjustment = max(0, min(0.1, len(content) / 10000))
            score = max(0.0, min(1.0, base_score - score_adjustment))

            # Find position of trigger
            start_pos = content_lower.find(trigger)
            end_pos = start_pos + len(trigger)

            detection = Detection(
                detection_type=detector_id,
                detection=threshold is None or score >= threshold,
                score=score,
                start=start_pos,
                end=end_pos,
                text=content[start_pos:end_pos] if start_pos >= 0 else None,
                evidence=[f"Found trigger word: {trigger}"],
            )
            detections.append(detection)

    # If no triggers found, return a negative detection
    if not detections:
        detections.append(
            Detection(
                detection_type=detector_id,
                detection=False,
                score=0.05,  # Low score for clean content
            )
        )

    return detections


# Authentication middleware
def verify_auth_token(authorization: Optional[str] = Header(None)) -> bool:
    """
    Verify the authentication token.

    Args:
        authorization: Authorization header value

    Returns:
        True if valid, raises HTTPException otherwise
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
        )

    # Simple token validation - in real implementation, this would validate against a real auth system
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Expected: Bearer <token>",
        )

    token = authorization.replace("Bearer ", "")

    # Accept any non-empty token for mock purposes
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Empty token provided",
        )

    return True


# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "IBM FMS Guardrails Mock Server"}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "IBM FMS Guardrails Orchestrator Mock",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "text_detection": "/api/v1/text/detection",
            "generation_detection": "/api/v1/text/generation/detection",
            "context_detection": "/api/v1/text/context/detection",
        },
        "available_detectors": list(MOCK_DETECTORS.keys()),
    }


@app.post("/api/v1/text/contents")
async def text_detection(
    request: TextDetectionRequest,
    detector_id: str = Header(None),  # query parameter
    authorization: Optional[str] = Header(None),
):
    """
    Detect potential issues in text content.

    Args:
        request: Detection request with content and detector ID
        detector_id: ID of detector
        authorization: Bearer token for authentication

    Returns:
        Detection results
    """
    verify_auth_token(authorization)

    detections = simulate_detection(
        detector_id=detector_id,
        content=request.contents,
        detector_params=request.detector_params,
    )

    return detections


@app.post("/api/v1/text/generation/detection", response_model=DetectionResponse)
async def text_generation_detection(
    request: TextGenerationDetectionRequest,
    authorization: Optional[str] = Header(None),
):
    """
    Detect potential issues in generated text.

    Args:
        request: Detection request with prompt and generated text
        authorization: Bearer token for authentication

    Returns:
        Detection results
    """
    verify_auth_token(authorization)

    # Analyze both prompt and generated text
    combined_content = f"{request.prompt} {request.generated_text}"

    detections = simulate_detection(
        detector_id=request.detector_id,
        content=combined_content,
        detector_params=request.detector_params,
    )

    return DetectionResponse(
        detections=detections,
        detection_id=str(uuid.uuid4()),
    )


@app.post("/api/v1/text/context/detection", response_model=DetectionResponse)
async def context_detection(
    request: ContextDetectionRequest,
    authorization: Optional[str] = Header(None),
):
    """
    Detect potential issues in text with additional context.

    Args:
        request: Detection request with content and context
        authorization: Bearer token for authentication

    Returns:
        Detection results
    """
    verify_auth_token(authorization)

    detections = simulate_detection(
        detector_id=request.detector_id,
        content=request.content,
        detector_params=request.detector_params,
    )

    return DetectionResponse(
        detections=detections,
        detection_id=str(uuid.uuid4()),
    )


@app.get("/api/v1/detectors")
async def list_detectors(authorization: Optional[str] = Header(None)):
    """
    List available detectors.

    Args:
        authorization: Bearer token for authentication

    Returns:
        List of available detectors
    """
    verify_auth_token(authorization)

    return {
        "detectors": [
            {
                "id": detector_id,
                "name": config["name"],
                "triggers": config["triggers"],
            }
            for detector_id, config in MOCK_DETECTORS.items()
        ]
    }


if __name__ == "__main__":
    print("🚀 Starting IBM FMS Guardrails Mock Server...")
    print("📍 Server will be available at: http://localhost:8001")
    print("📚 API docs at: http://localhost:8001/docs")
    print("\nAvailable detectors:")
    for detector_id, config in MOCK_DETECTORS.items():
        print(f"  - {detector_id}: {config['name']}")
    print("\n✨ Use any Bearer token for authentication in this mock server\n")

    uvicorn.run(app, host="0.0.0.0", port=8001)
