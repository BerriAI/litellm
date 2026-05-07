#!/usr/bin/env python3
"""
Mock Bedrock Guardrail API Server

This is a FastAPI server that mimics the AWS Bedrock Guardrail API for testing purposes.
It follows the same API spec as the real Bedrock guardrail endpoint.

Usage:
    python mock_bedrock_guardrail_server.py

The server will start on http://localhost:8080
"""

import os
import re
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ============================================================================
# Request/Response Models (matching Bedrock API spec)
# ============================================================================


class BedrockTextContent(BaseModel):
    text: str


class BedrockContentItem(BaseModel):
    text: BedrockTextContent


class BedrockRequest(BaseModel):
    source: Literal["INPUT", "OUTPUT"]
    content: List[BedrockContentItem] = Field(default_factory=list)


class BedrockGuardrailOutput(BaseModel):
    text: Optional[str] = None


class TopicPolicyItem(BaseModel):
    name: str
    type: str
    action: Literal["BLOCKED", "NONE"]


class TopicPolicy(BaseModel):
    topics: List[TopicPolicyItem] = Field(default_factory=list)


class ContentFilterItem(BaseModel):
    type: str
    confidence: str
    action: Literal["BLOCKED", "NONE"]


class ContentPolicy(BaseModel):
    filters: List[ContentFilterItem] = Field(default_factory=list)


class CustomWord(BaseModel):
    match: str
    action: Literal["BLOCKED", "NONE"]


class WordPolicy(BaseModel):
    customWords: List[CustomWord] = Field(default_factory=list)
    managedWordLists: List[Dict[str, Any]] = Field(default_factory=list)


class PiiEntity(BaseModel):
    type: str
    match: str
    action: Literal["BLOCKED", "ANONYMIZED", "NONE"]


class RegexMatch(BaseModel):
    name: str
    match: str
    regex: str
    action: Literal["BLOCKED", "ANONYMIZED", "NONE"]


class SensitiveInformationPolicy(BaseModel):
    piiEntities: List[PiiEntity] = Field(default_factory=list)
    regexes: List[RegexMatch] = Field(default_factory=list)


class ContextualGroundingFilter(BaseModel):
    type: str
    threshold: float
    score: float
    action: Literal["BLOCKED", "NONE"]


class ContextualGroundingPolicy(BaseModel):
    filters: List[ContextualGroundingFilter] = Field(default_factory=list)


class Assessment(BaseModel):
    topicPolicy: Optional[TopicPolicy] = None
    contentPolicy: Optional[ContentPolicy] = None
    wordPolicy: Optional[WordPolicy] = None
    sensitiveInformationPolicy: Optional[SensitiveInformationPolicy] = None
    contextualGroundingPolicy: Optional[ContextualGroundingPolicy] = None


class BedrockGuardrailResponse(BaseModel):
    usage: Dict[str, int] = Field(
        default_factory=lambda: {"topicPolicyUnits": 1, "contentPolicyUnits": 1}
    )
    action: Literal["NONE", "GUARDRAIL_INTERVENED"] = "NONE"
    outputs: List[BedrockGuardrailOutput] = Field(default_factory=list)
    assessments: List[Assessment] = Field(default_factory=list)


# ============================================================================
# Mock Guardrail Configuration
# ============================================================================


class GuardrailConfig(BaseModel):
    """Configuration for mock guardrail behavior"""

    blocked_words: List[str] = Field(
        default_factory=lambda: ["offensive", "inappropriate", "badword"]
    )
    blocked_topics: List[str] = Field(default_factory=lambda: ["violence", "illegal"])
    pii_patterns: Dict[str, str] = Field(
        default_factory=lambda: {
            "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "PHONE": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
            "CREDIT_CARD": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        }
    )
    anonymize_pii: bool = True  # If True, ANONYMIZE PII; if False, BLOCK it
    bearer_token: str = "mock-bedrock-token-12345"


# Global config
GUARDRAIL_CONFIG = GuardrailConfig()

# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title="Mock Bedrock Guardrail API",
    description="Mock server mimicking AWS Bedrock Guardrail API",
    version="1.0.0",
)


# ============================================================================
# Authentication
# ============================================================================


async def verify_bearer_token(authorization: Optional[str] = Header(None)) -> str:
    """
    Verify the Bearer token from the Authorization header.

    Args:
        authorization: The Authorization header value

    Returns:
        The token if valid

    Raises:
        HTTPException: If token is missing or invalid
    """
    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if it's a Bearer token
    parts = authorization.split()
    print(f"parts: {parts}")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = parts[1]

    # Verify token
    if token != GUARDRAIL_CONFIG.bearer_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid bearer token",
        )

    return token


# ============================================================================
# Guardrail Logic
# ============================================================================


def check_blocked_words(text: str) -> Optional[WordPolicy]:
    """Check if text contains blocked words"""
    found_words = []
    text_lower = text.lower()

    for word in GUARDRAIL_CONFIG.blocked_words:
        if word.lower() in text_lower:
            found_words.append(CustomWord(match=word, action="BLOCKED"))

    if found_words:
        return WordPolicy(customWords=found_words)
    return None


def check_blocked_topics(text: str) -> Optional[TopicPolicy]:
    """Check if text contains blocked topics"""
    found_topics = []
    text_lower = text.lower()

    for topic in GUARDRAIL_CONFIG.blocked_topics:
        if topic.lower() in text_lower:
            found_topics.append(
                TopicPolicyItem(name=topic, type=topic.upper(), action="BLOCKED")
            )

    if found_topics:
        return TopicPolicy(topics=found_topics)
    return None


def check_pii(text: str) -> tuple[Optional[SensitiveInformationPolicy], str]:
    """
    Check for PII in text and return policy + anonymized text

    Returns:
        Tuple of (SensitiveInformationPolicy or None, anonymized_text)
    """
    pii_entities = []
    anonymized_text = text
    action = "ANONYMIZED" if GUARDRAIL_CONFIG.anonymize_pii else "BLOCKED"

    for pii_type, pattern in GUARDRAIL_CONFIG.pii_patterns.items():
        try:
            # Compile the regex pattern with a timeout to prevent ReDoS attacks
            compiled_pattern = re.compile(pattern)
            matches = compiled_pattern.finditer(text)
            for match in matches:
                matched_text = match.group()
                pii_entities.append(
                    PiiEntity(type=pii_type, match=matched_text, action=action)
                )

                # Anonymize the text if configured
                if GUARDRAIL_CONFIG.anonymize_pii:
                    anonymized_text = anonymized_text.replace(
                        matched_text, f"[{pii_type}_REDACTED]"
                    )
        except re.error:
            # Invalid regex pattern - skip it and log a warning
            print(f"Warning: Invalid regex pattern for PII type {pii_type}: {pattern}")
            continue

    if pii_entities:
        return SensitiveInformationPolicy(piiEntities=pii_entities), anonymized_text

    return None, text


def process_guardrail_request(
    request: BedrockRequest,
) -> tuple[BedrockGuardrailResponse, List[str]]:
    """
    Process a guardrail request and return the response.

    Returns:
        Tuple of (response, list of output texts)
    """
    all_text_content = []
    output_texts = []

    # Extract all text from content items
    for content_item in request.content:
        if content_item.text and content_item.text.text:
            all_text_content.append(content_item.text.text)

    # Combine all text for analysis
    combined_text = " ".join(all_text_content)

    # Initialize response
    response = BedrockGuardrailResponse()
    assessment = Assessment()
    has_intervention = False

    # Check for blocked words
    word_policy = check_blocked_words(combined_text)
    if word_policy:
        assessment.wordPolicy = word_policy
        has_intervention = True

    # Check for blocked topics
    topic_policy = check_blocked_topics(combined_text)
    if topic_policy:
        assessment.topicPolicy = topic_policy
        has_intervention = True

    # Check for PII
    for text in all_text_content:
        pii_policy, anonymized_text = check_pii(text)
        if pii_policy:
            assessment.sensitiveInformationPolicy = pii_policy
            if GUARDRAIL_CONFIG.anonymize_pii:
                # If anonymizing, we don't block, we modify the text
                output_texts.append(anonymized_text)
                has_intervention = True
            else:
                # If not anonymizing PII, we block it
                output_texts.append(text)
                has_intervention = True
        else:
            output_texts.append(text)

    # Build response
    if has_intervention:
        response.action = "GUARDRAIL_INTERVENED"
        # Only add assessment if there were interventions
        response.assessments = [assessment]

    # Add outputs (modified or original text)
    response.outputs = [BedrockGuardrailOutput(text=txt) for txt in output_texts]

    return response, output_texts


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Mock Bedrock Guardrail API",
        "status": "running",
        "endpoint_format": "/guardrail/{guardrailIdentifier}/version/{guardrailVersion}/apply",
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


"""
LiteLLM exposes a basic guardrail API with the text extracted from the request and sent to the guardrail API, as well as the received request body for any further processing. 

This works across all LiteLLM endpoints (completion, anthropic /v1/messages, responses api, image generation, embedding, etc.)

This makes it easy to support your own guardrail API without having to make a PR to LiteLLM.

LiteLLM supports passing any provider specific params from LiteLLM config.yaml to the guardrail API.

Example:

```yaml
guardrails:
  - guardrail_name: "bedrock-content-guard"
    litellm_params:
      guardrail: generic_guardrail_api
      mode: "pre_call"
      api_key: os.environ/GUARDRAIL_API_KEY
      api_base: os.environ/GUARDRAIL_API_BASE
      additional_provider_specific_params:
        api_version: os.environ/GUARDRAIL_API_VERSION # additional provider specific params
```

This is a beta API. Please help us improve it. 
"""


class LitellmBasicGuardrailRequest(BaseModel):
    texts: List[str]
    images: Optional[List[str]] = None
    tools: Optional[List[dict]] = None
    tool_calls: Optional[List[dict]] = None
    request_data: Dict[str, Any] = Field(default_factory=dict)
    additional_provider_specific_params: Dict[str, Any] = Field(default_factory=dict)
    input_type: Literal["request", "response"]
    litellm_call_id: Optional[str] = None
    litellm_trace_id: Optional[str] = None
    structured_messages: Optional[List[Dict[str, Any]]] = None


class LitellmBasicGuardrailResponse(BaseModel):
    action: Literal[
        "BLOCKED", "NONE", "GUARDRAIL_INTERVENED"
    ]  # BLOCKED = litellm will raise an error, NONE = litellm will continue, GUARDRAIL_INTERVENED = litellm will continue, but the text was modified by the guardrail
    blocked_reason: Optional[str] = None  # only if action is BLOCKED, otherwise None
    texts: Optional[List[str]] = None
    images: Optional[List[str]] = None


@app.post(
    "/beta/litellm_basic_guardrail_api",
    response_model=LitellmBasicGuardrailResponse,
)
async def beta_litellm_basic_guardrail_api(
    request: LitellmBasicGuardrailRequest,
) -> LitellmBasicGuardrailResponse:
    """
    Apply guardrail to input or output content.

    This endpoint mimics the AWS Bedrock ApplyGuardrail API.

    Args:
        request: The guardrail request containing content to analyze
        token: Bearer token (verified by dependency)

    Returns:
        LitellmBasicGuardrailResponse with analysis results
    """
    print(f"request: {request}")
    if any("ishaan" in text.lower() for text in request.texts):
        return LitellmBasicGuardrailResponse(
            action="BLOCKED", blocked_reason="Ishaan is not allowed"
        )
    elif any("pii_value" in text for text in request.texts):
        return LitellmBasicGuardrailResponse(
            action="GUARDRAIL_INTERVENED",
            texts=[
                text.replace("pii_value", "pii_value_redacted")
                for text in request.texts
            ],
        )
    return LitellmBasicGuardrailResponse(action="NONE")


@app.post("/config/update")
async def update_config(
    config: GuardrailConfig, token: str = Depends(verify_bearer_token)
):
    """
    Update the guardrail configuration.

    This is a testing endpoint to modify the mock guardrail behavior.

    Args:
        config: New guardrail configuration
        token: Bearer token (verified by dependency)

    Returns:
        Updated configuration
    """
    global GUARDRAIL_CONFIG
    GUARDRAIL_CONFIG = config
    return {"status": "updated", "config": GUARDRAIL_CONFIG}


@app.get("/config")
async def get_config(token: str = Depends(verify_bearer_token)):
    """
    Get the current guardrail configuration.

    Args:
        token: Bearer token (verified by dependency)

    Returns:
        Current configuration
    """
    return GUARDRAIL_CONFIG


# ============================================================================
# Error Handlers
# ============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom error handler for HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
        headers=exc.headers,
    )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Get configuration from environment
    host = os.getenv("MOCK_BEDROCK_HOST", "0.0.0.0")
    port = int(os.getenv("MOCK_BEDROCK_PORT", "8080"))
    bearer_token = os.getenv("MOCK_BEDROCK_TOKEN", "mock-bedrock-token-12345")

    # Update config with environment token
    GUARDRAIL_CONFIG.bearer_token = bearer_token

    print("=" * 80)
    print("Mock Bedrock Guardrail API Server")
    print("=" * 80)
    print(f"Server starting on: http://{host}:{port}")
    print(f"Bearer Token: {bearer_token}")
    print(f"Endpoint: POST /guardrail/{{id}}/version/{{version}}/apply")
    print("=" * 80)
    print("\nExample curl command:")
    print(
        f"""
curl -X POST "http://{host}:{port}/guardrail/test-guardrail/version/1/apply" \\
  -H "Authorization: Bearer {bearer_token}" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "source": "INPUT",
    "content": [
      {{
        "text": {{
          "text": "Hello, my email is test@example.com"
        }}
      }}
    ]
  }}'
    """
    )
    print("=" * 80)

    uvicorn.run(app, host=host, port=port)
