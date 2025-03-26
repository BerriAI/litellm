import base64
import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pydantic
from pydantic import BaseModel
from typing_extensions import Annotated


# Define the TextualdetectionType enum
class TextualdetectionType(Enum):
    KEYWORD = "Keyword"
    PII = "PII"
    SECRET = "Secret"

# Define the Textualdetection class
@dataclass
class Textualdetection:
    type: TextualdetectionType
    name: str
    score: Optional[float]

class Modality(BaseModel):
    group: str
    type: str

# Define the Extraction class
class Extraction(BaseModel):
    r"""Represents the extracted information to log."""

    pi_is: Annotated[Optional[Dict[str, float]], pydantic.Field(alias="PIIs")] = None

    annotations: Optional[Dict[str, str]] = None

    categories: Optional[List[Modality]] = None

    confidentiality: Optional[float] = None

    data: Optional[str] = None

    detections: Optional[List[Textualdetection]] = None

    exploits: Optional[Dict[str, float]] = None

    hash: Optional[str] = None

    intent: Optional[Dict[str, float]] = None

    internal: Optional[bool] = None

    is_file: Annotated[Optional[bool], pydantic.Field(alias="isFile")] = None

    is_stored: Annotated[Optional[bool], pydantic.Field(alias="isStored")] = None

    keywords: Optional[Dict[str, float]] = None

    label: Optional[str] = None

    languages: Optional[Dict[str, float]] = None

    malcontents: Optional[Dict[str, float]] = None

    modalities: Optional[List[Modality]] = None

    relevance: Optional[float] = None

    secrets: Optional[Dict[str, float]] = None

    topics: Optional[Dict[str, float]] = None


# Define the GuardResult class
@dataclass
class GuardResult:
    matched: bool
    guard_name: str
    threshold: str
    actual_value: float
    match_count: int
    match_values: List[str]

# Define GuardName as an Enum instead of a class
class GuardName(Enum):
    PROMPT_INJECTION = "PROMPT_INJECTION"
    JAIL_BREAK = "JAIL_BREAK"
    MALICIOUS_URL = "MALICIOUS_URL"
    TOXIC = "TOXIC"
    BIASED = "BIASED"
    HARMFUL_CONTENT = "HARMFUL_CONTENT"
    LANGUAGE = "LANGUAGE"
    PII_DETECTOR = "PII_DETECTOR"
    SECRETS_DETECTOR = "SECRETS_DETECTOR"
    KEYWORD_DETECTOR = "KEYWORD_DETECTOR"

def decode_jwt(token: str) -> Optional[dict]:
    """
    Decodes a JWT token without validation
    """
    try:
        parts = token.split('.')
        if len(parts) != 3:
            return None

        # Convert Base64Url to Base64
        base64_str = parts[1].replace('-', '+').replace('_', '/')

        # Add padding if necessary
        padding = len(base64_str) % 4
        if padding:
            base64_str += '=' * (4 - padding)

        # Decode Base64 payload
        json_payload = base64.b64decode(base64_str).decode('utf-8')
        return json.loads(json_payload)
    except Exception:
        return None

def get_apex_url_from_token(token: str) -> Optional[str]:
    """
    Extracts apex URL from token
    """
    try:
        decoded_token = decode_jwt(token)
        if decoded_token and 'opaque' in decoded_token:
            return decoded_token['opaque'].get('apex-url')
        return None
    except Exception:
        return None

class ResponseHelper:
    def get_guard_value(self, lookup: Optional[Dict[str, float]], key: str) -> Tuple[bool, float]:
        """Gets the guard value from a lookup object"""
        if not lookup or key not in lookup:
            return False, 0.0
        return True, lookup[key]

    def get_text_detections(
        self,
        lookup: Optional[Dict[str, float]],
        threshold: float,
        detection_type: TextualdetectionType,
        detections: Optional[List[Textualdetection]],
        match_name: Optional[str] = None
    ) -> Tuple[bool, float, int, List[str]]:
        """Gets text detection values"""
        if match_name:
            if not detections:
                return False, 0.0, 0, []

            text_matches = [
                d.score for d in detections
                if (d.type == detection_type and
                    d.name == match_name and
                    d.score is not None and
                    isinstance(d.score, (int, float)) and
                    d.score >= threshold)
            ]

            count = len(text_matches)

            if count == 0 and lookup and match_name in lookup:
                lookup_value = lookup[match_name]
                if isinstance(lookup_value, (int, float)):
                    return True, lookup_value, 1, [match_name]

            if count == 0:
                return False, 0.0, 0, []

            max_score = max(text_matches)
            return True, max_score, count, [match_name]

        exists = bool(lookup and lookup.keys())
        return (
            exists,
            1.0 if exists else 0.0,
            len(lookup.keys()) if lookup else 0,
            list(lookup.keys()) if lookup else []
        )

    def evaluate(
        self,
        extraction: Extraction,
        guard_name: GuardName,
        threshold: float,
        match_name: Optional[str] = None
    ) -> GuardResult:
        """
        Evaluates a check condition using guard name and threshold.
        """
        exists = False
        value = 0.0
        match_count = 0
        match_values: List[str] = []

        try:
            if guard_name in [GuardName.PROMPT_INJECTION, GuardName.JAIL_BREAK, GuardName.MALICIOUS_URL]:
                exists, value = self.get_guard_value(
                    extraction.exploits,
                    guard_name.value.lower()
                )
            elif guard_name in [GuardName.TOXIC, GuardName.BIASED, GuardName.HARMFUL_CONTENT]:
                exists, value = self.get_guard_value(
                    extraction.malcontents,
                    guard_name.value.lower()
                )
            elif guard_name == GuardName.LANGUAGE:
                if match_name:
                    exists, value = self.get_guard_value(extraction.languages, match_name)
                elif extraction.languages:
                    exists = bool(extraction.languages)
                    value = 1.0
            elif guard_name == GuardName.PII_DETECTOR:
                exists, value, match_count, match_values = self.get_text_detections(
                    extraction.pi_is,
                    threshold,
                    TextualdetectionType.PII,
                    extraction.detections,
                    match_name
                )
            elif guard_name == GuardName.SECRETS_DETECTOR:
                exists, value, match_count, match_values = self.get_text_detections(
                    extraction.secrets,
                    threshold,
                    TextualdetectionType.SECRET,
                    extraction.detections,
                    match_name
                )
            elif guard_name == GuardName.KEYWORD_DETECTOR:
                exists, value, match_count, match_values = self.get_text_detections(
                    extraction.keywords,
                    threshold,
                    TextualdetectionType.KEYWORD,
                    extraction.detections,
                    match_name
                )

            matched = exists and value >= threshold

            return GuardResult(
                matched=matched,
                guard_name=guard_name.value,
                threshold=str(threshold),
                actual_value=value,
                match_count=match_count,
                match_values=match_values
            )
        except Exception as e:
            raise Exception(f"Error in evaluation: {str(e)}")
