"""
Prebuilt regex patterns for content filtering.

This module contains predefined regex patterns for detecting sensitive information
like SSNs, credit cards, API keys, etc.
"""

import re
from enum import Enum
from typing import Dict, List, Pattern

# US Social Security Number patterns
US_SSN_PATTERN = r"\b\d{3}-\d{2}-\d{4}\b"  # Format: 123-45-6789
US_SSN_NO_DASH_PATTERN = r"\b(?!000|666|9\d{2})\d{3}(?!00)\d{2}(?!0000)\d{4}\b"  # Format: 123456789 (with validation)

# Credit Card patterns
VISA_PATTERN = r"\b4\d{3}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"  # Visa starts with 4
MASTERCARD_PATTERN = r"\b5[1-5]\d{2}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"  # Mastercard starts with 51-55
AMEX_PATTERN = r"\b3[47]\d{2}[\s\-]?\d{6}[\s\-]?\d{5}\b"  # Amex starts with 34 or 37
DISCOVER_PATTERN = r"\b6(?:011|5\d{2})[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"  # Discover starts with 6011 or 65

# Email pattern
EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"

# Phone number patterns (US)
US_PHONE_PATTERN = r"\b(?:\+?1[\s.-]?)?\(?([0-9]{3})\)?[\s.-]?([0-9]{3})[\s.-]?([0-9]{4})\b"

# API Key patterns (common formats)
AWS_ACCESS_KEY_PATTERN = r"\b(AKIA[0-9A-Z]{16})\b"  # AWS Access Key ID
AWS_SECRET_KEY_PATTERN = r"\b([A-Za-z0-9/+=]{40})\b"  # AWS Secret Access Key (generic 40 char)
GITHUB_TOKEN_PATTERN = r"\b(gh[ps]_[a-zA-Z0-9]{36})\b"  # GitHub Personal Access Token
SLACK_TOKEN_PATTERN = r"\b(xox[pboa]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24,32})\b"  # Slack tokens
GENERIC_API_KEY_PATTERN = r"\b([Aa][Pp][Ii][-_]?[Kk][Ee][Yy][\s:=]+['\"]?[A-Za-z0-9_\-]{20,}['\"]?)\b"

# IP Address patterns
IPV4_PATTERN = r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
IPV6_PATTERN = r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"

# URL patterns
URL_PATTERN = r"\b(?:https?://|www\.)[^\s/$.?#].[^\s]*\b"


class PrebuiltPatternName(str, Enum):
    """Enum for prebuilt pattern names"""
    # SSN patterns
    US_SSN = "us_ssn"
    US_SSN_NO_DASH = "us_ssn_no_dash"
    
    # Credit card patterns
    VISA = "visa"
    MASTERCARD = "mastercard"
    AMEX = "amex"
    DISCOVER = "discover"
    CREDIT_CARD = "credit_card"
    
    # Contact information
    EMAIL = "email"
    US_PHONE = "us_phone"
    
    # API keys and secrets
    AWS_ACCESS_KEY = "aws_access_key"
    AWS_SECRET_KEY = "aws_secret_key"
    GITHUB_TOKEN = "github_token"
    SLACK_TOKEN = "slack_token"
    GENERIC_API_KEY = "generic_api_key"
    
    # Network identifiers
    IPV4 = "ipv4"
    IPV6 = "ipv6"
    URL = "url"


PREBUILT_PATTERNS: Dict[str, str] = {
    # SSN patterns
    PrebuiltPatternName.US_SSN.value: US_SSN_PATTERN,
    PrebuiltPatternName.US_SSN_NO_DASH.value: US_SSN_NO_DASH_PATTERN,
    
    # Credit card patterns
    PrebuiltPatternName.VISA.value: VISA_PATTERN,
    PrebuiltPatternName.MASTERCARD.value: MASTERCARD_PATTERN,
    PrebuiltPatternName.AMEX.value: AMEX_PATTERN,
    PrebuiltPatternName.DISCOVER.value: DISCOVER_PATTERN,
    PrebuiltPatternName.CREDIT_CARD.value: rf"(?:{VISA_PATTERN}|{MASTERCARD_PATTERN}|{AMEX_PATTERN}|{DISCOVER_PATTERN})",
    
    # Contact information
    PrebuiltPatternName.EMAIL.value: EMAIL_PATTERN,
    PrebuiltPatternName.US_PHONE.value: US_PHONE_PATTERN,
    
    # API keys and secrets
    PrebuiltPatternName.AWS_ACCESS_KEY.value: AWS_ACCESS_KEY_PATTERN,
    PrebuiltPatternName.AWS_SECRET_KEY.value: AWS_SECRET_KEY_PATTERN,
    PrebuiltPatternName.GITHUB_TOKEN.value: GITHUB_TOKEN_PATTERN,
    PrebuiltPatternName.SLACK_TOKEN.value: SLACK_TOKEN_PATTERN,
    PrebuiltPatternName.GENERIC_API_KEY.value: GENERIC_API_KEY_PATTERN,
    
    # Network identifiers
    PrebuiltPatternName.IPV4.value: IPV4_PATTERN,
    PrebuiltPatternName.IPV6.value: IPV6_PATTERN,
    PrebuiltPatternName.URL.value: URL_PATTERN,
}


def get_compiled_pattern(pattern_name: str) -> Pattern:
    """
    Get a compiled regex pattern by name.
    
    Args:
        pattern_name: Name of the prebuilt pattern
        
    Returns:
        Compiled regex pattern
        
    Raises:
        ValueError: If pattern_name is not found in PREBUILT_PATTERNS
    """
    if pattern_name not in PREBUILT_PATTERNS:
        available_patterns = ", ".join(PREBUILT_PATTERNS.keys())
        raise ValueError(
            f"Unknown pattern name: '{pattern_name}'. "
            f"Available patterns: {available_patterns}"
        )
    
    return re.compile(PREBUILT_PATTERNS[pattern_name], re.IGNORECASE)


def get_all_pattern_names():
    """
    Get a list of all available prebuilt pattern names.
    
    Returns:
        List of pattern names
    """
    return list(PREBUILT_PATTERNS.keys())


# Pattern categories for UI organization
PATTERN_CATEGORIES: Dict[str, List[str]] = {
    "PII Patterns": [
        PrebuiltPatternName.US_SSN.value,
        PrebuiltPatternName.EMAIL.value,
        PrebuiltPatternName.US_PHONE.value,
    ],
    "Payment Card Patterns": [
        PrebuiltPatternName.VISA.value,
        PrebuiltPatternName.MASTERCARD.value,
        PrebuiltPatternName.AMEX.value,
        PrebuiltPatternName.DISCOVER.value,
        PrebuiltPatternName.CREDIT_CARD.value,
    ],
    "Credential Patterns": [
        PrebuiltPatternName.AWS_ACCESS_KEY.value,
        PrebuiltPatternName.AWS_SECRET_KEY.value,
        PrebuiltPatternName.GITHUB_TOKEN.value,
        PrebuiltPatternName.SLACK_TOKEN.value,
        PrebuiltPatternName.GENERIC_API_KEY.value,
    ],
    "Network Patterns": [
        PrebuiltPatternName.IPV4.value,
        PrebuiltPatternName.IPV6.value,
        PrebuiltPatternName.URL.value,
    ],
}


# Pattern descriptions for UI display
PATTERN_DESCRIPTIONS: Dict[str, str] = {
    PrebuiltPatternName.US_SSN.value: "Detects US Social Security Numbers (XXX-XX-XXXX format)",
    PrebuiltPatternName.US_SSN_NO_DASH.value: "Detects US SSN without dashes (XXXXXXXXX format)",
    PrebuiltPatternName.EMAIL.value: "Detects email addresses",
    PrebuiltPatternName.US_PHONE.value: "Detects US phone numbers in various formats",
    PrebuiltPatternName.VISA.value: "Detects Visa credit card numbers",
    PrebuiltPatternName.MASTERCARD.value: "Detects Mastercard credit card numbers",
    PrebuiltPatternName.AMEX.value: "Detects American Express credit card numbers",
    PrebuiltPatternName.DISCOVER.value: "Detects Discover credit card numbers",
    PrebuiltPatternName.CREDIT_CARD.value: "Detects any major credit card number",
    PrebuiltPatternName.AWS_ACCESS_KEY.value: "Detects AWS access keys (AKIA...)",
    PrebuiltPatternName.AWS_SECRET_KEY.value: "Detects AWS secret keys (40 characters)",
    PrebuiltPatternName.GITHUB_TOKEN.value: "Detects GitHub personal access tokens",
    PrebuiltPatternName.SLACK_TOKEN.value: "Detects Slack API tokens",
    PrebuiltPatternName.GENERIC_API_KEY.value: "Detects generic API key patterns",
    PrebuiltPatternName.IPV4.value: "Detects IPv4 addresses",
    PrebuiltPatternName.IPV6.value: "Detects IPv6 addresses",
    PrebuiltPatternName.URL.value: "Detects URLs (http/https)",
}


def get_pattern_metadata() -> List[Dict[str, str]]:
    """
    Return pattern metadata for UI display.
    
    Returns:
        List of dictionaries containing pattern name, category, and description
    """
    prebuilt_patterns = []
    for category, pattern_names in PATTERN_CATEGORIES.items():
        for pattern_name in pattern_names:
            if pattern_name in PREBUILT_PATTERNS:
                prebuilt_patterns.append({
                    "name": pattern_name,
                    "category": category,
                    "description": PATTERN_DESCRIPTIONS.get(
                        pattern_name,
                        f"Detects {pattern_name.replace('_', ' ').title()}"
                    ),
                })
    return prebuilt_patterns

