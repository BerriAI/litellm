"""
Prebuilt regex patterns for content filtering.

This module contains predefined regex patterns for detecting sensitive information
like SSNs, credit cards, API keys, etc.
"""

import re
from typing import Dict, Pattern

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


PREBUILT_PATTERNS: Dict[str, str] = {
    # SSN patterns
    "us_ssn": US_SSN_PATTERN,
    "us_ssn_no_dash": US_SSN_NO_DASH_PATTERN,
    
    # Credit card patterns
    "visa": VISA_PATTERN,
    "mastercard": MASTERCARD_PATTERN,
    "amex": AMEX_PATTERN,
    "discover": DISCOVER_PATTERN,
    "credit_card": rf"(?:{VISA_PATTERN}|{MASTERCARD_PATTERN}|{AMEX_PATTERN}|{DISCOVER_PATTERN})",
    
    # Contact information
    "email": EMAIL_PATTERN,
    "us_phone": US_PHONE_PATTERN,
    
    # API keys and secrets
    "aws_access_key": AWS_ACCESS_KEY_PATTERN,
    "aws_secret_key": AWS_SECRET_KEY_PATTERN,
    "github_token": GITHUB_TOKEN_PATTERN,
    "slack_token": SLACK_TOKEN_PATTERN,
    "generic_api_key": GENERIC_API_KEY_PATTERN,
    
    # Network identifiers
    "ipv4": IPV4_PATTERN,
    "ipv6": IPV6_PATTERN,
    "url": URL_PATTERN,
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

