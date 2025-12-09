"""
Prebuilt regex patterns for content filtering.

This module loads predefined regex patterns from patterns.json for detecting 
sensitive information like SSNs, credit cards, API keys, etc.
"""

import json
import os
import re
from enum import Enum
from typing import Dict, List, Pattern


def _load_patterns_from_json() -> Dict:
    """Load pattern definitions from patterns.json file"""
    json_path = os.path.join(os.path.dirname(__file__), "patterns.json")
    with open(json_path, "r") as f:
        return json.load(f)


# Load patterns from JSON
_PATTERNS_DATA = _load_patterns_from_json()


class PrebuiltPatternName(str, Enum):
    """Enum for prebuilt pattern names - dynamically generated from JSON"""
    pass


# Dynamically create enum values from JSON
for pattern_data in _PATTERNS_DATA["patterns"]:
    setattr(PrebuiltPatternName, pattern_data["name"].upper(), pattern_data["name"])


# Build lookup dictionaries from JSON
PREBUILT_PATTERNS: Dict[str, str] = {
    pattern_data["name"]: pattern_data["pattern"]
    for pattern_data in _PATTERNS_DATA["patterns"]
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


def get_all_pattern_names() -> List[str]:
    """
    Get a list of all available prebuilt pattern names.
    
    Returns:
        List of pattern names
    """
    return list(PREBUILT_PATTERNS.keys())


# Build category mapping from JSON
PATTERN_CATEGORIES: Dict[str, List[str]] = {}
for pattern_data in _PATTERNS_DATA["patterns"]:
    category = pattern_data["category"]
    if category not in PATTERN_CATEGORIES:
        PATTERN_CATEGORIES[category] = []
    PATTERN_CATEGORIES[category].append(pattern_data["name"])


# Build display names mapping from JSON
PATTERN_DISPLAY_NAMES: Dict[str, str] = {
    pattern_data["name"]: pattern_data["display_name"]
    for pattern_data in _PATTERNS_DATA["patterns"]
}


# Build descriptions mapping from JSON
PATTERN_DESCRIPTIONS: Dict[str, str] = {
    pattern_data["name"]: pattern_data["description"]
    for pattern_data in _PATTERNS_DATA["patterns"]
}


def get_pattern_metadata() -> List[Dict[str, str]]:
    """
    Return pattern metadata for UI display.
    
    Returns:
        List of dictionaries containing pattern name, display_name, category, and description
    """
    return [
        {
            "name": pattern_data["name"],
            "display_name": pattern_data["display_name"],
            "category": pattern_data["category"],
            "description": pattern_data["description"],
        }
        for pattern_data in _PATTERNS_DATA["patterns"]
    ]

