"""
Prebuilt regex patterns for content filtering.

This module loads predefined regex patterns from patterns.json for detecting
sensitive information like SSNs, credit cards, API keys, etc.
"""

import json
import os
import re
from enum import Enum
from typing import Any, Dict, List, Pattern


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


# Capture any extra configuration declared per pattern (e.g., contextual keywords)
KNOWN_PATTERN_KEYS = {
    "name",
    "display_name",
    "pattern",
    "category",
    "action",
    "description",
}

PATTERN_EXTRA_CONFIG: Dict[str, Dict[str, Any]] = {}
for pattern_data in _PATTERNS_DATA["patterns"]:
    extra_config = {
        key: value
        for key, value in pattern_data.items()
        if key not in KNOWN_PATTERN_KEYS
    }
    PATTERN_EXTRA_CONFIG[pattern_data["name"]] = extra_config


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


def get_available_content_categories() -> List[Dict[str, str]]:
    """
    Return available content categories for UI display.

    Includes categories defined in .yaml/.yml files and in .json files
    (e.g. harm_toxic_abuse.json).

    Returns:
        List of dictionaries containing category name, display_name, and description
    """
    import yaml

    categories_dir = os.path.join(os.path.dirname(__file__), "categories")
    available_categories = []

    if not os.path.exists(categories_dir):
        return []

    # Scan the categories directory for YAML files
    for filename in os.listdir(categories_dir):
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            category_file_path = os.path.join(categories_dir, filename)
            try:
                with open(category_file_path, "r") as f:
                    category_data = yaml.safe_load(f)

                if category_data and "category_name" in category_data:
                    # Create display name from category name (convert harmful_self_harm -> Harmful Self Harm)
                    display_name = (
                        category_data["category_name"].replace("_", " ").title()
                    )

                    available_categories.append(
                        {
                            "name": category_data["category_name"],
                            "display_name": display_name,
                            "description": category_data.get("description", ""),
                            "default_action": category_data.get(
                                "default_action", "BLOCK"
                            ),
                        }
                    )
            except Exception as e:
                # Skip files that can't be loaded but log the error for debugging
                from litellm._logging import verbose_proxy_logger

                verbose_proxy_logger.warning(
                    f"Failed to load category file {filename}: {str(e)}"
                )
                continue
        elif filename.endswith(".json"):
            # JSON category files (e.g. harm_toxic_abuse.json) - no YAML header, use filename
            category_name = os.path.splitext(filename)[0]
            try:
                if category_name == "harm_toxic_abuse":
                    display_name = "Harmful Toxic Abuse"
                    description = (
                        "Detects harmful, toxic, or abusive language and content"
                    )
                else:
                    display_name = category_name.replace("_", " ").title()
                    description = f"Content category: {display_name}"
                available_categories.append(
                    {
                        "name": category_name,
                        "display_name": display_name,
                        "description": description,
                        "default_action": "BLOCK",
                    }
                )
            except Exception:
                continue

    # Sort by name for consistent ordering
    available_categories.sort(key=lambda x: x["name"])

    return available_categories
