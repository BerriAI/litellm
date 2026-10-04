"""
Prebuilt regex patterns for content filtering.

This module loads predefined regex patterns from patterns.json for detecting
sensitive information like SSNs, credit cards, API keys, etc.
"""

import json
import os
import re
from collections.abc import Iterator
from enum import Enum
from re import Pattern
from typing import Any, Final

from litellm.proxy.guardrails.content_filter_data import DATA_ROOTS, category_dirs


def _load_patterns_from_json() -> dict:
    """Load pattern definitions from patterns.json file"""
    json_path: Final = os.path.join(os.path.dirname(__file__), "patterns.json")
    with open(json_path, "r") as f:
        return json.load(f)


# Load patterns from JSON
_PATTERNS_DATA: Final = _load_patterns_from_json()


class PrebuiltPatternName(str, Enum):
    """Enum for prebuilt pattern names - dynamically generated from JSON"""


# Dynamically create enum values from JSON
for pattern_data in _PATTERNS_DATA["patterns"]:
    setattr(PrebuiltPatternName, pattern_data["name"].upper(), pattern_data["name"])


# Build lookup dictionaries from JSON
PREBUILT_PATTERNS: Final[dict[str, str]] = {
    pattern_data["name"]: pattern_data["pattern"] for pattern_data in _PATTERNS_DATA["patterns"]
}


# Capture any extra configuration declared per pattern (e.g., contextual keywords)
KNOWN_PATTERN_KEYS: Final = {
    "name",
    "display_name",
    "pattern",
    "category",
    "action",
    "description",
}

PATTERN_EXTRA_CONFIG: Final[dict[str, dict[str, Any]]] = {}
for pattern_data in _PATTERNS_DATA["patterns"]:
    extra_config = {key: value for key, value in pattern_data.items() if key not in KNOWN_PATTERN_KEYS}
    PATTERN_EXTRA_CONFIG[pattern_data["name"]] = extra_config


def get_compiled_pattern(pattern_name: str) -> Pattern[str]:
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
        available_patterns: Final = ", ".join(PREBUILT_PATTERNS.keys())
        raise ValueError(f"Unknown pattern name: '{pattern_name}'. Available patterns: {available_patterns}")

    return re.compile(PREBUILT_PATTERNS[pattern_name], re.IGNORECASE)


def get_all_pattern_names() -> list[str]:
    """
    Get a list of all available prebuilt pattern names.

    Returns:
        List of pattern names
    """
    return list(PREBUILT_PATTERNS.keys())


# Build category mapping from JSON
PATTERN_CATEGORIES: Final[dict[str, list[str]]] = {}
for pattern_data in _PATTERNS_DATA["patterns"]:
    category = pattern_data["category"]
    if category not in PATTERN_CATEGORIES:
        PATTERN_CATEGORIES[category] = []
    PATTERN_CATEGORIES[category].append(pattern_data["name"])


# Build display names mapping from JSON
PATTERN_DISPLAY_NAMES: Final[dict[str, str]] = {
    pattern_data["name"]: pattern_data["display_name"] for pattern_data in _PATTERNS_DATA["patterns"]
}


# Build descriptions mapping from JSON
PATTERN_DESCRIPTIONS: Final[dict[str, str]] = {
    pattern_data["name"]: pattern_data["description"] for pattern_data in _PATTERNS_DATA["patterns"]
}


def get_pattern_metadata() -> list[dict[str, str]]:
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


def _category_entry(categories_dir: str, filename: str) -> dict[str, str] | None:
    import yaml

    category_file_path: Final = os.path.join(categories_dir, filename)
    if filename.endswith((".yaml", ".yml")):
        try:
            with open(category_file_path, "r") as f:
                category_data = yaml.safe_load(f)
        except Exception as e:
            from litellm._logging import verbose_proxy_logger

            verbose_proxy_logger.warning("Failed to load category file %s: %s", filename, e)
            return None
        if not category_data or "category_name" not in category_data:
            return None
        return {
            "name": category_data["category_name"],
            "display_name": category_data.get("display_name")
            or category_data["category_name"].replace("_", " ").title(),
            "description": category_data.get("description", ""),
            "default_action": category_data.get("default_action", "BLOCK"),
        }
    if filename.endswith(".json"):
        category_name: Final = os.path.splitext(filename)[0]
        if category_name == "harm_toxic_abuse":
            return {
                "name": category_name,
                "display_name": "Harmful Toxic Abuse",
                "description": "Detects harmful, toxic, or abusive language and content",
                "default_action": "BLOCK",
            }
        display_name: Final = category_name.replace("_", " ").title()
        return {
            "name": category_name,
            "display_name": display_name,
            "description": f"Content category: {display_name}",
            "default_action": "BLOCK",
        }
    return None


def get_available_content_categories(roots: tuple[str, ...] = DATA_ROOTS) -> list[dict[str, str]]:
    """
    Return available content categories for UI display.

    Includes categories defined in .yaml/.yml files and in .json files
    (e.g. harm_toxic_abuse.json) under every data root, bundled first. A
    name that appears under several roots is listed once, from the first root.

    Returns:
        List of dictionaries containing category name, display_name, and description
    """
    entries: Final = tuple(e for e in (_category_entry(d, f) for d, f in _category_files(roots)) if e is not None)
    first_per_name: Final = {e["name"]: e for e in reversed(entries)}
    return sorted(first_per_name.values(), key=lambda x: x["name"])


def _category_files(roots: tuple[str, ...]) -> Iterator[tuple[str, str]]:
    for categories_dir in category_dirs(roots):
        for filename in sorted(os.listdir(categories_dir)):
            yield categories_dir, filename
