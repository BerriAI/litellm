"""Minimal JSON helpers used across LiteLLM extras and scenarios."""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Union

try:  # pragma: no cover - optional dependency
    from json_repair import repair_json
except ImportError:  # pragma: no cover - fallback when package missing
    repair_json = None  # type: ignore

__all__ = [
    "PathEncoder",
    "json_serialize",
    "load_json_file",
    "save_json_to_file",
    "parse_json",
    "clean_json_string",
]

logger = logging.getLogger(__name__)


class PathEncoder(json.JSONEncoder):
    """JSON encoder that turns ``pathlib.Path`` objects into strings."""

    def default(self, obj: Any) -> Any:  # noqa: D401 - short override
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def json_serialize(data: Any, *, handle_paths: bool = False, **kwargs: Any) -> str:
    """Serialise ``data`` to JSON, optionally handling ``Path`` instances."""

    if handle_paths:
        return json.dumps(data, cls=PathEncoder, **kwargs)
    return json.dumps(data, **kwargs)


def load_json_file(file_path: str) -> Any:
    """Load JSON from ``file_path``; returns ``None`` if the file is missing."""

    if not os.path.exists(file_path):
        logger.warning("File does not exist: %s", file_path)
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        logger.warning("JSON decoding error; retrying with utf-8-sig: %s", file_path)
        with open(file_path, "r", encoding="utf-8-sig") as file:
            return json.load(file)


def save_json_to_file(data: Any, file_path: str) -> None:
    """Persist ``data`` to ``file_path``, creating the directory when needed."""

    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)


_JSON_PATTERN = re.compile(r"(\[.*\]|\{.*\})", re.DOTALL)


def parse_json(content: str) -> Union[dict, list, str]:
    """Attempt to parse ``content`` as JSON; fall back to repairing the string."""

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    match = _JSON_PATTERN.search(content)
    if match:
        content = match.group(1)
    if repair_json is None:
        logger.debug("json_repair not installed; returning original content")
        return content

    try:
        repaired = repair_json(content, return_objects=True)
        if isinstance(repaired, (dict, list)):
            return repaired
        return json.loads(repaired)
    except Exception:
        logger.debug("Returning original content after repair failure")
        return content


def clean_json_string(content: Union[str, dict, list], *, return_dict: bool = False) -> Union[str, dict, list]:
    """Normalise ``content`` and optionally return a Python structure."""

    if isinstance(content, (dict, list)):
        return content if return_dict else json.dumps(content)

    cleaned = parse_json(content)
    if return_dict and isinstance(cleaned, (dict, list)):
        return cleaned
    if return_dict:
        return {}
    return json.dumps(cleaned) if isinstance(cleaned, (dict, list)) else str(cleaned)
