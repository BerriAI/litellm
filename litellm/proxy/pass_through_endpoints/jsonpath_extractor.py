"""
JSONPath Extractor Module

Extracts field values from data using simple JSONPath-like expressions.
"""

from collections.abc import Mapping, Sequence
from itertools import chain
from typing import Final

from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping

_JSON_VALUE: Final = TypeAdapter(object)


def _json_object(value: object) -> Mapping[str, object] | None:
    mapping: Final = as_str_mapping(value)
    return mapping if isinstance(value, dict) else None


def _matched_values(match: object) -> Sequence[object]:
    listed: Final = as_object_list(match)
    return listed if listed is not None else (match,)


class JsonPathExtractor:
    """Extracts field values from data using JSONPath-like expressions."""

    @staticmethod
    def extract_fields(
        data: dict,
        jsonpath_expressions: list[str],
    ) -> str:
        """
        Extract field values from data using JSONPath-like expressions.

        Supports simple expressions like:
        - "query" -> data["query"]
        - "documents[*].text" -> all text fields from documents array
        - "messages[*].content" -> all content fields from messages array

        Returns concatenated string of all extracted values.
        """
        extracted_values: Final[list[str]] = []

        for expr in jsonpath_expressions:
            try:
                value = JsonPathExtractor.evaluate(data, expr)
                if value:
                    listed = as_object_list(value)
                    if listed is not None:
                        extracted_values.extend([str(v) for v in listed if v])
                    else:
                        extracted_values.append(str(value))
            except Exception as e:
                verbose_proxy_logger.debug("Failed to extract field %s: %s", expr, str(e))

        return "\n".join(extracted_values)

    @staticmethod
    def evaluate(data: dict, expr: str) -> object:
        """
        Evaluate a simple JSONPath-like expression.

        Supports:
        - Simple key: "query" -> data["query"]
        - Nested key: "foo.bar" -> data["foo"]["bar"]
        - Array wildcard: "items[*].text" -> [item["text"] for item in data["items"]]
        """
        if not expr or not data:
            return None

        parts: Final = expr.replace("[*]", ".[*]").split(".")
        current: object = _JSON_VALUE.validate_python(data)

        for i, part in enumerate(parts):
            if current is None:
                return None

            if part == "[*]":
                # Wildcard - current should be a list
                items = as_object_list(current)
                if items is None:
                    return None

                # Get remaining path
                remaining_path = ".".join(parts[i + 1 :])
                if not remaining_path:
                    return current

                # Recursively evaluate remaining path for each item
                matches = (JsonPathExtractor.evaluate(item, remaining_path) for item in items if isinstance(item, dict))
                results = list(chain.from_iterable(_matched_values(match) for match in matches if match))
                return results if results else None

            mapping = _json_object(current)
            if mapping is None:
                return None
            current = mapping.get(part)

        return current
