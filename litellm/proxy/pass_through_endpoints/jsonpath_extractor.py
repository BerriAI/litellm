"""
JSONPath Extractor Module

Extracts field values from data using simple JSONPath-like expressions.
"""

from itertools import chain
from typing import Final

from pydantic import JsonValue

from litellm._logging import verbose_proxy_logger


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
                    if isinstance(value, list):
                        extracted_values.extend([str(v) for v in value if v])
                    else:
                        extracted_values.append(str(value))
            except Exception as e:
                verbose_proxy_logger.debug("Failed to extract field %s: %s", expr, str(e))

        return "\n".join(extracted_values)

    @staticmethod
    def evaluate(data: dict, expr: str) -> JsonValue:
        """
        Evaluate a simple JSONPath-like expression.

        Supports:
        - Simple key: "query" -> data["query"]
        - Nested key: "foo.bar" -> data["foo"]["bar"]
        - Array wildcard: "items[*].text" -> [item["text"] for item in data["items"]]
        - Nested array wildcards: "messages[*].content[*].text" -> flattened list of every block's text
        """
        body: Final[JsonValue] = data
        return JsonPathExtractor._walk(body, tuple(expr.replace("[*]", ".[*]").split(".")))

    @staticmethod
    def _walk(current: JsonValue, parts: tuple[str, ...]) -> JsonValue:
        if not parts:
            return current

        head, rest = parts[0], parts[1:]
        if head != "[*]":
            return JsonPathExtractor._walk(current.get(head), rest) if isinstance(current, dict) else None
        if not isinstance(current, list):
            return None
        if not rest:
            return current

        results: Final = (JsonPathExtractor._walk(item, rest) for item in current)
        return list(chain.from_iterable(r if isinstance(r, list) else (r,) for r in results if r)) or None
