"""
JSONPath Extractor Module

Extracts field values from data using simple JSONPath-like expressions.
"""

from typing import Any, Final

from litellm._logging import verbose_proxy_logger


def _normalize_wildcard_segments(expr: str) -> tuple[str, ...]:
    """Normalize ``key[*]`` into ``key`` + ``[*]`` segments.

    Idempotent: the recursion in :meth:`JsonPathExtractor.evaluate` passes
    already-normalized paths (e.g. ``content.[*].text``) back through here.
    Re-running ``expr.replace("[*]", ".[*]")`` on those injects empty
    segments (``content..[*].text``); dropping empty segments afterwards
    keeps the normalization stable for raw (``a[*].b``), adjacent
    (``a[*][*].b``) and already-normalized inputs alike, so no
    nested-wildcard expression is silently dropped and skips the
    guardrail (#46302).
    """
    segments: Final = tuple(
        segment for segment in expr.replace("[*]", ".[*]").split(".") if segment
    )
    return segments


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
    def evaluate(data: dict, expr: str) -> str | list[str] | None:
        """
        Evaluate a simple JSONPath-like expression.

        Supports:
        - Simple key: "query" -> data["query"]
        - Nested key: "foo.bar" -> data["foo"]["bar"]
        - Array wildcard: "items[*].text" -> [item["text"] for item in data["items"]]
        """
        if not expr or not data:
            return None

        parts: Final = _normalize_wildcard_segments(expr)
        current: Any = data

        for i, part in enumerate(parts):
            if current is None:
                return None

            if part == "[*]":
                # Wildcard - current should be a list
                if not isinstance(current, list):
                    return None

                # Get remaining path
                remaining_path = ".".join(parts[i + 1 :])
                if not remaining_path:
                    return current

                # Recursively evaluate remaining path for each item
                results = []
                for item in current:
                    if isinstance(item, dict):
                        result = JsonPathExtractor.evaluate(item, remaining_path)
                        if result:
                            if isinstance(result, list):
                                results.extend(result)
                            else:
                                results.append(result)
                return results if results else None

            elif isinstance(current, dict):
                current = current.get(part)
            else:
                return None

        return current
