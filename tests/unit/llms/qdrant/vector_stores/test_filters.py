from collections.abc import Mapping
from typing import Final

import pytest

from litellm.exceptions import BadRequestError
from litellm.llms.qdrant.vector_stores.filters import translate_filter


@pytest.mark.parametrize(
    "operator,value,expected",
    [
        ("eq", "fr", {"must": [{"key": "field", "match": {"value": "fr"}}]}),
        ("eq", True, {"must": [{"key": "field", "match": {"value": True}}]}),
        ("ne", 3, {"must_not": [{"key": "field", "match": {"value": 3}}]}),
        ("gt", 3.5, {"must": [{"key": "field", "range": {"gt": 3.5}}]}),
        ("gte", 3, {"must": [{"key": "field", "range": {"gte": 3}}]}),
        ("lt", 3, {"must": [{"key": "field", "range": {"lt": 3}}]}),
        ("lte", 3, {"must": [{"key": "field", "range": {"lte": 3}}]}),
        ("in", ["fr", "en"], {"must": [{"key": "field", "match": {"any": ["fr", "en"]}}]}),
        ("nin", [1, 2], {"must_not": [{"key": "field", "match": {"any": [1, 2]}}]}),
    ],
)
def test_comparison_filters_preserve_operator_semantics(
    operator: str,
    value: object,
    expected: Mapping[str, object],
) -> None:
    assert translate_filter({"type": operator, "key": "field", "value": value}) == expected


def test_nested_and_or_filters_preserve_boolean_grouping() -> None:
    assert translate_filter(
        {
            "type": "and",
            "filters": [
                {"type": "eq", "key": "language", "value": "fr"},
                {
                    "type": "or",
                    "filters": [
                        {"type": "gte", "key": "year", "value": 2025},
                        {"type": "ne", "key": "status", "value": "archived"},
                    ],
                },
            ],
        }
    ) == {
        "must": [
            {"must": [{"key": "language", "match": {"value": "fr"}}]},
            {
                "should": [
                    {"must": [{"key": "year", "range": {"gte": 2025}}]},
                    {"must_not": [{"key": "status", "match": {"value": "archived"}}]},
                ]
            },
        ]
    }


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"type": "contains", "key": "text", "value": "x"},
        {"type": "and", "filters": []},
        {"type": "eq", "key": "", "value": "x"},
        {"type": "eq", "key": "price", "value": 1.5},
        {"type": "gt", "key": "price", "value": True},
        {"type": "gt", "key": "price", "value": float("inf")},
        {"type": "in", "key": "language", "value": []},
        {"type": "eq", "key": "language", "value": "fr", "unexpected": "x"},
    ],
)
def test_invalid_filters_are_rejected_instead_of_dropped(value: Mapping[str, object]) -> None:
    with pytest.raises(BadRequestError):
        translate_filter(value)


def _nested_filter(depth: int) -> Mapping[str, object]:
    return {"type": "and", "filters": [_nested_filter(depth - 1)]} if depth else {"type": "eq", "key": "x", "value": 1}


def test_excessive_nesting_fails_with_a_clear_error() -> None:
    value: Final = _nested_filter(21)
    with pytest.raises(BadRequestError, match="at most 20 levels"):
        translate_filter(value)
