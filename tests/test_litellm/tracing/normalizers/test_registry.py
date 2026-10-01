from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from litellm.tracing.normalizers import (
    NORMALIZERS,
    GenAISemconvNormalizer,
    LangSmithNormalizer,
    OpenInferenceNormalizer,
    select_normalizer,
)
from litellm.tracing.types import SpanRow

_NO_ATTRIBUTES: Final[Mapping[str, str]] = MappingProxyType({})


def test_langsmith_scope_selects_langsmith_without_any_attributes():
    assert isinstance(select_normalizer("langsmith", _NO_ATTRIBUTES), LangSmithNormalizer)


def test_langsmith_kind_attribute_selects_langsmith_under_any_scope():
    assert isinstance(select_normalizer("other", MappingProxyType({"langsmith.span.kind": "llm"})), LangSmithNormalizer)


def test_langsmith_wins_over_openinference_when_both_markers_present():
    attributes: Final = MappingProxyType({"langsmith.span.kind": "llm", "openinference.span.kind": "LLM"})
    assert isinstance(select_normalizer("other", attributes), LangSmithNormalizer)


def test_openinference_kind_attribute_selects_openinference():
    assert isinstance(
        select_normalizer("other", MappingProxyType({"openinference.span.kind": "LLM"})), OpenInferenceNormalizer
    )


def test_unmarked_span_falls_back_to_genai():
    assert isinstance(
        select_normalizer("other", MappingProxyType({"gen_ai.operation.name": "chat"})), GenAISemconvNormalizer
    )


def test_empty_registry_falls_back_to_genai():
    assert isinstance(select_normalizer("langsmith", _NO_ATTRIBUTES, registry=()), GenAISemconvNormalizer)


def test_registry_names_are_unique():
    names: Final = tuple(n.name for n in NORMALIZERS)
    assert len(names) == len(frozenset(names))


@dataclass(frozen=True, slots=True)
class _CustomNormalizer:
    name: str = "custom"

    def matches(self, scope_name: str, attributes: Mapping[str, str]) -> bool:
        return scope_name == "custom-sdk"

    def normalize(self, row: SpanRow, attributes: Mapping[str, str]) -> None:
        return None


def test_normalizer_inserted_ahead_in_custom_registry_wins_only_where_it_matches():
    registry: Final = (_CustomNormalizer(), *NORMALIZERS)
    assert isinstance(
        select_normalizer("custom-sdk", MappingProxyType({"langsmith.span.kind": "llm"}), registry), _CustomNormalizer
    )
    assert isinstance(select_normalizer("langsmith", _NO_ATTRIBUTES, registry), LangSmithNormalizer)
