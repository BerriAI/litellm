"""Per-convention span normalizers, tried in order: the first whose `matches()` is true wins."""

from collections.abc import Mapping, Sequence
from typing import Final

from litellm.tracing.normalizers.base import SpanNormalizer
from litellm.tracing.normalizers.genai import GenAISemconvNormalizer
from litellm.tracing.normalizers.langsmith import LangSmithNormalizer
from litellm.tracing.normalizers.openinference import OpenInferenceNormalizer

NORMALIZERS: Final[tuple[SpanNormalizer, ...]] = (
    LangSmithNormalizer(),
    OpenInferenceNormalizer(),
    GenAISemconvNormalizer(),
)
_FALLBACK: Final[SpanNormalizer] = GenAISemconvNormalizer()


def select_normalizer(
    scope_name: str, attributes: Mapping[str, str], registry: Sequence[SpanNormalizer] = NORMALIZERS
) -> SpanNormalizer:
    return next((n for n in registry if n.matches(scope_name, attributes)), _FALLBACK)


__all__ = (
    "NORMALIZERS",
    "GenAISemconvNormalizer",
    "LangSmithNormalizer",
    "OpenInferenceNormalizer",
    "SpanNormalizer",
    "select_normalizer",
)
