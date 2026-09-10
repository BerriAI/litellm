from collections.abc import Sequence
from typing import Literal

from typing_extensions import ReadOnly, TypedDict


class SonioxContextGeneralEntry(TypedDict):
    key: ReadOnly[str]
    value: ReadOnly[str]


class SonioxTranslationTerm(TypedDict):
    source: ReadOnly[str]
    target: ReadOnly[str]


class SonioxContext(TypedDict, total=False):
    """Soniox `context` request field: https://soniox.com/docs/stt/concepts/context"""

    general: ReadOnly[Sequence[SonioxContextGeneralEntry]]
    text: ReadOnly[str]
    terms: ReadOnly[Sequence[str]]
    translation_terms: ReadOnly[Sequence[SonioxTranslationTerm]]


class SonioxTranslation(TypedDict, total=False):
    """Soniox `translation` request field: https://soniox.com/docs/translation/stt-translation"""

    type: ReadOnly[Literal["one_way", "two_way"]]
    target_language: ReadOnly[str]
    language_a: ReadOnly[str]
    language_b: ReadOnly[str]
