from collections.abc import Sequence

from pydantic import ConfigDict, with_config
from typing_extensions import ReadOnly, TypedDict


class SonioxContextGeneralEntry(TypedDict):
    key: ReadOnly[str]
    value: ReadOnly[str]


class SonioxTranslationTerm(TypedDict):
    source: ReadOnly[str]
    target: ReadOnly[str]


@with_config(ConfigDict(extra="forbid"))
class SonioxContext(TypedDict, total=False):
    """Soniox `context` request field: https://soniox.com/docs/stt/concepts/context"""

    general: ReadOnly[Sequence[SonioxContextGeneralEntry]]
    text: ReadOnly[str]
    terms: ReadOnly[Sequence[str]]
    translation_terms: ReadOnly[Sequence[SonioxTranslationTerm]]
