"""The typed boundary to the optional `privaite` package.

`privaite` is not a LiteLLM dependency, so it is imported lazily and nothing it returns
has a type here. Each object it hands back is checked once against the protocol below
that names what this guardrail calls on it.
"""

import asyncio
import importlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import MappingProxyType, ModuleType
from typing import Final, Protocol, TypeVar, runtime_checkable

INSTALL_HINT: Final = (
    "The PrivAiTe guardrail requires the 'privaite' package. Install it with: pip install 'privaite>=0.7.2'"
)

SPACY_MODELS: Final = MappingProxyType(
    {
        "en": "en_core_web_lg",
        "fr": "fr_core_news_md",
        "de": "de_core_news_md",
        "es": "es_core_news_md",
        "it": "it_core_news_md",
        "pt": "pt_core_news_md",
        "nl": "nl_core_news_md",
    }
)

_Narrowed = TypeVar("_Narrowed")  # rebind-ok: a TypeVar cannot be declared Final


@runtime_checkable
class RestoreMapping(Protocol):
    @property
    def is_empty(self) -> bool: ...

    def add(self, original: str, fake: str, entity_type: str) -> None: ...

    def get_all_fakes(self) -> Mapping[str, str]: ...


@runtime_checkable
class PrivaiteEngine(Protocol):
    async def initialize(self) -> None: ...

    async def process_request(self, messages: Sequence[object]) -> tuple[Sequence[object], RestoreMapping]: ...

    async def process_response(self, text: str, mapping: RestoreMapping) -> str: ...


@runtime_checkable
class RestoreBuffer(Protocol):
    def feed(self, text: str) -> str: ...

    def flush(self) -> str: ...


def _module(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(INSTALL_HINT) from exc


def _factory(module: ModuleType, name: str) -> Callable[..., object]:
    value: Final = getattr(module, name, None)
    if callable(value):
        return value
    if hasattr(module, name):
        raise TypeError(f"{module.__name__}.{name} is not callable. {INSTALL_HINT}")
    raise RuntimeError(f"{module.__name__}.{name} is missing. {INSTALL_HINT}")


def _narrow(value: object, kind: type[_Narrowed]) -> _Narrowed:
    if not isinstance(value, kind):
        raise TypeError(f"privaite returned {type(value).__name__} where {kind.__name__} was expected")
    return value


def _declares_field(model: object, name: str) -> bool:
    fields: Final = getattr(model, "model_fields", None)
    return isinstance(fields, Mapping) and name in fields


def _is_missing_spacy_model(exc: OSError) -> bool:
    """spaCy's E050. Permission, disk and network errors must not trigger a download."""
    message: Final = str(exc)
    return "E050" in message or "Can't find model" in message


async def _download_spacy_models(languages: Sequence[str]) -> None:
    """Pulls hundreds of MB through pip's synchronous machinery, so it runs off the event loop."""
    download: Final = _factory(importlib.import_module("spacy.cli"), "download")
    for language in languages:
        model = SPACY_MODELS.get(language)
        if model:
            await asyncio.to_thread(download, model)


async def build_engine(
    preset: str, languages: Sequence[str], deanonymize: bool, block_entities: Sequence[str]
) -> PrivaiteEngine:
    schema: Final = _module("privaite.config.schema")
    make_engine: Final = _factory(_module("privaite.pii.engine"), "PIIEngine")
    pii_config: Final = _factory(schema, "PIIConfig")
    if block_entities and not _declares_field(pii_config, "block_entities"):
        # PIIConfig allows extra fields, so an older privaite would accept block_entities,
        # ignore it and forward the PII. Refusing is the only safe answer.
        raise RuntimeError(
            "block_entities is set but the installed privaite does not support it; "
            "upgrade privaite to a version that enforces pii.block_entities."
        )
    presidio: Final = _factory(schema, "PresidioDetectorConfig")(enabled=True, languages=list(languages))
    config: Final = pii_config(
        enabled=True,
        preset=preset,
        detectors=_factory(schema, "DetectorsConfig")(presidio=presidio),
        anonymization=_factory(schema, "AnonymizationConfig")(method="placeholder"),
        deanonymization=_factory(schema, "DeanonymizationConfig")(enabled=deanonymize),
        block_entities=list(block_entities),
    )
    engine: Final = _narrow(make_engine(config), PrivaiteEngine)
    try:
        await engine.initialize()
    except OSError as exc:
        if not _is_missing_spacy_model(exc):
            raise
        await _download_spacy_models(languages)
        retried: Final = _narrow(make_engine(config), PrivaiteEngine)
        await retried.initialize()
        return retried
    return engine


def blocked_error() -> type[Exception]:
    """The exception privaite raises when a request carries a type listed in block_entities."""
    error: Final = getattr(_module("privaite.pii.engine"), "PIIBlockedError", None)
    if not isinstance(error, type) or not issubclass(error, Exception):
        raise TypeError(f"privaite.pii.engine.PIIBlockedError is missing. {INSTALL_HINT}")
    return error


def restore_mapping(pairs: Iterable[tuple[str, str]]) -> RestoreMapping:
    """A privaite mapping holding each (fake, original) pair."""
    mapping: Final = _narrow(_factory(_module("privaite.pii.mapping"), "PIIMapping")(), RestoreMapping)
    for fake, original in pairs:
        mapping.add(original, fake, "PII")
    return mapping


def restore_buffer_factory() -> Callable[[RestoreMapping], RestoreBuffer]:
    """privaite's streaming de-anonymizer, which holds back a placeholder split across chunks."""
    make_buffer: Final = _factory(_module("privaite.streaming.buffer"), "StreamingDeAnonymizer")

    def _buffer(mapping: RestoreMapping) -> RestoreBuffer:
        return _narrow(make_buffer(mapping), RestoreBuffer)

    return _buffer
