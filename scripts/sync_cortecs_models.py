"""Sync the cortecs entries of model_prices_and_context_window.json with Cortecs's live catalog.

Pulls ``GET https://api.cortecs.ai/v1/models?currency=USD`` (public, no credentials), maps the catalog's
pricing and modality/feature metadata onto registry fields, and diffs the result against the registry. Dry
run (the default) prints the diff summary and the generated PR body; ``--write`` splices the result into the
root cost map and its ``litellm/`` backup copy, leaving every entry outside the cortecs block byte-identical.

Policy highlights:
- Prices are requested in USD because the registry is USD-only (`?currency` is Cortecs's documented pricing
  conversion, see https://docs.cortecs.ai/api/models), and token prices are per million tokens; the sync
  normalizes them to clean per-token values.
- Modality and feature flags come from ``input_modalities`` / ``supported_features``, and a flag is only
  written when the catalog advertises it. Terms with no registry field are surfaced as warnings instead of
  being guessed at.
- The catalog hides every non-default tag, so the sync requests each of ``CATALOG_TAGS`` and merges the results.
- The entry type follows the model's output: text is chat, embeddings is embedding, speech is audio_speech,
  transcription is audio_transcription, an OCR price makes it ocr and decision is evaluation (the mode the
  Decisions routes price from). A model with any other output, or without the price its type bills by, is
  reported and skipped instead of being registered with a wrong or zero price.
- A cortecs entry the catalog no longer lists is reported, never deleted: delisting is a human call.
- Fields the sync does not own (anything a human added by hand) are preserved across updates.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from typing_extensions import assert_never

MODELS_URL: Final = "https://api.cortecs.ai/v1/models?currency=USD"
SOURCE_URL: Final = "https://api.cortecs.ai/v1/models"
PROVIDER: Final = "cortecs"
PREFIX: Final = "cortecs/"
CATALOG_TAGS: Final = ("Instruct", "Embedding", "Speech", "Transcription", "OCR", "Decision")
COST_MAP_RELPATHS: Final = (
    "model_prices_and_context_window.json",
    "litellm/model_prices_and_context_window_backup.json",
)
ENTRY_INDENT: Final = " " * 4

# Catalog terms that the registry has a field for. Everything else the catalog advertises is reported so a
# human can decide whether it maps to a field this script does not know about yet.
_ALL_FLAGS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "audio": ("supports_audio_input",),
        "file": ("supports_pdf_input",),
        "image": ("supports_vision",),
        "reasoning": ("supports_reasoning",),
        "tools": ("supports_function_calling", "supports_tool_choice"),
        "video": ("supports_video_input",),
    }
)
# Catalog terms with no registry field. `text` is the baseline modality every entry already describes, and
# `json_mode` is response_format={"type": "json_object"}, which the registry has no field for
# (`supports_response_schema` is structured outputs, a different contract).
_TERMS_WITHOUT_A_FIELD: Final[frozenset[str]] = frozenset({"json_mode", "text"})

_OWNED_FIELDS: Final[frozenset[str]] = frozenset(
    (
        "cache_creation_input_token_cost",
        "annotation_cost_per_page",
        "cache_read_input_token_cost",
        "input_cost_per_audio_per_second",
        "input_cost_per_character",
        "input_cost_per_second",
        "input_cost_per_token",
        "litellm_provider",
        "max_input_tokens",
        "max_output_tokens",
        "max_tokens",
        "mode",
        "ocr_cost_per_page",
        "output_cost_per_token",
        "source",
        "supports_prompt_caching",
        *chain.from_iterable(_ALL_FLAGS.values()),
    )
)


class SyncError(RuntimeError):
    pass


class CatalogPricing(BaseModel):
    model_config = ConfigDict(frozen=True)

    currency: str
    input_token: float
    output_token: float
    cache_read_cost: float | None = None
    cache_write_cost: float | None = None
    audio_cost: float | None = None
    speech_cost: float | None = None
    ocr_cost: float | None = None
    ocr_annotated_cost: float | None = None
    output_cost_per_image: float | None = None
    output_cost_per_additional_megapixel: float | None = None


class CatalogModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    context_size: int
    max_output_tokens: int | None = None
    input_modalities: tuple[str, ...]
    output_modalities: tuple[str, ...]
    supported_features: tuple[str, ...]
    pricing: CatalogPricing


CATALOG_ADAPTER: Final[TypeAdapter[list[CatalogModel]]] = TypeAdapter(list[CatalogModel])


@dataclass(frozen=True, slots=True)
class SyncOutcome:
    cost_map: Mapping[str, object]
    added: tuple[str, ...] = ()
    updated: tuple[str, ...] = ()
    stale: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.updated)


@dataclass(frozen=True, slots=True)
class _Splice:
    start: int
    end: int
    text: str


Kind = Literal["chat", "embedding", "speech", "transcription", "ocr", "decision"]

_CONSUMED_PRICING: Final[Mapping[Kind, frozenset[str]]] = MappingProxyType(
    {
        "chat": frozenset({"input_token", "output_token", "cache_read_cost", "cache_write_cost", "audio_cost"}),
        "embedding": frozenset({"input_token", "output_token"}),
        "speech": frozenset({"speech_cost"}),
        "transcription": frozenset({"audio_cost"}),
        "ocr": frozenset({"ocr_cost", "ocr_annotated_cost"}),
        "decision": frozenset({"input_token", "output_token"}),
    }
)


def per_token(usd_per_million: float) -> float:
    return float(Decimal(f"{usd_per_million:.12g}") / 1_000_000)


def per_page(usd_per_thousand: float) -> float:
    return float(Decimal(f"{usd_per_thousand:.12g}") / 1_000)


def kind_of(model: CatalogModel) -> Kind | None:
    if model.pricing.ocr_cost is not None:
        return "ocr"
    match model.output_modalities:
        case ("text",):
            return "chat"
        case ("embeddings",):
            return "embedding"
        case ("audio", "speech"):
            return "speech"
        case ("text", "transcription"):
            return "transcription"
        case ("decision",):
            return "decision"
        case _:
            return None


def _flags(model: CatalogModel) -> Mapping[str, bool]:
    advertised: Final = frozenset(model.input_modalities) | frozenset(model.supported_features)
    matching: Final = (flags for term, flags in _ALL_FLAGS.items() if term in advertised)
    return MappingProxyType({flag: True for flag in chain.from_iterable(matching)})


def _priced_fields(pricing: CatalogPricing) -> Mapping[str, object]:
    cached_read: Final = pricing.cache_read_cost
    cached_write: Final = pricing.cache_write_cost
    audio: Final = pricing.audio_cost
    return MappingProxyType(
        {
            **({"supports_prompt_caching": True} if cached_read is not None else {}),
            **({"cache_read_input_token_cost": per_token(cached_read)} if cached_read is not None else {}),
            **({"cache_creation_input_token_cost": per_token(cached_write)} if cached_write is not None else {}),
            **({"input_cost_per_audio_per_second": float(f"{audio:.6g}")} if audio is not None else {}),
        }
    )


def _base_fields(mode: str) -> Mapping[str, object]:
    return {"litellm_provider": PROVIDER, "mode": mode, "source": SOURCE_URL}


def _token_fields(model: CatalogModel) -> Mapping[str, object]:
    return {
        "input_cost_per_token": per_token(model.pricing.input_token),
        "max_input_tokens": model.context_size,
        "max_tokens": model.context_size,
        "output_cost_per_token": per_token(model.pricing.output_token),
    }


def _chat_fields(model: CatalogModel) -> Mapping[str, object] | None:
    if model.max_output_tokens is None:
        return None
    return {
        **_base_fields("chat"),
        "input_cost_per_token": per_token(model.pricing.input_token),
        "max_input_tokens": model.context_size,
        "max_output_tokens": model.max_output_tokens,
        "max_tokens": model.max_output_tokens,
        "output_cost_per_token": per_token(model.pricing.output_token),
        **_flags(model),
        **_priced_fields(model.pricing),
    }


def _speech_fields(model: CatalogModel) -> Mapping[str, object] | None:
    price: Final = model.pricing.speech_cost
    if price is None:
        return None
    return {**_base_fields("audio_speech"), "input_cost_per_character": per_token(price)}


def _transcription_fields(model: CatalogModel) -> Mapping[str, object] | None:
    price: Final = model.pricing.audio_cost
    if price is None:
        return None
    return {**_base_fields("audio_transcription"), "input_cost_per_second": float(f"{price:.6g}")}


def _ocr_fields(model: CatalogModel) -> Mapping[str, object] | None:
    price: Final = model.pricing.ocr_cost
    annotated: Final = model.pricing.ocr_annotated_cost
    if price is None:
        return None
    return {
        **_base_fields("ocr"),
        "ocr_cost_per_page": per_page(price),
        **({"annotation_cost_per_page": per_page(annotated)} if annotated is not None else {}),
    }


def _fields_for(model: CatalogModel, kind: Kind) -> Mapping[str, object] | None:
    match kind:
        case "chat":
            return _chat_fields(model)
        case "embedding":
            return {**_base_fields("embedding"), **_token_fields(model)}
        case "decision":
            return {**_base_fields("evaluation"), **_token_fields(model)}
        case "speech":
            return _speech_fields(model)
        case "transcription":
            return _transcription_fields(model)
        case "ocr":
            return _ocr_fields(model)
        case _:
            assert_never(kind)


def entry_for(model: CatalogModel) -> Mapping[str, object] | None:
    kind: Final = kind_of(model)
    if kind is None:
        return None
    fields: Final = _fields_for(model, kind)
    return None if fields is None else dict(sorted(fields.items()))


def _merged(existing: Mapping[str, object], desired: Mapping[str, object]) -> Mapping[str, object]:
    kept: Final = {field: value for field, value in existing.items() if field not in _OWNED_FIELDS}
    return dict(sorted({**kept, **desired}.items()))


def _unmapped_terms(catalog: Sequence[CatalogModel]) -> tuple[str, ...]:
    advertised: Final = chain.from_iterable(
        chain.from_iterable((model.input_modalities, model.supported_features))
        for model in catalog
        if kind_of(model) == "chat"
    )
    return tuple(sorted({term for term in advertised if term not in _ALL_FLAGS and term not in _TERMS_WITHOUT_A_FIELD}))


def _optional_prices(pricing: CatalogPricing) -> Mapping[str, float | None]:
    return {
        "audio_cost": pricing.audio_cost,
        "cache_read_cost": pricing.cache_read_cost,
        "cache_write_cost": pricing.cache_write_cost,
        "ocr_annotated_cost": pricing.ocr_annotated_cost,
        "ocr_cost": pricing.ocr_cost,
        "output_cost_per_additional_megapixel": pricing.output_cost_per_additional_megapixel,
        "output_cost_per_image": pricing.output_cost_per_image,
        "speech_cost": pricing.speech_cost,
    }


def _unmapped_pricing_fields(model: CatalogModel) -> tuple[str, ...]:
    """Catalog prices that the registry entry for this model's type does not carry."""
    kind: Final = kind_of(model)
    if kind is None:
        return ()
    consumed: Final = _CONSUMED_PRICING[kind]
    return tuple(
        field for field, price in _optional_prices(model.pricing).items() if price is not None and field not in consumed
    )


def _unmapped_pricing(catalog: Sequence[CatalogModel]) -> tuple[str, ...]:
    return tuple(sorted(set(chain.from_iterable(_unmapped_pricing_fields(model) for model in catalog))))


def _warnings(catalog: Sequence[CatalogModel], stale: tuple[str, ...]) -> tuple[str, ...]:
    skipped: Final = tuple(model.id for model in catalog if entry_for(model) is None)
    unmapped: Final = _unmapped_terms(catalog)
    unmapped_pricing: Final = _unmapped_pricing(catalog)
    return (
        *(
            f"`{PREFIX}{model_id}` no longer appears in the Cortecs catalog; left in the registry for a human "
            "deprecation call"
            for model_id in stale
        ),
        *(
            (
                f"the catalog advertises {list(unmapped)} with no registry field in this sync; add a mapping if a "
                "field exists for it",
            )
            if unmapped
            else ()
        ),
        *(
            (f"the catalog now prices {list(unmapped_pricing)}; this sync does not write those registry fields",)
            if unmapped_pricing
            else ()
        ),
        *(
            (
                f"no registry entry is written for {list(skipped)}; each has an output type this sync does not "
                "register or lacks the price its type is billed by",
            )
            if skipped
            else ()
        ),
    )


def _desired_entries(catalog: Sequence[CatalogModel]) -> Mapping[str, Mapping[str, object]]:
    entries: Final = ((model, entry_for(model)) for model in catalog)
    return MappingProxyType({f"{PREFIX}{model.id}": entry for model, entry in entries if entry is not None})


def compute_sync(cost_map: Mapping[str, object], catalog: Sequence[CatalogModel]) -> SyncOutcome:
    desired: Final = _desired_entries(catalog)
    existing: Final[Mapping[str, Mapping[str, object]]] = MappingProxyType(
        {key: entry for key, entry in cost_map.items() if key.startswith(PREFIX) and isinstance(entry, Mapping)}
    )
    added: Final = tuple(sorted(key for key in desired if key not in existing))
    updated: Final = tuple(
        sorted(
            key for key, entry in desired.items() if key in existing and _merged(existing[key], entry) != existing[key]
        )
    )
    stale: Final = tuple(sorted(key for key in existing if key not in desired))
    synced: Final[Mapping[str, object]] = MappingProxyType(
        {
            **cost_map,
            **{key: _merged(existing[key], desired[key]) for key in existing if key in desired},
            **{key: desired[key] for key in added},
        }
    )
    return SyncOutcome(
        cost_map=synced,
        added=added,
        updated=updated,
        stale=stale,
        warnings=_warnings(catalog, stale),
    )


def _render_entry(key: str, entry: object) -> str:
    rendered: Final = json.dumps({key: entry}, indent=4)
    return "\n".join(rendered.splitlines()[1:-1])


def _entry_span(text: str, key: str) -> tuple[int, int]:
    head: Final = f"\n{ENTRY_INDENT}{json.dumps(key)}: "
    start: Final = text.find(head)
    if start < 0:
        raise SyncError(f"cannot find `{key}` in the cost map; refusing to rewrite a file in an unexpected shape")
    value_start: Final = start + len(head)
    return start, json.JSONDecoder().raw_decode(text, value_start)[1]


def _cut(text: str, key: str, entry: object) -> _Splice:
    start, end = _entry_span(text, key)
    return _Splice(start=start, end=end, text=f"\n{_render_entry(key, entry)}")


def _splice(text: str, cuts: Sequence[_Splice], offset: int = 0) -> str:
    if not cuts:
        return text[offset:]
    first: Final = cuts[0]
    return text[offset : first.start] + first.text + _splice(text, cuts[1:], first.end)


def _replace_entries(text: str, entries: Mapping[str, object]) -> str:
    cuts: Final = tuple(sorted((_cut(text, key, entries[key]) for key in entries), key=lambda cut: cut.start))
    return _splice(text, cuts)


def _append_entries(text: str, entries: Mapping[str, object]) -> str:
    if not entries:
        return text
    body: Final = text.rstrip()
    if not body.endswith("}"):
        raise SyncError("the cost map does not end with a closing brace; refusing to write it")
    block: Final = ",\n".join(_render_entry(key, entries[key]) for key in sorted(entries))
    last_entry: Final = body[:-1].removesuffix("\n")
    return f"{last_entry},\n{block}\n}}\n"


def render_cost_map_text(text: str, outcome: SyncOutcome) -> str:
    replaced: Final = _replace_entries(text, {key: outcome.cost_map[key] for key in outcome.updated})
    return _append_entries(replaced, {key: outcome.cost_map[key] for key in outcome.added})


def _section(title: str, lines: Sequence[str]) -> str:
    bullets: Final = "\n".join(f"- `{line}`" for line in lines) or "- none"
    return f"### {title} ({len(lines)})\n{bullets}\n"


def render_pr_body(outcome: SyncOutcome) -> str:
    return (
        "Automated daily sync of the cortecs entries in model_prices_and_context_window.json against "
        f"`GET {MODELS_URL}` for the tags {', '.join(CATALOG_TAGS)} by scripts/sync_cortecs_models.py.\n"
        "\n"
        f"{_section('Added', outcome.added)}"
        "\n"
        f"{_section('Updated', outcome.updated)}"
        "\n"
        f"{_section('Warnings needing a human call', outcome.warnings)}"
        "\n"
    )


def render_summary(outcome: SyncOutcome) -> str:
    return f"added={len(outcome.added)} updated={len(outcome.updated)} warnings={len(outcome.warnings)}"


def load_catalog(raw: bytes) -> list[CatalogModel]:
    parsed: Final = json.loads(raw)
    entries: Final = parsed.get("data") if isinstance(parsed, dict) else parsed
    try:
        catalog: Final = CATALOG_ADAPTER.validate_python(entries)
    except ValidationError as error:
        raise SyncError(f"the catalog response no longer matches the expected shape: {error}") from error
    if not catalog:
        raise SyncError("the catalog response contains no models; refusing to sync an empty catalog")
    currencies: Final = sorted({model.pricing.currency for model in catalog})
    if currencies != ["USD"]:
        raise SyncError(
            f"the catalog prices in {currencies} despite the currency=USD request; the registry is USD-only, so "
            "the sync needs a conversion before it can write prices"
        )
    return catalog


def catalog_urls() -> tuple[str, ...]:
    return tuple(f"{MODELS_URL}&tag={tag}" for tag in CATALOG_TAGS)


def load_catalogs(fetch: Callable[[str], bytes]) -> list[CatalogModel]:
    models: Final = chain.from_iterable(load_catalog(fetch(url)) for url in catalog_urls())
    return list({model.id: model for model in models}.values())


def _fetch(url: str) -> bytes:
    response: Final = httpx.get(url, timeout=30, follow_redirects=True)
    if response.status_code != 200:
        raise SyncError(f"GET {url} returned {response.status_code}")
    return response.content


def main(argv: Sequence[str]) -> int:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="apply the sync to the cost map files (default: dry run)")
    parser.add_argument("--models-json", type=Path, help="recorded catalog response to use instead of the live API")
    parser.add_argument("--pr-body-file", type=Path, help="write the generated PR body to this path")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args: Final = parser.parse_args(argv)

    catalog: Final = (
        load_catalog(args.models_json.read_bytes()) if args.models_json is not None else load_catalogs(_fetch)
    )
    cost_map_path: Final = args.repo_root / COST_MAP_RELPATHS[0]
    original: Final = cost_map_path.read_text()
    outcome: Final = compute_sync(json.loads(original), catalog)
    rendered: Final = render_cost_map_text(original, outcome)
    if json.loads(rendered) != outcome.cost_map:
        raise SyncError("the spliced cost map does not parse back to the synced map; refusing to write it")
    body: Final = render_pr_body(outcome)

    if args.pr_body_file is not None:
        args.pr_body_file.write_text(body)
    if args.write and outcome.has_changes:
        for relpath in COST_MAP_RELPATHS:
            (args.repo_root / relpath).write_text(rendered)
    print(render_summary(outcome))
    print()
    print(body)
    if not args.write:
        print("dry run: no files were touched")
    elif not outcome.has_changes:
        print("registry already in sync: no files were touched")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except SyncError as error:
        print(f"SYNC FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error
