from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final

from pydantic import JsonValue, TypeAdapter

ROOT: Final = Path(__file__).resolve().parents[1]
COST_MAP_PATH: Final = ROOT / "model_prices_and_context_window.json"
SOURCE_PATH: Final = ROOT / "provider_endpoints_support.json"
OUTPUT_PATHS: Final = (
    SOURCE_PATH,
    ROOT / "litellm/provider_endpoints_support_backup.json",
)
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])

MODE_TO_FLAG: Final[Mapping[str, str]] = MappingProxyType(
    {
        "chat": "chat_completions",
        "responses": "responses",
        "completion": "text_completion",
        "embedding": "embeddings",
        "image_generation": "image_generations",
        "image_edit": "image_edits",
        "audio_transcription": "audio_transcriptions",
        "audio_speech": "audio_speech",
        "rerank": "rerank",
        "ocr": "ocr",
        "search": "search",
        "moderation": "moderations",
        "realtime": "realtime",
        "video_generation": "video_generations",
    }
)
SUPPORTED_ENDPOINT_TO_FLAG: Final[Mapping[str, str]] = MappingProxyType(
    {
        "/v1/chat/completions": "chat_completions",
        "/v1/responses": "responses",
        "/v1/completions": "text_completion",
        "/v1/embeddings": "embeddings",
        "/v1/images/generations": "image_generations",
        "/v1/images/edits": "image_edits",
        "/v1/audio/transcriptions": "audio_transcriptions",
        "/v1/audio/speech": "audio_speech",
        "/v1/ocr": "ocr",
        "/v1/realtime": "realtime",
        "/v1/videos": "video_generations",
        "/v1/batch": "batches",
        "/v1/messages": "messages",
        "/v1beta/interactions": "interactions",
        "/v1/systemone": "systemone",
    }
)


def _json_object(value: JsonValue, name: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _provider_slug(provider: str) -> str:
    if provider.startswith("vertex_ai-"):
        return "vertex_ai"
    if provider == "fireworks_ai-embedding-models":
        return "fireworks_ai"
    if provider == "bedrock_converse":
        return "bedrock"
    if provider == "text-completion-inception":
        return "inception"
    return provider


def _model_provider_slug(model: JsonValue) -> str | None:
    if not isinstance(model, dict):
        return None
    provider: Final = model.get("litellm_provider")
    if not isinstance(provider, str) or not provider:
        return None
    return _provider_slug(provider)


def _model_flags(model: Mapping[str, JsonValue]) -> tuple[str, ...]:
    mode: Final = model.get("mode")
    mode_flag: Final = MODE_TO_FLAG.get(mode) if isinstance(mode, str) else None
    supported_endpoints: Final = model.get("supported_endpoints")
    endpoint_flags: Final = (
        tuple(
            SUPPORTED_ENDPOINT_TO_FLAG[endpoint]
            for endpoint in supported_endpoints
            if isinstance(endpoint, str) and endpoint in SUPPORTED_ENDPOINT_TO_FLAG
        )
        if isinstance(supported_endpoints, list)
        else ()
    )
    return ((mode_flag,) if mode_flag is not None else ()) + endpoint_flags


def _bridge_field(endpoint: JsonValue) -> str | None:
    if not isinstance(endpoint, dict) or endpoint.get("bridges_to_chat_completion") is not True:
        return None
    provider_json_field: Final = endpoint.get("provider_json_field")
    return provider_json_field if isinstance(provider_json_field, str) else None


def _bridged_fields(endpoints: Mapping[str, JsonValue]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            provider_json_field
            for endpoint in endpoints.values()
            if (provider_json_field := _bridge_field(endpoint)) is not None
        )
    )


def _derived_keys(bridged_fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*MODE_TO_FLAG.values(), *SUPPORTED_ENDPOINT_TO_FLAG.values(), *bridged_fields)))


def _model_evidence(
    model: JsonValue, bridged_fields: tuple[str, ...]
) -> Iterator[tuple[str, str]]:
    provider: Final = _model_provider_slug(model)
    if provider is None or not isinstance(model, dict):
        return
    flags: Final = _model_flags(model)
    yield from ((provider, flag) for flag in flags)
    if "chat_completions" not in flags:
        return
    yield from ((provider, flag) for flag in bridged_fields)


def _provider_evidence(
    cost_map: Mapping[str, JsonValue], bridged_fields: tuple[str, ...]
) -> Iterator[tuple[str, str]]:
    return chain.from_iterable(
        _model_evidence(model, bridged_fields)
        for model_name, model in cost_map.items()
        if model_name != "sample_spec"
    )


def _provider_slugs(cost_map: Mapping[str, JsonValue]) -> frozenset[str]:
    return frozenset(
        provider_slug
        for model_name, model in cost_map.items()
        if model_name != "sample_spec" and (provider_slug := _model_provider_slug(model)) is not None
    )


def _generated_provider(
    slug: str,
    value: JsonValue,
    derived_keys: tuple[str, ...],
    derived_key_set: frozenset[str],
    evidence: frozenset[tuple[str, str]],
) -> dict[str, JsonValue]:
    provider: Final = _json_object(value, f"provider {slug}")
    current_flags_value: Final = provider.get("endpoints")
    current_flags: Final = (
        _json_object(current_flags_value, f"provider {slug} endpoints")
        if current_flags_value is not None
        else {}
    )
    derived_flags: Final = {flag: (slug, flag) in evidence for flag in derived_keys}
    handwritten_flags: Final = {
        flag: value for flag, value in current_flags.items() if flag not in derived_key_set
    }
    return {
        **provider,
        "endpoints": {**derived_flags, **handwritten_flags},
    }


def generate_provider_endpoints_support(
    cost_map: Mapping[str, JsonValue],
    current_json: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    current_providers: Final = _json_object(
        current_json.get("providers"), "provider_endpoints_support providers"
    )
    endpoints: Final = _json_object(current_json.get("endpoints"), "provider_endpoints_support endpoints")
    bridged_fields: Final = _bridged_fields(endpoints)
    derived_keys: Final = _derived_keys(bridged_fields)
    derived_key_set: Final = frozenset(derived_keys)
    provider_slugs: Final = _provider_slugs(cost_map)
    missing_slugs: Final = tuple(sorted(provider_slugs - frozenset(current_providers)))
    if missing_slugs:
        raise ValueError(f"Cost map providers missing from provider_endpoints_support.json: {', '.join(missing_slugs)}")
    evidence: Final = frozenset(_provider_evidence(cost_map, bridged_fields))
    providers: Final = {
        slug: _generated_provider(slug, value, derived_keys, derived_key_set, evidence)
        for slug, value in current_providers.items()
    }
    return {**current_json, "providers": providers}


def generate_files(
    cost_map: Mapping[str, JsonValue],
    current_json: Mapping[str, JsonValue],
    output_paths: tuple[Path, ...],
    check: bool,
) -> int:
    generated: Final = generate_provider_endpoints_support(cost_map, current_json)
    content: Final = json.dumps(generated, indent=2, ensure_ascii=False) + "\n"
    stale_paths: Final = tuple(
        path for path in output_paths if not path.exists() or path.read_text(encoding="utf-8") != content
    )
    if check:
        if not stale_paths:
            return 0
        sys.stderr.write(
            "provider endpoint support is stale in "
            + ", ".join(str(path) for path in stale_paths)
            + "\nRun `uv run --no-sync python scripts/generate_provider_endpoints_support.py` to regenerate it\n"
        )
        return 1
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return 0


def main(argv: Sequence[str] | None = None, *, repo_root: Path = ROOT) -> int:
    parser: Final = argparse.ArgumentParser(description="Generate provider endpoint support from the model cost map")
    parser.add_argument("--check", action="store_true", help="fail if either generated file is stale")
    args: Final = parser.parse_args(argv)
    cost_map: Final = JSON_OBJECT.validate_json((repo_root / COST_MAP_PATH.name).read_bytes())
    current_json: Final = JSON_OBJECT.validate_json((repo_root / SOURCE_PATH.name).read_bytes())
    return generate_files(
        cost_map,
        current_json,
        (
            repo_root / SOURCE_PATH.name,
            repo_root / OUTPUT_PATHS[1].relative_to(ROOT),
        ),
        args.check,
    )


if __name__ == "__main__":
    sys.exit(main())
