# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

from litellm.python_extension.generated.v1 import extension_host_pb2 as pb


@dataclass(frozen=True, slots=True)
class ExtensionManifest:
    revision_id: str
    specs: tuple[pb.ExtensionSpec, ...]
    callback_ids: Mapping[str, str]
    guardrail_ids: Mapping[tuple[str, str], str]


def build_manifest(config: Mapping[str, object]) -> ExtensionManifest:
    specs: list[  # mutable-ok: LiteLLM compatibility payload # rebind-ok: invocation-scoped RPC state
        pb.ExtensionSpec
    ] = []  # mutable-ok: LiteLLM compatibility payload # rebind-ok: invocation-scoped RPC state
    callback_ids: dict[  # mutable-ok: LiteLLM compatibility payload # rebind-ok: invocation-scoped RPC state
        str, str
    ] = {}  # mutable-ok: LiteLLM compatibility payload # rebind-ok: invocation-scoped RPC state
    guardrail_ids: dict[  # mutable-ok: LiteLLM compatibility payload # rebind-ok: invocation-scoped RPC state
        tuple[str, str], str
    ] = {}  # mutable-ok: LiteLLM compatibility payload # rebind-ok: invocation-scoped RPC state
    callback_events: dict[  # mutable-ok: LiteLLM compatibility payload # rebind-ok: invocation-scoped RPC state
        str, set[str]
    ] = {}  # mutable-ok: LiteLLM compatibility payload # rebind-ok: invocation-scoped RPC state
    litellm_settings: Final = config.get("litellm_settings")
    settings: Final = (
        litellm_settings if isinstance(litellm_settings, Mapping) else {}  # mutable-ok: LiteLLM compatibility payload
    )  # mutable-ok: LiteLLM compatibility payload
    for setting, event in (
        ("callbacks", None),
        ("success_callback", "success"),
        ("failure_callback", "failure"),
    ):
        for entrypoint in _string_entries(settings.get(setting)):
            if _is_customer_entrypoint(entrypoint):
                events = callback_events.setdefault(
                    entrypoint,
                    set(),  # mutable-ok: LiteLLM compatibility payload
                )
                events.update(("success", "failure") if event is None else (event,))
    for entrypoint, events in sorted(callback_events.items()):
        extension_id = _stable_id("callback", entrypoint)
        callback_ids[entrypoint] = extension_id
        specs.append(
            pb.ExtensionSpec(
                id=extension_id,
                kind=pb.EXTENSION_KIND_CALLBACK,
                entrypoint=entrypoint,
                constructor_json=_canonical_json(
                    {"callback_events": sorted(events)}  # mutable-ok: LiteLLM compatibility payload
                ),  # mutable-ok: LiteLLM compatibility payload
            )
        )
    guardrail_configs: Final = _guardrail_configs(config, settings)
    for guardrail in guardrail_configs:
        name = guardrail.get("guardrail_name")
        params = guardrail.get("litellm_params")
        if not isinstance(name, str) or not isinstance(params, Mapping):
            continue
        entrypoint = params.get("guardrail")
        if not isinstance(entrypoint, str) or not _is_customer_entrypoint(entrypoint):
            continue
        extension_id = _stable_id("guardrail", entrypoint, name)
        guardrail_ids[(entrypoint, name)] = extension_id
        kwargs = dict(params)  # mutable-ok: LiteLLM compatibility payload
        kwargs.pop("guardrail", None)
        mode = kwargs.pop("mode", None)
        default_on = kwargs.pop("default_on", False)
        kwargs.update(guardrail_name=name, event_hook=mode, default_on=default_on)
        specs.append(
            pb.ExtensionSpec(
                id=extension_id,
                kind=pb.EXTENSION_KIND_GUARDRAIL,
                entrypoint=entrypoint,
                constructor_json=_canonical_json({"kwargs": kwargs}),  # mutable-ok: LiteLLM compatibility payload
            )
        )
    canonical_specs: Final = b"".join(spec.SerializeToString(deterministic=True) for spec in specs)
    revision_id: Final = hashlib.sha256(canonical_specs).hexdigest()[:24]
    return ExtensionManifest(revision_id, tuple(specs), callback_ids, guardrail_ids)


def manifest_json_from_config_path(config_path: str) -> str:
    import yaml

    with open(config_path, encoding="utf-8") as config_file:
        raw: Final = yaml.safe_load(config_file) or {}  # mutable-ok: LiteLLM compatibility payload
    if not isinstance(raw, dict):
        raise TypeError("LiteLLM config must contain an object")
    manifest: Final = build_manifest(raw)
    extensions: Final = [  # mutable-ok: LiteLLM compatibility payload
        {  # mutable-ok: LiteLLM compatibility payload
            "id": spec.id,
            "kind": "guardrail" if spec.kind == pb.EXTENSION_KIND_GUARDRAIL else "callback",
            "entrypoint": spec.entrypoint,
            "constructor": json.loads(spec.constructor_json or b"{}"),
        }
        for spec in manifest.specs
    ]
    return json.dumps(
        {  # mutable-ok: LiteLLM compatibility payload
            "revision_id": manifest.revision_id,
            "extensions": extensions,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _guardrail_configs(
    config: Mapping[str, object], settings: Mapping[str, object]
) -> tuple[Mapping[str, object], ...]:
    values: list[  # mutable-ok: LiteLLM compatibility payload # rebind-ok: invocation-scoped RPC state
        Mapping[str, object]
    ] = []  # mutable-ok: LiteLLM compatibility payload # rebind-ok: invocation-scoped RPC state
    for candidate in (config.get("guardrails"), settings.get("guardrails")):
        if isinstance(candidate, list):
            values.extend(item for item in candidate if isinstance(item, Mapping))
    return tuple(values)


def _string_entries(value: object) -> Iterable[str]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return (item for item in value if isinstance(item, str))
    return ()


def _is_customer_entrypoint(value: str) -> bool:
    return ("." in value or ":" in value) and not value.startswith(("http://", "https://"))


def _stable_id(kind: str, entrypoint: str, name: str = "") -> str:
    digest: Final = hashlib.sha256(f"{kind}\0{entrypoint}\0{name}".encode()).hexdigest()[:16]
    return f"{kind}-{digest}"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, default=str).encode()
