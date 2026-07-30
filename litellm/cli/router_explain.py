"""``litellm router-explain`` — offline summary of a proxy router config.

Loads a proxy ``config.yaml`` (or ``model_list`` JSON) and reports the
resolved shape of the router: total deployments, unique ``model_name``
count, the set of providers in use, the configured routing strategy,
the per-group deployment count, and a small set of static anomalies
the operator is likely to care about (single-deployment groups, typo'd
routing strategy, negative router settings, duplicate litellm_params
across groups, empty model_list, orphan model-group aliases).

Offline-only. No network calls, no provider credentials required.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import click

Severity = Literal["warn", "fail"]


# Routing strategies that litellm/router.py accepts. Mirrors the
# allowlist in config_validate so the two CLIs stay in lockstep.
_KNOWN_ROUTING_STRATEGIES: frozenset[str] = frozenset(
    [
        "simple-shuffle",
        "least-busy",
        "latency-based-routing",
        "cost-based-routing",
        "usage-based-routing",
        "usage-based-routing-v2",
        "provider-budget-routing",
        "lar1",
    ]
)


@dataclass(frozen=True)
class Anomaly:
    """A single static finding about the resolved router."""

    name: str
    severity: Severity
    details: str


@dataclass(frozen=True)
class ModelGroup:
    """A single ``model_name`` and the deployments that share it."""

    name: str
    deployment_count: int
    providers: tuple[str, ...]


@dataclass(frozen=True)
class RouterExplanation:
    """Result of a full ``explain_router`` run."""

    path: str
    deployments: int
    model_names: tuple[str, ...]
    providers: tuple[str, ...]
    routing_strategy: str | None
    num_retries: int | None
    timeout: float | None
    cooldown_time: float | None
    model_access_groups: tuple[str, ...]
    model_group_aliases: Mapping[str, str]
    model_groups: tuple[ModelGroup, ...]
    anomalies: tuple[Anomaly, ...] = field(default_factory=tuple)

    @property
    def has_anomalies(self) -> bool:
        return bool(self.anomalies)

    def to_jsonable(self) -> Mapping[str, object]:
        return {
            "path": self.path,
            "deployments": self.deployments,
            "model_names": list(self.model_names),
            "providers": list(self.providers),
            "routing_strategy": self.routing_strategy,
            "num_retries": self.num_retries,
            "timeout": self.timeout,
            "cooldown_time": self.cooldown_time,
            "model_access_groups": list(self.model_access_groups),
            "model_group_aliases": dict(self.model_group_aliases),
            "model_groups": [
                {
                    "name": g.name,
                    "deployment_count": g.deployment_count,
                    "providers": list(g.providers),
                }
                for g in self.model_groups
            ],
            "anomalies": [asdict(a) for a in self.anomalies],
        }


def _read_stdin() -> str:
    return sys.stdin.read()


def _parse_yaml_or_json(text: str, path: str) -> tuple[Mapping[str, object] | None, str | None]:
    """Parse the file as YAML or JSON. Return (parsed_dict, error_message)."""
    suffix = Path(path).suffix.lower() if path not in {"-", ""} else ""
    if suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            return None, f"invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"
    else:
        try:
            import yaml
        except ImportError:
            return None, "PyYAML is not installed; cannot parse YAML"
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            return None, f"invalid YAML: {exc}"
    if data is None:
        return {}, None
    if not isinstance(data, dict):
        return None, f"top-level value must be a mapping, got {type(data).__name__}"
    return data, None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _provider_from_model(model: object) -> str | None:
    """Extract the provider prefix from a ``litellm_params.model`` string.

    The convention in litellm is ``<provider>/<model>`` (e.g. ``openai/gpt-4o``,
    ``anthropic/claude-sonnet-4-6``, ``bedrock/anthropic.claude-3-sonnet``).
    Returns the provider segment, or ``None`` if the model string is not
    shaped that way.
    """
    if not isinstance(model, str) or not model:
        return None
    if "/" not in model:
        return None
    provider = model.split("/", 1)[0].strip()
    return provider or None


def _build_explanation(source: str, parsed: Mapping[str, object]) -> RouterExplanation:
    """Build the structured explanation from a parsed config."""
    raw_model_list = parsed.get("model_list") or []
    model_list: tuple[Mapping[str, object], ...] = tuple(
        entry for entry in raw_model_list if isinstance(entry, Mapping)
    )

    # Unique model names, preserving first-seen order. A model_name only
    # counts when its entry has a well-formed litellm_params.model; an
    # entry with a name but no usable model is a malformed deployment,
    # not a real group.
    # dict.fromkeys preserves first-seen order and dedupes; no append loop.
    model_names: tuple[str, ...] = tuple(
        dict.fromkeys(
            entry["model_name"]
            for entry in model_list
            if isinstance(entry.get("model_name"), str)
            and entry["model_name"]
            and isinstance(entry.get("litellm_params"), Mapping)
            and isinstance(entry["litellm_params"].get("model"), str)
            and entry["litellm_params"]["model"]
        )
    )

    # Provider set, extracted from each entry's litellm_params.model.
    providers: tuple[str, ...] = tuple(
        sorted(
            frozenset(
                _provider_from_model(entry["litellm_params"]["model"])
                for entry in model_list
                if isinstance(entry.get("litellm_params"), Mapping)
                and isinstance(entry["litellm_params"].get("model"), str)
                and entry["litellm_params"]["model"]
                and _provider_from_model(entry["litellm_params"]["model"]) is not None
            )
        )
    )

    # Per-group deployment count and provider set. Mirrors the
    # model_names filter: a deployment only counts when it has a real
    # model_name AND a real litellm_params.model. The dict/set are
    # local accumulators that never escape the function; mutable-ok
    # is the right suppression here (see config_validate._iter_credential_refs
    # for the same pattern).
    group_providers: dict[str, set[str]] = defaultdict(set)  # mutable-ok: per-group accumulator
    group_counts: dict[str, int] = defaultdict(int)  # mutable-ok: per-group counter
    for entry in model_list:
        name = entry.get("model_name")
        if not isinstance(name, str) or not name:
            continue
        params = entry.get("litellm_params")
        if not isinstance(params, Mapping):
            continue
        model_field = params.get("model")
        if not isinstance(model_field, str) or not model_field:
            continue
        group_counts[name] += 1
        provider = _provider_from_model(model_field)
        if provider is not None:
            group_providers[name].add(provider)

    model_groups = tuple(
        ModelGroup(
            name=name,
            deployment_count=group_counts[name],
            providers=tuple(sorted(group_providers.get(name, set()))),
        )
        for name in model_names
    )

    # Router settings (top-level values, not deep validation).
    router_settings_raw = parsed.get("router_settings")
    router_settings: Mapping[str, object] = router_settings_raw if isinstance(router_settings_raw, Mapping) else {}
    routing_strategy_raw = router_settings.get("routing_strategy")
    routing_strategy: str | None = (
        routing_strategy_raw if isinstance(routing_strategy_raw, str) and routing_strategy_raw else None
    )
    num_retries = _coerce_int(router_settings.get("num_retries"))
    timeout = _coerce_float(router_settings.get("timeout"))
    cooldown_time = _coerce_float(router_settings.get("cooldown_time"))

    # model_access_groups at the top level (litellm_settings.model_access_groups).
    litellm_settings_raw = parsed.get("litellm_settings")
    litellm_settings: Mapping[str, object] = litellm_settings_raw if isinstance(litellm_settings_raw, Mapping) else {}
    access_groups_raw = litellm_settings.get("model_access_groups")
    model_access_groups: tuple[str, ...]
    if isinstance(access_groups_raw, list):
        model_access_groups = tuple(str(g) for g in access_groups_raw if isinstance(g, (str, int)))
    else:
        model_access_groups = ()

    # model_group_aliases (general_settings.model_group_alias).
    general_settings_raw = parsed.get("general_settings")
    general_settings: Mapping[str, object] = general_settings_raw if isinstance(general_settings_raw, Mapping) else {}
    aliases_raw = general_settings.get("model_group_alias")
    # Local accumulator; mutable-ok is correct because the dict never escapes.
    model_group_aliases: dict[str, str] = {}  # mutable-ok: alias accumulator
    if isinstance(aliases_raw, Mapping):
        for k, v in aliases_raw.items():
            if isinstance(k, str) and isinstance(v, str):
                model_group_aliases[k] = v

    anomalies = _compute_anomalies(
        model_list=model_list,
        model_names=model_names,
        group_counts=group_counts,
        routing_strategy=routing_strategy,
        num_retries=num_retries,
        timeout=timeout,
        cooldown_time=cooldown_time,
        model_group_aliases=model_group_aliases,
    )

    return RouterExplanation(
        path=source,
        deployments=len(model_list),
        model_names=tuple(model_names),
        providers=providers,
        routing_strategy=routing_strategy,
        num_retries=num_retries,
        timeout=timeout,
        cooldown_time=cooldown_time,
        model_access_groups=model_access_groups,
        model_group_aliases=model_group_aliases,
        model_groups=model_groups,
        anomalies=anomalies,
    )


def _compute_anomalies(
    *,
    model_list: tuple[Mapping[str, object], ...],
    model_names: tuple[str, ...],
    group_counts: Mapping[str, int],
    routing_strategy: str | None,
    num_retries: int | None,
    timeout: float | None,
    cooldown_time: float | None,
    model_group_aliases: Mapping[str, str],
) -> tuple[Anomaly, ...]:
    """Build the ordered tuple of anomalies for a parsed config."""
    # Local accumulator; mutable-ok because `found` never escapes the function
    # and is frozen to a tuple at return.
    found: list[Anomaly] = []  # mutable-ok: anomaly accumulator

    # Empty model_list is a warn, not a fail, because the proxy can still start.
    if not model_list:
        found.append(
            Anomaly(
                name="empty-model-list",
                severity="warn",
                details="model_list is empty; no deployments will be available",
            )
        )

    # Single-deployment model groups: a fallback group with one deployment
    # is usually a typo (the operator meant to add a second backend).
    singles = tuple(sorted(n for n, c in group_counts.items() if c == 1))
    if singles:
        preview = ", ".join(repr(s) for s in singles[:3])
        suffix = "" if len(singles) <= 3 else f" (+{len(singles) - 3} more)"
        found.append(
            Anomaly(
                name="single-deployment-group",
                severity="warn",
                details=f"{len(singles)} model_name(s) have only 1 deployment (no fallback): {preview}{suffix}",
            )
        )

    # Duplicate litellm_params: two model_name entries share the same
    # underlying (model, api_key, api_base) tuple. Sometimes intentional
    # (aliasing), usually a config error. The two dicts are local
    # accumulators; mutable-ok is correct (same pattern as the DFS
    # walker in config_validate._iter_credential_refs).
    fingerprint_counts: dict[tuple[object, ...], int] = defaultdict(int)  # mutable-ok: fingerprint tally
    fingerprint_to_names: dict[tuple[object, ...], list[str]] = defaultdict(list)  # mutable-ok: names per fingerprint
    for entry in model_list:
        name = entry.get("model_name")
        if not isinstance(name, str):
            continue
        params = entry.get("litellm_params")
        if not isinstance(params, Mapping):
            continue
        fingerprint = (
            params.get("model"),
            params.get("api_key"),
            params.get("api_base"),
            params.get("api_version"),
        )
        fingerprint_counts[fingerprint] += 1
        fingerprint_to_names[fingerprint].append(name)
    duplicate_groups: tuple[tuple[str, ...], ...] = tuple(
        tuple(fingerprint_to_names[fp])
        for fp, count in fingerprint_counts.items()
        if count > 1 and len(fingerprint_to_names[fp]) > 1
    )
    if duplicate_groups:
        names_preview = ", ".join("/".join(names[:3]) for names in duplicate_groups[:2])
        suffix = "" if len(duplicate_groups) <= 2 else f" (+{len(duplicate_groups) - 2} more)"
        found.append(
            Anomaly(
                name="duplicate-litellm-params",
                severity="warn",
                details=f"{len(duplicate_groups)} litellm_params fingerprint(s) shared across model_name(s): {names_preview}{suffix}",
            )
        )

    # Routing strategy must be in the known allowlist (or absent).
    if routing_strategy is not None and routing_strategy not in _KNOWN_ROUTING_STRATEGIES:
        found.append(
            Anomaly(
                name="unknown-routing-strategy",
                severity="fail",
                details=f"routing_strategy {routing_strategy!r} is not a known strategy; expected one of {sorted(_KNOWN_ROUTING_STRATEGIES)}",
            )
        )

    # Router settings must be non-negative / positive where they make sense.
    if num_retries is not None and num_retries < 0:
        found.append(
            Anomaly(
                name="negative-num-retries",
                severity="fail",
                details=f"num_retries must be >= 0, got {num_retries}",
            )
        )
    if timeout is not None and timeout <= 0:
        found.append(
            Anomaly(
                name="non-positive-timeout",
                severity="fail",
                details=f"timeout must be > 0, got {timeout}",
            )
        )
    if cooldown_time is not None and cooldown_time < 0:
        found.append(
            Anomaly(
                name="negative-cooldown-time",
                severity="fail",
                details=f"cooldown_time must be >= 0, got {cooldown_time}",
            )
        )

    # Orphan model-group aliases: alias -> name, but name not in model_list.
    # Use a frozenset so the membership test does not require a mutable set.
    name_set = frozenset(model_names)
    orphans = tuple(alias for alias, target in model_group_aliases.items() if target not in name_set)
    if orphans:
        preview = ", ".join(repr(o) for o in orphans[:3])
        suffix = "" if len(orphans) <= 3 else f" (+{len(orphans) - 3} more)"
        found.append(
            Anomaly(
                name="orphan-model-group-alias",
                severity="warn",
                details=f"{len(orphans)} model_group_alias target(s) not present in model_list: {preview}{suffix}",
            )
        )

    return tuple(found)


def explain_router(source: str) -> RouterExplanation:
    """Public helper for programmatic use. Returns a ``RouterExplanation``.

    ``source`` is either a path to the config file or ``-`` to read from
    stdin. Raises :class:`ValueError` for invalid input (empty source).
    File or parse errors are surfaced as ``Anomaly`` entries on the
    returned ``RouterExplanation``, not raised.
    """
    if not source:
        raise ValueError("source is required (path or '-' for stdin)")

    if source == "-":
        try:
            text = _read_stdin()
        except OSError as exc:
            return RouterExplanation(
                path=source,
                deployments=0,
                model_names=(),
                providers=(),
                routing_strategy=None,
                num_retries=None,
                timeout=None,
                cooldown_time=None,
                model_access_groups=(),
                model_group_aliases={},
                model_groups=(),
                anomalies=(
                    Anomaly(
                        name="file",
                        severity="fail",
                        details=f"could not read stdin: {exc}",
                    ),
                ),
            )
        source_label = "<stdin>"
    else:
        path = Path(source)
        if not path.is_file():
            return RouterExplanation(
                path=source,
                deployments=0,
                model_names=(),
                providers=(),
                routing_strategy=None,
                num_retries=None,
                timeout=None,
                cooldown_time=None,
                model_access_groups=(),
                model_group_aliases={},
                model_groups=(),
                anomalies=(
                    Anomaly(
                        name="file",
                        severity="fail",
                        details=f"{source!r} does not exist or is not a regular file",
                    ),
                ),
            )
        if not os.access(path, os.R_OK):
            return RouterExplanation(
                path=source,
                deployments=0,
                model_names=(),
                providers=(),
                routing_strategy=None,
                num_retries=None,
                timeout=None,
                cooldown_time=None,
                model_access_groups=(),
                model_group_aliases={},
                model_groups=(),
                anomalies=(
                    Anomaly(
                        name="file",
                        severity="fail",
                        details=f"{source!r} is not readable",
                    ),
                ),
            )
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            return RouterExplanation(
                path=source,
                deployments=0,
                model_names=(),
                providers=(),
                routing_strategy=None,
                num_retries=None,
                timeout=None,
                cooldown_time=None,
                model_access_groups=(),
                model_group_aliases={},
                model_groups=(),
                anomalies=(
                    Anomaly(
                        name="file",
                        severity="fail",
                        details=f"could not read: {exc}",
                    ),
                ),
            )
        source_label = source

    parsed, parse_error = _parse_yaml_or_json(text, source)
    if parse_error is not None:
        return RouterExplanation(
            path=source_label,
            deployments=0,
            model_names=(),
            providers=(),
            routing_strategy=None,
            num_retries=None,
            timeout=None,
            cooldown_time=None,
            model_access_groups=(),
            model_group_aliases={},
            model_groups=(),
            anomalies=(Anomaly(name="parse", severity="fail", details=parse_error),),
        )

    explanation = _build_explanation(source_label, parsed)
    return explanation


def _exit_code(explanation: RouterExplanation) -> int:
    if explanation.has_anomalies:
        return 1
    return 0


def _render_summary(explanation: RouterExplanation, console: object) -> None:
    from rich.console import Console
    from rich.table import Table

    if not isinstance(console, Console):
        raise TypeError(f"console must be a rich.Console, got {type(console).__name__}")

    console.print(f"router-explain: {explanation.path}", style="bold")
    console.print()

    summary = Table(title="Summary", show_lines=False)
    summary.add_column("field", style="cyan", no_wrap=True)
    summary.add_column("value")
    summary.add_row("Deployments", str(explanation.deployments))
    summary.add_row("Model names", f"{len(explanation.model_names)} unique")
    summary.add_row(
        "Providers",
        ", ".join(explanation.providers) if explanation.providers else "(none)",
    )
    summary.add_row(
        "Routing strategy",
        explanation.routing_strategy if explanation.routing_strategy is not None else "(default)",
    )
    summary.add_row(
        "Num retries",
        str(explanation.num_retries) if explanation.num_retries is not None else "(default)",
    )
    summary.add_row(
        "Timeout",
        f"{explanation.timeout} s" if explanation.timeout is not None else "(default)",
    )
    summary.add_row(
        "Cooldown",
        f"{explanation.cooldown_time} s" if explanation.cooldown_time is not None else "(default)",
    )
    summary.add_row(
        "Model access groups",
        str(len(explanation.model_access_groups))
        + (
            f" ({', '.join(explanation.model_access_groups[:3])}{'...' if len(explanation.model_access_groups) > 3 else ''})"
            if explanation.model_access_groups
            else ""
        ),
    )
    if explanation.model_group_aliases:
        aliases_preview = ", ".join(f"{k}->{v}" for k, v in list(explanation.model_group_aliases.items())[:3])
        suffix = (
            "" if len(explanation.model_group_aliases) <= 3 else f" (+{len(explanation.model_group_aliases) - 3} more)"
        )
        summary.add_row("Model group aliases", f"{len(explanation.model_group_aliases)} ({aliases_preview}{suffix})")
    else:
        summary.add_row("Model group aliases", "0 configured")
    console.print(summary)
    console.print()

    groups = Table(title=f"Model groups ({len(explanation.model_groups)})", show_lines=False)
    groups.add_column("name", style="cyan", no_wrap=True)
    groups.add_column("deployments", justify="right", no_wrap=True)
    groups.add_column("providers")
    for g in explanation.model_groups:
        groups.add_row(
            g.name,
            str(g.deployment_count),
            ", ".join(g.providers) if g.providers else "(none)",
        )
    console.print(groups)
    console.print()

    if explanation.anomalies:
        anomalies = Table(title=f"Anomalies ({len(explanation.anomalies)})", show_lines=False)
        anomalies.add_column("severity", no_wrap=True)
        anomalies.add_column("name", style="cyan", no_wrap=True)
        anomalies.add_column("details")
        severity_styles: Mapping[str, str] = {"warn": "yellow", "fail": "red"}
        for a in explanation.anomalies:
            anomalies.add_row(
                f"[{severity_styles[a.severity]}]{a.severity}[/]",
                a.name,
                a.details,
            )
        console.print(anomalies)
    else:
        console.print("Anomalies: none", style="green")


@click.command(name="router-explain")
@click.option(
    "--config",
    "config",
    required=True,
    help="Path to the proxy config file (.yaml, .yml, .json). Pass '-' to read from stdin.",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Emit a JSON object instead of a human-readable summary.",
)
def cli(
    config: str,
    output_json: bool,
) -> None:
    """Summarise the resolved shape of a proxy router config."""
    try:
        explanation = explain_router(config)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        click.echo(f"router-explain failed: {type(exc).__name__}: {exc}", err=True)
        sys.exit(1)

    if output_json:
        click.echo(json.dumps(explanation.to_jsonable()))
    else:
        from rich.console import Console

        _render_summary(explanation, Console())
    sys.exit(_exit_code(explanation))
