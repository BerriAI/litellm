"""``litellm config-validate`` — offline validation of a proxy config file.

Loads a proxy ``config.yaml`` (or ``model_list`` JSON) and runs a small
set of static checks that catch the most common misconfigurations
before the proxy is started. Offline-only. No network calls, no
provider credentials required.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import click

Status = Literal["pass", "warn", "fail"]


# Top-level keys the proxy reads from the config file. Anything else
# at the top level is treated as a warn (likely a typo or a feature
# this CLI does not yet know about) and a fail under --strict.
_KNOWN_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    [
        "model_list",
        "litellm_settings",
        "general_settings",
        "router_settings",
        "environment_variables",
        "callback_settings",
        "guardrails",
        "prompts",
        "credential_list",
        "search_tools",
        "sandbox_tools",
        "mcp_tools",
        "mcp_servers",
        "agent_list",
        "assistant_settings",
        "finetune_settings",
        "files_settings",
        "default_vertex_config",
        "vector_store_registry",
        "worker_registry",
        "policies",
        "include",
        # accepted at the top level for proxy startup but not load-bearing
        "model_name",
        "litellm_params",
    ]
)

# Routing strategies that litellm/router.py accepts. Includes the
# legacy aliases ("simple-shuffle", "lar1") honored at the top of
# Router.__init__.
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

# Matches `os.environ/VAR_NAME` and `os.environ//VAR_NAME` (the
# double-slash typo is also caught so a deployment does not silently
# look up an empty env var name).
_ENV_VAR_REF_RE = re.compile(r"^os\.environ/+([A-Za-z_][A-Za-z0-9_]*)$")


@dataclass(frozen=True)
class Check:
    """Result of a single validation check."""

    name: str
    status: Status
    details: str

    @property
    def ok(self) -> bool:
        return self.status == "pass"


@dataclass(frozen=True)
class ConfigValidation:
    """Result of a full ``validate_config`` run."""

    path: str
    checks: tuple[Check, ...] = ()

    @property
    def has_failures(self) -> bool:
        return any(c.status == "fail" for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(c.status == "warn" for c in self.checks)

    def to_jsonable(self) -> Mapping[str, object]:
        return {
            "path": self.path,
            "checks": [asdict(c) for c in self.checks],
        }


def _read_stdin() -> str:
    return sys.stdin.read()


def _parse_yaml_or_json(text: str, path: str) -> tuple[Mapping[str, object] | None, str | None]:
    """Parse the file as YAML or JSON. Return (parsed_dict, error_message).

    YAML is tried first because config.yaml is the canonical proxy
    format. JSON is tried if the file extension is .json. A parse
    error is returned as (None, error_message) and the caller turns
    it into a fail-level check.
    """
    suffix = Path(path).suffix.lower()
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


def _check_file_pass(source: str) -> Check:
    """Build the file-level pass check for known-good sources."""
    if source == "-":
        return Check(name="file", status="pass", details="reading from stdin")
    return Check(name="file", status="pass", details=str(source))


def _check_file(source: str) -> Check:
    """Confirm the file exists and is readable."""
    if source == "-":
        return Check(name="file", status="pass", details="reading from stdin")
    path = Path(source)
    if not path.is_file():
        return Check(name="file", status="fail", details=f"{source!r} does not exist or is not a regular file")
    if not os.access(path, os.R_OK):
        return Check(name="file", status="fail", details=f"{source!r} is not readable")
    suffix = path.suffix.lower()
    if suffix not in {".yaml", ".yml", ".json"}:
        return Check(
            name="file",
            status="warn",
            details=f"{source!r} has suffix {suffix!r}; expected .yaml, .yml, or .json",
        )
    return Check(name="file", status="pass", details=str(path))


def _file_failure_for_missing(source: str) -> Check:
    """Build a file-level fail check with a consistent 'does not exist' message."""
    return Check(
        name="file",
        status="fail",
        details=f"{source!r} does not exist or is not a regular file",
    )


def _check_top_level_keys(data: Mapping[str, object]) -> Check:
    """Warn on any top-level key that is not in the known allowlist."""
    unknown = tuple(sorted(k for k in data.keys() if k not in _KNOWN_TOP_LEVEL_KEYS))
    if not unknown:
        return Check(name="top-level-keys", status="pass", details=f"{len(data)} known keys")
    if len(unknown) == 1:
        details = f"unknown top-level key {unknown[0]!r} (typo?)"
    else:
        details = f"unknown top-level keys: {', '.join(repr(k) for k in unknown)}"
    return Check(name="top-level-keys", status="warn", details=details)


def _check_model_list_shape(data: Mapping[str, object]) -> Check:
    """Confirm model_list is a list of mappings with model_name and litellm_params."""
    if "model_list" not in data:
        return Check(name="model-list-shape", status="pass", details="not present")
    model_list = data["model_list"]
    if not isinstance(model_list, list):
        return Check(
            name="model-list-shape",
            status="fail",
            details=f"model_list must be a list, got {type(model_list).__name__}",
        )
    if not model_list:
        return Check(name="model-list-shape", status="warn", details="model_list is empty")

    def _bad_message(idx: int, entry: object) -> str | None:
        if not isinstance(entry, dict):
            return f"[{idx}]: not a mapping"
        if not isinstance(entry.get("model_name"), str) or not entry["model_name"]:
            return f"[{idx}]: missing or non-string model_name"
        litellm_params = entry.get("litellm_params")
        if not isinstance(litellm_params, dict):
            return f"[{idx}]: missing or non-mapping litellm_params"
        return None

    bad = tuple(filter(None, (_bad_message(idx, entry) for idx, entry in enumerate(model_list))))
    if bad:
        preview = "; ".join(bad[:3])
        suffix = "" if len(bad) <= 3 else f" (+{len(bad) - 3} more)"
        return Check(
            name="model-list-shape",
            status="fail",
            details=f"{len(bad)} malformed entries: {preview}{suffix}",
        )
    return Check(name="model-list-shape", status="pass", details=f"{len(model_list)} entries")


def _check_model_name_uniqueness(data: Mapping[str, object]) -> Check:
    """Confirm each model_name in model_list is unique."""
    model_list = data.get("model_list") or []
    if not isinstance(model_list, list):
        return Check(name="model-name-uniq", status="pass", details="skipped (no model_list)")
    names = tuple(
        entry["model_name"]
        for entry in model_list
        if isinstance(entry, dict) and isinstance(entry.get("model_name"), str)
    )
    counts: Mapping[str, int] = {n: names.count(n) for n in set(names)}
    dupes = tuple(sorted(n for n, c in counts.items() if c > 1))
    if not dupes:
        return Check(name="model-name-uniq", status="pass", details=f"{len(counts)} unique")
    preview = ", ".join(repr(d) for d in dupes[:3])
    suffix = "" if len(dupes) <= 3 else f" (+{len(dupes) - 3} more)"
    return Check(
        name="model-name-uniq",
        status="fail",
        details=f"{len(dupes)} duplicates: {preview}{suffix}",
    )


def _check_litellm_params_model(data: Mapping[str, object]) -> Check:
    """Confirm litellm_params.model is a non-empty string in every entry."""
    model_list = data.get("model_list") or []
    if not isinstance(model_list, list):
        return Check(name="litellm-params-model", status="pass", details="skipped (no model_list)")

    def _is_bad(idx: int, entry: object) -> bool:
        if not isinstance(entry, dict):
            return False
        params = entry.get("litellm_params")
        if not isinstance(params, dict):
            return False
        model = params.get("model")
        return not isinstance(model, str) or not model

    bad = tuple(idx for idx, entry in enumerate(model_list) if _is_bad(idx, entry))
    if not bad:
        return Check(
            name="litellm-params-model",
            status="pass",
            details=f"{len(model_list)} entries have non-empty model",
        )
    preview = ", ".join(str(i) for i in bad[:5])
    suffix = "" if len(bad) <= 5 else f" (+{len(bad) - 5} more)"
    return Check(
        name="litellm-params-model",
        status="fail",
        details=f"{len(bad)} entries have non-string or empty model: index {preview}{suffix}",
    )


def _check_router_strategy(data: Mapping[str, object]) -> Check:
    """Confirm router_settings.routing_strategy is a known value if set."""
    router_settings = data.get("router_settings")
    if not isinstance(router_settings, dict):
        return Check(name="router-strategy", status="pass", details="not present")
    if "routing_strategy" not in router_settings:
        return Check(name="router-strategy", status="pass", details="not set")
    strategy = router_settings["routing_strategy"]
    if not isinstance(strategy, str):
        return Check(
            name="router-strategy",
            status="fail",
            details=f"routing_strategy must be a string, got {type(strategy).__name__}",
        )
    if strategy in _KNOWN_ROUTING_STRATEGIES:
        return Check(name="router-strategy", status="pass", details=strategy)
    return Check(
        name="router-strategy",
        status="fail",
        details=f"{strategy!r} is not a known routing strategy; expected one of {sorted(_KNOWN_ROUTING_STRATEGIES)}",
    )


def _iter_credential_refs(node: object) -> tuple[tuple[str, str], ...]:
    """Walk a JSON-like value and collect every ``os.environ/VAR`` reference.

    Returns a tuple of (parent_path, var_name) pairs. The parent_path
    is a dotted path for diagnostics; var_name is the env var being
    referenced.
    """

    def walk(value: object, path: str) -> tuple[tuple[str, str], ...]:
        if isinstance(value, dict):
            out: tuple[tuple[str, str], ...] = ()
            for k, v in value.items():
                out += walk(v, f"{path}.{k}" if path else str(k))
            return out
        if isinstance(value, list):
            out = ()
            for i, v in enumerate(value):
                out += walk(v, f"{path}[{i}]")
            return out
        if isinstance(value, str):
            m = _ENV_VAR_REF_RE.match(value)
            if m:
                return ((path, m.group(1)),)
            return ()
        return ()

    return walk(node, "")


def _check_credential_pattern(data: Mapping[str, object]) -> Check:
    """Warn on any ``os.environ/VAR`` reference whose env var is unset."""
    refs = _iter_credential_refs(data)
    if not refs:
        return Check(name="cred-pattern", status="pass", details="no env-var references")
    unset = tuple(name for _, name in refs if not os.environ.get(name))
    if not unset:
        return Check(
            name="cred-pattern",
            status="pass",
            details=f"{len(refs)} env-var references; all set",
        )
    preview = ", ".join(unset[:3])
    suffix = "" if len(unset) <= 3 else f" (+{len(unset) - 3} more)"
    return Check(
        name="cred-pattern",
        status="warn",
        details=f"{len(unset)} referenced env var(s) unset: {preview}{suffix}",
    )


def _run_all_checks(source: str, parsed: Mapping[str, object] | None) -> tuple[Check, ...]:
    """Build the ordered tuple of checks for a parsed config."""
    file_check = _check_file(source)
    if parsed is None:
        return (file_check,)
    return (
        file_check,
        Check(name="parse", status="pass", details=f"{len(parsed)} top-level keys"),
        _check_top_level_keys(parsed),
        _check_model_list_shape(parsed),
        _check_model_name_uniqueness(parsed),
        _check_litellm_params_model(parsed),
        _check_router_strategy(parsed),
        _check_credential_pattern(parsed),
    )


def validate_config(source: str, *, strict: bool = False) -> ConfigValidation:
    """Public helper for programmatic use. Returns a ``ConfigValidation``.

    ``source`` is either a path to the config file or ``-`` to read from
    stdin. ``strict=True`` promotes warn-level findings to fail in the
    returned ``ConfigValidation.has_failures`` sense (caller decides how
    to surface the strict mode; the CLI exits 1 in that case).

    Raises :class:`ValueError` for invalid input (empty source). File
    or parse errors are surfaced as fail-level checks in the returned
    ``ConfigValidation``, not raised.
    """
    if not source:
        raise ValueError("source is required (path or '-' for stdin)")
    if source == "-":
        try:
            text = _read_stdin()
        except OSError as exc:
            return ConfigValidation(
                path=source,
                checks=(Check(name="file", status="fail", details=f"could not read stdin: {exc}"),),
            )
        checks: tuple[Check, ...] = (_check_file_pass(source),)
    else:
        path = Path(source)
        if not path.is_file():
            return ConfigValidation(path=source, checks=(_file_failure_for_missing(source),))
        if not os.access(path, os.R_OK):
            return ConfigValidation(
                path=source,
                checks=(Check(name="file", status="fail", details=f"{source!r} is not readable"),),
            )
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            return ConfigValidation(
                path=source,
                checks=(Check(name="file", status="fail", details=f"could not read: {exc}"),),
            )
        checks = (_check_file_pass(source),)
    parsed, parse_error = _parse_yaml_or_json(text, source)
    if parse_error is not None:
        return ConfigValidation(
            path=source,
            checks=checks + (Check(name="parse", status="fail", details=parse_error),),
        )
    checks = _run_all_checks(source, parsed)
    if strict:
        checks = tuple(
            Check(name=c.name, status=("fail" if c.status == "warn" else c.status), details=c.details) for c in checks
        )
    return ConfigValidation(path=source, checks=checks)


def _exit_code(validation: ConfigValidation) -> int:
    if validation.has_failures:
        return 1
    return 0


def _render_table(validation: ConfigValidation, console: object) -> None:
    from rich.console import Console
    from rich.table import Table

    if not isinstance(console, Console):
        raise TypeError(f"console must be a rich.Console, got {type(console).__name__}")
    table = Table(title=f"config-validate: {validation.path}", show_lines=False)
    table.add_column("check", style="cyan", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("details")
    status_styles: Mapping[str, str] = {"pass": "green", "warn": "yellow", "fail": "red"}
    for check in validation.checks:
        table.add_row(
            check.name,
            f"[{status_styles[check.status]}]{check.status}[/]",
            check.details,
        )
    console.print(table)


@click.command()
@click.option(
    "--config",
    "config",
    required=True,
    help="Path to the proxy config file (.yaml, .yml, .json). Pass '-' to read from stdin.",
)
@click.option(
    "--strict",
    "strict",
    is_flag=True,
    help="Treat warnings as failures (exit 1 instead of 0).",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Emit a JSON object instead of a table.",
)
def cli(
    config: str,
    strict: bool,
    output_json: bool,
) -> None:
    """Validate a proxy config file without starting the proxy."""
    try:
        validation = validate_config(config, strict=strict)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    except click.UsageError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        click.echo(f"config-validate failed: {type(exc).__name__}: {exc}", err=True)
        sys.exit(1)

    if output_json:
        click.echo(json.dumps(validation.to_jsonable()))
    else:
        from rich.console import Console

        _render_table(validation, Console())
    sys.exit(_exit_code(validation))
