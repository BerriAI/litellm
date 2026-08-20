"""``litellm doctor`` — offline diagnostic checks for the local litellm SDK setup.

Runs a fixed battery of checks (Python version, package import, env, API
keys, model price map, token counter) and prints a table of results.
Exit code is 0 if all checks pass, 1 if any check failed, 2 if any check
returned a warning.

The checks are intentionally offline — no outbound HTTP — and never
print secret values, only the names of env vars that are set.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable, List, Literal

import click
from rich.console import Console
from rich.table import Table

Status = Literal["pass", "warn", "fail"]


@dataclass(frozen=True)
class CheckResult:
    """Result of a single diagnostic check."""

    name: str
    status: Status
    details: str

    @property
    def ok(self) -> bool:
        return self.status == "pass"


_STATUS_STYLES: dict[Status, str] = {
    "pass": "green",
    "warn": "yellow",
    "fail": "red",
}


def _check_python() -> CheckResult:
    """Verify the Python interpreter is in the range litellm supports."""
    supported_min = (3, 10)
    supported_max = (3, 15)
    info = sys.implementation.name
    actual = sys.version_info
    in_range = supported_min <= (actual.major, actual.minor) < supported_max
    details = f"{info} {actual.major}.{actual.minor}.{actual.micro}"
    if not in_range:
        return CheckResult(
            name="python",
            status="fail",
            details=f"{details} (litellm requires >= {supported_min[0]}.{supported_min[1]}, < {supported_max[0]}.{supported_max[1]})",
        )
    return CheckResult(name="python", status="pass", details=details)


def _check_litellm() -> CheckResult:
    """Verify the litellm package is importable and report its version."""
    try:
        pkg_version = version("litellm")
    except PackageNotFoundError:
        return CheckResult(
            name="litellm",
            status="fail",
            details="litellm is not importable in the current environment",
        )
    try:
        import_module("litellm")
    except Exception as exc:
        return CheckResult(
            name="litellm",
            status="fail",
            details=f"version {pkg_version} but import raised {type(exc).__name__}: {exc}",
        )
    return CheckResult(name="litellm", status="pass", details=f"version {pkg_version}")


def _check_env_file() -> CheckResult:
    """Warn (do not fail) if no .env file is present in cwd or any parent."""
    cwd = Path.cwd()
    for directory in (cwd, *cwd.parents):
        if (directory / ".env").is_file():
            return CheckResult(
                name="env",
                status="pass",
                details=f"found {directory / '.env'}",
            )
    return CheckResult(
        name="env",
        status="warn",
        details=f"no .env file in {cwd} or any parent directory",
    )


_KNOWN_KEY_ENV_VARS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "AZURE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "VERTEXAI_API_KEY",
    "COHERE_API_KEY",
    "MISTRAL_API_KEY",
    "GROQ_API_KEY",
    "TOGETHERAI_API_KEY",
    "FIREWORKS_AI_API_KEY",
    "DEEPSEEK_API_KEY",
    "XAI_API_KEY",
    "PERPLEXITYAI_API_KEY",
    "REPLICATE_API_KEY",
    "HUGGINGFACE_API_KEY",
    "OPENROUTER_API_KEY",
    "DATABRICKS_API_KEY",
    "BEDROCK_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
)


def _check_api_keys() -> CheckResult:
    """List which known provider API key env vars are set.

    Never prints values, only the names of env vars that resolve to a
    non-empty string. Uses ``os.getenv`` directly rather than
    ``litellm.get_secret`` so the doctor command has no side effects on
    litellm's own secret cache.
    """
    set_keys: tuple[str, ...] = tuple(name for name in _KNOWN_KEY_ENV_VARS if os.getenv(name))
    if not set_keys:
        return CheckResult(
            name="api-keys",
            status="warn",
            details="no known provider API key env vars are set",
        )
    return CheckResult(
        name="api-keys",
        status="pass",
        details=f"{len(set_keys)} known provider key(s) set: {', '.join(set_keys)}",
    )


def _check_model_costs() -> CheckResult:
    """Verify model_prices_and_context_window.json loads and is non-trivial.

    Reads the local cost map shipped with the package, so this check is
    fully offline. The remote URL fetch is not exercised.
    """
    try:
        from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
    except Exception as exc:
        return CheckResult(
            name="model-costs",
            status="fail",
            details=f"could not import GetModelCostMap: {type(exc).__name__}",
        )
    try:
        cost_map = GetModelCostMap.load_local_model_cost_map()
    except Exception as exc:
        return CheckResult(
            name="model-costs",
            status="fail",
            details=f"local model cost map failed to load: {type(exc).__name__}: {exc}",
        )
    try:
        json.dumps(cost_map)
    except (TypeError, ValueError) as exc:
        return CheckResult(
            name="model-costs",
            status="fail",
            details=f"model cost map is not JSON-serializable: {exc}",
        )
    model_count = len(cost_map)
    if model_count < 50:
        return CheckResult(
            name="model-costs",
            status="warn",
            details=f"only {model_count} models in cost map — file may be stale or truncated",
        )
    return CheckResult(
        name="model-costs",
        status="pass",
        details=f"{model_count} models loaded (local copy)",
    )


def _check_token_counter() -> CheckResult:
    """Verify litellm.token_counter runs on a sample input without raising."""
    try:
        from litellm import token_counter
    except Exception as exc:
        return CheckResult(
            name="token-counter",
            status="fail",
            details=f"could not import token_counter: {type(exc).__name__}: {exc}",
        )
    try:
        count = token_counter(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "litellm doctor check"}],
        )
    except Exception as exc:
        return CheckResult(
            name="token-counter",
            status="fail",
            details=f"token_counter raised {type(exc).__name__}: {exc}",
        )
    if count <= 0:
        return CheckResult(
            name="token-counter",
            status="fail",
            details=f"token_counter returned non-positive value: {count!r}",
        )
    return CheckResult(name="token-counter", status="pass", details=f"{count} tokens")


_CHECKS: tuple[Callable[[], CheckResult], ...] = (
    _check_python,
    _check_litellm,
    _check_env_file,
    _check_api_keys,
    _check_model_costs,
    _check_token_counter,
)


def run_all_checks() -> List[CheckResult]:
    """Run every registered check and return the results in registration order."""
    return [check() for check in _CHECKS]


def _render_table(results: List[CheckResult], console: Console) -> None:
    table = Table(title="litellm doctor", show_lines=False)
    table.add_column("check", style="cyan", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("details")
    for result in results:
        table.add_row(
            result.name,
            f"[{_STATUS_STYLES[result.status]}]{result.status}[/]",
            result.details,
        )
    console.print(table)


def _exit_code(results: List[CheckResult]) -> int:
    if any(result.status == "fail" for result in results):
        return 1
    if any(result.status == "warn" for result in results):
        return 2
    return 0


@click.command()
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Emit results as a JSON array (one object per check) instead of a table.",
)
def cli(output_json: bool) -> None:
    """Run diagnostic checks on the local litellm SDK setup."""
    results = run_all_checks()
    if output_json:
        import json

        click.echo(json.dumps([{"name": r.name, "status": r.status, "details": r.details} for r in results]))
    else:
        _render_table(results, Console())
    sys.exit(_exit_code(results))


if __name__ == "__main__":
    cli()
