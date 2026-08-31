import json
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import urlparse

import click
from pydantic import TypeAdapter

from litellm.litellm_core_utils.private_json import ensure_private_dir, write_private_json

HIDDEN_COMMANDS_KEY: Final = "hidden_commands"

_config_adapter: Final[TypeAdapter[Mapping[str, str]]] = TypeAdapter(Mapping[str, str])


def get_config_file_path() -> str:
    """Get the path to the persistent CLI config file"""
    home_dir: Final = Path.home()
    config_dir: Final = home_dir / ".litellm"
    return str(config_dir / "config.json")


def load_config() -> Mapping[str, str]:
    """Load CLI config from file; returns {} if missing or unreadable"""
    try:
        config_file: Final = get_config_file_path()
    except RuntimeError:
        return {}
    if not os.path.exists(config_file):
        return {}
    try:
        with open(config_file, "r") as f:
            return _config_adapter.validate_python(json.load(f))
    except (OSError, ValueError) as e:
        click.echo(f"Warning: ignoring invalid config file {config_file}: {e}", err=True)
        return {}


def save_config(config: Mapping[str, str]) -> None:
    """Save CLI config to file"""
    config_file: Final = Path(get_config_file_path())
    ensure_private_dir(config_file.parent)
    write_private_json(str(config_file), config)


def get_config_value(key: str) -> str | None:
    """Get a single value from the persistent CLI config"""
    return load_config().get(key)


def parse_hidden_commands(raw: str | None) -> frozenset[str]:
    """Split a stored `hidden_commands` value, e.g. "codex, opencode"."""
    return frozenset(name.strip() for name in (raw or "").split(",") if name.strip())


def hidden_command_names() -> frozenset[str]:
    """Top-level commands the operator chose to keep out of `lite`'s listings."""
    return parse_hidden_commands(get_config_value(HIDDEN_COMMANDS_KEY))


def _normalize_base_url(value: str) -> str:
    parsed: Final = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise click.UsageError("base_url must be a full http:// or https:// URL including a host")
    if "?" in value or "#" in value:
        raise click.UsageError("base_url must not include a query string or fragment")
    return value.rstrip("/")


def _normalize_hidden_commands(value: str) -> str:
    names: Final = parse_hidden_commands(value)
    if not names:
        raise click.UsageError(
            f"{HIDDEN_COMMANDS_KEY} must be a comma-separated list of command names, e.g. "
            f"`lite config set {HIDDEN_COMMANDS_KEY} codex,opencode`. To list everything again, "
            f"run `lite config unset {HIDDEN_COMMANDS_KEY}`"
        )
    if any(" " in name for name in names):
        raise click.UsageError(f"{HIDDEN_COMMANDS_KEY} entries must be single command names, without spaces")
    return ",".join(sorted(names))


_NORMALIZERS: Final[Mapping[str, Callable[[str], str]]] = MappingProxyType(
    {
        "base_url": _normalize_base_url,
        HIDDEN_COMMANDS_KEY: _normalize_hidden_commands,
    }
)

ALLOWED_CONFIG_KEYS: Final[tuple[str, ...]] = tuple(_NORMALIZERS)


@click.group(name="config")
def config_commands() -> None:
    """Manage persistent CLI configuration (~/.litellm/config.json)"""


@config_commands.command(name="set")
@click.argument("key")
@click.argument("value")
def set_config(key: str, value: str) -> None:
    """Set a config KEY to VALUE (e.g. `lite config set base_url https://your-proxy.example.com`)"""
    normalizer: Final = _NORMALIZERS.get(key)
    if normalizer is None:
        raise click.UsageError(f"Unknown config key '{key}'. Allowed keys: {', '.join(ALLOWED_CONFIG_KEYS)}")

    normalized_value: Final = normalizer(value)
    save_config({**load_config(), key: normalized_value})
    click.echo(f"Set {key} = {normalized_value} in {get_config_file_path()}")


@config_commands.command(name="get")
@click.argument("key", required=False)
def get_config(key: str | None) -> None:
    """Print the value of KEY, or all stored config when KEY is omitted"""
    config: Final = load_config()

    if key is not None:
        value: Final = config.get(key)
        if value is None:
            click.echo(f"{key} is not set", err=True)
            sys.exit(1)
        click.echo(value)
        return

    if not config:
        click.echo("(no config set)")
        return

    for entry_key, entry_value in config.items():
        click.echo(f"{entry_key} = {entry_value}")


@config_commands.command(name="unset")
@click.argument("key")
def unset_config(key: str) -> None:
    """Remove KEY from the config file"""
    config: Final = load_config()
    if key not in config:
        click.echo(f"{key} was not set")
        return

    save_config({k: v for k, v in config.items() if k != key})
    click.echo(f"Removed {key} from {get_config_file_path()}")
