import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

import click
from pydantic import TypeAdapter, ValidationError

ALLOWED_CONFIG_KEYS: tuple[str, ...] = ("base_url",)

_config_adapter: TypeAdapter[Mapping[str, str]] = TypeAdapter(Mapping[str, str])


def get_config_file_path() -> str:
    """Get the path to the persistent CLI config file"""
    home_dir = Path.home()
    config_dir = home_dir / ".litellm"
    return str(config_dir / "config.json")


def load_config() -> Mapping[str, str]:
    """Load CLI config from file; returns {} if missing or unreadable"""
    try:
        config_file = get_config_file_path()
        if not os.path.exists(config_file):
            return {}
        with open(config_file, "r") as f:
            return _config_adapter.validate_python(json.load(f))
    except (OSError, RuntimeError, json.JSONDecodeError, ValidationError):
        return {}


def save_config(config: Mapping[str, str]) -> None:
    """Save CLI config to file"""
    config_file = get_config_file_path()
    Path(config_file).parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    os.chmod(config_file, 0o600)


def get_config_value(key: str) -> str | None:
    """Get a single value from the persistent CLI config"""
    return load_config().get(key)


@click.group(name="config")
def config_commands() -> None:
    """Manage persistent CLI configuration (~/.litellm/config.json)"""


@config_commands.command(name="set")
@click.argument("key")
@click.argument("value")
def set_config(key: str, value: str) -> None:
    """Set a config KEY to VALUE (e.g. `lite config set base_url https://your-proxy.example.com`)"""
    if key not in ALLOWED_CONFIG_KEYS:
        raise click.UsageError(f"Unknown config key '{key}'. Allowed keys: {', '.join(ALLOWED_CONFIG_KEYS)}")

    if key == "base_url":
        parsed = urlparse(value)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise click.UsageError("base_url must be a full http:// or https:// URL including a host")

    normalized_value = value.rstrip("/")
    save_config({**load_config(), key: normalized_value})
    click.echo(f"Set {key} = {normalized_value} in {get_config_file_path()}")


@config_commands.command(name="get")
@click.argument("key", required=False)
def get_config(key: str | None) -> None:
    """Print the value of KEY, or all stored config when KEY is omitted"""
    config = load_config()

    if key is not None:
        value = config.get(key)
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
    config = load_config()
    if key not in config:
        click.echo(f"{key} was not set")
        return

    save_config({k: v for k, v in config.items() if k != key})
    click.echo(f"Removed {key} from {get_config_file_path()}")
