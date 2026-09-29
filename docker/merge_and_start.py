from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import cast

import yaml

DEFAULT_BASE_CONFIG = Path("/etc/litellm/config.yaml")
DEFAULT_MCP_CONFIG = Path("/app/mcp/mcp-servers.yml")


class ConfigMergeError(ValueError):
    pass


def _as_string_mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ConfigMergeError(f"{name} must be a mapping")

    raw_mapping = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in raw_mapping):
        raise ConfigMergeError(f"{name} must use string keys")

    return {key: item for key, item in raw_mapping.items() if isinstance(key, str)}


def load_config(path: Path) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as config_file:
            parsed: object = yaml.safe_load(config_file)
    except OSError as exc:
        raise ConfigMergeError(f"could not read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigMergeError(f"could not parse {path}: {exc}") from exc

    if parsed is None:
        return {}

    return _as_string_mapping(parsed, name=str(path))


def merge_configs(base_config: dict[str, object], mcp_config: dict[str, object]) -> dict[str, object]:
    base_servers = _as_string_mapping(base_config.get("mcp_servers", {}), name="base mcp_servers")
    mcp_servers = _as_string_mapping(mcp_config.get("mcp_servers", {}), name="MCP mcp_servers")
    collisions = sorted(base_servers.keys() & mcp_servers.keys())
    if collisions:
        names = ", ".join(collisions)
        raise ConfigMergeError(f"MCP server name collision: {names}")

    merged_config = dict(base_config)
    for key, value in mcp_config.items():
        if key == "mcp_servers":
            continue
        if key in merged_config and merged_config[key] != value:
            raise ConfigMergeError(f"conflicting top-level config key: {key}")
        merged_config[key] = value

    merged_config["mcp_servers"] = {**base_servers, **mcp_servers}
    return merged_config


def write_merged_config(config: dict[str, object]) -> Path:
    try:
        merged_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="litellm-merged-",
            suffix=".yaml",
            delete=False,
        )
        with merged_file:
            yaml.safe_dump(config, merged_file, sort_keys=False)
        os.chmod(merged_file.name, 0o600)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigMergeError(f"could not write merged config: {exc}") from exc

    return Path(merged_file.name)


def parse_arguments(argv: list[str] | None = None) -> tuple[Path, tuple[str, ...]]:
    parser = argparse.ArgumentParser(description="Merge the bundled MCP config before starting LiteLLM")
    parser.add_argument("--config", type=Path, default=DEFAULT_BASE_CONFIG, help="Base LiteLLM config path")
    parsed, passthrough = parser.parse_known_args(argv)
    return parsed.config, tuple(passthrough)


def build_litellm_command(merged_config: Path, passthrough: tuple[str, ...]) -> list[str]:
    return ["litellm", "--config", str(merged_config), *passthrough]


def main(argv: list[str] | None = None) -> int:
    merged_config: Path | None = None
    try:
        base_config_path, passthrough = parse_arguments(argv)
        base_config = load_config(base_config_path)
        mcp_config = load_config(DEFAULT_MCP_CONFIG)
        merged_config = write_merged_config(merge_configs(base_config, mcp_config))
        os.execvp("litellm", build_litellm_command(merged_config, passthrough))
    except (ConfigMergeError, OSError) as exc:
        if merged_config is not None:
            merged_config.unlink(missing_ok=True)
        sys.stderr.write(f"merge-and-start: {exc}\n")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
