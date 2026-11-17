"""Shared rig for the `test_dec1_cfgload_*` contracts (config loading, ahead of the Rust port).

The Rust gateway now loads config.yaml through its own crates: `litellm-yaml` (a port of PyYAML's
`safe_load`, #52023), `config-load` (the `include` merge and `os.environ/` resolution, #52025) and
`litellm-config` (the typed sections). Every contract here pins what the Python proxy does today with a
config the test writes, so the same file can be pointed at the Rust loader.

This module adds what `dec1_config_support` (another wave's helper, imported, never edited) lacks:

`write_yaml_text` writes YAML text exactly as given, for contracts about YAML scalar resolution that a
`yaml.safe_dump` round trip would erase (`yes`, `0755`, anchors, merge keys, tags, repeated keys).

`refused_with_exit` boots a proxy that must fail and returns its exit code and log; it is
`refused_boot_log` from `_support/process.py` with the exit code kept. It uses that module's private
launch helpers, which is the assumption to check first if it breaks.

`validate_config` runs (with no gateway) the proxy CLI's `--validate_config` one-shot command, which loads the config the
way the server does (`ProxyConfig.load_config`) and exits 0 with `LiteLLM: config OK (<n> models)` or
exits 1 with `LiteLLM: config validation failed: <error>` (`litellm/proxy/proxy_cli.py`,
`run_config_validation`). It needs no database and no listening socket.

`s3_bucket` is a diverted object store: a loopback HTTP peer that answers S3 path-style
`GET /<bucket>/<key>` from a dict, reached by pointing boto3 at it with `AWS_ENDPOINT_URL_S3`. No real
cloud service is contacted.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import unquote, urlsplit

import httpx

from tests.integration._support.budget import readiness_seconds
from tests.integration._support.client import Gateway
from tests.integration._support.process import (
    DB_PUSH,
    _is_ready,
    _launch,
    _owns_listener,
    _proxy_environment,
    _proxy_root,
    _stop_launch,
)
from tests.integration._support.wire import Reply, Request, Wire, wire_server

NO_ENVIRONMENT: Final[Mapping[str, str]] = MappingProxyType({})
BUCKET: Final = "dec1-cfgload-bucket"
FAKE_AWS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "AWS_ACCESS_KEY_ID": "AKIADEC1CFGLOADFAKE00",
        "AWS_SECRET_ACCESS_KEY": "dec1-cfgload-fake-secret-access-key-000000",
        "AWS_DEFAULT_REGION": "us-east-1",
        "AWS_REGION_NAME": "us-east-1",
    }
)
VALIDATION_OK_PREFIX: Final = "LiteLLM: config OK ("
VALIDATION_FAILED_PREFIX: Final = "LiteLLM: config validation failed: "


def write_yaml_text(directory: Path, text: str, name: str = "dec1-cfgload.yaml") -> Path:
    """Write `text` byte for byte (UTF-8), so YAML 1.1 spellings survive to the loader."""
    directory.mkdir(parents=True, exist_ok=True)
    path: Final = directory / name
    path.write_text(text, encoding="utf-8")
    return path


@dataclass(frozen=True, slots=True)
class Refusal:
    exit_code: int
    log: str


def refused_with_exit(
    gateway: Gateway,
    directory: Path,
    config: Path,
    environment: Mapping[str, str] = NO_ENVIRONMENT,
    *,
    remove_environment: tuple[str, ...] = (),
) -> Refusal:
    """Start the proxy on `config` and return its exit code and log once it exits without becoming ready."""
    command: Final = (
        sys.executable,
        "-m",
        "integration._support.proxy",
        "--config",
        str(config),
        "--host",
        "127.0.0.1",
        "--num_workers",
        "1",
        *DB_PUSH,
    )
    output: Final = Path(os.environ.get("INTEGRATION_RESULTS_DIR", str(directory)))
    output.mkdir(parents=True, exist_ok=True)
    launch: Final = _launch(
        command, _proxy_root(), _proxy_environment(gateway, environment, remove_environment), output
    )
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{launch.port}", timeout=15, trust_env=False) as client:
            deadline: Final = time.monotonic() + readiness_seconds(1)
            while launch.process.poll() is None:
                assert not (_owns_listener(launch) and _is_ready(client)), (
                    f"Proxy became ready instead of refusing to boot:\n{launch.log.read_text()}"
                )
                assert time.monotonic() < deadline, "Proxy neither exited nor became ready within the deadline"
                time.sleep(0.1)  # test-quality-ok: polling a child process, same loop as refused_boot_log
        return Refusal(launch.process.returncode, launch.log.read_text())
    finally:
        _stop_launch(launch)


@dataclass(frozen=True, slots=True)
class Validation:
    exit_code: int
    stdout: str
    stderr: str


CLI_MASTER_KEY: Final = "sk-dec1-cfgload-cli-master"


def validate_config(
    config: Path | None,
    environment: Mapping[str, str] = NO_ENVIRONMENT,
    *,
    remove_environment: tuple[str, ...] = (),
) -> Validation:
    """Run `litellm --validate_config [--config <config>]` to completion in the checkout under test.

    It needs no proxy, database or socket, so it takes no gateway fixture: the environment is this
    process's minus `remove_environment` and `DATABASE_URL`, with dotenv loading off and a fixed master key.
    """
    inherited: Final = {
        name: value
        for name, value in os.environ.items()
        if name not in remove_environment and name not in ("DATABASE_URL", "DATABASE_URL_READ_REPLICA")
    }
    completed: Final = subprocess.run(  # test-quality-ok: the CLI one-shot command is the surface under test
        (
            sys.executable,
            "-c",
            "from litellm import run_server; run_server()",
            "--validate_config",
            *(() if config is None else ("--config", str(config))),
        ),
        cwd=_proxy_root(),
        env={**inherited, "PYTHON_DOTENV_DISABLED": "1", "LITELLM_MASTER_KEY": CLI_MASTER_KEY, **environment},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    return Validation(completed.returncode, completed.stdout, completed.stderr)


def _no_such_key(key: str) -> Reply:
    return Reply(
        status=404,
        content_type="application/xml",
        body=(
            "<?xml version='1.0' encoding='UTF-8'?><Error><Code>NoSuchKey</Code>"
            f"<Message>The specified key does not exist.</Message><Key>{key}</Key></Error>"
        ).encode(),
    )


def bucket_object_key(request: Request) -> str | None:
    """The object key of a path-style S3 `GET /<bucket>/<key>`, or None for anything else."""
    path: Final = unquote(urlsplit(request.target).path)
    prefix: Final = f"/{BUCKET}/"
    if request.method != "GET" or not path.startswith(prefix):
        return None
    return path.removeprefix(prefix)


@contextmanager
def s3_bucket(objects: Mapping[str, bytes]) -> Iterator[Wire]:
    """A path-style S3 endpoint on loopback holding `objects` (key to body) in bucket `BUCKET`."""

    def respond(request: Request) -> Reply:
        key: Final = bucket_object_key(request)
        if key is None or key not in objects:
            return _no_such_key(key or "")
        return Reply(body=objects[key], content_type="application/x-yaml")

    with wire_server(respond) as wire:
        yield wire


def s3_environment(wire: Wire, object_key: str, **extra: str) -> Mapping[str, str]:
    """The `LITELLM_CONFIG_BUCKET_*` variables naming `object_key` in the diverted bucket, plus fake AWS
    credentials and the endpoint override boto3 reads (`AWS_ENDPOINT_URL_S3`, botocore >= 1.31)."""
    return MappingProxyType(
        {
            **FAKE_AWS,
            "AWS_ENDPOINT_URL_S3": wire.url,
            "LITELLM_CONFIG_BUCKET_TYPE": "s3",
            "LITELLM_CONFIG_BUCKET_NAME": BUCKET,
            "LITELLM_CONFIG_BUCKET_OBJECT_KEY": object_key,
            **extra,
        }
    )


def requested_keys(requests: tuple[Request, ...]) -> tuple[str, ...]:
    """The object keys the proxy asked the diverted bucket for, in order."""
    return tuple(key for key in map(bucket_object_key, requests) if key is not None)


def nested_mapping(depth: int, leaf: Mapping[str, object]) -> dict[str, object]:
    """`leaf` wrapped in `depth` levels of `{"k": ...}`."""
    built: dict[str, object] = dict(leaf)
    for _ in range(depth):
        built = {"k": built}
    return built


def leaf_of(value: object) -> object:
    """The innermost value of a `nested_mapping` read back as JSON."""
    current: object = value
    while isinstance(current, dict) and "k" in current:
        current = current["k"]
    return current["leaf"] if isinstance(current, dict) else current


Responder = Callable[[Request], Reply]
