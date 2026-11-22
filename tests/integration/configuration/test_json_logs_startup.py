"""JSON logging keeps every line the proxy writes during boot, serving and shutdown machine-parseable.

A deployment that turns on JSON logs ships the proxy's stdout to a log pipeline, and every line that is
not a JSON object breaks that pipeline's parser. This contract boots an owned proxy with JSON logs on,
serves a chat request on every configured model plus one rejected request, stops the proxy and
rejects any line of its output that is not one JSON record, while checking that the
startup information those lines used to carry is still present as JSON records. JSON mode is switched
on through the environment and through the config, at one and two uvicorn workers, at the INFO, the
default and the WARNING log level, under a concurrent burst that lands on both workers, under an operator
--log_config that names the proxy logger or gives it a handler of its own, and with a policy list longer
than the stdout cap. The model list keeps a model name the secret patterns would match. The deprecated
--enforce_prisma_migration_check notice is one JSON record under a config-only json_logs, and a boot whose
config cannot be loaded still prints it before failing. A success callback that raises in a destructor, raises in a
thread and drops an open aiohttp session leaves one JSON record per traceback, whether JSON logs came from the
environment or the config.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import uuid
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import pytest
import yaml
from pydantic import JsonValue, TypeAdapter, ValidationError

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.process import OwnedProxy, owned_proxy_process, private_state, refused_boot_log

JSON_RECORD: Final = TypeAdapter(dict[str, JsonValue])
PLAIN_LOG_LINE: Final = re.compile(r"^\d\d:\d\d:\d\d - LiteLLM")
PRISMA_ENGINE_TEXT: Final = re.compile(r"query-engine [0-9a-f]{40}")
STARTED_WORKER: Final = "Started server process"
MODEL_LIST_HEADER: Final = "Proxy initialized with Config, Set models:"
CACHE_LINE: Final = "Setting Cache on Proxy"
CONNECTION_BUDGET_LINE: Final = "Postgres connection budget"
REDIS_AUTH_CACHE_LINE: Final = "enable_redis_auth_cache"
BANNER_FRAGMENT: Final = "██╗"
FEEDBACK_BOX_LINE: Final = "Give Feedback / Get Help"
WORKER_COUNTS: Final = (1, 2)
UNKNOWN_MODEL_STATUS: Final = 400
BURST_SIZE: Final = 16
CACHE_SETTINGS: Final[Mapping[str, JsonValue]] = {"cache": True, "cache_params": {"type": "redis"}}
STDOUT_CAP: Final = 4096
POLICY_COUNT: Final = 60
POLICY_LIST_HEADER: Final = "LiteLLM Policy Engine: Loaded"
TRUNCATION_MARKER: Final = "litellm_truncated"
DEPRECATED_FLAG: Final = "--enforce_prisma_migration_check"
DEPRECATED_FLAG_NOTICE: Final = "--enforce_prisma_migration_check is deprecated and has no effect"
UNPARSEABLE_CONFIG: Final = "model_list: [\n  - bad: yaml: here\n"
OPERATOR_LOG_CONFIG: Final[Mapping[str, JsonValue]] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"json": {"()": "litellm._logging.JsonFormatter"}},
    "handlers": {"stdout": {"class": "logging.StreamHandler", "stream": "ext://sys.stdout", "formatter": "json"}},
    "loggers": {"LiteLLM Proxy": {"level": "INFO"}},
    "root": {"level": "INFO", "handlers": ["stdout"]},
}
OPERATOR_SILENCING_LOG_CONFIG: Final[Mapping[str, JsonValue]] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"json": {"()": "litellm._logging.JsonFormatter"}},
    "handlers": {
        "stdout": {"class": "logging.StreamHandler", "stream": "ext://sys.stdout", "formatter": "json"},
        "silence": {"class": "logging.NullHandler"},
    },
    "loggers": {"LiteLLM Proxy": {"level": "INFO", "handlers": ["silence"], "propagate": False}},
    "root": {"level": "INFO", "handlers": ["stdout"]},
}


NOISY_CALLBACK_MODULE: Final = "noisy_callback"
NOISY_CALLBACK_SOURCE: Final = """
import threading

import aiohttp

from litellm.integrations.custom_logger import CustomLogger


class _FailsInDestructor:
    def __del__(self):
        raise RuntimeError("callback resource cleanup failed")


def _fails_in_background():
    raise ValueError("callback background job failed")


class NoisyCallback(CustomLogger):
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        aiohttp.ClientSession()
        threading.Thread(target=_fails_in_background, name="callback-background-job").start()
        _FailsInDestructor()


proxy_handler_instance = NoisyCallback()
"""
DESTRUCTOR_TRACEBACK_TAIL: Final = "RuntimeError: callback resource cleanup failed"
THREAD_RECORD_MESSAGE: Final = "Exception in thread callback-background-job"
THREAD_TRACEBACK_TAIL: Final = "ValueError: callback background job failed"
UNCLOSED_SESSION_MESSAGE: Final = "Unclosed client session"
CALLBACK_RECORD_SECONDS: Final = 30


def _models() -> tuple[str, str]:
    stamp: Final = uuid.uuid4().hex
    return (f"json-logs-first-{stamp}", f"json-logs-second-{stamp}")


def _policies() -> dict[str, JsonValue]:
    stamp: Final = uuid.uuid4().hex
    return {
        f"startup-listing-policy-{n:02d}-{stamp}": {
            "description": f"policy {n:02d} of a listing that runs past the stdout cap meant for request payloads"
        }
        for n in range(POLICY_COUNT)
    }


def _write_config(
    directory: Path,
    upstream_url: str,
    models: tuple[str, ...],
    litellm_settings: Mapping[str, JsonValue],
    extra_sections: Mapping[str, JsonValue] | None = None,
) -> Path:
    config: Final = directory / f"json_logs_{uuid.uuid4().hex}.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model_list": [
                    {
                        "model_name": model,
                        "litellm_params": {
                            "model": f"openai/{model}",
                            "api_base": f"{upstream_url}/v1",
                            "api_key": "integration-provider-key",
                        },
                    }
                    for model in models
                ],
                "litellm_settings": dict(litellm_settings),
                "router_settings": {"num_retries": 0},
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "disable_prisma_schema_update": True,
                },
                **(extra_sections or {}),
            },
            sort_keys=False,
        )
    )
    return config


def _record(line: str) -> dict[str, JsonValue] | None:
    stripped: Final = line.strip()
    if not stripped.startswith("{"):
        return None
    try:
        return JSON_RECORD.validate_json(stripped)
    except ValidationError:
        return None


def _lines(log: str) -> tuple[str, ...]:
    return tuple(line for line in PRISMA_ENGINE_TEXT.sub("", log).splitlines() if line.strip())


def _messages(log: str) -> tuple[str, ...]:
    records: Final = tuple(record for record in (_record(line) for line in _lines(log)) if record is not None)
    return tuple(str(record["message"]) for record in records if "message" in record)


def _non_json_lines(log: str) -> tuple[str, ...]:
    return tuple(line for line in _lines(log) if _record(line) is None)


@contextmanager
def _initialized_proxy(
    gateway: Gateway,
    directory: Path,
    config: Path,
    overrides: Mapping[str, str],
    workers: int,
    extra_arguments: tuple[str, ...] = (),
) -> Iterator[OwnedProxy]:
    with private_state(directory, os.environ) as state:
        prepared: Final = subprocess.run(
            [
                sys.executable,
                "-I",
                "-m",
                "prisma",
                "db",
                "push",
                "--schema",
                "litellm/proxy/schema.prisma",
                "--skip-generate",
            ],
            env={**os.environ, **state.environment},
            capture_output=True,
            text=True,
            check=False,
        )
        assert prepared.returncode == 0, prepared.stdout + prepared.stderr
        with owned_proxy_process(
            gateway,
            directory,
            {**state.environment, **overrides},
            config=config,
            workers=workers,
            database_setup=(),
            extra_arguments=extra_arguments,
        ) as owned:
            yield owned


def _boot_serve_and_stop(
    gateway: Gateway,
    directory: Path,
    config: Path,
    overrides: Mapping[str, str],
    workers: int,
    models: tuple[str, ...],
    extra_arguments: tuple[str, ...] = (),
) -> str:
    with _initialized_proxy(gateway, directory, config, overrides, workers, extra_arguments) as owned:
        _serve(owned.gateway, models)
        log: Final = owned.log
    return log.read_text()


def _serve(gateway: Gateway, models: tuple[str, ...]) -> None:
    """One chat per model and one rejected request, so the log holds the request-time lines of served
    and failed requests next to the boot lines."""
    for model in models:
        reply: Final = gateway.chat(model, text=f"json logs {uuid.uuid4().hex}")
        assert reply["model"] == model, reply
    rejected: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": f"missing-{uuid.uuid4().hex}", "messages": [{"role": "user", "content": "json logs"}]},
    )
    assert rejected.status_code == UNKNOWN_MODEL_STATUS, rejected.text


def _assert_every_line_is_json(
    log: str, workers: int, models: tuple[str, ...], *, redis_auth_cache_line: bool = True
) -> None:
    plaintext: Final = _non_json_lines(log)
    assert plaintext == (), "plaintext lines with JSON logs on:\n" + "\n".join(plaintext)
    messages: Final = _messages(log)
    assert sum(STARTED_WORKER in message for message in messages) == workers, messages
    assert any(MODEL_LIST_HEADER in message for message in messages), messages
    for model in models:
        assert any(model in message for message in messages), (model, messages)
    assert any(CACHE_LINE in message for message in messages), messages
    if redis_auth_cache_line:
        assert any(REDIS_AUTH_CACHE_LINE in message for message in messages), messages
    assert any(CONNECTION_BUDGET_LINE in message for message in messages), messages
    assert sum(BANNER_FRAGMENT in message for message in messages) == workers, messages
    assert sum(FEEDBACK_BOX_LINE in message for message in messages) == workers, messages


@pytest.mark.parametrize("workers", WORKER_COUNTS)
def test_json_logs_env_keeps_every_line_json(gateway: Gateway, tmp_path: Path, workers: int) -> None:
    models: Final = _models()
    config: Final = _write_config(tmp_path, gateway.upstream_url, models, CACHE_SETTINGS)
    log: Final = _boot_serve_and_stop(
        gateway, tmp_path, config, {"JSON_LOGS": "true", "LITELLM_LOG": "INFO"}, workers, models
    )
    _assert_every_line_is_json(log, workers, models)


@pytest.mark.parametrize("workers", WORKER_COUNTS)
def test_json_logs_config_keeps_every_line_json(gateway: Gateway, tmp_path: Path, workers: int) -> None:
    models: Final = _models()
    config: Final = _write_config(tmp_path, gateway.upstream_url, models, {**CACHE_SETTINGS, "json_logs": True})
    log: Final = _boot_serve_and_stop(gateway, tmp_path, config, {"LITELLM_LOG": "INFO"}, workers, models)
    _assert_every_line_is_json(log, workers, models)


@pytest.mark.parametrize("workers", WORKER_COUNTS)
def test_json_logs_with_operator_log_config_keeps_every_line_json(
    gateway: Gateway, tmp_path: Path, workers: int
) -> None:
    """uvicorn applies an operator `--log_config` after the CLI process switched JSON on and before
    `initialize()` switches it on again through the config; a config that names the proxy logger leaves it
    without handlers in that window, where the banner and the feedback box are emitted. They still arrive,
    as JSON, one per worker, and the operator's own formatter keeps uvicorn's lines JSON. The Redis
    auth-cache line is a regular logger line written in that same window; this log_config removed the
    proxy logger's handlers and left propagation off, so it is lost at one worker on main and here alike."""
    models: Final = _models()
    config: Final = _write_config(tmp_path, gateway.upstream_url, models, {**CACHE_SETTINGS, "json_logs": True})
    log_config: Final = tmp_path / "operator_log_config.yaml"
    log_config.write_text(yaml.safe_dump(dict(OPERATOR_LOG_CONFIG)))
    log: Final = _boot_serve_and_stop(
        gateway, tmp_path, config, {"LITELLM_LOG": "INFO"}, workers, models, ("--log_config", str(log_config))
    )
    _assert_every_line_is_json(log, workers, models, redis_auth_cache_line=False)


@pytest.mark.parametrize("workers", WORKER_COUNTS)
def test_json_logs_with_operator_handler_on_the_proxy_logger_keeps_the_startup_notices(
    gateway: Gateway, tmp_path: Path, workers: int
) -> None:
    """An operator `--log_config` that gives the proxy logger a handler of its own, here a NullHandler that
    silences it, is in force in the window where the banner and the feedback box are emitted. The prints
    they replace wrote to stdout whatever the logging config said, so both still arrive there as JSON, one
    per worker. The Redis auth-cache line is a regular logger line written in that same window and goes
    to the operator's handler, so it is silenced on main and here alike."""
    models: Final = _models()
    config: Final = _write_config(tmp_path, gateway.upstream_url, models, {**CACHE_SETTINGS, "json_logs": True})
    log_config: Final = tmp_path / "operator_silencing_log_config.yaml"
    log_config.write_text(yaml.safe_dump(dict(OPERATOR_SILENCING_LOG_CONFIG)))
    log: Final = _boot_serve_and_stop(
        gateway, tmp_path, config, {"LITELLM_LOG": "INFO"}, workers, models, ("--log_config", str(log_config))
    )
    _assert_every_line_is_json(log, workers, models, redis_auth_cache_line=False)


def test_json_logs_keeps_a_startup_listing_longer_than_the_stdout_cap_whole(gateway: Gateway, tmp_path: Path) -> None:
    """A startup listing longer than MAX_STRING_LENGTH_STDOUT_LOG reaches the pipeline whole. The policy list
    stands in for every such listing (the required migration SQL on a drifted database is the other one):
    the print it replaces showed it whole, and the stdout cap exists for echoed request payloads, so the JSON
    record must not be middle-truncated."""
    models: Final = _models()
    policies: Final = _policies()
    config: Final = _write_config(
        tmp_path, gateway.upstream_url, models, {**CACHE_SETTINGS, "json_logs": True}, {"policies": policies}
    )

    log: Final = _boot_serve_and_stop(gateway, tmp_path, config, {"LITELLM_LOG": "INFO"}, 1, models)

    _assert_every_line_is_json(log, 1, models)
    listings: Final = tuple(message for message in _messages(log) if POLICY_LIST_HEADER in message)
    assert len(listings) == 1, listings
    assert len(listings[0]) > STDOUT_CAP, len(listings[0])
    assert TRUNCATION_MARKER not in listings[0], listings[0]
    for policy in policies:
        assert policy in listings[0], (policy, listings[0])


def test_json_logs_model_list_keeps_a_model_name_the_secret_patterns_would_match(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The model list printed every model name as configured. The sk- key pattern matches inside a name such as
    `risk-assessment-...`, so the JSON record must carry that name as the print did, not `riREDACTED`."""
    models: Final = (f"risk-assessment-{uuid.uuid4().hex}",)
    config: Final = _write_config(tmp_path, gateway.upstream_url, models, {**CACHE_SETTINGS, "json_logs": True})

    log: Final = _boot_serve_and_stop(gateway, tmp_path, config, {"LITELLM_LOG": "INFO"}, 1, models)

    _assert_every_line_is_json(log, 1, models)
    assert f"    {models[0]}" in _messages(log), _messages(log)


def test_json_logs_default_level_keeps_every_line_json(gateway: Gateway, tmp_path: Path) -> None:
    """Without LITELLM_LOG the proxy loggers sit at WARNING; the startup notices a plain-mode operator
    always sees still arrive, as JSON, and nothing arrives as plaintext."""
    models: Final = _models()
    config: Final = _write_config(tmp_path, gateway.upstream_url, models, {**CACHE_SETTINGS, "json_logs": True})
    log: Final = _boot_serve_and_stop(gateway, tmp_path, config, {}, 1, models)
    plaintext: Final = _non_json_lines(log)
    assert plaintext == (), "plaintext lines with JSON logs on:\n" + "\n".join(plaintext)
    messages: Final = _messages(log)
    assert sum(STARTED_WORKER in message for message in messages) == 1, messages
    for fragment in (
        BANNER_FRAGMENT,
        FEEDBACK_BOX_LINE,
        MODEL_LIST_HEADER,
        CACHE_LINE,
        CONNECTION_BUDGET_LINE,
        *models,
    ):
        assert any(fragment in message for message in messages), (fragment, messages)


def test_json_logs_warning_level_keeps_startup_notices(gateway: Gateway, tmp_path: Path) -> None:
    """`LITELLM_LOG=WARNING` drops the INFO logger lines; the startup notices a plain-mode operator sees at
    every level still arrive, as JSON, and nothing arrives as plaintext."""
    models: Final = _models()
    config: Final = _write_config(tmp_path, gateway.upstream_url, models, {**CACHE_SETTINGS, "json_logs": True})
    log: Final = _boot_serve_and_stop(gateway, tmp_path, config, {"LITELLM_LOG": "WARNING"}, 1, models)
    plaintext: Final = _non_json_lines(log)
    assert plaintext == (), "plaintext lines with JSON logs on:\n" + "\n".join(plaintext)
    messages: Final = _messages(log)
    assert not any(REDIS_AUTH_CACHE_LINE in message for message in messages), messages
    for fragment in (
        BANNER_FRAGMENT,
        FEEDBACK_BOX_LINE,
        MODEL_LIST_HEADER,
        CACHE_LINE,
        CONNECTION_BUDGET_LINE,
        *models,
    ):
        assert any(fragment in message for message in messages), (fragment, messages)


def test_json_logs_burst_on_two_workers_keeps_every_line_json(gateway: Gateway, tmp_path: Path) -> None:
    """Concurrent requests spread over both workers interleave their log writes; every line must still be
    one complete JSON record."""
    models: Final = _models()
    config: Final = _write_config(tmp_path, gateway.upstream_url, models, {**CACHE_SETTINGS, "json_logs": True})
    with _initialized_proxy(gateway, tmp_path, config, {"LITELLM_LOG": "INFO"}, 2) as owned:
        with ThreadPoolExecutor(max_workers=BURST_SIZE) as pool:
            replies: Final = tuple(
                pool.map(lambda index: owned.gateway.chat(models[index % 2], text=f"burst {index}"), range(BURST_SIZE))
            )
        assert tuple(reply["model"] for reply in replies) == tuple(models[index % 2] for index in range(BURST_SIZE))
        log: Final = owned.log
    text: Final = log.read_text()
    plaintext: Final = _non_json_lines(text)
    assert plaintext == (), "plaintext lines with JSON logs on:\n" + "\n".join(plaintext)
    assert sum(STARTED_WORKER in message for message in _messages(text)) == 2, _messages(text)


def test_json_logs_config_makes_the_deprecated_flag_notice_one_json_record(gateway: Gateway, tmp_path: Path) -> None:
    """The notice waits until the config is read, so a json_logs set only there makes it a JSON record."""
    models: Final = _models()
    config: Final = _write_config(tmp_path, gateway.upstream_url, models, {"json_logs": True})
    log: Final = _boot_serve_and_stop(gateway, tmp_path, config, {"LITELLM_LOG": "INFO"}, 1, models, (DEPRECATED_FLAG,))
    plaintext: Final = _non_json_lines(log)
    assert plaintext == (), "plaintext lines with JSON logs on:\n" + "\n".join(plaintext)
    assert sum(DEPRECATED_FLAG_NOTICE in message for message in _messages(log)) == 1, _messages(log)


@pytest.mark.parametrize("config_text", (None, UNPARSEABLE_CONFIG), ids=("missing", "unparseable"))
def test_a_config_that_fails_to_load_still_prints_the_deprecated_flag_notice(
    gateway: Gateway, tmp_path: Path, config_text: str | None
) -> None:
    """A boot with the deprecated flag and a config file that does not exist or does not parse exits
    non-zero, and it still prints the notice once before the error, as it did before the notice waited
    for the config."""
    config: Final = tmp_path / f"unloadable_{uuid.uuid4().hex}.yaml"
    if config_text is not None:
        config.write_text(config_text)
    log: Final = refused_boot_log(gateway, tmp_path, {}, config=config, extra_arguments=(DEPRECATED_FLAG,))
    assert log.count(DEPRECATED_FLAG_NOTICE) == 1, log


def test_plain_logs_control_prints_startup_text(gateway: Gateway, tmp_path: Path) -> None:
    models: Final = _models()
    config: Final = _write_config(tmp_path, gateway.upstream_url, models, CACHE_SETTINGS)
    log: Final = _boot_serve_and_stop(gateway, tmp_path, config, {"LITELLM_LOG": "INFO"}, 1, models)
    plaintext: Final = _non_json_lines(log)
    assert any(BANNER_FRAGMENT in line for line in plaintext), log
    assert any(FEEDBACK_BOX_LINE in line for line in plaintext), log
    assert any(CACHE_LINE in line for line in plaintext), log
    assert any(MODEL_LIST_HEADER in line for line in plaintext), log
    for model in models:
        assert any(line.strip().endswith(model) or line.strip().endswith(f"{model}\x1b[0m") for line in plaintext), (
            model,
            log,
        )
    assert any(PLAIN_LOG_LINE.match(line) and REDIS_AUTH_CACHE_LINE in line for line in plaintext), log
    assert any(line.startswith(f"LiteLLM Proxy: {CONNECTION_BUDGET_LINE}") for line in plaintext), log


def _records(log: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(record for record in (_record(line) for line in _lines(log)) if record is not None)


def _callback_records(log: str) -> tuple[tuple[dict[str, JsonValue], ...], ...]:
    records: Final = _records(log)
    return (
        tuple(record for record in records if str(record.get("message", "")).startswith("Exception ignored")),
        tuple(record for record in records if record.get("message") == THREAD_RECORD_MESSAGE),
        tuple(record for record in records if str(record.get("message", "")).startswith(UNCLOSED_SESSION_MESSAGE)),
    )


@pytest.mark.parametrize(
    ("overrides", "settings", "workers"),
    [
        pytest.param({"JSON_LOGS": "true"}, {}, 1, id="env-one-worker"),
        pytest.param({"JSON_LOGS": "true"}, {}, 2, id="env-two-workers"),
        pytest.param({}, {"json_logs": True}, 2, id="config-two-workers"),
    ],
)
def test_json_logs_turn_every_callback_traceback_into_one_record(
    gateway: Gateway,
    tmp_path: Path,
    overrides: Mapping[str, str],
    settings: Mapping[str, JsonValue],
    workers: int,
) -> None:
    """Python prints an exception raised in a destructor or in a thread, and asyncio prints an unclosed aiohttp
    session, as plain multi-line text on stderr, and a log shipper files every line of it as its own error entry.
    With JSON logs on, a served request whose success callback does all three leaves exactly one JSON record for
    each, with the traceback inside it, and no plaintext line."""
    models: Final = _models()[:1]
    (tmp_path / f"{NOISY_CALLBACK_MODULE}.py").write_text(NOISY_CALLBACK_SOURCE)
    config: Final = _write_config(
        tmp_path,
        gateway.upstream_url,
        models,
        {**settings, "callbacks": f"{NOISY_CALLBACK_MODULE}.proxy_handler_instance"},
    )

    with _initialized_proxy(gateway, tmp_path, config, {**overrides, "LITELLM_LOG": "WARNING"}, workers) as owned:
        reply: Final = owned.gateway.chat(models[0], text=f"json logs {uuid.uuid4().hex}")
        assert reply["model"] == models[0], reply
        eventually(
            lambda: _callback_records(owned.log.read_text()),
            lambda found: all(found),
            seconds=CALLBACK_RECORD_SECONDS,
            return_last_on_timeout=True,
        )
        log_path: Final = owned.log
    log: Final = log_path.read_text()

    plaintext: Final = _non_json_lines(log)
    assert plaintext == (), "plaintext lines with JSON logs on:\n" + "\n".join(plaintext)
    destructor, thread, session = _callback_records(log)
    assert [str(record["stacktrace"]).rstrip().endswith(DESTRUCTOR_TRACEBACK_TAIL) for record in destructor] == [True]
    assert "_FailsInDestructor.__del__" in str(destructor[0]["message"])
    assert [str(record["stacktrace"]).rstrip().endswith(THREAD_TRACEBACK_TAIL) for record in thread] == [True]
    assert [record["level"] for record in (*destructor, *thread, *session)] == ["ERROR", "ERROR", "ERROR"]
