"""JSON logging keeps every line the proxy writes during boot, serving and shutdown machine-parseable.

A deployment that turns on JSON logs ships the proxy's stdout to a log pipeline, and every line that is
not a JSON object breaks that pipeline's parser. This contract boots an owned proxy with JSON logs on,
serves a chat request on every configured model plus one rejected request, stops the proxy and
rejects any line of its output that is not one JSON record, while checking that the
startup information those lines used to carry is still present as JSON records. JSON mode is switched
on through the environment and through the config, at one and two uvicorn workers, at the INFO, the
default and the WARNING log level, and under a concurrent burst that lands on both workers.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import pytest
import yaml
from pydantic import JsonValue, TypeAdapter, ValidationError

from tests.integration._support.client import Gateway
from tests.integration._support.process import owned_proxy_process

JSON_RECORD: Final = TypeAdapter(dict[str, JsonValue])
PLAIN_LOG_LINE: Final = re.compile(r"^\d\d:\d\d:\d\d - LiteLLM")
# prisma-client-py starts the Prisma query engine on the proxy's own stdout, and the engine announces
# its build hash there in two writes (the text, then the newline), so with two workers the two
# announcements land glued together or glued to a neighbor's line. The text is not litellm's and is
# removed before the lines are read.
PRISMA_ENGINE_TEXT: Final = re.compile(r"query-engine [0-9a-f]{40}")
STARTED_WORKER: Final = "Started server process"
MODEL_LIST_HEADER: Final = "Proxy initialized with Config, Set models:"
CACHE_LINE: Final = "Setting Cache on Proxy"
# printed by the CLI process before any worker exists, so it proves the launcher itself logs JSON
CONNECTION_BUDGET_LINE: Final = "Postgres connection budget"
REDIS_AUTH_CACHE_LINE: Final = "enable_redis_auth_cache"
BANNER_FRAGMENT: Final = "██╗"
FEEDBACK_BOX_LINE: Final = "Give Feedback / Get Help"
WORKER_COUNTS: Final = (1, 2)
UNKNOWN_MODEL_STATUS: Final = 400
BURST_SIZE: Final = 16
CACHE_SETTINGS: Final[Mapping[str, JsonValue]] = {"cache": True, "cache_params": {"type": "redis"}}


def _models() -> tuple[str, str]:
    stamp: Final = uuid.uuid4().hex
    return (f"json-logs-first-{stamp}", f"json-logs-second-{stamp}")


def _write_config(
    directory: Path, upstream_url: str, models: tuple[str, ...], litellm_settings: Mapping[str, JsonValue]
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
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "disable_prisma_schema_update": True,
                },
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


def _boot_serve_and_stop(
    gateway: Gateway,
    directory: Path,
    config: Path,
    overrides: Mapping[str, str],
    workers: int,
    models: tuple[str, ...],
) -> str:
    with owned_proxy_process(gateway, directory, overrides, config=config, workers=workers, database_setup=()) as owned:
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


def _assert_every_line_is_json(log: str, workers: int, models: tuple[str, ...]) -> None:
    plaintext: Final = _non_json_lines(log)
    assert plaintext == (), "plaintext lines with JSON logs on:\n" + "\n".join(plaintext)
    messages: Final = _messages(log)
    assert sum(STARTED_WORKER in message for message in messages) == workers, messages
    assert any(MODEL_LIST_HEADER in message for message in messages), messages
    for model in models:
        assert any(model in message for message in messages), (model, messages)
    assert any(CACHE_LINE in message for message in messages), messages
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
    # cache is listed before json_logs on purpose: the cache branch logs the Redis auth-cache line
    # while the config is still being read, so JSON mode must not depend on the key order.
    config: Final = _write_config(tmp_path, gateway.upstream_url, models, {**CACHE_SETTINGS, "json_logs": True})
    log: Final = _boot_serve_and_stop(gateway, tmp_path, config, {"LITELLM_LOG": "INFO"}, workers, models)
    _assert_every_line_is_json(log, workers, models)


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
    with owned_proxy_process(
        gateway, tmp_path, {"LITELLM_LOG": "INFO"}, config=config, workers=2, database_setup=()
    ) as owned:
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
