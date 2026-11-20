from __future__ import annotations

import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest
import yaml
from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.process import owned_proxy_process

STARTED_WORKER: Final = "Started server process"
RELOAD_LINE: Final = "search tool(s) into router"
RELOADS_PER_WORKER: Final = 3
RELOAD_WAIT_SECONDS: Final = 60
SEARCH_TOOLS_HEADER: Final = "Proxy initialized with Search Tools:"
STARTUP_MESSAGES: Final = (
    "██╗",
    "Give Feedback / Get Help",
    "Proxy initialized with Config, Set models:",
    "Setting Cache on Proxy",
    SEARCH_TOOLS_HEADER,
    "Postgres connection budget",
)
ANSI_ESCAPE: Final = "\x1b["
WORKER_COUNTS: Final = (1, 2)
CACHE_SETTINGS: Final[Mapping[str, JsonValue]] = {"cache": True, "cache_params": {"type": "redis"}}


def _lines_with(log: str, text: str) -> int:
    return sum(text in line for line in log.splitlines())


def _write_config(
    directory: Path, upstream_url: str, model: str, search_tool: str, litellm_settings: Mapping[str, JsonValue]
) -> Path:
    config: Final = directory / f"startup_output_{uuid.uuid4().hex}.yaml"
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
                ],
                "search_tools": [
                    {
                        "search_tool_name": search_tool,
                        "litellm_params": {"search_provider": "tavily", "api_key": "integration-search-key"},
                    }
                ],
                "litellm_settings": dict(litellm_settings),
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "proxy_config_reload_interval_seconds": 1,
                },
            },
            sort_keys=False,
        )
    )
    return config


def _boot_reload_and_stop(
    gateway: Gateway,
    directory: Path,
    litellm_settings: Mapping[str, JsonValue],
    overrides: Mapping[str, str],
    workers: int,
) -> tuple[str, str]:
    stamp: Final = uuid.uuid4().hex
    model: Final = f"startup-output-{stamp}"
    search_tool: Final = f"startup-output-search-{stamp}"
    config: Final = _write_config(directory, gateway.upstream_url, model, search_tool, litellm_settings)
    with owned_proxy_process(
        gateway,
        directory,
        {"LITELLM_LOG": "INFO", **overrides},
        config=config,
        workers=workers,
    ) as owned:
        reply: Final = owned.gateway.chat(model, text=f"startup output {stamp}")
        assert reply["model"] == model, reply
        reloads: Final = eventually(
            lambda: _lines_with(owned.log.read_text(), RELOAD_LINE),
            lambda count: count >= RELOADS_PER_WORKER * workers,
            seconds=RELOAD_WAIT_SECONDS,
            return_last_on_timeout=True,
        )
        assert reloads >= RELOADS_PER_WORKER * workers, reloads
        log: Final = owned.log
    return log.read_text(), search_tool


@pytest.mark.parametrize("workers", WORKER_COUNTS)
def test_startup_messages_reach_a_piped_stdout_once_and_without_color_codes(
    gateway: Gateway, tmp_path: Path, workers: int
) -> None:
    log, search_tool = _boot_reload_and_stop(gateway, tmp_path, CACHE_SETTINGS, {}, workers)

    for message in STARTUP_MESSAGES:
        assert message in log, (message, log)
    assert _lines_with(log, SEARCH_TOOLS_HEADER) == workers, log
    assert _lines_with(log, f"{search_tool} (tavily)") == workers, log
    colored: Final = tuple(line for line in log.splitlines() if ANSI_ESCAPE in line)
    assert colored == (), "lines with ANSI color codes on a piped stdout:\n" + "\n".join(colored)


@pytest.mark.parametrize(
    ("litellm_settings", "overrides"),
    [
        pytest.param(CACHE_SETTINGS, {"LITELLM_SUPPRESS_STARTUP_OUTPUT": "true"}, id="env-plain"),
        pytest.param({**CACHE_SETTINGS, "json_logs": True, "suppress_startup_output": True}, {}, id="config-json"),
    ],
)
def test_suppress_startup_output_leaves_out_every_startup_message(
    gateway: Gateway, tmp_path: Path, litellm_settings: Mapping[str, JsonValue], overrides: Mapping[str, str]
) -> None:
    workers: Final = 2
    log, search_tool = _boot_reload_and_stop(gateway, tmp_path, litellm_settings, overrides, workers)

    assert _lines_with(log, STARTED_WORKER) == workers, log
    printed: Final = tuple(message for message in (*STARTUP_MESSAGES, search_tool) if message in log)
    assert printed == (), (printed, log)
