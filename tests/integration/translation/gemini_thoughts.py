from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, list_value, object_value
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Request
from pydantic import JsonValue

GEMINI_2_5_FLASH_DEPLOYMENT: Final[Mapping[str, JsonValue]] = {
    "model_name": "gemini/gemini-2.5-flash",
    "litellm_params": {
        "model": "gemini/gemini-2.5-flash",
        "api_base": "http://127.0.0.1:8191",
        "api_key": "synthetic-gemini-key",
    },
}
LONG_SUMMARY: Final = "detailed " * 600


def with_thinking_config(
    request: Mapping[str, JsonValue], thinking_config: Mapping[str, JsonValue]
) -> Mapping[str, JsonValue]:
    return {
        **request,
        "generationConfig": {**object_value(request["generationConfig"]), "thinkingConfig": dict(thinking_config)},
    }


def sent_thinking_config(request: Request) -> Mapping[str, JsonValue]:
    body: Final = JSON_OBJECT.validate_json(request.body)
    return object_value(object_value(body["generationConfig"])["thinkingConfig"])


@contextmanager
def gemini_thoughts_proxy(
    gateway: Gateway, directory: Path, default: JsonValue, *, workers: int = 1
) -> Iterator[OwnedProxy]:
    stock: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final = directory / "gemini-thoughts.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                **stock,
                "model_list": [*list_value(stock["model_list"]), dict(GEMINI_2_5_FLASH_DEPLOYMENT)],
                "litellm_settings": {
                    **object_value(stock["litellm_settings"]),
                    "gemini_include_thoughts_default": default,
                },
            }
        )
    )
    with owned_proxy_process(gateway, directory, {}, config=config, workers=workers) as owned:
        yield owned


@pytest.fixture(scope="module")
def thoughts_off(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    with gemini_thoughts_proxy(session_gateway, tmp_path_factory.mktemp("gemini-thoughts-off"), False) as owned:
        yield owned.gateway
