"""The proxy must serve with every package the FIPS image drops absent."""

import json
import os
import signal
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import httpx
import psutil
import yaml
from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually, object_value, string_value
from tests.integration._support.process import OwnedProxy, group_members, owned_proxy_process

SITECUSTOMIZE: Final = """
import importlib.abc
import sys

_ROOTS = frozenset({"nacl", "granian", "pyroscope", "hf_xet", "awscrt", "_awscrt", "xmlsec", "lxml"})
_EXACT = frozenset({"litellm.rust_bridge._native"})


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in _EXACT or fullname.split(".", 1)[0] in _ROOTS:
            raise ImportError(f"FIPS footprint test blocks {fullname}")
        return None


sys.meta_path.insert(0, _Blocker())
"""


@contextmanager
def fips_proxy(gateway: Gateway, directory: Path, *, workers: int = 2) -> Iterator[OwnedProxy]:
    blockers: Final = directory / "blockers"
    blockers.mkdir()
    (blockers / "sitecustomize.py").write_text(SITECUSTOMIZE)
    config_source: Final = yaml.safe_load((Path(__file__).resolve().parents[1] / "proxy_config.yaml").read_text())
    config_source["general_settings"]["encryption_algorithm"] = "aes-256-gcm"
    config: Final = directory / "fips_config.yaml"
    config.write_text(yaml.safe_dump(config_source))
    overrides: Final = {
        "PYTHONPATH": f"{blockers}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    }
    with owned_proxy_process(gateway, directory, overrides, config=config, workers=workers) as owned:
        yield owned


def upstream_requests(gateway: Gateway) -> tuple[dict[str, JsonValue], ...]:
    response: Final = httpx.get(f"{gateway.upstream_url}/__observations", timeout=5, trust_env=False)
    response.raise_for_status()
    return tuple(object_value(value) for value in response.json()["requests"])


def chat_response(
    proxy: OwnedProxy,
    model: str,
    *,
    key: str | None = None,
    body: Mapping[str, JsonValue] | None = None,
) -> httpx.Response:
    return proxy.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": "fips footprint"}], **(body or {})},
        key=key,
    )


def test_proxy_serves_chat_completions_with_fips_dropped_packages_absent(gateway: Gateway, tmp_path: Path) -> None:
    with fips_proxy(gateway, tmp_path) as owned, owned.gateway.scenario() as scenario:
        model: Final = scenario.model()
        upstream_requests(gateway)
        response: Final = chat_response(owned, model)
        assert response.status_code == 200, response.text
        assert string_value(response.json()["id"])
        requests: Final = upstream_requests(gateway)
        assert len(requests) == 1, requests
        assert "chat/completions" in requests[0]["path"]
        log_text: Final = owned.log.read_text()
        assert "ModuleNotFoundError" not in log_text and "Traceback" not in log_text, log_text[-2000:]


def test_streaming_chat_completes_with_fips_dropped_packages_absent(gateway: Gateway, tmp_path: Path) -> None:
    with fips_proxy(gateway, tmp_path) as owned, owned.gateway.scenario() as scenario:
        model: Final = scenario.model()
        with owned.gateway.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "fips footprint"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            headers={"Authorization": f"Bearer {owned.gateway.key}"},
        ) as response:
            assert response.status_code == 200, response.read().decode()
            chunks: Final = tuple(
                line[len("data: ") :] for line in response.iter_lines() if line.startswith("data: ")
            )
        assert chunks, "stream produced no data chunks"
        assert chunks[-1] == "[DONE]", chunks
        final: Final = object_value(json.loads(chunks[-2]))
        assert final["object"] == "chat.completion.chunk" and "usage" in final, chunks[-2]


def test_key_generated_on_the_owned_proxy_serves_chat(gateway: Gateway, tmp_path: Path) -> None:
    with fips_proxy(gateway, tmp_path) as owned, owned.gateway.scenario() as scenario:
        model: Final = scenario.model()
        created: Final = owned.gateway.post("/key/generate", {})
        token: Final = string_value(created["key"])
        response: Final = chat_response(owned, model, key=token)
        assert response.status_code == 200, response.text
        assert string_value(response.json()["id"])


def test_surviving_worker_keeps_serving_after_one_worker_is_killed(gateway: Gateway, tmp_path: Path) -> None:
    with fips_proxy(gateway, tmp_path) as owned, owned.gateway.scenario() as scenario:
        model: Final = scenario.model()
        children: Final = tuple(
            process for process in group_members(owned.process.pid) if process.pid != owned.process.pid
        )
        assert len(children) >= 2, [process.pid for process in group_members(owned.process.pid)]
        victim: Final = children[0]
        victim.send_signal(signal.SIGKILL)

        def victim_dead() -> bool:
            try:
                return victim.status() in {psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD} or not victim.is_running()
            except psutil.NoSuchProcess:
                return True

        eventually(victim_dead, lambda dead: dead, seconds=10)
        with ThreadPoolExecutor(max_workers=20) as pool:
            responses: Final = tuple(pool.map(lambda _: chat_response(owned, model), range(20)))
        assert all(response.status_code == 200 for response in responses), [
            (response.status_code, response.text) for response in responses
        ]
        eventually(
            lambda: owned.gateway.client.get("/health/readiness").status_code,
            lambda status: status == 200,
            seconds=30,
        )
