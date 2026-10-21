import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, gateway_from_environment
from integration._support.forward_proxy import ForwardProxy, refusing_forward_proxy
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration.providers._api_base_trailing_slash_support import KEY, new_marker
from pydantic import JsonValue

_AZURE_HOST: Final = "anthropic-prod.services.ai.azure.com"
_EMBEDDINGS: Final = "host-embeddings"
_RERANK: Final = "host-rerank"
_AZURE_CLAUDE: Final = "host-azure-claude"
_AZURE_COUNT: Final = "host-azure-count"
_EMPTY_BASE: Final = "host-empty-base"
_SCRUBBED_PREFIXES: Final = ("ANTHROPIC_", "AZURE_AI_", "INFINITY_")
_PROXY_VARIABLES: Final = frozenset({"HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY"})
_load_yaml: Final[Callable[[str], object]] = yaml.safe_load

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)


def _deployment(name: str, model: str, api_base: str) -> dict[str, JsonValue]:
    return {"model_name": name, "litellm_params": {"model": model, "api_base": api_base, "api_key": KEY}}


def _config(directory: Path) -> Path:
    base: Final = JSON_OBJECT.validate_python(_load_yaml(Path("tests/integration/proxy_config.yaml").read_text()))
    configuration: Final = {
        **base,
        "model_list": [
            _deployment(_EMBEDDINGS, "infinity/BAAI/bge-small-en-v1.5", "http://embeddings"),
            _deployment(_RERANK, "infinity/BAAI/bge-reranker-base", "http://rerank"),
            _deployment(_AZURE_CLAUDE, "azure_ai/claude-haiku-5-5", f"http://{_AZURE_HOST}/anthropic"),
            _deployment(_AZURE_COUNT, "azure_ai/claude-haiku-5-5", f"http://{_AZURE_HOST}"),
            _deployment(_EMPTY_BASE, "anthropic/claude-haiku-5-5", ""),
        ],
        "router_settings": {"disable_cooldowns": True, "num_retries": 0},
    }
    path: Final = directory / "host-named-like-the-route.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


def _scrubbed() -> tuple[str, ...]:
    return tuple(
        name
        for name in os.environ
        if name.upper() in _PROXY_VARIABLES or name.upper().startswith(_SCRUBBED_PREFIXES)
    )


@pytest.fixture(scope="module")
def forward() -> Iterator[ForwardProxy]:
    with refusing_forward_proxy() as proxy:
        yield proxy


@pytest.fixture(scope="module")
def isolated(forward: ForwardProxy, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("url-join-host")
    with gateway_from_environment() as shared, owned_proxy(
        shared,
        directory,
        {
            "HTTP_PROXY": forward.url,
            "HTTPS_PROXY": forward.url,
            "NO_PROXY": "127.0.0.1,localhost",
            "PROXY_BATCH_POLLING_ENABLED": "false",
        },
        config=_config(directory),
        remove_environment=_scrubbed(),
    ) as candidate:
        yield candidate


def _targets_after(forward: ForwardProxy, call: Callable[[], object]) -> frozenset[str]:
    forward.drain()
    call()
    targets: Final = frozenset(forward.targets())
    assert targets, "the forward proxy saw no outbound request"
    return targets


def _chat(isolated: Gateway, model: str) -> object:
    return isolated.request(
        "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": new_marker()}]}
    )


def _messages(isolated: Gateway, model: str, path: str) -> object:
    return isolated.request(
        "POST",
        path,
        {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": new_marker()}]},
        headers={"anthropic-version": "2023-06-01"},
    )


def test_embedding_host_named_embeddings_gets_the_route_appended(isolated: Gateway, forward: ForwardProxy) -> None:
    targets: Final = _targets_after(
        forward, lambda: isolated.request("POST", "/v1/embeddings", {"model": _EMBEDDINGS, "input": new_marker()})
    )
    assert targets == {"http://embeddings/embeddings"}


def test_rerank_host_named_rerank_gets_the_route_appended(isolated: Gateway, forward: ForwardProxy) -> None:
    body: Final[dict[str, JsonValue]] = {"model": _RERANK, "query": new_marker(), "documents": ["only document"]}
    targets: Final = _targets_after(forward, lambda: isolated.request("POST", "/v1/rerank", body))
    assert targets == {"http://rerank/rerank"}


def test_azure_claude_chat_on_a_host_starting_with_anthropic_keeps_the_host(
    isolated: Gateway, forward: ForwardProxy
) -> None:
    targets: Final = _targets_after(forward, lambda: _chat(isolated, _AZURE_CLAUDE))
    assert targets == {f"http://{_AZURE_HOST}/anthropic/v1/messages"}


def test_azure_claude_messages_on_a_host_starting_with_anthropic_keeps_the_host(
    isolated: Gateway, forward: ForwardProxy
) -> None:
    targets: Final = _targets_after(forward, lambda: _messages(isolated, _AZURE_CLAUDE, "/v1/messages"))
    assert targets == {f"http://{_AZURE_HOST}/anthropic/v1/messages"}


def test_azure_claude_count_tokens_on_a_bare_host_starting_with_anthropic_adds_the_anthropic_path(
    isolated: Gateway, forward: ForwardProxy
) -> None:
    targets: Final = _targets_after(forward, lambda: _messages(isolated, _AZURE_COUNT, "/v1/messages/count_tokens"))
    assert targets == {f"http://{_AZURE_HOST}/anthropic/v1/messages/count_tokens"}


def test_empty_api_base_falls_back_to_the_provider_default_host(isolated: Gateway, forward: ForwardProxy) -> None:
    targets: Final = _targets_after(forward, lambda: _chat(isolated, _EMPTY_BASE))
    assert targets == {"api.anthropic.com:443"}
