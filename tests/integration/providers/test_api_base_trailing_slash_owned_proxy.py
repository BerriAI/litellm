import base64
from collections.abc import Iterator
from typing import Final

import openai
import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment, list_value, object_value
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Wire, wire_server
from integration.providers._api_base_trailing_slash_support import (
    KEY,
    TRAILING,
    TRAILING_IDS,
    VECTOR,
    Row,
    assert_one_request,
    body_of,
    deployment,
    file_id,
    first_item,
    new_marker,
    of_marker,
    provider_peer,
    reply_text,
    request_json,
    result_url,
)

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_FLAG_ROWS: Final = (
    Row("moonshot/kimi-k2-0905-preview", "/ms/v1", "/chat/completions"),
    Row("dashscope/qwen-plus", "/dc/compatible-mode/v1", "/chat/completions"),
    Row("qwencloud/qwen-plus", "/qc/compatible-mode/v1", "/chat/completions"),
    Row("qwen_ai_platform/qwen-plus", "/qp/compatible-mode/v1", "/chat/completions"),
    Row("modelscope/Qwen/Qwen3-8B", "/mo/v1", "/chat/completions"),
)
_MOONSHOT: Final = _FLAG_ROWS[0]
_PNG_HEADER: Final = b"\x89PNG\r\n\x1a\n"


def _overrides(wire_url: str) -> dict[str, str]:
    return {
        "EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER": "true",
        "GIGACHAT_ACCESS_TOKEN": KEY,
        "GIGACHAT_API_BASE": f"{wire_url}/gcenv/v1/",
        "PERPLEXITY_API_BASE": f"{wire_url}/pxenv/",
        "GRADIENT_AI_AGENT_ENDPOINT": f"{wire_url}/gragent/",
    }


@pytest.fixture(scope="module")
def shared_wire() -> Iterator[Wire]:
    with wire_server(provider_peer) as wire:
        yield wire


@pytest.fixture(scope="module")
def owned(shared_wire: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("api-base-trailing-slash")
    with gateway_from_environment() as gateway:
        with owned_proxy_process(gateway, directory, _overrides(shared_wire.url), workers=2) as proxy:
            yield proxy


def _openai(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0)


def _chat(gateway: Gateway, model: str, marker: str) -> str | None:
    completion: Final = _openai(gateway).chat.completions.create(
        model=model, messages=[{"role": "user", "content": marker}]
    )
    return completion.choices[0].message.content


@pytest.mark.parametrize("trailing", TRAILING, ids=TRAILING_IDS)
@pytest.mark.parametrize("row", _FLAG_ROWS, ids=tuple(row.model for row in _FLAG_ROWS))
def test_flag_gated_chat_reaches_one_single_slash_url(
    owned: OwnedProxy, shared_wire: Wire, row: Row, trailing: str
) -> None:
    marker: Final = new_marker()
    with owned.gateway.scenario() as scenario:
        model: Final = deployment(scenario, row, f"{shared_wire.url}{row.base}{trailing}")
        assert _chat(owned.gateway, model, marker) == reply_text(marker)
    assert_one_request(of_marker(shared_wire.drain(), marker), row.target, marker)


@pytest.mark.parametrize("trailing", TRAILING, ids=TRAILING_IDS)
def test_flag_gated_streamed_chat_reaches_one_single_slash_url(
    owned: OwnedProxy, shared_wire: Wire, trailing: str
) -> None:
    marker: Final = new_marker()
    with owned.gateway.scenario() as scenario:
        model: Final = deployment(scenario, _MOONSHOT, f"{shared_wire.url}{_MOONSHOT.base}{trailing}")
        stream: Final = _openai(owned.gateway).chat.completions.create(
            model=model, messages=[{"role": "user", "content": marker}], stream=True
        )
        text: Final = "".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)
        assert text == reply_text(marker)
    forwarded: Final = assert_one_request(of_marker(shared_wire.drain(), marker), _MOONSHOT.target, marker)
    assert request_json(forwarded)["stream"] is True


@pytest.mark.parametrize("trailing", TRAILING, ids=TRAILING_IDS)
def test_gigachat_image_upload_and_chat_reach_single_slash_urls(
    owned: OwnedProxy, shared_wire: Wire, trailing: str
) -> None:
    marker: Final = new_marker()
    image: Final = base64.b64encode(_PNG_HEADER + marker.encode()).decode()
    with owned.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="gigachat/GigaChat-Pro", api_base=f"{shared_wire.url}/gc/v1{trailing}", api_key=KEY
        )
        completion: Final = _openai(owned.gateway).chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": marker},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
                    ],
                }
            ],
        )
        assert completion.choices[0].message.content == reply_text(marker)
    upload, chat = of_marker(shared_wire.drain(), marker)
    assert (upload.method, upload.target) == ("POST", "/gc/v1/files"), upload.target
    assert upload.headers["authorization"] == f"Bearer {KEY}", sorted(upload.headers)
    assert (chat.method, chat.target) == ("POST", "/gc/v1/chat/completions"), chat.target
    messages: Final = request_json(chat)["messages"]
    assert isinstance(messages, list)
    assert object_value(messages[-1])["attachments"] == [file_id(upload)], chat.body


@pytest.mark.parametrize("trailing", TRAILING, ids=TRAILING_IDS)
def test_gigachat_embedding_reaches_one_single_slash_url(owned: OwnedProxy, shared_wire: Wire, trailing: str) -> None:
    marker: Final = new_marker()
    with owned.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="gigachat/Embeddings", api_base=f"{shared_wire.url}/gc/v1{trailing}", api_key=KEY
        )
        response: Final = owned.gateway.request("POST", "/v1/embeddings", {"model": model, "input": marker})
        assert first_item(body_of(response)["data"])["embedding"] == list(VECTOR)
    assert_one_request(of_marker(shared_wire.drain(), marker), "/gc/v1/embeddings", marker)


def test_perplexity_search_with_no_api_base_uses_the_env_base_without_a_double_slash(
    owned: OwnedProxy, shared_wire: Wire
) -> None:
    marker: Final = new_marker()
    name: Final = f"slash-tool-{new_marker()}"
    with owned.gateway.scenario() as scenario:
        created: Final = owned.gateway.post(
            "/search_tools",
            {
                "search_tool": {
                    "search_tool_name": name,
                    "litellm_params": {"search_provider": "perplexity", "api_key": KEY},
                }
            },
        )
        scenario.cleanups.callback(owned.gateway.request, "DELETE", f"/search_tools/{created['search_tool_id']}")
        response: Final = owned.gateway.request("POST", f"/v1/search/{name}", {"query": marker})
        assert [object_value(result)["url"] for result in list_value(body_of(response)["results"])] == [
            result_url(marker)
        ]
    assert_one_request(of_marker(shared_wire.drain(), marker), "/pxenv/search", marker)


def test_gradient_with_no_api_base_uses_the_agent_endpoint_env_without_a_double_slash(
    owned: OwnedProxy, shared_wire: Wire
) -> None:
    marker: Final = new_marker()
    with owned.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="gradient_ai/llama3.3-70b-instruct", api_base=None, api_key=KEY, drop_params=True
        )
        assert _chat(owned.gateway, model, marker) == reply_text(marker)
    assert_one_request(of_marker(shared_wire.drain(), marker), "/gragent/api/v1/chat/completions", marker)


def test_gigachat_pass_through_without_a_model_joins_the_env_base_once(owned: OwnedProxy, shared_wire: Wire) -> None:
    marker: Final = new_marker()
    response: Final = owned.gateway.request("POST", "/gigachat/models", {"probe": marker})
    assert body_of(response)["data"] == [{"id": marker, "object": "model"}], response.text
    assert_one_request(of_marker(shared_wire.drain(), marker), "/gcenv/v1/models", marker)


def test_worker_kill_leaves_the_survivor_and_the_replacement_building_single_slash_urls(
    owned: OwnedProxy, shared_wire: Wire
) -> None:
    victim: Final = min(eventually(owned.worker_pids, lambda pids: len(pids) == 2, seconds=60))
    with owned.gateway.scenario() as scenario:
        model: Final = deployment(scenario, _MOONSHOT, f"{shared_wire.url}{_MOONSHOT.base}/")
        with owned.replacing_worker(victim) as kill:
            kill()
            during: Final = tuple(new_marker() for _ in range(4))
            assert [_chat(owned.gateway, model, marker) for marker in during] == [reply_text(m) for m in during]
        after: Final = tuple(new_marker() for _ in range(8))
        assert [_chat(owned.gateway, model, marker) for marker in after] == [reply_text(m) for m in after]
    received: Final = shared_wire.drain()
    for marker in (*during, *after):
        assert_one_request(of_marker(received, marker), _MOONSHOT.target, marker)
    assert victim not in owned.worker_pids()
    assert len(owned.worker_pids()) == 2
