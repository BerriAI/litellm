import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import anthropic
import httpx
import pytest
import yaml
from anthropic.types import Message
from integration._support.client import JSON_OBJECT, Gateway, gateway_from_environment
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.wire import Reply, Request, wire_server

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_DENIAL: Final = "This request is not allowed by policy."
_CALL_ID_HEADER: Final = "x-litellm-call-id"


@dataclass(frozen=True, slots=True)
class _Rail:
    gateway: Gateway
    names: tuple[str, str]


@dataclass(frozen=True, slots=True)
class _Ask:
    model: str
    rails: tuple[str, str]
    call_id: str


@pytest.fixture(scope="module")
def denying_rail(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rail]:
    directory: Final = tmp_path_factory.mktemp("messages-guardrail-block")
    name: Final = f"messages-deny-{uuid.uuid4().hex}"
    post_call_name: Final = f"messages-deny-replies-{uuid.uuid4().hex}"
    stock: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final = directory / "messages-guardrail-block.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                **stock,
                "guardrails": [
                    {
                        "guardrail_name": name,
                        "litellm_params": {
                            "guardrail": "custom_code",
                            "mode": "pre_call",
                            "default_on": False,
                            "custom_code": (
                                f"def apply_guardrail(inputs, request_data, input_type):\n    return block({_DENIAL!r})\n"
                            ),
                        },
                    },
                    {
                        "guardrail_name": post_call_name,
                        "litellm_params": {
                            "guardrail": "custom_code",
                            "mode": "post_call",
                            "default_on": False,
                            "custom_code": (
                                "def apply_guardrail(inputs, request_data, input_type):\n"
                                "    return block('A post-call rail scanned the block message.')\n"
                            ),
                        },
                    },
                ],
            }
        )
    )
    with gateway_from_environment() as shared, owned_proxy(shared, directory, {}, config=config) as candidate:
        yield _Rail(candidate, (name, post_call_name))


def _provider_must_not_answer(request: Request) -> Reply:
    return Reply(status=500, body=f"unexpected {request.method} {request.target}".encode())


def _streamed(client: anthropic.Anthropic, ask: _Ask) -> tuple[Message, httpx.Headers]:
    with client.messages.stream(
        model=ask.model,
        max_tokens=32,
        messages=[{"role": "user", "content": "count from 1 to 100"}],
        extra_body={"guardrails": list(ask.rails)},
        extra_headers={_CALL_ID_HEADER: ask.call_id},
    ) as stream:
        return stream.get_final_message(), stream.response.headers


def _created(client: anthropic.Anthropic, ask: _Ask) -> tuple[Message, httpx.Headers]:
    raw: Final = client.messages.with_raw_response.create(
        model=ask.model,
        max_tokens=32,
        messages=[{"role": "user", "content": "count from 1 to 100"}],
        extra_body={"guardrails": list(ask.rails)},
        extra_headers={_CALL_ID_HEADER: ask.call_id},
    )
    return raw.parse(), raw.headers


@pytest.mark.parametrize("send", (_streamed, _created), ids=("stream", "json"))
def test_anthropic_sdk_reads_a_pre_call_block_as_a_finished_message_past_post_call_rails(
    denying_rail: _Rail, send: Callable[[anthropic.Anthropic, _Ask], tuple[Message, httpx.Headers]]
) -> None:
    call_id: Final = f"messages-block-{uuid.uuid4().hex}"
    with (
        wire_server(_provider_must_not_answer) as provider,
        denying_rail.gateway.scenario() as scenario,
        httpx.Client(timeout=30, trust_env=False) as transport,
        anthropic.Anthropic(
            api_key=denying_rail.gateway.key,
            base_url=str(denying_rail.gateway.client.base_url).rstrip("/"),
            max_retries=0,
            http_client=transport,
        ) as client,
    ):
        model: Final = scenario.model(
            model="anthropic/claude-haiku-5-5", api_base=provider.url, api_key="synthetic-anthropic-key"
        )
        message, headers = send(client, _Ask(model, denying_rail.names, call_id))
        sent: Final = provider.drain()

    assert [(block.type, getattr(block, "text", None)) for block in message.content] == [("text", _DENIAL)], message
    assert message.stop_reason == "end_turn", message
    assert headers.get(_CALL_ID_HEADER) == call_id, dict(headers)
    assert sent == (), sent
