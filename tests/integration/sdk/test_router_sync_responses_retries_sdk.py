from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from typing import Final

import pytest
from integration._support.anthropic_sse import Attempts, message_json
from integration._support.openai_wire import chat_reply, openai_error, posted_targets, responses_reply
from integration._support.wire import Reply, Request, wire_server
from litellm import Router
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.router import DeploymentTypedDict

_MODEL: Final = "gpt-5.4"
_API_KEY: Final = "sync-responses-retry-test-key"
_CLAUDE: Final = "claude-sonnet-4-5"
_TEXT: Final = "served after the blip"


def _deployment(model_name: str, api_base: str, model: str = f"openai/{_MODEL}") -> DeploymentTypedDict:
    return DeploymentTypedDict(
        model_name=model_name,
        litellm_params={"model": model, "api_base": api_base, "api_key": _API_KEY},
    )


def _upstream(fails: Callable[[str, int], bool]) -> Callable[[Request], Reply]:
    attempts: Final = Attempts()

    def respond(request: Request) -> Reply:
        group, _, route = request.target.lstrip("/").partition("/")
        assert request.method == "POST", request
        if fails(group, attempts.record(group)):
            return openai_error(503)
        identity: Final = f"{group}_{uuid.uuid4().hex}"
        match route:
            case "responses":
                return responses_reply(f"resp_{identity}", _MODEL, _TEXT, stream=False)
            case "chat/completions":
                return chat_reply(f"chatcmpl-{identity}", _MODEL, _TEXT, stream=False)
            case "v1/messages":
                return Reply(body=message_json(f"msg_{identity}", _CLAUDE, _TEXT))
        raise AssertionError(request)

    return respond


def _output_text(response: object) -> str:
    assert isinstance(response, ResponsesAPIResponse), response
    return str(response.output[0].content[0].text)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]  # the output item shape is the wire's


def test_sync_responses_retries_the_group_after_a_503() -> None:
    with wire_server(_upstream(lambda _group, attempt: attempt == 1)) as wire:
        router: Final = Router(
            model_list=[_deployment("my-model", f"{wire.url}/my-model")],
            num_retries=1,
            retry_after=0,
            disable_cooldowns=True,
        )
        response: Final = router.responses(model="my-model", input=f"retry {uuid.uuid4().hex}")
        targets: Final = posted_targets(wire)

    assert _output_text(response) == _TEXT
    assert targets == ("/my-model/responses", "/my-model/responses")


def test_sync_responses_falls_back_when_the_primary_group_keeps_failing() -> None:
    with wire_server(_upstream(lambda group, _attempt: group == "primary")) as wire:
        router: Final = Router(
            model_list=[
                _deployment("primary", f"{wire.url}/primary"),
                _deployment("backup", f"{wire.url}/backup"),
            ],
            fallbacks=[{"primary": ["backup"]}],
            num_retries=0,
            disable_cooldowns=True,
        )
        response: Final = router.responses(model="primary", input=f"fallback {uuid.uuid4().hex}")
        targets: Final = posted_targets(wire)

    assert _output_text(response) == _TEXT
    assert targets == ("/primary/responses", "/backup/responses")


@pytest.mark.parametrize(
    ("model", "route", "call"),
    [
        pytest.param(
            f"openai/{_MODEL}",
            "/my-model/chat/completions",
            lambda router: router.completion(model="my-model", messages=[{"role": "user", "content": "hi"}]),
            id="completion",
        ),
        pytest.param(
            f"anthropic/{_CLAUDE}",
            "/my-model/v1/messages",
            lambda router: asyncio.run(
                router.anthropic_messages(model="my-model", messages=[{"role": "user", "content": "hi"}], max_tokens=16)
            ),
            id="anthropic_messages",
        ),
        pytest.param(
            f"openai/{_MODEL}",
            "/my-model/responses",
            lambda router: router.responses(model="my-model", input="hi"),
            id="responses",
        ),
    ],
)
def test_every_entry_point_retries_the_group_after_a_503_the_same_way(
    model: str, route: str, call: Callable[[Router], object]
) -> None:
    with wire_server(_upstream(lambda _group, attempt: attempt == 1)) as wire:
        router: Final = Router(
            model_list=[_deployment("my-model", f"{wire.url}/my-model", model)],
            num_retries=1,
            retry_after=0,
            disable_cooldowns=True,
        )
        _ = call(router)
        targets: Final = posted_targets(wire)

    assert targets == (route, route)
