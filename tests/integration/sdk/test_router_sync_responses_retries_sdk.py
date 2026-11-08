from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Final

from integration._support.anthropic_sse import Attempts
from integration._support.openai_wire import openai_error, posted_targets, responses_reply
from integration._support.wire import Reply, Request, wire_server
from litellm import Router
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.router import DeploymentTypedDict

_MODEL: Final = "gpt-5.4"
_API_KEY: Final = "sync-responses-retry-test-key"
_TEXT: Final = "served after the blip"


def _deployment(model_name: str, api_base: str) -> DeploymentTypedDict:
    return DeploymentTypedDict(
        model_name=model_name,
        litellm_params={"model": f"openai/{_MODEL}", "api_base": api_base, "api_key": _API_KEY},
    )


def _upstream(fails: Callable[[str, int], bool]) -> Callable[[Request], Reply]:
    attempts: Final = Attempts()

    def respond(request: Request) -> Reply:
        group, _, route = request.target.lstrip("/").partition("/")
        assert (request.method, route) == ("POST", "responses"), request
        if fails(group, attempts.record(group)):
            return openai_error(503)
        return responses_reply(f"resp_{group}_{uuid.uuid4().hex}", _MODEL, _TEXT, stream=False)

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
