from typing import Final, Literal

import pytest

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy.mcp_server.events.guardrails import Blocked, Screened, screen_event
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

from .fakes import CALLER, occurrence


class _Masker(CustomGuardrail):
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: object = None,
    ) -> GenericGuardrailAPIInputs:
        return {"texts": [text.replace("secret", "[REDACTED]") for text in inputs.get("texts", [])]}


class _Rejecter(CustomGuardrail):
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: object = None,
    ) -> GenericGuardrailAPIInputs:
        raise ValueError("prompt injection")


def _screen(callbacks: tuple[object, ...]):  # noqa: ANN202  # test helper
    return screen_event(
        occurrence(data={"title": "the secret plan", "count": 3}),
        server_name="ci",
        gateway_name="ci-build.failed",
        subscription_id="sub_1",
        caller=CALLER,
        callbacks=lambda: callbacks,
    )


@pytest.mark.asyncio
async def test_an_event_guardrail_masks_payload_strings() -> None:
    masker: Final = _Masker(guardrail_name="mask", event_hook=GuardrailEventHooks.mcp_event, default_on=True)
    assert await _screen((masker,)) == Screened({"title": "the [REDACTED] plan", "count": 3})


@pytest.mark.asyncio
async def test_an_event_guardrail_that_rejects_blocks_the_event() -> None:
    rejecter: Final = _Rejecter(guardrail_name="pi", event_hook=GuardrailEventHooks.mcp_event, default_on=True)
    assert await _screen((rejecter,)) == Blocked("pi")


@pytest.mark.parametrize(
    "callback",
    (
        _Masker(guardrail_name="tools-only", event_hook=GuardrailEventHooks.post_mcp_call, default_on=True),
        _Masker(guardrail_name="not-default", event_hook=GuardrailEventHooks.mcp_event, default_on=False),
        CustomGuardrail(guardrail_name="no-apply", event_hook=GuardrailEventHooks.mcp_event, default_on=True),
        object(),
    ),
)
@pytest.mark.asyncio
async def test_guardrails_that_did_not_opt_into_events_leave_the_payload_alone(callback: object) -> None:
    assert await _screen((callback,)) == Screened({"title": "the secret plan", "count": 3})
