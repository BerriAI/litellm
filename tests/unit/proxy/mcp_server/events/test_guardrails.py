from typing import Final, Literal

import pytest

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.mcp_server.events.guardrails import Blocked, Screened, screen_event
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

from .fixtures import CALLER, occurrence


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


@pytest.mark.asyncio
async def test_a_guardrail_assigned_to_the_key_screens_events_without_default_on() -> None:
    assigned: Final = _Masker(guardrail_name="assigned", event_hook=GuardrailEventHooks.mcp_event, default_on=False)

    async def from_key_metadata(data: dict[str, object], caller: UserAPIKeyAuth) -> None:
        data["metadata"] = {"guardrails": list((caller.metadata or {}).get("guardrails") or [])}

    outcome: Final = await screen_event(
        occurrence(data={"title": "the secret plan", "count": 3}),
        server_name="ci",
        gateway_name="ci-build.failed",
        subscription_id="sub_1",
        caller=UserAPIKeyAuth(api_key="sk-a", metadata={"guardrails": ["assigned"]}),
        callbacks=lambda: (assigned,),
        assign=from_key_metadata,
    )
    assert outcome == Screened({"title": "the [REDACTED] plan", "count": 3})


@pytest.mark.asyncio
async def test_an_assignment_that_cannot_be_resolved_blocks_the_event() -> None:
    async def broken(_data: dict[str, object], _caller: UserAPIKeyAuth) -> None:
        raise RuntimeError("policy registry unavailable")

    outcome: Final = await screen_event(
        occurrence(data={"title": "x"}),
        server_name="ci",
        gateway_name="ci-build.failed",
        subscription_id=None,
        caller=CALLER,
        callbacks=lambda: (),
        assign=broken,
    )
    assert outcome == Blocked("guardrail-assignment")


@pytest.mark.asyncio
async def test_a_policy_attached_to_the_key_assigns_its_guardrail_to_events() -> None:
    from litellm.proxy.policy_engine.attachment_registry import get_attachment_registry
    from litellm.proxy.policy_engine.policy_registry import get_policy_registry

    assigned: Final = _Masker(guardrail_name="assigned", event_hook=GuardrailEventHooks.mcp_event, default_on=False)
    policies: Final = get_policy_registry()
    attachments: Final = get_attachment_registry()
    policies.load_policies({"event-policy": {"guardrails": {"add": ["assigned"]}}})
    attachments.load_attachments([{"policy": "event-policy", "keys": ["events-key"]}])
    try:
        outcome: Final = await screen_event(
            occurrence(data={"title": "the secret plan", "count": 3}),
            server_name="ci",
            gateway_name="ci-build.failed",
            subscription_id="sub_1",
            caller=UserAPIKeyAuth(api_key="sk-a", key_alias="events-key"),
            callbacks=lambda: (assigned,),
        )
    finally:
        attachments.clear()
        policies.clear()
    assert outcome == Screened({"title": "the [REDACTED] plan", "count": 3})


@pytest.mark.parametrize(
    "caller",
    (
        UserAPIKeyAuth(api_key="sk-a", metadata={"disable_global_guardrails": True}),
        UserAPIKeyAuth(api_key="sk-a", team_metadata={"opted_out_global_guardrails": ["mask"]}),
    ),
)
@pytest.mark.asyncio
async def test_a_caller_that_opted_out_of_a_global_guardrail_is_not_screened_by_it(caller: UserAPIKeyAuth) -> None:
    masker: Final = _Masker(guardrail_name="mask", event_hook=GuardrailEventHooks.mcp_event, default_on=True)
    outcome: Final = await screen_event(
        occurrence(data={"title": "the secret plan"}),
        server_name="ci",
        gateway_name="ci-build.failed",
        subscription_id=None,
        caller=caller,
        callbacks=lambda: (masker,),
    )
    assert outcome == Screened({"title": "the secret plan"})


def test_event_screening_carries_the_caller_tags_and_admin_metadata() -> None:
    from litellm.proxy.mcp_server.events.guardrails import caller_metadata

    caller: Final = UserAPIKeyAuth(
        api_key="sk-a", metadata={"tags": ["key-tag"]}, team_metadata={"tags": ["team-tag", "key-tag"]}
    )
    metadata: Final = caller_metadata(caller)
    assert metadata["tags"] == ["key-tag", "team-tag"]
    assert metadata["user_api_key_metadata"] == {"tags": ["key-tag"]}
