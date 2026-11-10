import pytest


from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks, Mode


def test_custom_guardrail_with_mode_default_list(monkeypatch):
    """Test Mode with default as a list of modes (e.g. default: ["pre_call", "post_call"])"""
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True)
    cg = CustomGuardrail(
        guardrail_name="test_guardrail",
        supported_event_hooks=[
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
            GuardrailEventHooks.logging_only,
        ],
        event_hook=Mode(
            tags={"test_tag": "logging_only"},
            default=["pre_call", "post_call"],
        ),
        default_on=True,
    )

    # No tag match → default fires for pre_call
    assert (
        cg.should_run_guardrail(
            data={"messages": [{"role": "user", "content": "test"}]},
            event_type=GuardrailEventHooks.pre_call,
        )
        is True
    )

    # No tag match → default fires for post_call
    assert (
        cg.should_run_guardrail(
            data={"messages": [{"role": "user", "content": "test"}]},
            event_type=GuardrailEventHooks.post_call,
        )
        is True
    )

    # No tag match → logging_only NOT in default list, should not fire
    assert (
        cg.should_run_guardrail(
            data={"messages": [{"role": "user", "content": "test"}]},
            event_type=GuardrailEventHooks.logging_only,
        )
        is False
    )

    # Tag matches → only logging_only should fire
    assert (
        cg.should_run_guardrail(
            data={
                "messages": [{"role": "user", "content": "test"}],
                "litellm_metadata": {"tags": ["test_tag"]},
            },
            event_type=GuardrailEventHooks.logging_only,
        )
        is True
    )

    # Tag matches → pre_call should NOT fire (tag says logging_only)
    assert (
        cg.should_run_guardrail(
            data={
                "messages": [{"role": "user", "content": "test"}],
                "litellm_metadata": {"tags": ["test_tag"]},
            },
            event_type=GuardrailEventHooks.pre_call,
        )
        is False
    )

    # Tag matches → post_call should NOT fire (tag says logging_only)
    assert (
        cg.should_run_guardrail(
            data={
                "messages": [{"role": "user", "content": "test"}],
                "litellm_metadata": {"tags": ["test_tag"]},
            },
            event_type=GuardrailEventHooks.post_call,
        )
        is False
    )


def test_custom_guardrail_with_mode_no_default(monkeypatch):
    """Test Mode with no default — guardrail only fires when tag matches"""
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True)
    cg = CustomGuardrail(
        guardrail_name="test_guardrail",
        supported_event_hooks=[
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.logging_only,
        ],
        event_hook=Mode(
            tags={"test_tag": "logging_only"},
        ),
        default_on=True,
    )

    # No tag, no default → nothing fires
    assert (
        cg.should_run_guardrail(
            data={"messages": [{"role": "user", "content": "test"}]},
            event_type=GuardrailEventHooks.pre_call,
        )
        is False
    )

    assert (
        cg.should_run_guardrail(
            data={"messages": [{"role": "user", "content": "test"}]},
            event_type=GuardrailEventHooks.logging_only,
        )
        is False
    )

    # Tag matches → only logging_only fires
    assert (
        cg.should_run_guardrail(
            data={
                "messages": [{"role": "user", "content": "test"}],
                "litellm_metadata": {"tags": ["test_tag"]},
            },
            event_type=GuardrailEventHooks.logging_only,
        )
        is True
    )


def test_custom_guardrail_with_mode_tag_value_list(monkeypatch):
    """Test Mode with tag value as a list of modes (e.g. tags: {"tag": ["pre_call", "post_call"]})"""
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True)
    cg = CustomGuardrail(
        guardrail_name="test_guardrail",
        supported_event_hooks=[
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
            GuardrailEventHooks.logging_only,
        ],
        event_hook=Mode(
            tags={"test_tag": ["pre_call", "post_call"]},
            default="logging_only",
        ),
        default_on=True,
    )

    # Tag matches → pre_call should fire (in tag's list)
    assert (
        cg.should_run_guardrail(
            data={
                "messages": [{"role": "user", "content": "test"}],
                "litellm_metadata": {"tags": ["test_tag"]},
            },
            event_type=GuardrailEventHooks.pre_call,
        )
        is True
    )

    # Tag matches → post_call should fire (in tag's list)
    assert (
        cg.should_run_guardrail(
            data={
                "messages": [{"role": "user", "content": "test"}],
                "litellm_metadata": {"tags": ["test_tag"]},
            },
            event_type=GuardrailEventHooks.post_call,
        )
        is True
    )

    # Tag matches → logging_only should NOT fire (not in tag's list)
    assert (
        cg.should_run_guardrail(
            data={
                "messages": [{"role": "user", "content": "test"}],
                "litellm_metadata": {"tags": ["test_tag"]},
            },
            event_type=GuardrailEventHooks.logging_only,
        )
        is False
    )

    # No tag match → default fires for logging_only
    assert (
        cg.should_run_guardrail(
            data={"messages": [{"role": "user", "content": "test"}]},
            event_type=GuardrailEventHooks.logging_only,
        )
        is True
    )

    # No tag match → pre_call should NOT fire (not in default)
    assert (
        cg.should_run_guardrail(
            data={"messages": [{"role": "user", "content": "test"}]},
            event_type=GuardrailEventHooks.pre_call,
        )
        is False
    )


def test_custom_guardrail_with_mode(monkeypatch):
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.premium_user", True
    )  # Set premium_user to True
    cg = CustomGuardrail(
        guardrail_name="test_guardrail",
        supported_event_hooks=[
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.logging_only,
        ],
        event_hook=Mode(
            tags={"test_tag": "pre_call"},
            default="logging_only",
        ),
        default_on=True,
    )

    assert (
        cg.should_run_guardrail(
            data={
                "messages": [{"role": "user", "content": "test_message"}],
                "litellm_metadata": {"tags": ["test_tag"]},
            },
            event_type=GuardrailEventHooks.pre_call,
        )
        is True
    )

    assert (
        cg.should_run_guardrail(
            data={
                "messages": [{"role": "user", "content": "test_message"}],
            },
            event_type=GuardrailEventHooks.pre_call,
        )
        is False
    )

    assert (
        cg.should_run_guardrail(
            data={
                "messages": [{"role": "user", "content": "test_message"}],
                "litellm_metadata": {"tags": ["test_tag"]},
            },
            event_type=GuardrailEventHooks.logging_only,
        )
        is False
    )


@pytest.mark.parametrize("caller_tags,runs_pre_call", [((), True), (("test_tag",), False)])
def test_custom_guardrail_mode_ignores_baggage_mapped_tags_the_caller_did_not_send(
    monkeypatch, caller_tags, runs_pre_call
):
    from types import SimpleNamespace
    from typing import cast

    from fastapi import Request

    from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup

    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True)
    cg = CustomGuardrail(
        guardrail_name="test_guardrail",
        supported_event_hooks=[GuardrailEventHooks.pre_call, GuardrailEventHooks.logging_only],
        event_hook=Mode(tags={"test_tag": "logging_only"}, default="pre_call"),
        default_on=True,
    )
    request = cast(Request, SimpleNamespace(state=SimpleNamespace(litellm_baggage_mapped_tags=("test_tag",))))
    data = {"messages": [{"role": "user", "content": "test"}], "metadata": {"tags": []}}
    LiteLLMProxyRequestSetup.apply_authenticated_baggage_tags(request, data)
    data["metadata"]["caller_tags"] = caller_tags
    assert cg.should_run_guardrail(data=data, event_type=GuardrailEventHooks.pre_call) is runs_pre_call


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "team_tags,deployment_tags,runs_post_call",
    [
        (["policy:sensitive"], ["usecase:dev"], True),
        ([], ["policy:sensitive", "usecase:dev"], True),
        (["policy:sensitive", "usecase:dev"], [], True),
        ([], ["usecase:dev"], False),
        ([], [], True),
    ],
)
async def test_baggage_never_changes_the_guardrail_mode_key_team_and_deployment_tags_select(
    monkeypatch, team_tags, deployment_tags, runs_post_call
):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import litellm
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True)
    cg = CustomGuardrail(
        guardrail_name="test_guardrail",
        supported_event_hooks=[GuardrailEventHooks.post_call, GuardrailEventHooks.logging_only],
        event_hook=Mode(tags={"policy:sensitive": "post_call", "usecase:dev": "logging_only"}, default="post_call"),
        default_on=True,
    )
    router = litellm.Router(
        model_list=[{"model_name": "gpt-4", "litellm_params": {"model": "gpt-4o", "tags": deployment_tags}}]
    )

    async def runs_post_call_guardrail(baggage_tags: tuple[str, ...]) -> bool:
        request = MagicMock()
        request.url.path = "/chat/completions"
        request.method = "POST"
        request.query_params = {}
        request.headers = {"Content-Type": "application/json"}
        request.client.host = "127.0.0.1"
        request.state = SimpleNamespace(litellm_baggage_mapped_tags=baggage_tags)
        data = await add_litellm_data_to_request(
            data={"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]},
            request=request,
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed-key", team_metadata={"tags": team_tags}),
            proxy_config=MagicMock(),
            general_settings={},
        )
        await router.acompletion(**data, mock_response="hi")
        return cg.should_run_guardrail(data=data, event_type=GuardrailEventHooks.post_call)

    assert await runs_post_call_guardrail(()) is runs_post_call
    assert await runs_post_call_guardrail(("usecase:dev",)) is runs_post_call
