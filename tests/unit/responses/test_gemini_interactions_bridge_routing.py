from collections.abc import Mapping
import json
from typing import Final

import httpx
import pytest

import litellm
import litellm.completion_extras
from litellm import Router
from litellm.completion_extras.litellm_interactions_transformation import (
    InteractionsToCompletionBridgeHandler,
)
from litellm.exceptions import BadRequestError, UnsupportedParamsError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.vertex_ai.interactions.transformation import VertexAIInteractionsConfig
from litellm.responses.bridge_check import keep_registered_mode, should_route_to_interactions_api
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES
from litellm.types.utils import Choices, ModelResponse
from litellm.utils import (
    validate_chat_completion_tool_choice,  # pyright: ignore[reportUnknownVariableType]  # upstream signature uses bare dict
)


def test_toggle_is_off_by_default_and_excludes_non_gemini_vertex_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "route_all_chat_gemini_to_interactions", False)
    monkeypatch.delenv("LITELLM_ROUTE_ALL_CHAT_GEMINI_TO_INTERACTIONS", raising=False)

    assert should_route_to_interactions_api(model="gemini-3-flash-preview", custom_llm_provider="gemini") is False
    assert should_route_to_interactions_api(model="gemini-3-flash-preview", custom_llm_provider="vertex_ai") is False

    monkeypatch.setattr(litellm, "route_all_chat_gemini_to_interactions", True)

    assert should_route_to_interactions_api(model="gemini-3-flash-preview", custom_llm_provider="gemini") is True
    assert should_route_to_interactions_api(model="gemini-3-flash-preview", custom_llm_provider="vertex_ai") is True
    assert should_route_to_interactions_api(model="claude-opus-4-7", custom_llm_provider="vertex_ai") is False
    assert should_route_to_interactions_api(model="gpt-5.4", custom_llm_provider="openai") is False
    assert (
        should_route_to_interactions_api(
            model="gemini-3-flash-preview",
            custom_llm_provider="gemini",
            deployment_model_info={"mode": "chat"},
        )
        is False
    )
    assert (
        should_route_to_interactions_api(
            model="gemini-3-flash-preview",
            custom_llm_provider="gemini",
            optional_params={"use_interactions": False},
        )
        is False
    )


def test_completion_routes_to_interactions_when_global_toggle_is_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "route_all_chat_gemini_to_interactions", True)
    captured: Final[list[httpx.Request]] = []  # mutable-ok: transport spy

    def responder(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(
            200,
            json={
                "id": "v1_e2e_unit_1",
                "status": "completed",
                "steps": [{"type": "model_output", "content": [{"type": "text", "text": "BRIDGED_OK"}]}],
                "usage": {"total_input_tokens": 8, "total_output_tokens": 3, "total_tokens": 11},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client:
        response: Final = litellm.completion(
            model="gemini/gemini-3-flash-preview",
            messages=[{"role": "user", "content": "Say BRIDGED_OK"}],
            store=False,
            api_key="unit-gemini-key",
            api_base="https://generativelanguage.googleapis.com",
            client=HTTPHandler(client=raw_client),
        )

    assert isinstance(response, ModelResponse)
    choice: Final = response.choices[0]
    assert isinstance(choice, Choices)
    assert choice.message.content == "BRIDGED_OK"
    assert len(captured) == 1
    assert captured[0].url.path == "/v1beta/interactions"
    assert json.loads(captured[0].content) == {
        "model": "gemini-3-flash-preview",
        "input": [{"type": "user_input", "content": [{"type": "text", "text": "Say BRIDGED_OK"}]}],
        "store": False,
    }


@pytest.mark.asyncio
async def test_router_deployment_mode_pins_isolate_interactions_and_generate_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "route_all_chat_gemini_to_interactions", False)
    seen_paths: Final[list[str]] = []  # mutable-ok: transport path spy

    def responder(req: httpx.Request) -> httpx.Response:
        seen_paths.append(req.url.path)
        if req.url.path == "/v1beta/interactions":
            return httpx.Response(
                200,
                json={
                    "id": "v1_router_int",
                    "status": "completed",
                    "steps": [{"type": "model_output", "content": [{"type": "text", "text": "FROM_INTERACTIONS"}]}],
                    "usage": {"total_input_tokens": 4, "total_output_tokens": 2, "total_tokens": 6},
                },
            )
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"role": "model", "parts": [{"text": "FROM_GENERATE_CONTENT"}]}, "finishReason": "STOP"}
                ],
                "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2, "totalTokenCount": 6},
            },
        )

    async_handler: Final = AsyncHTTPHandler()
    async_handler.client = httpx.AsyncClient(transport=httpx.MockTransport(responder))

    router: Final = Router(
        model_list=[
            {
                "model_name": "gemini-interactions-deployment",
                "litellm_params": {
                    "model": "gemini/gemini-3-flash-preview",
                    "api_key": "router-key",
                },
                "model_info": {"mode": "interactions"},
            },
            {
                "model_name": "gemini-chat-deployment",
                "litellm_params": {
                    "model": "gemini/gemini-3-flash-preview",
                    "api_key": "router-key",
                },
                "model_info": {"mode": "chat"},
            },
        ]
    )

    resp_int: Final = await router.acompletion(  # pyright: ignore[reportUnknownMemberType]  # Router.acompletion overload carries untyped kwargs
        model="gemini-interactions-deployment",
        messages=[{"role": "user", "content": "hi"}],
        client=async_handler,
    )
    resp_chat: Final = await router.acompletion(  # pyright: ignore[reportUnknownMemberType]  # Router.acompletion overload carries untyped kwargs
        model="gemini-chat-deployment",
        messages=[{"role": "user", "content": "hi"}],
        client=async_handler,
    )

    assert isinstance(resp_int, ModelResponse)
    assert isinstance(resp_chat, ModelResponse)
    assert resp_int.choices[0].message.content == "FROM_INTERACTIONS"  # pyright: ignore[reportAttributeAccessIssue]  # Choices union
    assert resp_chat.choices[0].message.content == "FROM_GENERATE_CONTENT"  # pyright: ignore[reportAttributeAccessIssue]  # Choices union
    assert len(seen_paths) == 2
    assert seen_paths[0] == "/v1beta/interactions"
    assert seen_paths[1].endswith("/models/gemini-3-flash-preview:generateContent")


@pytest.mark.asyncio
async def test_vertex_completion_and_acompletion_forward_credentials_project_and_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "route_all_chat_gemini_to_interactions", True)
    seen_mints: Final[list[tuple[VERTEX_CREDENTIALS_TYPES | None, str | None]]] = []  # mutable-ok: minter spy
    seen_requests: Final[list[tuple[str, str]]] = []  # mutable-ok: transport spy

    def minter(creds: VERTEX_CREDENTIALS_TYPES | None, project_id: str | None) -> tuple[str, str]:
        seen_mints.append((creds, project_id))
        return f"vertex-minted-{project_id}", project_id or "missing-project"

    def responder(req: httpx.Request) -> httpx.Response:
        seen_requests.append((str(req.url), req.headers.get("authorization", "")))
        return httpx.Response(
            200,
            json={
                "id": "v1_vertex_fwd",
                "status": "completed",
                "steps": [{"type": "model_output", "content": [{"type": "text", "text": "VERTEX_FWD_OK"}]}],
                "usage": {"total_input_tokens": 5, "total_output_tokens": 2, "total_tokens": 7},
            },
        )

    injected_bridge: Final = InteractionsToCompletionBridgeHandler(
        provider_config=VertexAIInteractionsConfig(mint_access_token=minter)
    )
    monkeypatch.setattr(litellm.completion_extras, "interactions_api_bridge", injected_bridge)

    sync_creds: Final = {"type": "service_account", "client_email": "svc@sync-proj.iam.gserviceaccount.com"}
    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_sync:
        sync_resp: Final = litellm.completion(
            model="vertex_ai/gemini-3-flash-preview",
            messages=[{"role": "user", "content": "ping"}],
            vertex_credentials=sync_creds,
            vertex_project="sync-proj",
            vertex_location="us-central1",
            client=HTTPHandler(client=raw_sync),
        )

    async_creds: Final = '{"type":"service_account","project_id":"async-proj"}'
    async_handler: Final = AsyncHTTPHandler()
    async_handler.client = httpx.AsyncClient(transport=httpx.MockTransport(responder))
    async_resp: Final = await litellm.acompletion(
        model="vertex_ai/gemini-3-flash-preview",
        messages=[{"role": "user", "content": "ping"}],
        vertex_ai_credentials=async_creds,
        vertex_ai_project="async-proj",
        vertex_ai_location="europe-west4",
        client=async_handler,
    )

    assert isinstance(sync_resp, ModelResponse)
    assert isinstance(async_resp, ModelResponse)
    assert (sync_creds, "sync-proj") in seen_mints
    assert (async_creds, "async-proj") in seen_mints
    assert all(creds is not None for creds, _ in seen_mints)
    assert seen_requests == [
        (
            "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/sync-proj/locations/us-central1/interactions",
            "Bearer vertex-minted-sync-proj",
        ),
        (
            "https://europe-west4-aiplatform.googleapis.com/v1beta1/projects/async-proj/locations/europe-west4/interactions",
            "Bearer vertex-minted-async-proj",
        ),
    ]


def test_top_level_interactions_kwargs_pass_preprocessing_with_drop_params_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    monkeypatch.setattr(litellm, "route_all_chat_gemini_to_interactions", False)
    captured: Final[list[httpx.Request]] = []  # mutable-ok: transport spy

    def responder(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        if req.url.path == "/v1beta/interactions":
            return httpx.Response(
                200,
                json={
                    "id": "v1_top_kwargs",
                    "status": "completed",
                    "steps": [{"type": "model_output", "content": [{"type": "text", "text": '{"ok": true}'}]}],
                    "usage": {"total_input_tokens": 5, "total_output_tokens": 3, "total_tokens": 8},
                },
            )
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"role": "model", "parts": [{"text": "GC_OK"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2, "totalTokenCount": 7},
            },
        )

    schema: Final = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client:
        http_handler: Final = HTTPHandler(client=raw_client)
        bridged: Final = litellm.completion(
            model="gemini/gemini-3-flash-preview",
            messages=[{"role": "user", "content": "continue"}],
            use_interactions=True,
            previous_interaction_id="v1_prev_99",
            background=True,
            environment="browser",
            audio={"voice": "alloy", "format": "pcm16"},
            response_json_schema=schema,
            api_key="unit-gemini-key",
            api_base="https://generativelanguage.googleapis.com",
            client=http_handler,
        )

        monkeypatch.setattr(litellm, "route_all_chat_gemini_to_interactions", True)
        opted_out: Final = litellm.completion(
            model="gemini/gemini-3-flash-preview",
            messages=[{"role": "user", "content": "plain chat"}],
            use_interactions=False,
            api_key="unit-gemini-key",
            api_base="https://generativelanguage.googleapis.com",
            client=http_handler,
        )

    assert isinstance(bridged, ModelResponse)
    assert isinstance(opted_out, ModelResponse)
    assert len(captured) == 2
    assert captured[0].url.path == "/v1beta/interactions"
    assert json.loads(captured[0].content) == {
        "model": "gemini-3-flash-preview",
        "input": [{"type": "user_input", "content": [{"type": "text", "text": "continue"}]}],
        "previous_interaction_id": "v1_prev_99",
        "background": True,
        "environment": "browser",
        "response_format": {"type": "text", "mime_type": "application/json", "schema": schema},
        "generation_config": {"speech_config": [{"voice": "Kore"}]},
    }
    assert captured[1].url.path.endswith("/models/gemini-3-flash-preview:generateContent")
    assert "use_interactions" not in json.loads(captured[1].content)


@pytest.mark.parametrize(
    ("extra_kwargs", "expected_fields"),
    (
        (
            {"tool_choice": "any"},
            {"generation_config": {"tool_choice": "any"}},
        ),
        (
            {"tool_choice": "validated"},
            {"generation_config": {"tool_choice": "validated"}},
        ),
        (
            {"tool_choice": {"allowed_tools": {"mode": "any", "tools": ["get_weather"]}}},
            {"generation_config": {"tool_choice": {"allowed_tools": {"mode": "any", "tools": ["get_weather"]}}}},
        ),
        (
            {"audio": {"voice": "alloy", "format": "wav"}},
            {
                "response_format": {"type": "audio"},
                "generation_config": {"speech_config": [{"voice": "Kore"}]},
            },
        ),
    ),
)
def test_interactions_native_tool_choice_and_wav_audio_bypass_legacy_gemini_validation(
    extra_kwargs: Mapping[str, object],
    expected_fields: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    captured: Final[list[httpx.Request]] = []  # mutable-ok: transport spy

    def responder(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(
            200,
            json={
                "id": "v1_tc_audio",
                "status": "completed",
                "steps": [{"type": "model_output", "content": [{"type": "text", "text": "ok"}]}],
                "usage": {"total_input_tokens": 4, "total_output_tokens": 2, "total_tokens": 6},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client:
        resp: Final = litellm.completion(
            model="gemini/gemini-3-flash-preview",
            messages=[{"role": "user", "content": "test"}],
            use_interactions=True,
            api_key="unit-gemini-key",
            api_base="https://generativelanguage.googleapis.com",
            client=HTTPHandler(client=raw_client),
            **extra_kwargs,  # pyright: ignore[reportArgumentType]  # dynamic completion kwargs under test
        )

    assert isinstance(resp, ModelResponse)
    assert len(captured) == 1
    assert json.loads(captured[0].content) == {
        "model": "gemini-3-flash-preview",
        "input": [{"type": "user_input", "content": [{"type": "text", "text": "test"}]}],
        **expected_fields,
    }


def test_runtime_false_toggle_overrides_env_and_interactions_mode_does_not_leak_to_unregistered_cost_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_ROUTE_ALL_CHAT_GEMINI_TO_INTERACTIONS", "true")
    monkeypatch.setattr(litellm, "route_all_chat_gemini_to_interactions", False)

    assert should_route_to_interactions_api(model="gemini-3-flash-preview", custom_llm_provider="gemini") is False

    registered_row: Final = keep_registered_mode(
        "gemini/gemini-3.1-flash-lite-preview",
        {"litellm_provider": "gemini", "mode": "interactions", "input_cost_per_token": 1e-7},
    )
    assert registered_row == {"litellm_provider": "gemini", "mode": "chat", "input_cost_per_token": 1e-7}

    unregistered_endpoint_row: Final = keep_registered_mode(
        "vertex_ai/endpoints/1234567890",
        {"litellm_provider": "vertex_ai", "mode": "interactions", "input_cost_per_token": 1e-7},
    )
    assert unregistered_endpoint_row == {"litellm_provider": "vertex_ai", "input_cost_per_token": 1e-7}


@pytest.mark.parametrize(
    ("model", "extra_kwargs", "expected_provider"),
    (
        ("gemini/gemini-3-flash-preview", {"use_interactions": False}, "gemini"),
        ("anthropic/claude-sonnet-4-6", {}, "anthropic"),
        ("openai/gpt-5.4", {}, "openai"),
        ("bedrock/anthropic.claude-sonnet-4-6", {}, "bedrock"),
    ),
)
def test_allowed_tools_tool_choice_is_rejected_for_non_bridged_providers_and_direct_validator(
    model: str,
    extra_kwargs: Mapping[str, object],
    expected_provider: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    monkeypatch.setattr(litellm, "route_all_chat_gemini_to_interactions", False)
    allowed_tools_choice: Final = {"allowed_tools": {"mode": "auto", "tools": ["get_weather"]}}
    tools_payload: Final = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]

    with pytest.raises(BadRequestError, match="Invalid tool choice") as direct_exc:
        validate_chat_completion_tool_choice(allowed_tools_choice, model=model)
    assert direct_exc.value.status_code == 400

    with pytest.raises(UnsupportedParamsError, match="Invalid tool choice") as completion_exc:
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "weather?"}],
            tools=tools_payload,  # pyright: ignore[reportArgumentType]  # ChatCompletionToolParam dict shape
            tool_choice=allowed_tools_choice,
            api_key="unit-test-key",
            **extra_kwargs,  # pyright: ignore[reportArgumentType]  # dynamic completion kwargs under test
        )
    assert completion_exc.value.status_code == 400
    assert getattr(completion_exc.value, "llm_provider", None) == expected_provider


def test_non_bridged_gemini_drops_allowed_tools_tool_choice_cleanly_when_drop_params_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "route_all_chat_gemini_to_interactions", False)
    captured: Final[list[httpx.Request]] = []  # mutable-ok: transport spy

    def responder(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"role": "model", "parts": [{"text": "OK"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 1, "totalTokenCount": 5},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client:
        resp: Final = litellm.completion(
            model="gemini/gemini-3-flash-preview",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
            tool_choice={"allowed_tools": {"mode": "auto", "tools": ["get_weather"]}},
            use_interactions=False,
            drop_params=True,
            api_key="unit-gemini-key",
            api_base="https://generativelanguage.googleapis.com",
            client=HTTPHandler(client=raw_client),
        )

    assert isinstance(resp, ModelResponse)
    assert len(captured) == 1
    assert "toolConfig" not in json.loads(captured[0].content)


def test_bridged_gemini_translates_openai_allowed_tools_tool_choice_while_unbridged_rejects_with_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "route_all_chat_gemini_to_interactions", False)
    captured: Final[list[httpx.Request]] = []  # mutable-ok: transport spy

    def responder(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(
            200,
            json={
                "id": "v1_openai_allowed_tools",
                "status": "completed",
                "steps": [{"type": "model_output", "content": [{"type": "text", "text": "OK"}]}],
                "usage": {"total_input_tokens": 4, "total_output_tokens": 1, "total_tokens": 5},
            },
        )

    openai_allowed_tools: Final = {
        "type": "allowed_tools",
        "allowed_tools": {
            "mode": "required",
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        },
    }
    tools_payload: Final = [
        {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object"}}},
    ]

    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client:
        http_handler: Final = HTTPHandler(client=raw_client)
        bridged: Final = litellm.completion(
            model="gemini/gemini-3-flash-preview",
            messages=[{"role": "user", "content": "hi"}],
            tools=tools_payload,  # pyright: ignore[reportArgumentType]  # ChatCompletionToolParam dict shape
            tool_choice=openai_allowed_tools,
            use_interactions=True,
            api_key="unit-gemini-key",
            api_base="https://generativelanguage.googleapis.com",
            client=http_handler,
        )

        with pytest.raises(BadRequestError, match="allowed_tools") as unbridged_exc:
            litellm.completion(
                model="gemini/gemini-3-flash-preview",
                messages=[{"role": "user", "content": "hi"}],
                tools=tools_payload,  # pyright: ignore[reportArgumentType]  # ChatCompletionToolParam dict shape
                tool_choice=openai_allowed_tools,
                use_interactions=False,
                drop_params=True,
                api_key="unit-gemini-key",
                api_base="https://generativelanguage.googleapis.com",
                client=http_handler,
            )

    assert isinstance(bridged, ModelResponse)
    assert len(captured) == 1
    assert captured[0].url.path == "/v1beta/interactions"
    assert json.loads(captured[0].content)["generation_config"] == {
        "tool_choice": {"allowed_tools": {"mode": "any", "tools": ["get_weather"]}}
    }
    assert unbridged_exc.value.status_code == 400
