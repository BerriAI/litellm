import json
from collections.abc import Mapping
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
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
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


def test_runtime_false_toggle_keeps_older_gemini_off_interactions_and_interactions_mode_does_not_leak_to_unregistered_cost_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


@pytest.mark.parametrize(
    ("model", "provider", "extra", "expected"),
    (
        ("gemini-4-neon", "gemini", {}, True),
        ("gemini-4-neon", "vertex_ai", {}, True),
        ("gemini-4.5-flash", "gemini", {}, True),
        ("gemini-12.5-flash", "gemini", {}, True),
        ("gemini-10-pro", "gemini", {}, True),
        ("gemini-3.8-flash", "gemini", {}, False),
        ("gemini-3-flash-preview", "vertex_ai", {}, False),
        ("gemini-2.5-pro", "gemini", {}, False),
        ("gemini-flash-latest", "gemini", {}, False),
        ("gemini-40", "openai", {}, False),
        ("gemini-4-neon", "gemini", {"optional_params": {"use_interactions": False}}, False),
        ("gemini-4-neon", "gemini", {"optional_params": {"extra_body": {"use_interactions": False}}}, False),
        ("gemini-4-neon", "gemini", {"deployment_model_info": {"mode": "chat"}}, False),
        ("gemini-4-neon", "gemini", {"skip_bridge": True}, False),
        ("gemini-4-neon", "gemini", {"optional_params": {"safety_settings": [{"category": "c"}]}}, False),
        (
            "gemini-4-neon",
            "gemini",
            {
                "optional_params": {
                    "extra_body": {
                        "safetySettings": [
                            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_ONLY_HIGH"}
                        ]
                    }
                }
            },
            False,
        ),
        (
            "gemini-4-neon",
            "gemini",
            {
                "optional_params": {
                    "safety_settings": [
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "HARM_BLOCK_THRESHOLD_UNSPECIFIED"}
                    ]
                }
            },
            False,
        ),
        (
            "gemini-4-neon",
            "gemini",
            {
                "optional_params": {
                    "safety_settings": [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"}]
                }
            },
            True,
        ),
        ("gemini-4-neon", "gemini", {"optional_params": {"top_k": 40}}, True),
        ("gemini-4-neon", "gemini", {"asks_for_penalty": True}, False),
        ("gemini-4-neon", "gemini", {"optional_params": {"video_metadata": {"fps": 1}}}, False),
        ("gemini-4-neon", "gemini", {"optional_params": {"extra_body": {"generationConfig": {}}}}, False),
        ("gemini-4-neon", "gemini", {"optional_params": {"cached_content": "cachedContents/abc"}}, False),
        ("gemini-4-neon", "gemini", {"optional_params": {"previous_interaction_id": "v1_other_keys_turn"}}, False),
        ("gemini-4-neon", "gemini", {"optional_params": {"environment": "env_other_keys_sandbox"}}, False),
        (
            "gemini-4-neon",
            "gemini",
            {"optional_params": {"extra_body": {"environment": {"environment_id": "e1"}}}},
            False,
        ),
        (
            "gemini-4-neon",
            "gemini",
            {
                "optional_params": {
                    "tools": [{"type": "file_search", "file_search_store_names": ["fileSearchStores/other-team"]}]
                }
            },
            False,
        ),
        (
            "gemini-4-neon",
            "gemini",
            {"optional_params": {"tools": [{"type": "function", "function": {"name": "lookup"}}]}},
            True,
        ),
        (
            "gemini-4-neon",
            "gemini",
            {"optional_params": {"extra_body": {"cachedContent": "cachedContents/abc"}}},
            False,
        ),
    ),
)
def test_gemini_4_and_newer_route_to_interactions_by_default_and_older_models_do_not(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    provider: str,
    extra: Mapping[str, object],
    expected: bool,
) -> None:
    monkeypatch.setattr(litellm, "route_all_chat_gemini_to_interactions", False)

    assert should_route_to_interactions_api(model=model, custom_llm_provider=provider, **extra) is expected


def test_globally_configured_safety_settings_keep_gemini_4_on_generate_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "route_all_chat_gemini_to_interactions", False)
    monkeypatch.setattr(litellm, "vertex_ai_safety_settings", {"HARM_CATEGORY_HARASSMENT": "BLOCK_NONE"})

    assert should_route_to_interactions_api(model="gemini-4-neon", custom_llm_provider="vertex_ai") is False


@pytest.mark.parametrize(
    ("provider", "request_params", "expected"),
    (
        ("vertex_ai", {}, False),
        ("vertex_ai_beta", {}, False),
        ("vertex_ai", {"litellm_params": {"vertex_location": "us-central1"}}, True),
        ("vertex_ai", {"optional_params": {"vertex_ai_location": "us-central1"}}, True),
        ("gemini", {}, True),
    ),
)
@pytest.mark.parametrize("ambient", ("module", "VERTEXAI_LOCATION", "VERTEX_LOCATION"))
def test_vertex_gemini_4_relying_on_an_ambient_location_stays_on_generate_content(
    monkeypatch: pytest.MonkeyPatch,
    ambient: str,
    provider: str,
    request_params: Mapping[str, object],
    expected: bool,
) -> None:
    monkeypatch.setattr(litellm, "route_all_chat_gemini_to_interactions", False)
    monkeypatch.setattr(litellm, "vertex_location", "europe-west4" if ambient == "module" else None)
    for name in ("VERTEXAI_LOCATION", "VERTEX_LOCATION"):
        monkeypatch.delenv(name, raising=False)
    if ambient != "module":
        monkeypatch.setenv(ambient, "europe-west4")

    assert (
        should_route_to_interactions_api(model="gemini-4-neon", custom_llm_provider=provider, **request_params)
        is expected
    )


def test_vertex_gemini_4_with_no_configured_location_routes_to_interactions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "route_all_chat_gemini_to_interactions", False)
    monkeypatch.setattr(litellm, "vertex_location", None)
    for name in ("VERTEXAI_LOCATION", "VERTEX_LOCATION"):
        monkeypatch.delenv(name, raising=False)

    assert should_route_to_interactions_api(model="gemini-4-neon", custom_llm_provider="vertex_ai") is True


@pytest.mark.parametrize(
    "token_kwargs",
    ({"continuation_token": "tok_resume"}, {"extra_body": {"continuation_token": "tok_resume"}}),
)
def test_callers_continuation_token_reaches_interactions_from_the_top_level_or_extra_body(
    monkeypatch: pytest.MonkeyPatch, token_kwargs: Mapping[str, object]
) -> None:
    monkeypatch.setattr(litellm, "route_all_chat_gemini_to_interactions", False)
    captured: Final[list[httpx.Request]] = []  # mutable-ok: transport spy

    def responder(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(
            200,
            json={
                "id": "v1_resumed",
                "status": "completed",
                "steps": [{"type": "model_output", "content": [{"type": "text", "text": "the rest"}]}],
                "usage": {"total_input_tokens": 5, "total_output_tokens": 3, "total_tokens": 8},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client:
        resumed: Final = litellm.completion(
            model="gemini/gemini-4-neon",
            messages=[{"role": "user", "content": "write a long story"}],
            api_key="unit-gemini-key",
            api_base="https://generativelanguage.googleapis.com",
            client=HTTPHandler(client=raw_client),
            **token_kwargs,
        )

    assert isinstance(resumed, ModelResponse)
    assert [req.url.path for req in captured] == ["/v1beta/interactions"]
    assert json.loads(captured[0].content)["continuation_token"] == "tok_resume"


def _answering_either_gemini_route(captured: list[httpx.Request]) -> httpx.MockTransport:
    def responder(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        if req.url.path.endswith("/interactions"):
            return httpx.Response(
                200,
                json={
                    "id": "v1_route_choice",
                    "status": "completed",
                    "steps": [{"type": "model_output", "content": [{"type": "text", "text": "ANSWER"}]}],
                    "usage": {"total_input_tokens": 8, "total_output_tokens": 3, "total_tokens": 11},
                },
            )
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"role": "model", "parts": [{"text": "ANSWER"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 3, "totalTokenCount": 11},
            },
        )

    return httpx.MockTransport(responder)


@pytest.mark.parametrize(
    "request_fields",
    (
        {"tools": [{"type": "file_search", "file_search_store_names": ["fileSearchStores/team"]}]},
        {"service_tier": "flex"},
        {"presence_penalty": 0.5},
        {"frequency_penalty": 0.5},
        {"safety_settings": [{"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_ONLY_HIGH"}]},
    ),
    ids=("file_search_tool", "service_tier", "presence_penalty", "frequency_penalty", "unmapped_safety_category"),
)
def test_gemini_4_completion_with_a_field_interactions_cannot_carry_is_sent_to_generate_content(
    monkeypatch: pytest.MonkeyPatch, request_fields: Mapping[str, object]
) -> None:
    monkeypatch.setattr(litellm, "route_all_chat_gemini_to_interactions", False)
    captured: Final[list[httpx.Request]] = []  # mutable-ok: transport spy

    with httpx.Client(transport=_answering_either_gemini_route(captured)) as raw_client:
        litellm.completion(  # pyright: ignore[reportUnknownMemberType]  # completion is untyped upstream
            model="gemini/gemini-4-neon",
            messages=[{"role": "user", "content": "answer"}],
            api_key="unit-gemini-key",
            api_base="https://generativelanguage.googleapis.com",
            client=HTTPHandler(client=raw_client),
            **request_fields,
        )

    assert [req.url.path.rsplit(":", 1)[-1] for req in captured] == ["generateContent"]


@pytest.mark.parametrize(
    ("provider_model", "request_fields", "expected_extra"),
    (
        ("gemini/gemini-4-neon", {"n": 2}, {}),
        ("gemini/gemini-4-neon", {"top_k": 40}, {}),
        ("gemini/gemini-4-neon", {"logprobs": True, "top_logprobs": 2}, {}),
        (
            "gemini/gemini-4-neon",
            {
                "safety_settings": [
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE", "method": "SEVERITY"},
                ]
            },
            {
                "safety_settings": [
                    {"type": "dangerous_content", "threshold": "block_only_high"},
                    {"type": "harassment", "threshold": "off", "method": "severity"},
                ]
            },
        ),
    ),
    ids=("n", "top_k", "logprobs", "safety_settings"),
)
def test_gemini_4_completion_reaches_interactions_without_the_fields_it_has_no_place_for_and_with_translated_safety(
    monkeypatch: pytest.MonkeyPatch,
    provider_model: str,
    request_fields: Mapping[str, object],
    expected_extra: Mapping[str, object],
) -> None:
    monkeypatch.setattr(litellm, "route_all_chat_gemini_to_interactions", False)
    monkeypatch.setattr(litellm, "drop_params", False)
    captured: Final[list[httpx.Request]] = []  # mutable-ok: transport spy

    with httpx.Client(transport=_answering_either_gemini_route(captured)) as raw_client:
        response: Final = litellm.completion(  # pyright: ignore[reportUnknownMemberType]  # completion is untyped upstream
            model=provider_model,
            messages=[{"role": "user", "content": "answer"}],
            api_key="unit-gemini-key",
            api_base="https://generativelanguage.googleapis.com",
            client=HTTPHandler(client=raw_client),
            **request_fields,
        )

    assert [req.url.path.rsplit("/", 1)[-1] for req in captured] == ["interactions"]
    assert json.loads(captured[0].content) == {
        "model": "gemini-4-neon",
        "input": [{"type": "user_input", "content": [{"type": "text", "text": "answer"}]}],
        **expected_extra,
    }
    assert len(response.choices) == 1


def test_gemini_4_stream_asked_to_be_replayed_makes_one_complete_interactions_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "route_all_chat_gemini_to_interactions", False)
    captured: Final[list[httpx.Request]] = []  # mutable-ok: transport spy

    with httpx.Client(transport=_answering_either_gemini_route(captured)) as raw_client:
        stream: Final = litellm.completion(  # pyright: ignore[reportUnknownMemberType]  # completion is untyped upstream
            model="gemini/gemini-4-neon",
            messages=[{"role": "user", "content": "answer"}],
            api_key="unit-gemini-key",
            api_base="https://generativelanguage.googleapis.com",
            client=HTTPHandler(client=raw_client),
            stream=True,
            fake_stream=True,
        )
        assert isinstance(stream, CustomStreamWrapper)
        chunks: Final = tuple(stream)
        text: Final = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)

    assert text == "ANSWER"
    assert [(req.url.path, req.url.query) for req in captured] == [("/v1beta/interactions", b"")]
    assert "stream" not in json.loads(captured[0].content)


def test_non_gemini_vertex_beta_model_is_never_sent_to_the_gemini_interactions_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "route_all_chat_gemini_to_interactions", True)

    assert should_route_to_interactions_api(model="gemini-4-neon", custom_llm_provider="vertex_ai_beta") is True
    assert should_route_to_interactions_api(model="claude-opus-4-7", custom_llm_provider="vertex_ai_beta") is False
    assert (
        should_route_to_interactions_api(
            model="claude-opus-4-7", custom_llm_provider="vertex_ai_beta", optional_params={"use_interactions": True}
        )
        is False
    )


@pytest.mark.parametrize(
    "api_base",
    ("https://gateway.example", "https://gateway.example/v1beta", "https://gateway.example/v1beta/"),
)
def test_gemini_4_completion_reaches_one_interactions_path_whether_or_not_the_api_base_names_the_version(
    monkeypatch: pytest.MonkeyPatch, api_base: str
) -> None:
    monkeypatch.setattr(litellm, "route_all_chat_gemini_to_interactions", False)
    captured: Final[list[httpx.Request]] = []  # mutable-ok: transport spy

    with httpx.Client(transport=_answering_either_gemini_route(captured)) as raw_client:
        litellm.completion(  # pyright: ignore[reportUnknownMemberType]  # completion is untyped upstream
            model="gemini/gemini-4-neon",
            messages=[{"role": "user", "content": "answer"}],
            api_key="unit-gemini-key",
            api_base=api_base,
            client=HTTPHandler(client=raw_client),
        )

    assert [str(req.url) for req in captured] == ["https://gateway.example/v1beta/interactions"]


_SECRET_NAMED_SCHEMA: Final = {
    "type": "object",
    "properties": {"api_key": {"type": "string"}, "access_token": {"type": "string"}, "user": {"type": "string"}},
    "required": ["api_key", "access_token", "user"],
}
_JSON_TEXT_FORMAT: Final = {"type": "text", "mime_type": "application/json"}


@pytest.mark.parametrize(
    ("chat_params", "expected_in_body"),
    [
        pytest.param(
            {"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": _SECRET_NAMED_SCHEMA}}},
            {"response_format": {**_JSON_TEXT_FORMAT, "schema": _SECRET_NAMED_SCHEMA}},
            id="schema-properties-named-like-secrets-survive",
        ),
        pytest.param(
            {"tools": [{"type": "web_search"}]},
            {"tools": [{"type": "google_search"}]},
            id="web-search-tool-becomes-google-search",
        ),
        pytest.param(
            {"tools": [{"type": "web_search_preview"}]},
            {"tools": [{"type": "google_search"}]},
            id="web-search-preview-tool-becomes-google-search",
        ),
        pytest.param(
            {"response_mime_type": "application/json"},
            {"response_format": _JSON_TEXT_FORMAT},
            id="json-mime-type-becomes-a-json-text-format",
        ),
    ],
)
def test_bridged_chat_request_carries_the_callers_schema_search_tool_and_json_mime_type(
    chat_params: dict[str, object], expected_in_body: dict[str, object]
) -> None:
    captured: Final[list[httpx.Request]] = []  # mutable-ok: transport spy

    def responder(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(
            200,
            json={
                "id": "v1_unit_carry",
                "status": "completed",
                "steps": [{"type": "model_output", "content": [{"type": "text", "text": "{}"}]}],
                "usage": {"total_input_tokens": 8, "total_output_tokens": 3, "total_tokens": 11},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client:
        litellm.completion(
            model="gemini/gemini-3-flash-preview",
            messages=[{"role": "user", "content": "hi"}],
            use_interactions=True,
            store=False,
            api_key="unit-gemini-key",
            api_base="https://generativelanguage.googleapis.com",
            client=HTTPHandler(client=raw_client),
            **chat_params,
        )

    assert json.loads(captured[0].content) == {
        "model": "gemini-3-flash-preview",
        "input": [{"type": "user_input", "content": [{"type": "text", "text": "hi"}]}],
        "store": False,
        **expected_in_body,
    }
