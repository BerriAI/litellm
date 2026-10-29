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
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.vertex_ai.interactions.transformation import VertexAIInteractionsConfig
from litellm.responses.bridge_check import should_route_to_interactions_api
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES
from litellm.types.utils import Choices, ModelResponse


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
    seen_urls: Final[list[str]] = []  # mutable-ok: transport spy

    def minter(creds: VERTEX_CREDENTIALS_TYPES | None, project_id: str | None) -> tuple[str, str]:
        seen_mints.append((creds, project_id))
        return "vertex-minted-token", project_id or "missing-project"

    def responder(req: httpx.Request) -> httpx.Response:
        seen_urls.append(str(req.url))
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
    assert seen_mints == [
        (sync_creds, "sync-proj"),
        (sync_creds, "sync-proj"),
        (async_creds, "async-proj"),
        (async_creds, "async-proj"),
    ]
    assert seen_urls == [
        "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/sync-proj/locations/us-central1/interactions",
        "https://europe-west4-aiplatform.googleapis.com/v1beta1/projects/async-proj/locations/europe-west4/interactions",
    ]
