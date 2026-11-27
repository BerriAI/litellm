# Test the following scenarios:
# 1. Generate a Key, and use it to make a call


import asyncio
import base64
import hashlib
import itertools
import json
import logging
import socket
import threading
import traceback
import uuid
from collections.abc import Callable, Coroutine, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Final, NoReturn
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import httpx
import pytest
import respx
import tiktoken
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

load_dotenv()

# this file is to test litellm/proxy


from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.types import Message

import litellm
from litellm import Router
from litellm._logging import JsonFormatter, verbose_logger, verbose_proxy_logger
from litellm.litellm_core_utils.get_model_cost_map import (
    GetModelCostMap,
    ModelCostMapReloaded,
    install_model_cost_map,
)
from litellm.litellm_core_utils.tokenizer import HuggingFace
from litellm.llms.anthropic.count_tokens import handler as anthropic_count_tokens_handler
from litellm.llms.anthropic.messages.adapters.transformation import LiteLLMAnthropicMessagesAdapter
from litellm.llms.anthropic.messages.responses_adapters.transformation import LiteLLMAnthropicToResponsesAPIAdapter
from litellm.llms.azure_ai.anthropic.count_tokens import handler as azure_ai_count_tokens_http
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.bedrock.count_tokens import handler as bedrock_count_tokens_http
from litellm.llms.bedrock.count_tokens import mantle_handler as bedrock_mantle_count_tokens_http
from litellm.llms.bedrock.count_tokens.bedrock_token_counter import BedrockTokenCounter
from litellm.llms.bedrock.count_tokens.handler import BedrockCountTokensHandler
from litellm.llms.bedrock.count_tokens.kimi_k3 import (
    KIMI_K3_RANKS_SHA256,
    KimiK3Encodings,
    KimiK3TokenizerSource,
    kimi_k3_template_tokens,
    kimi_k3_tokens,
    load_kimi_k3_encoding,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.gemini.count_tokens import handler as gemini_count_tokens_handler
from litellm.llms.mistral.count_tokens import handler as mistral_count_tokens_http
from litellm.llms.openai.responses.count_tokens import handler as openai_count_tokens_http
from litellm.llms.vertex_ai.vertex_ai_partner_models.count_tokens.handler import VertexAIPartnerModelsTokenCounter
from litellm.proxy._types import LitellmUserRoles, ProxyException, TokenCountRequest, UserAPIKeyAuth
from litellm.proxy.anthropic_endpoints.endpoints import (
    count_tokens as anthropic_count_tokens,
)
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
from litellm.proxy.common_utils.token_count_settings import (
    ProviderCountFailure,
    TokenCountModel,
    TokenCountSettings,
    cost_map_entry,
    resolve_token_count_settings,
)
from litellm.proxy.google_endpoints.endpoints import _google_count_tokens_request
from litellm.proxy.proxy_server import (
    _estimate_or_provider_error,
    _provider_token_count,
    _try_provider_token_count,
    count_request_tokens,
    token_counter,
)
from litellm.responses.litellm_completion_transformation.session_handler import ResponsesSessionHandler
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.rust_bridge.fork_guard import ForkedAfterNativeRuntimeStarted
from litellm.types.llms.anthropic import ANTHROPIC_BETA_HEADER_VALUES, AnthropicMessagesRequest
from litellm.types.secret_managers.main import KeyManagementSettings, KeyManagementSystem
from litellm.types.utils import (
    ADDRESSED_RESPONSE_ID_FIELD,
    SelectTokenizerResponse,
    SpecialEnums,
    TokenCountMessagesFormat,
    TokenCountResponse,
)
from litellm.utils import _HUGGINGFACE_TOKENIZER_LOAD_OUTCOMES

from tests.unit.proxy.router_entry_point_stubs import route_entry_points_to_stubs


def _hugging_face_offline() -> NoReturn:
    raise OSError("unit tests never download the Kimi K3 tokenizer")


@pytest.fixture(autouse=True)
def _kimi_k3_tokenizer_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "_KIMI_K3_ENCODINGS",
        KimiK3Encodings(
            source=KimiK3TokenizerSource(ranks_path=_hugging_face_offline, sha256=KIMI_K3_RANKS_SHA256),
            retry_after_seconds=60,
        ),
    )


@pytest.mark.asyncio
async def test_vLLM_token_counting():
    """
    Test Token counter for vLLM models
    - User passes model="special-alias"
    - token_counter should infer that special_alias -> maps to wolfram/miquliz-120b-v2.0
    -> token counter should use hugging face tokenizer
    """

    llm_router = Router(
        model_list=[
            {
                "model_name": "special-alias",
                "litellm_params": {
                    "model": "openai/wolfram/miquliz-120b-v2.0",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    response = await count_request_tokens(
        request=TokenCountRequest(
            model="special-alias",
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    print("response: ", response)

    assert response.tokenizer_type == "openai_tokenizer"  # SHOULD use the default tokenizer
    assert response.model_used == "wolfram/miquliz-120b-v2.0"


@pytest.mark.asyncio
async def test_token_counting_model_not_in_model_list():
    """
    Test Token counter - when a model is not in model_list
    -> should use the default OpenAI tokenizer
    """

    llm_router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4",
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    response = await count_request_tokens(
        request=TokenCountRequest(
            model="special-alias",
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    print("response: ", response)

    assert response.tokenizer_type == "openai_tokenizer"  # SHOULD use the OpenAI tokenizer
    assert response.model_used == "special-alias"


def _team_scoped_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "shared-model",
                "litellm_params": {"model": "groq/global-backend", "api_key": "sk-test"},
                "model_info": {"id": "global-deployment"},
            },
            {
                "model_name": "model_name_team-a_1",
                "litellm_params": {"model": "groq/team-a-backend", "api_key": "sk-test"},
                "model_info": {"id": "team-a-deployment", "team_id": "team-a", "team_public_model_name": "shared-model"},
            },
            {
                "model_name": "model_name_team-b_2",
                "litellm_params": {"model": "groq/team-b-backend", "api_key": "sk-test"},
                "model_info": {"id": "team-b-deployment", "team_id": "team-b", "team_public_model_name": "team-only-model"},
            },
            {
                "model_name": "model_name_team-c_3",
                "litellm_params": {"model": "groq/team-c-backend", "api_key": "sk-test"},
                "model_info": {"id": "team-c-deployment", "team_id": "team-c", "team_public_model_name": "shared-model"},
            },
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("caller", "model", "expected_model_used"),
    [
        (UserAPIKeyAuth(team_id="team-a"), "shared-model", "team-a-backend"),
        (UserAPIKeyAuth(team_id="team-c"), "shared-model", "team-c-backend"),
        (UserAPIKeyAuth(team_id="team-b"), "team-only-model", "team-b-backend"),
        (UserAPIKeyAuth(team_id="team-b"), "shared-model", "global-backend"),
        (UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN), "team-only-model", "team-b-backend"),
        (UserAPIKeyAuth(team_id="team-b", aliases={"opus": "team-only-model"}), "opus", "team-b-backend"),
        (UserAPIKeyAuth(team_id="team-b", team_model_aliases={"fast": "team-only-model"}), "fast", "team-b-backend"),
        (
            UserAPIKeyAuth(team_id="team-b", router_settings={"model_group_alias": {"big": "team-only-model"}}),
            "big",
            "team-b-backend",
        ),
        (UserAPIKeyAuth(team_id="team-a"), "legacy-name", "team-a-backend"),
        (None, "shared-model", "global-backend"),
    ],
)
async def test_count_runs_on_the_deployment_the_callers_team_routes_to(
    monkeypatch: pytest.MonkeyPatch, caller: UserAPIKeyAuth | None, model: str, expected_model_used: str
):
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", _team_scoped_router())
    monkeypatch.setattr(litellm, "model_alias_map", {"legacy-name": "shared-model"})

    response = await count_request_tokens(
        request=TokenCountRequest(model=model, messages=[{"role": "user", "content": "hello"}]), caller=caller
    )

    assert response.model_used == expected_model_used


@pytest.mark.asyncio
async def test_count_refuses_a_saved_alias_target_the_key_may_not_call(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", _team_scoped_router())

    with pytest.raises(ProxyException) as refused:
        await count_request_tokens(
            request=TokenCountRequest(model="public-model", messages=[{"role": "user", "content": "hello"}]),
            caller=UserAPIKeyAuth(
                models=["public-model"], router_settings={"model_group_alias": {"public-model": "shared-model"}}
            ),
        )

    assert str(refused.value.code) == "403"


def _tag_routed_router(tag_routing_prefix: str, model_name: str = "tagged-model") -> Router:
    return Router(
        model_list=[
            {
                "model_name": model_name,
                "litellm_params": {"model": "groq/default-backend", "api_key": "sk-test", "tags": ["default"]},
                "model_info": {"id": "default-deployment"},
            },
            {
                "model_name": model_name,
                "litellm_params": {"model": "groq/team-x-backend", "api_key": "sk-test", "tags": ["team-x"]},
                "model_info": {"id": "team-x-deployment"},
            },
            {
                "model_name": model_name,
                "litellm_params": {
                    "model": "groq/claude-code-backend",
                    "api_key": "sk-test",
                    "tag_regex": ["^User-Agent: claude-cli/"],
                },
                "model_info": {"id": "claude-code-deployment"},
            },
        ],
        enable_tag_filtering=True,
        tag_routing_prefix=tag_routing_prefix,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("tag_routing_prefix", ["", "route:"])
@pytest.mark.parametrize(
    ("caller", "expected_model_used"),
    [
        (UserAPIKeyAuth(metadata={"tags": ["team-x"]}), "team-x-backend"),
        (UserAPIKeyAuth(team_id="team-x", team_metadata={"tags": ["team-x"]}), "team-x-backend"),
        (UserAPIKeyAuth(project_metadata={"tags": ["team-x"]}), "team-x-backend"),
        (UserAPIKeyAuth(), "default-backend"),
    ],
)
async def test_count_runs_on_the_deployment_the_callers_inherited_tags_route_to(
    monkeypatch: pytest.MonkeyPatch, tag_routing_prefix: str, caller: UserAPIKeyAuth, expected_model_used: str
):
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", _tag_routed_router(tag_routing_prefix))

    response = await count_request_tokens(
        request=TokenCountRequest(model="tagged-model", messages=[{"role": "user", "content": "hello"}]),
        caller=caller,
    )

    assert response.model_used == expected_model_used


def _router_settings_scoped_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "scoped-model",
                "litellm_params": {
                    "model": "groq/default-backend",
                    "api_key": "sk-test",
                    "tags": ["default"],
                    "order": 1,
                    "weight": 100,
                },
                "model_info": {"id": "default-deployment"},
            },
            {
                "model_name": "scoped-model",
                "litellm_params": {
                    "model": "groq/team-x-backend",
                    "api_key": "sk-test",
                    "tags": ["team-x"],
                    "order": 1,
                    "weight": 0,
                },
                "model_info": {"id": "team-x-deployment"},
            },
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("caller", "expected_model_used"),
    [
        (UserAPIKeyAuth(metadata={"tags": ["team-x"]}), "default-backend"),
        (
            UserAPIKeyAuth(metadata={"tags": ["team-x"]}, router_settings={"enable_tag_filtering": True}),
            "team-x-backend",
        ),
        (
            UserAPIKeyAuth(
                router_settings={"weights": {"scoped-model": {"default-deployment": 0, "team-x-deployment": 100}}}
            ),
            "team-x-backend",
        ),
    ],
    ids=("no-settings", "key-enables-tag-filtering", "key-weights"),
)
async def test_count_routes_with_the_callers_router_settings(
    monkeypatch: pytest.MonkeyPatch, caller: UserAPIKeyAuth, expected_model_used: str
):
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", _router_settings_scoped_router())

    response = await count_request_tokens(
        request=TokenCountRequest(model="scoped-model", messages=[{"role": "user", "content": "hello"}]),
        caller=caller,
    )

    assert response.model_used == expected_model_used



def _rpm_capped_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "capped-model",
                "litellm_params": {"model": "groq/capped-backend", "api_key": "sk-test", "rpm": 1},
                "model_info": {"id": "capped-deployment"},
            }
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["capped-model", "capped-deployment"], ids=("model-group", "deployment-id"))
async def test_a_count_under_the_callers_usage_based_strategy_leaves_the_deployments_rpm_to_inference(
    monkeypatch: pytest.MonkeyPatch, model: str
):
    router: Final = _rpm_capped_router()
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)

    response: Final = await count_request_tokens(
        request=TokenCountRequest(model=model, messages=[{"role": "user", "content": "hello"}]),
        caller=UserAPIKeyAuth(router_settings={"routing_strategy": "usage-based-routing-v2"}),
    )
    picked: Final = await router.select(model, {"routing_strategy": "usage-based-routing-v2"})

    assert response.model_used == "capped-backend"
    assert picked["model_info"]["id"] == "capped-deployment"


async def _count_on_utils_route(
    caller: UserAPIKeyAuth, headers: Mapping[str, str] = MappingProxyType({})
) -> object:
    return await token_counter(
        request=TokenCountRequest(model="shared-model", messages=[{"role": "user", "content": "hello"}]),
        http_request=_json_request({}, "python-httpx/0.28.1", headers),
        fastapi_response=Response(),
        user_api_key_dict=caller,
    )


async def _count_on_messages_route(
    caller: UserAPIKeyAuth, headers: Mapping[str, str] = MappingProxyType({})
) -> object:
    body: Final = {"model": "shared-model", "messages": [{"role": "user", "content": "hello"}]}
    return await anthropic_count_tokens(_json_request(body, "python-httpx/0.28.1", headers), Response(), caller)


async def _count_on_responses_route(
    caller: UserAPIKeyAuth, headers: Mapping[str, str] = MappingProxyType({})
) -> object:
    from litellm.proxy.response_api_endpoints.endpoints import responses_input_tokens

    return await responses_input_tokens(
        request=_json_request({"model": "shared-model", "input": "hello"}, "python-httpx/0.28.1", headers),
        fastapi_response=Response(),
        general_settings=MappingProxyType({}),
        token_counter=count_request_tokens,
        llm_router=litellm.proxy.proxy_server.llm_router,
        user_api_key_dict=caller,
        response_id_security=None,
        managed_files=None,
    )


async def _count_on_google_route(
    caller: UserAPIKeyAuth, headers: Mapping[str, str] = MappingProxyType({})
) -> object:
    from litellm.proxy.google_endpoints.endpoints import google_count_tokens

    body: Final = {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]}
    return await google_count_tokens(
        _json_request(body, "python-httpx/0.28.1", headers), "shared-model", Response(), caller
    )


_EVERY_COUNT_ROUTE: Final = pytest.mark.parametrize(
    "count_on_route",
    [_count_on_utils_route, _count_on_messages_route, _count_on_responses_route, _count_on_google_route],
    ids=["utils", "messages", "responses", "google"],
)


@pytest.mark.asyncio
@_EVERY_COUNT_ROUTE
async def test_every_count_route_counts_on_the_callers_team_deployment(
    monkeypatch: pytest.MonkeyPatch,
    token_count_log_lines: Callable[[], tuple[dict[str, object], ...]],
    count_on_route: Callable[[UserAPIKeyAuth], Coroutine[None, None, object]],
):
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", _team_scoped_router())

    await count_on_route(UserAPIKeyAuth(team_id="team-a"))

    assert tuple(line["model_id"] for line in token_count_log_lines()) == ("team-a-deployment",)


@pytest.mark.asyncio
@_EVERY_COUNT_ROUTE
async def test_every_count_route_counts_on_the_deployment_the_request_tags_header_routes_to(
    monkeypatch: pytest.MonkeyPatch,
    token_count_log_lines: Callable[[], tuple[dict[str, object], ...]],
    count_on_route: Callable[[UserAPIKeyAuth, Mapping[str, str]], Coroutine[None, None, object]],
):
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", _tag_routed_router("", model_name="shared-model"))

    await count_on_route(UserAPIKeyAuth(), {"x-litellm-tags": "team-x"})

    assert tuple(line["model_id"] for line in token_count_log_lines()) == ("team-x-deployment",)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_agent", "expected_model_id"),
    [("claude-cli/2.1.299 (external, cli)", "claude-code-deployment"), ("python-httpx/0.28.1", "default-deployment")],
)
async def test_messages_count_tokens_counts_on_the_deployment_the_user_agent_tag_regex_routes_to(
    monkeypatch: pytest.MonkeyPatch,
    token_count_log_lines: Callable[[], tuple[dict[str, object], ...]],
    user_agent: str,
    expected_model_id: str,
):
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", _tag_routed_router("", model_name="shared-model"))
    body: Final = {"model": "shared-model", "messages": [{"role": "user", "content": "hello"}]}

    await anthropic_count_tokens(_json_request(body, user_agent), Response(), UserAPIKeyAuth())

    assert tuple(line["model_id"] for line in token_count_log_lines()) == (expected_model_id,)


def _affinity_router(pre_call_check: str) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "pinned-model",
                "litellm_params": {"model": f"groq/backend-{deployment_id}", "api_key": "sk-test"},
                "model_info": {"id": deployment_id},
            }
            for deployment_id in ("deployment-1", "deployment-2", "deployment-3", "deployment-4")
        ],
        optional_pre_call_checks=[pre_call_check],
    )


_PINNED_SESSION: Final = "2f1c9d4e-7b3a-4c61-9e0f-5d8a2b6c4e17"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pre_call_check", "session_id", "session_headers", "session_body"),
    [
        ("deployment_affinity", None, {}, {}),
        ("session_affinity", _PINNED_SESSION, {"x-claude-code-session-id": _PINNED_SESSION}, {}),
        ("session_affinity", _PINNED_SESSION, {"baggage": f"user.id=42,session.id={_PINNED_SESSION}"}, {}),
        ("session_affinity", _PINNED_SESSION, {}, {"litellm_session_id": _PINNED_SESSION}),
        ("session_affinity", _PINNED_SESSION, {}, {"metadata": {"user_id": f"user_abc_account__session_{_PINNED_SESSION}"}}),
    ],
)
async def test_messages_count_tokens_counts_on_the_deployment_inference_pinned_for_the_caller(
    monkeypatch: pytest.MonkeyPatch,
    token_count_log_lines: Callable[[], tuple[dict[str, object], ...]],
    pre_call_check: str,
    session_id: str | None,
    session_headers: Mapping[str, str],
    session_body: Mapping[str, object],
):
    router: Final = _affinity_router(pre_call_check)
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
    caller: Final = UserAPIKeyAuth(api_key="sk-pinned-caller", user_id="pinned-user")
    inference: Final = await router.acompletion(
        model="pinned-model",
        messages=[{"role": "user", "content": "hello"}],
        mock_response="ok",
        metadata={
            "user_api_key_hash": LiteLLMProxyRequestSetup.get_logged_api_key(caller),
            **({"session_id": session_id} if session_id else {}),
        },
    )
    body: Final = {"model": "pinned-model", "messages": [{"role": "user", "content": "hello"}], **session_body}

    for _ in range(8):
        await anthropic_count_tokens(_json_request(body, "claude-cli/2.1.299", session_headers), Response(), caller)

    assert {line["model_id"] for line in token_count_log_lines()} == {inference._hidden_params["model_id"]}


@pytest.mark.asyncio
async def test_asking_the_provider_endpoint_leaves_an_openrouter_deployments_count_unchanged(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "or-gpt",
                    "litellm_params": {
                        "model": "openai/gpt-4o",
                        "custom_llm_provider": "openrouter",
                        "api_key": "sk-or-token-count",
                    },
                }
            ]
        ),
    )
    request: Final = TokenCountRequest(
        model="or-gpt", messages=[{"role": "user", "content": "Count these tokens, ünïcödé 你好 🙂"}]
    )

    via_provider_endpoint: Final = await count_request_tokens(request=request, call_endpoint=True)
    counted_locally: Final = await count_request_tokens(request=request, call_endpoint=False)

    assert (via_provider_endpoint.total_tokens, via_provider_endpoint.model_used) == (
        counted_locally.total_tokens,
        counted_locally.model_used,
    )


@pytest.mark.asyncio
async def test_gpt_token_counting():
    """
    Test Token counter
    -> should work for gpt-4
    """

    llm_router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4",
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    response = await count_request_tokens(
        request=TokenCountRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    print("response: ", response)

    assert response.tokenizer_type == "openai_tokenizer"  # SHOULD use the OpenAI tokenizer
    assert response.request_model == "gpt-4"


@pytest.mark.asyncio
async def test_token_counter_leaves_the_routers_deployment_without_the_credentials_it_resolves(tmp_path, monkeypatch):
    """token_counter reads the router's own deployment, so loading the named credential into it would leave the
    token read from the file on the shared deployment for every later request."""
    from litellm.types.utils import CredentialItem

    monkeypatch.setenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", str(tmp_path))
    token_file = tmp_path / "token"
    token_file.write_text("sa-token-from-file")
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="mistral-sa",
                credential_values={"api_key": f"oidc/file/{token_file}"},
                credential_info={},
            )
        ],
    )
    llm_router = Router(
        model_list=[
            {
                "model_name": "devstral",
                "litellm_params": {"model": "mistral/devstral-latest", "litellm_credential_name": "mistral-sa"},
            }
        ]
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    response = await count_request_tokens(
        request=TokenCountRequest(model="devstral", messages=[{"role": "user", "content": "hello"}])
    )

    assert response.total_tokens > 0
    stored = (await llm_router.async_get_available_deployment(model="devstral", request_kwargs={}))["litellm_params"]
    assert "api_key" not in stored
    assert stored["litellm_credential_name"] == "mistral-sa"


@pytest.mark.asyncio
@pytest.mark.parametrize("call_endpoint", [False, True], ids=["local tokenizer", "provider count api"])
async def test_token_counter_reads_an_oidc_api_key_only_when_it_calls_the_provider(
    tmp_path, monkeypatch, call_endpoint
):
    monkeypatch.setenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", str(tmp_path))
    llm_router = Router(
        model_list=[
            {
                "model_name": "claude",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-5-5",
                    "api_key": f"oidc/file/{tmp_path / 'missing-token'}",
                },
            }
        ]
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", llm_router)
    request = TokenCountRequest(model="claude", messages=[{"role": "user", "content": "hello"}])

    if not call_endpoint:
        assert (await count_request_tokens(request=request, call_endpoint=call_endpoint)).total_tokens > 0
        return
    with pytest.raises(ProxyException) as excinfo:
        await count_request_tokens(request=request, call_endpoint=call_endpoint)
    assert excinfo.value.code == "401"
    assert str(tmp_path) not in excinfo.value.message


@pytest.mark.asyncio
async def test_token_counter_answers_503_when_the_metadata_server_behind_an_oidc_google_api_key_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    audience: Final = f"https://count.test/{uuid.uuid4().hex}"
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "claude",
                    "litellm_params": {"model": "anthropic/claude-sonnet-5-5", "api_key": f"oidc/google/{audience}"},
                }
            ]
        ),
    )

    with respx.mock(assert_all_called=True) as upstream:
        upstream.get(url__regex=r"/computeMetadata/v1/instance/service-accounts/default/identity").respond(
            500, text="metadata server unavailable"
        )
        with pytest.raises(ProxyException) as excinfo:
            await count_request_tokens(
                request=TokenCountRequest(model="claude", messages=[{"role": "user", "content": "hello"}]),
                call_endpoint=True,
            )

    assert excinfo.value.code == "503"
    assert "oidc/google secret reference could not be resolved" in excinfo.value.message
    assert audience.removeprefix("https://") not in excinfo.value.message


@pytest.mark.asyncio
async def test_native_openai_responses_count_sends_the_token_an_oidc_api_key_points_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", str(tmp_path))
    token_file: Final = tmp_path / "token"
    token_file.write_text("sa-token-from-file")
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "gpt-oidc",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": f"oidc/file/{token_file}",
                        "api_base": "https://count.test/v1",
                    },
                }
            ]
        ),
    )
    authorizations: Final[list[str | None]] = []  # mutable-ok: collects what the mock peer receives

    def count_peer(request: httpx.Request) -> httpx.Response:
        authorizations.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 9}, request=request)

    with patch.object(
        openai_count_tokens_http,
        "get_async_httpx_client",
        return_value=AsyncHTTPHandler(transport=httpx.MockTransport(count_peer)),
    ):
        result: Final = await count_request_tokens(
            request=TokenCountRequest(model="gpt-oidc", messages=[{"role": "user", "content": "hello"}]),
            call_endpoint=True,
            responses_request={"model": "gpt-oidc", "input": "hello"},
        )

    assert (result.error, result.total_tokens) == (False, 9)
    assert authorizations == ["Bearer sa-token-from-file"]


@pytest.mark.asyncio
async def test_anthropic_messages_count_tokens_endpoint():
    """
    Test /v1/messages/count_tokens endpoint with Anthropic model
    - Should return response in Anthropic format: {"input_tokens": <count>}
    - Should work as wrapper around internal token_counter function
    """
    from unittest.mock import MagicMock

    from fastapi import Request

    from litellm.proxy.anthropic_endpoints.endpoints import count_tokens

    # Mock request object
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {}
    mock_request_data = {
        "model": "claude-3-sonnet-20240229",
        "messages": [{"role": "user", "content": "Hello Claude!"}],
    }

    # Mock the _read_request_body function
    async def mock_read_request_body(request):
        return mock_request_data

    # Mock UserAPIKeyAuth
    mock_user_api_key_dict = MagicMock()

    # Patch the _read_request_body function
    import litellm.proxy.anthropic_endpoints.endpoints as anthropic_endpoints

    original_read_request_body = anthropic_endpoints.read_request_body
    anthropic_endpoints.read_request_body = mock_read_request_body

    # Mock the internal token_counter function to return a controlled response
    async def mock_token_counter(
        request,
        call_endpoint=False,
        messages_format="openai",
        output_format=None,
        context_management=None,
        request_litellm_params=None,
        litellm_call_id=None,
        client_oauth_token=None,
        count_options=None,
        caller=None,
        request_routing=None,
    ):
        assert call_endpoint == True, "Should be called with call_endpoint=True for Anthropic endpoint"
        assert request.model == "claude-3-sonnet-20240229"
        assert request.messages == [{"role": "user", "content": "Hello Claude!"}]

        from litellm.types.utils import TokenCountResponse

        return TokenCountResponse(
            total_tokens=15,
            request_model="claude-3-sonnet-20240229",
            model_used="claude-3-sonnet-20240229",
            tokenizer_type="openai_tokenizer",
        )

    # Patch the imported token_counter function from proxy_server
    import litellm.proxy.proxy_server as proxy_server

    original_token_counter = proxy_server.count_request_tokens
    proxy_server.count_request_tokens = mock_token_counter

    try:
        # Call the endpoint
        response = await count_tokens(mock_request, Response(), mock_user_api_key_dict)

        # Verify response format matches Anthropic spec
        assert isinstance(response, dict)
        assert "input_tokens" in response
        assert response["input_tokens"] == 15
        assert len(response) == 1  # Should only contain input_tokens

        print("✅ Anthropic endpoint test passed!")

    finally:
        # Restore original functions
        anthropic_endpoints.read_request_body = original_read_request_body
        proxy_server.count_request_tokens = original_token_counter


@pytest.mark.asyncio
async def test_anthropic_messages_count_tokens_with_non_anthropic_model():
    """
    Test /v1/messages/count_tokens endpoint with non-Anthropic model (GPT-4)
    - Should still work and return Anthropic format
    - Should call internal token_counter with from_anthropic_endpoint=True
    """
    from unittest.mock import MagicMock

    from fastapi import Request

    from litellm.proxy.anthropic_endpoints.endpoints import count_tokens

    # Mock request object
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {}
    mock_request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello GPT!"}],
    }

    # Mock the _read_request_body function
    async def mock_read_request_body(request):
        return mock_request_data

    # Mock UserAPIKeyAuth
    mock_user_api_key_dict = MagicMock()

    # Patch the _read_request_body function
    import litellm.proxy.anthropic_endpoints.endpoints as anthropic_endpoints

    original_read_request_body = anthropic_endpoints.read_request_body
    anthropic_endpoints.read_request_body = mock_read_request_body

    # Mock the internal token_counter function to return a controlled response
    async def mock_token_counter(
        request,
        call_endpoint=True,
        messages_format="openai",
        output_format=None,
        context_management=None,
        request_litellm_params=None,
        litellm_call_id=None,
        client_oauth_token=None,
        count_options=None,
        caller=None,
        request_routing=None,
    ):
        assert call_endpoint == True, "Should be called with call_endpoint=True for Anthropic endpoint"
        assert request.model == "gpt-4"
        assert request.messages == [{"role": "user", "content": "Hello GPT!"}]

        from litellm.types.utils import TokenCountResponse

        return TokenCountResponse(
            total_tokens=12,
            request_model="gpt-4",
            model_used="gpt-4",
            tokenizer_type="openai_tokenizer",
        )

    # Patch the imported token_counter function from proxy_server
    import litellm.proxy.proxy_server as proxy_server

    original_token_counter = proxy_server.count_request_tokens
    proxy_server.count_request_tokens = mock_token_counter

    try:
        # Call the endpoint
        response = await count_tokens(mock_request, Response(), mock_user_api_key_dict)

        # Verify response format matches Anthropic spec
        assert isinstance(response, dict)
        assert "input_tokens" in response
        assert response["input_tokens"] == 12
        assert len(response) == 1  # Should only contain input_tokens

        print("✅ Non-Anthropic model test passed!")

    finally:
        # Restore original functions
        anthropic_endpoints.read_request_body = original_read_request_body
        proxy_server.count_request_tokens = original_token_counter


@pytest.mark.asyncio
async def test_internal_token_counter_anthropic_provider_detection():
    """
    Test that the internal token_counter correctly detects Anthropic providers
    and handles the from_anthropic_endpoint flag appropriately
    """

    # Test with Anthropic provider
    llm_router = Router(
        model_list=[
            {
                "model_name": "claude-test",
                "litellm_params": {
                    "model": "anthropic/claude-3-sonnet-20240229",
                    "api_key": "test-key",
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    # Test with is_direct_request=False (simulating call from Anthropic endpoint)
    # The provider counter is engaged: its API error surfaces instead of a local count
    with pytest.raises(ProxyException) as exc_info:
        await count_request_tokens(
            request=TokenCountRequest(
                model="claude-test",
                messages=[{"role": "user", "content": "hello"}],
            ),
            call_endpoint=True,
        )
    assert exc_info.value.type == "token_counting_error"

    # Test with non-Anthropic provider
    llm_router = Router(
        model_list=[
            {
                "model_name": "gpt-test",
                "litellm_params": {
                    "model": "gpt-4",
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    with pytest.raises(ProxyException) as missing_key:
        await count_request_tokens(
            request=TokenCountRequest(
                model="gpt-test",
                messages=[{"role": "user", "content": "hello"}],
            ),
            call_endpoint=True,
        )
    assert missing_key.value.code == "401"
    assert "No OpenAI API key is set for model gpt-test" in missing_key.value.message


@pytest.mark.asyncio
async def test_anthropic_endpoint_error_handling():
    """
    Test error handling in the /v1/messages/count_tokens endpoint
    """
    from unittest.mock import MagicMock

    from fastapi import Request

    from litellm.proxy.anthropic_endpoints.endpoints import count_tokens

    # Mock request object
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {}
    mock_request.scope = {}
    mock_user_api_key_dict = MagicMock()

    # Test missing model parameter
    mock_request_data = {
        "messages": [{"role": "user", "content": "Hello!"}]
        # Missing "model" key
    }

    async def mock_read_request_body(request):
        return mock_request_data

    import litellm.proxy.anthropic_endpoints.endpoints as anthropic_endpoints

    original_read_request_body = anthropic_endpoints.read_request_body
    anthropic_endpoints.read_request_body = mock_read_request_body

    try:
        response = await count_tokens(mock_request, Response(), mock_user_api_key_dict)

        assert response.status_code == 400
        assert json.loads(response.body) == {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "model parameter is required"},
        }

        print("✅ Error handling test passed!")

    finally:
        anthropic_endpoints.read_request_body = original_read_request_body


@pytest.mark.asyncio
async def test_factory_anthropic_endpoint_calls_anthropic_counter():
    """Test that /v1/messages/count_tokens with Anthropic model uses Anthropic counter."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from fastapi.testclient import TestClient

    from litellm.proxy.proxy_server import app

    # Mock the global handler instance in token_counter module
    mock_handler = MagicMock()
    mock_handler.handle_count_tokens_request = AsyncMock(return_value={"input_tokens": 42})

    with patch(
        "litellm.llms.anthropic.count_tokens.token_counter.anthropic_count_tokens_handler",
        mock_handler,
    ):
        # Mock router to return Anthropic deployment
        with patch("litellm.proxy.proxy_server.llm_router") as mock_router:
            route_entry_points_to_stubs(mock_router)
            mock_router.model_list = [
                {
                    "model_name": "claude-3-5-sonnet",
                    "litellm_params": {"model": "anthropic/claude-3-5-sonnet-20241022"},
                    "model_info": {},
                }
            ]

            # Mock the async method properly
            mock_router.async_get_available_deployment = AsyncMock(
                return_value={
                    "model_name": "claude-3-5-sonnet",
                    "litellm_params": {"model": "anthropic/claude-3-5-sonnet-20241022"},
                    "model_info": {},
                }
            )
            mock_router.get_routable_deployments = MagicMock(return_value=[mock_router.async_get_available_deployment.return_value])

            # Set ANTHROPIC_API_KEY for the test
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
                client = TestClient(app)

                response = client.post(
                    "/v1/messages/count_tokens",
                    json={
                        "model": "claude-3-5-sonnet",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                    headers={"Authorization": "Bearer test-key"},
                )

                assert response.status_code == 200
                data = response.json()
                assert data["input_tokens"] == 42

                # Verify that Anthropic handler was called
                mock_handler.handle_count_tokens_request.assert_called_once()


@pytest.mark.asyncio
async def test_factory_gpt4_endpoint_does_not_call_anthropic_counter():
    """Test that /v1/messages/count_tokens with GPT-4 does NOT use Anthropic counter."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from fastapi.testclient import TestClient

    from litellm.proxy.proxy_server import app

    # Mock the global handler instance in token_counter module
    mock_handler = MagicMock()
    mock_handler.handle_count_tokens_request = AsyncMock(return_value={"input_tokens": 42})

    with patch(
        "litellm.llms.anthropic.count_tokens.token_counter.anthropic_count_tokens_handler",
        mock_handler,
    ):
        # Mock litellm token counter
        with patch("litellm.token_counter") as mock_litellm_counter:
            mock_litellm_counter.return_value = 50

            # Mock router to return GPT-4 deployment
            with patch("litellm.proxy.proxy_server.llm_router") as mock_router:
                route_entry_points_to_stubs(mock_router)
                mock_router.model_list = [
                    {
                        "model_name": "gpt-4",
                        "litellm_params": {"model": "databricks/gpt-4"},
                        "model_info": {},
                    }
                ]

                # Mock the async method properly
                mock_router.async_get_available_deployment = AsyncMock(
                    return_value={
                        "model_name": "gpt-4",
                        "litellm_params": {"model": "databricks/gpt-4"},
                        "model_info": {},
                    }
                )
                mock_router.get_routable_deployments = MagicMock(return_value=[mock_router.async_get_available_deployment.return_value])

                client = TestClient(app)

                response = client.post(
                    "/v1/messages/count_tokens",
                    json={
                        "model": "gpt-4",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                    headers={"Authorization": "Bearer test-key"},
                )

                assert response.status_code == 200
                data = response.json()
                assert data["input_tokens"] == 50

                # Verify that Anthropic handler was NOT called
                mock_handler.handle_count_tokens_request.assert_not_called()


@pytest.mark.asyncio
async def test_factory_normal_token_counter_endpoint_does_not_call_anthropic():
    """Test that /utils/token_counter does NOT use Anthropic counter even with Anthropic model."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from fastapi.testclient import TestClient

    from litellm.proxy.proxy_server import app

    # Mock the global handler instance in token_counter module
    mock_handler = MagicMock()
    mock_handler.handle_count_tokens_request = AsyncMock(return_value={"input_tokens": 42})

    with patch(
        "litellm.llms.anthropic.count_tokens.token_counter.anthropic_count_tokens_handler",
        mock_handler,
    ):
        # Mock litellm token counter
        with patch("litellm.token_counter") as mock_litellm_counter:
            mock_litellm_counter.return_value = 35

            # Mock router to return Anthropic deployment
            with patch("litellm.proxy.proxy_server.llm_router") as mock_router:
                route_entry_points_to_stubs(mock_router)
                mock_router.model_list = [
                    {
                        "model_name": "claude-3-5-sonnet",
                        "litellm_params": {"model": "anthropic/claude-3-5-sonnet-20241022"},
                        "model_info": {},
                    }
                ]

                # Mock the async method properly
                mock_router.async_get_available_deployment = AsyncMock(
                    return_value={
                        "model_name": "claude-3-5-sonnet",
                        "litellm_params": {"model": "anthropic/claude-3-5-sonnet-20241022"},
                        "model_info": {},
                    }
                )
                mock_router.get_routable_deployments = MagicMock(return_value=[mock_router.async_get_available_deployment.return_value])

                client = TestClient(app)

                response = client.post(
                    "/utils/token_counter",
                    json={
                        "model": "claude-3-5-sonnet",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                    headers={"Authorization": "Bearer test-key"},
                )

                assert response.status_code == 200
                data = response.json()
                assert data["total_tokens"] == 35

                # Verify that Anthropic handler was NOT called (since call_endpoint=False)
                mock_handler.handle_count_tokens_request.assert_not_called()


@pytest.mark.asyncio
async def test_factory_registration():
    """Test that the new factory pattern correctly provides counters."""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    # Test Anthropic ModelInfo provides token counter
    anthropic_model_info = AnthropicModelInfo()
    counter = anthropic_model_info.get_token_counter()
    assert counter is not None

    # Create test deployments
    anthropic_deployment = {"litellm_params": {"model": "anthropic/claude-3-5-sonnet-20241022"}}

    non_anthropic_deployment = {"litellm_params": {"model": "openai/gpt-4"}}

    # Test Anthropic counter supports provider
    assert counter.should_use_token_counting_api(custom_llm_provider="anthropic")
    assert not counter.should_use_token_counting_api(custom_llm_provider="openai")

    # Test non-Anthropic provider
    assert not counter.should_use_token_counting_api(custom_llm_provider="openai")

    # Test None deployment
    assert not counter.should_use_token_counting_api(custom_llm_provider=None)


@pytest.mark.asyncio
async def test_bedrock_count_tokens_endpoint():
    """
    Test that Bedrock CountTokens endpoint correctly extracts model from request body.
    """
    from litellm.router import Router

    # Mock the Bedrock CountTokens handler
    async def mock_count_tokens_handler(request_data, litellm_params, resolved_model):
        # Verify the correct model was resolved
        assert resolved_model == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert request_data["model"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert request_data["messages"] == [{"role": "user", "content": "Hello!"}]

        return {"input_tokens": 25}

    # Set up router with Bedrock model
    llm_router = Router(
        model_list=[
            {
                "model_name": "claude-bedrock",
                "litellm_params": {"model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"},
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    # Test the mock handler directly to verify correct parameter extraction
    request_data = {
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "messages": [{"role": "user", "content": "Hello!"}],
    }

    # Test the mock handler directly to verify correct parameter extraction
    await mock_count_tokens_handler(request_data, {}, "anthropic.claude-3-sonnet-20240229-v1:0")


@pytest.mark.asyncio
async def test_vertex_ai_anthropic_token_counting():
    """
    Unit test for Vertex AI Anthropic token counting with mocked API calls.

    This tests the token counting implementation for Vertex AI partner models
    without making actual API calls. Mocks at the handler level to test the full flow.
    """
    from unittest.mock import patch

    # Mock the Vertex AI partner models token counter response
    mock_token_response = {
        "input_tokens": 15,
        "tokenizer_used": "vertex_ai_partner_models",
    }

    llm_router = Router(
        model_list=[
            {
                "model_name": "vertex_ai/claude-3-5-sonnet-20241022",
                "litellm_params": {
                    "model": "vertex_ai/claude-3-5-sonnet-20241022",
                    "vertex_project": "test-project",
                    "vertex_location": "us-east5",
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    # Mock the lower level handler method
    with patch(
        "litellm.llms.vertex_ai.vertex_ai_partner_models.count_tokens.handler.VertexAIPartnerModelsTokenCounter.handle_count_tokens_request"
    ) as mock_handle_count_tokens:
        mock_handle_count_tokens.return_value = mock_token_response

        # Test with messages format and call_endpoint=True
        response = await count_request_tokens(
            request=TokenCountRequest(
                model="vertex_ai/claude-3-5-sonnet-20241022",
                messages=[
                    {
                        "role": "user",
                        "content": "Hello Claude on Vertex AI! How are you?",
                    }
                ],
            ),
            call_endpoint=True,
        )

        # Validate that handle_count_tokens_request was called
        assert mock_handle_count_tokens.called

        # Verify the call arguments
        call_args = mock_handle_count_tokens.call_args
        assert call_args is not None
        assert call_args.kwargs["model"] == "claude-3-5-sonnet-20241022"
        assert "messages" in call_args.kwargs["request_data"]
        assert call_args.kwargs["request_data"]["messages"][0]["content"] == "Hello Claude on Vertex AI! How are you?"

        # Validate response structure
        assert response.model_used == "claude-3-5-sonnet-20241022"
        assert response.request_model == "vertex_ai/claude-3-5-sonnet-20241022"
        assert response.total_tokens == 15
        assert response.tokenizer_type == "vertex_ai_partner_models"

        # Validate original response contains input_tokens
        assert response.original_response is not None
        assert "input_tokens" in response.original_response
        assert response.original_response["input_tokens"] == 15


@pytest.mark.parametrize("vertex_location", ["global", "us-central1"])
def test_vertex_ai_partner_models_token_counting_endpoint(vertex_location):
    """
    Test that the VertexAIPartnerModelsTokenCounter builds the correct endpoint URL
    for different vertex locations, including the special 'global' location.
    """
    from litellm.llms.vertex_ai.vertex_ai_partner_models.count_tokens.handler import (
        VertexAIPartnerModelsTokenCounter,
    )

    endpoint = VertexAIPartnerModelsTokenCounter()._build_count_tokens_endpoint(
        model="claude-3-5-sonnet-20241022",
        project_id="test-project",
        vertex_location=vertex_location,
        api_base=None,
    )
    if vertex_location == "global":
        assert endpoint.startswith("https://aiplatform.googleapis.com")
    else:
        assert endpoint.startswith(f"https://{vertex_location}-aiplatform.googleapis.com")


class _FailingRuntimeHandler(BedrockCountTokensHandler):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self._error = error

    async def handle_count_tokens_request(
        self,
        request_data: dict[str, object],
        litellm_params: dict[str, object],
        resolved_model: str,
        client: AsyncHTTPHandler | None = None,
    ) -> dict[str, JsonValue]:
        raise self._error


@pytest.mark.asyncio
async def test_bedrock_token_counter_error_propagation_bedrock_error():
    """
    Test that BedrockTokenCounter properly returns error response when BedrockError is raised.
    Verifies that the status code and error message are preserved.
    """
    counter = BedrockTokenCounter(
        runtime_handler=_FailingRuntimeHandler(BedrockError(status_code=429, message="Rate limit exceeded"))
    )

    result = await counter.count_tokens(
        model_to_use="anthropic.claude-3-sonnet",
        messages=[{"role": "user", "content": "hello"}],
        contents=None,
        deployment={"litellm_params": {}},
        request_model="bedrock/anthropic.claude-3-sonnet",
    )

    assert result is not None
    assert result.error is True
    assert result.status_code == 429
    assert "Rate limit exceeded" in result.error_message
    assert result.tokenizer_type == "bedrock_api"
    assert result.total_tokens == 0


@pytest.mark.asyncio
async def test_bedrock_token_counter_error_propagation_generic_exception():
    """
    Test that BedrockTokenCounter returns error response with 500 status for generic exceptions.
    """
    counter = BedrockTokenCounter(runtime_handler=_FailingRuntimeHandler(Exception("Unexpected error")))

    result = await counter.count_tokens(
        model_to_use="anthropic.claude-3-sonnet",
        messages=[{"role": "user", "content": "hello"}],
        contents=None,
        deployment={"litellm_params": {}},
        request_model="bedrock/anthropic.claude-3-sonnet",
    )

    assert result is not None
    assert result.error is True
    assert result.status_code == 500
    assert "Unexpected error" in result.error_message


@pytest.mark.asyncio
async def test_bedrock_handler_httpx_error_status_code_propagation():
    """
    Test that BedrockCountTokensHandler properly extracts status code from httpx.HTTPStatusError.
    """
    handler = BedrockCountTokensHandler()

    # Create a mock httpx response with 403 status
    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_response.text = "Forbidden - Invalid credentials"

    # Create HTTPStatusError
    http_error = httpx.HTTPStatusError(
        message="Client error '403 Forbidden'",
        request=MagicMock(),
        response=mock_response,
    )

    with patch.object(handler, "_get_aws_region_name", return_value="us-west-2"):
        with patch.object(handler, "transform_anthropic_to_bedrock_count_tokens", return_value={}):
            with patch.object(
                handler,
                "get_bedrock_count_tokens_endpoint",
                return_value="https://example.com",
            ):
                with patch.object(handler, "_sign_request", return_value=({}, "{}")):
                    with patch("litellm.llms.bedrock.count_tokens.handler.get_async_httpx_client") as mock_client:
                        mock_async_client = AsyncMock()
                        mock_async_client.post = AsyncMock(side_effect=http_error)
                        mock_client.return_value = mock_async_client

                        with pytest.raises(BedrockError) as exc_info:
                            await handler.handle_count_tokens_request(
                                request_data={
                                    "model": "test",
                                    "messages": [{"role": "user", "content": "hello"}],
                                },
                                litellm_params={},
                                resolved_model="anthropic.claude-3-sonnet",
                            )

                        assert exc_info.value.status_code == 403
                        # Message should be the raw response text
                        assert exc_info.value.message == "Forbidden - Invalid credentials"


@pytest.mark.asyncio
async def test_token_counter_httpx_status_error_raises_proxy_exception():
    """
    When provider_counter.count_tokens() raises httpx.HTTPStatusError,
    the token_counter endpoint should catch it and raise a ProxyException
    with the upstream status code and error message.
    """

    upstream_status = 429
    upstream_message = "Rate limit exceeded"
    response = httpx.Response(
        status_code=upstream_status,
        request=httpx.Request("POST", "https://provider.example.com/count"),
    )
    http_error = httpx.HTTPStatusError(
        message=upstream_message,
        request=response.request,
        response=response,
    )

    mock_counter = MagicMock()
    mock_counter.should_use_token_counting_api.return_value = True
    mock_counter.count_tokens = AsyncMock(side_effect=http_error)

    # Save originals
    original_get_provider_token_counter = litellm.proxy.proxy_server._get_provider_token_counter
    original_router = litellm.proxy.proxy_server.llm_router

    try:

        def mock_get_provider_token_counter(deployment, model_to_use):
            return (mock_counter, "claude-4-6-sonnet", "vertex_ai")

        litellm.proxy.proxy_server._get_provider_token_counter = mock_get_provider_token_counter

        mock_router = route_entry_points_to_stubs(MagicMock())
        mock_router.async_get_available_deployment = AsyncMock(
            return_value={
                "litellm_params": {
                    "model": "vertex_ai/claude-4-6-sonnet",
                    "api_key": "fake-key",
                },
                "model_info": {},
            }
        )
        mock_router.get_routable_deployments = MagicMock(return_value=[mock_router.async_get_available_deployment.return_value])
        litellm.proxy.proxy_server.llm_router = mock_router

        with pytest.raises(ProxyException) as exc_info:
            await count_request_tokens(
                request=TokenCountRequest(
                    model="claude-4-6-sonnet",
                    messages=[{"role": "user", "content": "hello"}],
                ),
                call_endpoint=True,
            )

        assert exc_info.value.code == str(upstream_status)
        assert upstream_message in exc_info.value.message
        assert exc_info.value.type == "token_counting_error"
        assert exc_info.value.param == "model"
    finally:
        litellm.proxy.proxy_server._get_provider_token_counter = original_get_provider_token_counter
        litellm.proxy.proxy_server.llm_router = original_router


@pytest.mark.asyncio
async def test_proxy_token_counter_error_raises_exception_when_disabled():
    """
    Test that proxy token_counter raises ProxyException when disable_token_counter=True
    and provider returns an error response.
    """
    # Create error response
    error_response = TokenCountResponse(
        total_tokens=0,
        request_model="bedrock/anthropic.claude-3-sonnet",
        model_used="anthropic.claude-3-sonnet",
        tokenizer_type="bedrock_api",
        error=True,
        error_message="Rate limit exceeded",
        status_code=429,
    )

    # Create mock router that returns a deployment
    mock_deployment = {
        "litellm_params": {
            "model": "bedrock/anthropic.claude-3-sonnet",
        },
        "model_info": {},
    }

    mock_router = route_entry_points_to_stubs(MagicMock())
    mock_router.async_get_available_deployment = AsyncMock(return_value=mock_deployment)
    mock_router.get_routable_deployments = MagicMock(return_value=[mock_router.async_get_available_deployment.return_value])

    setattr(litellm.proxy.proxy_server, "llm_router", mock_router)

    # Save original value and function
    original_disable = litellm.disable_token_counter
    original_get_provider_token_counter = litellm.proxy.proxy_server._get_provider_token_counter

    try:
        litellm.disable_token_counter = True

        # Create a mock counter that returns an error response
        mock_counter = MagicMock(spec=BedrockTokenCounter)
        mock_counter.should_use_token_counting_api.return_value = True
        mock_counter.count_tokens = AsyncMock(return_value=error_response)

        # Replace the function directly
        def mock_get_provider_token_counter(deployment, model_to_use):
            return (mock_counter, "anthropic.claude-3-sonnet", "bedrock")

        litellm.proxy.proxy_server._get_provider_token_counter = mock_get_provider_token_counter

        with pytest.raises(ProxyException) as exc_info:
            await count_request_tokens(
                request=TokenCountRequest(
                    model="claude-bedrock",
                    messages=[{"role": "user", "content": "hello"}],
                ),
                call_endpoint=True,
            )

        assert exc_info.value.code == "429"
        assert "Rate limit exceeded" in exc_info.value.message
    finally:
        litellm.disable_token_counter = original_disable
        litellm.proxy.proxy_server._get_provider_token_counter = original_get_provider_token_counter


@pytest.mark.asyncio
async def test_proxy_token_counter_error_raises_when_enabled():
    """
    Test that proxy token_counter raises ProxyException when disable_token_counter=False
    and the provider returns an error response, never falling back to the local tokenizer.
    """
    # Create error response
    error_response = TokenCountResponse(
        total_tokens=0,
        request_model="bedrock/anthropic.claude-3-sonnet",
        model_used="anthropic.claude-3-sonnet",
        tokenizer_type="bedrock_api",
        error=True,
        error_message="Rate limit exceeded",
        status_code=429,
    )

    # Create mock router that returns a deployment
    mock_deployment = {
        "litellm_params": {
            "model": "bedrock/anthropic.claude-3-sonnet",
        },
        "model_info": {},
    }

    mock_router = route_entry_points_to_stubs(MagicMock())
    mock_router.async_get_available_deployment = AsyncMock(return_value=mock_deployment)
    mock_router.get_routable_deployments = MagicMock(return_value=[mock_router.async_get_available_deployment.return_value])

    setattr(litellm.proxy.proxy_server, "llm_router", mock_router)

    # Save original value and function
    original_disable = litellm.disable_token_counter
    original_get_provider_token_counter = litellm.proxy.proxy_server._get_provider_token_counter

    try:
        litellm.disable_token_counter = False

        # Create a mock counter that returns an error response
        mock_counter = MagicMock(spec=BedrockTokenCounter)
        mock_counter.should_use_token_counting_api.return_value = True
        mock_counter.count_tokens = AsyncMock(return_value=error_response)

        # Replace the function directly
        def mock_get_provider_token_counter(deployment, model_to_use):
            return (mock_counter, "anthropic.claude-3-sonnet", "bedrock")

        litellm.proxy.proxy_server._get_provider_token_counter = mock_get_provider_token_counter

        with patch("litellm.token_counter") as mock_local_counter:
            with pytest.raises(ProxyException) as exc_info:
                await count_request_tokens(
                    request=TokenCountRequest(
                        model="claude-bedrock",
                        messages=[{"role": "user", "content": "hello"}],
                    ),
                    call_endpoint=True,
                )

        assert exc_info.value.code == "429"
        assert "Rate limit exceeded" in exc_info.value.message
        assert exc_info.value.type == "token_counting_error"
        mock_local_counter.assert_not_called()
    finally:
        litellm.disable_token_counter = original_disable
        litellm.proxy.proxy_server._get_provider_token_counter = original_get_provider_token_counter


@pytest.mark.asyncio
async def test_anthropic_endpoint_returns_anthropic_error_format():
    """
    Test that /v1/messages/count_tokens returns errors in Anthropic format.
    """
    import litellm.proxy.anthropic_endpoints.endpoints as anthropic_endpoints
    import litellm.proxy.proxy_server as proxy_server

    # Mock request object
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {}
    mock_request_data = {
        "model": "claude-bedrock",
        "messages": [{"role": "user", "content": "Hello!"}],
    }

    async def mock_read_request_body(request):
        return mock_request_data

    mock_user_api_key_dict = MagicMock()

    original_read_request_body = anthropic_endpoints.read_request_body
    anthropic_endpoints.read_request_body = mock_read_request_body

    original_token_counter = proxy_server.count_request_tokens

    # Mock token_counter to raise ProxyException with Bedrock-style error
    async def mock_token_counter_error(
        request,
        call_endpoint=False,
        messages_format="openai",
        output_format=None,
        context_management=None,
        request_litellm_params=None,
        litellm_call_id=None,
        client_oauth_token=None,
        count_options=None,
        caller=None,
        request_routing=None,
    ):
        raise ProxyException(
            message='{"detail":{"message":"Input is too long for requested model."}}',
            type="token_counting_error",
            param="model",
            code=400,
        )

    proxy_server.count_request_tokens = mock_token_counter_error

    try:
        response: Final[object] = await anthropic_count_tokens(mock_request, Response(), mock_user_api_key_dict)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400
        detail = json.loads(bytes(response.body))
        assert detail["type"] == "error"
        assert detail["error"]["type"] == "invalid_request_error"
        assert detail["error"]["message"] == "Input is too long for requested model."
    finally:
        anthropic_endpoints.read_request_body = original_read_request_body
        proxy_server.count_request_tokens = original_token_counter


@pytest.mark.asyncio
async def test_anthropic_endpoint_403_permission_error_format():
    """
    Test that 403 errors are returned as permission_error in Anthropic format.
    """
    import litellm.proxy.anthropic_endpoints.endpoints as anthropic_endpoints
    import litellm.proxy.proxy_server as proxy_server

    mock_request = MagicMock(spec=Request)
    mock_request.headers = {}
    mock_request_data = {
        "model": "claude-bedrock",
        "messages": [{"role": "user", "content": "Hello!"}],
    }

    async def mock_read_request_body(request):
        return mock_request_data

    mock_user_api_key_dict = MagicMock()

    original_read_request_body = anthropic_endpoints.read_request_body
    anthropic_endpoints.read_request_body = mock_read_request_body

    original_token_counter = proxy_server.count_request_tokens

    # Mock token_counter to raise ProxyException with 403 error
    async def mock_token_counter_error(
        request,
        call_endpoint=False,
        messages_format="openai",
        output_format=None,
        context_management=None,
        request_litellm_params=None,
        litellm_call_id=None,
        client_oauth_token=None,
        count_options=None,
        caller=None,
        request_routing=None,
    ):
        raise ProxyException(
            message='{"Message":"Bearer Token has expired"}',
            type="token_counting_error",
            param="model",
            code=403,
        )

    proxy_server.count_request_tokens = mock_token_counter_error

    try:
        response: Final[object] = await anthropic_count_tokens(mock_request, Response(), mock_user_api_key_dict)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 403
        detail = json.loads(bytes(response.body))
        assert detail["type"] == "error"
        assert detail["error"]["type"] == "permission_error"
        assert detail["error"]["message"] == "Bearer Token has expired"
    finally:
        anthropic_endpoints.read_request_body = original_read_request_body
        proxy_server.count_request_tokens = original_token_counter


@pytest.mark.asyncio
async def test_anthropic_endpoint_429_rate_limit_error_format():
    """
    Test that 429 errors are returned as rate_limit_error in Anthropic format.
    """
    import litellm.proxy.anthropic_endpoints.endpoints as anthropic_endpoints
    import litellm.proxy.proxy_server as proxy_server

    mock_request = MagicMock(spec=Request)
    mock_request.headers = {}
    mock_request_data = {
        "model": "claude-bedrock",
        "messages": [{"role": "user", "content": "Hello!"}],
    }

    async def mock_read_request_body(request):
        return mock_request_data

    mock_user_api_key_dict = MagicMock()

    original_read_request_body = anthropic_endpoints.read_request_body
    anthropic_endpoints.read_request_body = mock_read_request_body

    original_token_counter = proxy_server.count_request_tokens

    # Mock token_counter to raise ProxyException with 429 error
    async def mock_token_counter_error(
        request,
        call_endpoint=False,
        messages_format="openai",
        output_format=None,
        context_management=None,
        request_litellm_params=None,
        litellm_call_id=None,
        client_oauth_token=None,
        count_options=None,
        caller=None,
        request_routing=None,
    ):
        raise ProxyException(
            message="Rate limit exceeded",
            type="token_counting_error",
            param="model",
            code=429,
        )

    proxy_server.count_request_tokens = mock_token_counter_error

    try:
        response: Final[object] = await anthropic_count_tokens(mock_request, Response(), mock_user_api_key_dict)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 429
        detail = json.loads(bytes(response.body))
        assert detail["type"] == "error"
        assert detail["error"]["type"] == "rate_limit_error"
        assert detail["error"]["message"] == "Rate limit exceeded"
    finally:
        anthropic_endpoints.read_request_body = original_read_request_body
        proxy_server.count_request_tokens = original_token_counter


def _json_request(
    body: Mapping[str, object], user_agent: str, headers: Mapping[str, str] = MappingProxyType({})
) -> Request:
    payload: Final = json.dumps(body).encode()

    async def receive() -> Message:
        return {"type": "http.request", "body": payload, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/messages/count_tokens",
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/json"),
                (b"user-agent", user_agent.encode()),
                *((name.encode(), value.encode()) for name, value in headers.items()),
            ],
        },
        receive,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_drop_params", "user_agent"),
    (({"drop_params": True}, "python-httpx/0.28.1"), ({}, "claude-cli/2.1.0 (external, cli)")),
)
async def test_anthropic_count_tokens_drops_a_malformed_tool_result_text_when_drop_params_applies(
    monkeypatch: pytest.MonkeyPatch, request_drop_params: dict[str, bool], user_agent: str
) -> None:
    import litellm.proxy.proxy_server as proxy_server

    body: Final = {
        "model": "gpt-tool-result",
        **request_drop_params,
        "messages": [
            {"role": "user", "content": "run the build"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_log", "name": "bash", "input": {"cmd": "make"}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_log",
                        "content": [{"type": "text", "text": "build ok"}, {"type": "text", "text": None}],
                    }
                ],
            },
        ],
    }
    router: Final = Router(
        model_list=[
            {"model_name": "gpt-tool-result", "litellm_params": {"model": "openai/gpt-5.6", "api_key": "sk-test"}}
        ]
    )
    request: Final = _json_request(body, user_agent)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "proxy_config", proxy_server.ProxyConfig())
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    with respx.mock:
        route: Final = respx.post("https://api.openai.com/v1/responses/input_tokens").mock(
            return_value=httpx.Response(200, json={"input_tokens": 9})
        )
        response: Final = await anthropic_count_tokens(request, Response(), UserAPIKeyAuth())

    assert response["input_tokens"] == 9
    sent: Final = json.loads(route.calls.last.request.content)
    assert {"type": "function_call_output", "call_id": "toolu_log", "output": "build ok"} in sent["input"]


def _build_log_messages(log_text: JsonValue) -> list[dict[str, JsonValue]]:
    return [
        {"role": "user", "content": "run the build"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_log", "name": "bash", "input": {"cmd": "make"}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_log",
                    "content": [{"type": "text", "text": "build ok"}, {"type": "text", "text": log_text}],
                }
            ],
        },
    ]


async def _count_on_a_model_without_a_count_api(
    monkeypatch: pytest.MonkeyPatch, body: Mapping[str, object], router_defaults: dict[str, JsonValue] | None = None
) -> JSONResponse | dict[str, JsonValue]:
    import litellm.proxy.proxy_server as proxy_server

    router: Final = Router(
        model_list=[
            {
                "model_name": "mistral-tool-result",
                "litellm_params": {"model": "mistral/mistral-large-4", "api_key": "sk-test"},
            }
        ],
        default_litellm_params=router_defaults,
    )
    request: Final = _json_request(body, "python-httpx/0.28.1")
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "proxy_config", proxy_server.ProxyConfig())
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    with respx.mock:
        respx.post("https://api.mistral.ai/v1/responses/input_tokens").mock(
            return_value=httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 7})
        )
        return await anthropic_count_tokens(request, Response(), UserAPIKeyAuth())


@pytest.mark.asyncio
@pytest.mark.parametrize("log_text", (None, 5))
async def test_count_tokens_without_a_count_api_rejects_a_non_string_tool_result_text(
    monkeypatch: pytest.MonkeyPatch, log_text: JsonValue
) -> None:
    response: Final = await _count_on_a_model_without_a_count_api(
        monkeypatch, {"model": "mistral-tool-result", "messages": _build_log_messages(log_text)}
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert json.loads(bytes(response.body))["error"] == {
        "type": "invalid_request_error",
        "message": "tool_result toolu_log content.1.text: Input should be a valid string",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ("mistral-tool-result", "model-not-in-the-router"))
async def test_count_tokens_without_a_count_api_counts_a_non_string_tool_result_text_under_drop_params(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    response: Final = await _count_on_a_model_without_a_count_api(
        monkeypatch, {"model": model, "drop_params": True, "messages": _build_log_messages(None)}
    )

    assert isinstance(response, dict)
    assert isinstance(response["input_tokens"], int) and response["input_tokens"] > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(("body_drop_params", "expected_status"), (({}, None), ({"drop_params": False}, 400)))
async def test_count_tokens_honors_router_default_drop_params_unless_the_body_overrides_it(
    monkeypatch: pytest.MonkeyPatch, body_drop_params: dict[str, JsonValue], expected_status: int | None
) -> None:
    response: Final = await _count_on_a_model_without_a_count_api(
        monkeypatch,
        {"model": "mistral-tool-result", "messages": _build_log_messages(None), **body_drop_params},
        router_defaults={"drop_params": True},
    )

    match response:
        case JSONResponse():
            assert response.status_code == expected_status
        case _:
            assert expected_status is None
            assert isinstance(response["input_tokens"], int) and response["input_tokens"] > 0


@pytest.mark.asyncio
async def test_anthropic_count_tokens_on_a_claude_deployment_drops_a_malformed_tool_result_text_under_drop_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.proxy_server as proxy_server

    router: Final = Router(
        model_list=[
            {
                "model_name": "claude-native-drop",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-5-5",
                    "api_key": "sk-ant-test",
                    "drop_params": True,
                },
            }
        ]
    )
    request: Final = _json_request(
        {"model": "claude-native-drop", "messages": _build_log_messages(5)}, "python-httpx/0.28.1"
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "proxy_config", proxy_server.ProxyConfig())
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    with respx.mock:
        route: Final = respx.post("https://api.anthropic.com/v1/messages/count_tokens").mock(
            return_value=httpx.Response(200, json={"input_tokens": 7})
        )
        response: Final = await anthropic_count_tokens(request, Response(), UserAPIKeyAuth())

    assert response["input_tokens"] == 7
    sent: Final = json.loads(route.calls.last.request.content)
    assert sent["messages"][2]["content"][0]["content"] == [{"type": "text", "text": "build ok"}]


def _prompt_beside(text_block: Mapping[str, JsonValue]) -> list[dict[str, JsonValue]]:
    return [{"role": "user", "content": [{"type": "text", "text": "run the build"}, dict(text_block)]}]


_CLEAN_PROMPT: Final[list[dict[str, JsonValue]]] = [
    {"role": "user", "content": [{"type": "text", "text": "run the build"}]}
]

_BLANK_BEFORE_NULL_PROMPT: Final[list[dict[str, JsonValue]]] = [
    {
        "role": "user",
        "content": [{"type": "text", "text": ""}, {"type": "text", "text": None}, {"type": "text", "text": "run the build"}],
    }
]


async def _body_sent_to_anthropic_count(
    monkeypatch: pytest.MonkeyPatch, body: Mapping[str, object], deployment_params: Mapping[str, object]
) -> Mapping[str, JsonValue]:
    import litellm.proxy.proxy_server as proxy_server

    router: Final = Router(
        model_list=[
            {
                "model_name": "claude-count",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-5-5",
                    "api_key": "sk-ant-test",
                    **deployment_params,
                },
            }
        ]
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "proxy_config", proxy_server.ProxyConfig())
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    api_base: Final = deployment_params.get("api_base", "https://api.anthropic.com")
    with respx.mock:
        route: Final = respx.post(f"{api_base}/v1/messages/count_tokens").mock(
            return_value=httpx.Response(200, json={"input_tokens": 7})
        )
        response: Final = await anthropic_count_tokens(
            _json_request({"model": "claude-count", **body}, "python-httpx/0.28.1"), Response(), UserAPIKeyAuth()
        )

    assert response["input_tokens"] == 7
    return json.loads(route.calls.last.request.content)


async def _messages_sent_to_anthropic_count(
    monkeypatch: pytest.MonkeyPatch, body: Mapping[str, object], deployment_params: Mapping[str, object]
) -> JsonValue:
    return (await _body_sent_to_anthropic_count(monkeypatch, body, deployment_params))["messages"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text_block",
    ({"type": "text", "text": ""}, {"type": "text", "text": " \n"}, {"type": "text"}),
    ids=("empty", "blank", "missing"),
)
async def test_anthropic_count_tokens_on_a_claude_deployment_strips_empty_user_text_like_inference(
    monkeypatch: pytest.MonkeyPatch, text_block: Mapping[str, JsonValue]
) -> None:
    sent: Final = await _messages_sent_to_anthropic_count(monkeypatch, {"messages": _prompt_beside(text_block)}, {})

    assert sent == _CLEAN_PROMPT


@pytest.mark.asyncio
@pytest.mark.parametrize("text", (None, 5), ids=("null", "number"))
@pytest.mark.parametrize(
    ("request_params", "deployment_params"),
    (({"drop_params": True}, {}), ({}, {"drop_params": True})),
    ids=("body", "deployment"),
)
async def test_anthropic_count_tokens_on_a_claude_deployment_drops_a_non_string_user_text_under_drop_params(
    monkeypatch: pytest.MonkeyPatch,
    text: JsonValue,
    request_params: Mapping[str, JsonValue],
    deployment_params: Mapping[str, JsonValue],
) -> None:
    sent: Final = await _messages_sent_to_anthropic_count(
        monkeypatch, {"messages": _prompt_beside({"type": "text", "text": text}), **request_params}, deployment_params
    )

    assert sent == _CLEAN_PROMPT


_SYSTEM_TEXT: Final[dict[str, JsonValue]] = {"type": "text", "text": "Answer in one word"}


@pytest.mark.asyncio
@pytest.mark.parametrize("text", (None, 5), ids=("null", "number"))
@pytest.mark.parametrize(
    ("request_params", "deployment_params"),
    (({"drop_params": True}, {}), ({}, {"drop_params": True})),
    ids=("body", "deployment"),
)
async def test_anthropic_count_tokens_on_a_claude_deployment_drops_a_non_string_system_text_under_drop_params(
    monkeypatch: pytest.MonkeyPatch,
    text: JsonValue,
    request_params: Mapping[str, JsonValue],
    deployment_params: Mapping[str, JsonValue],
) -> None:
    non_string: Final[dict[str, JsonValue]] = {"type": "text", "text": text}

    dropped: Final = await _body_sent_to_anthropic_count(
        monkeypatch,
        {"messages": _CLEAN_PROMPT, "system": [non_string, _SYSTEM_TEXT], **request_params},
        deployment_params,
    )
    emptied: Final = await _body_sent_to_anthropic_count(
        monkeypatch, {"messages": _CLEAN_PROMPT, "system": [non_string], **request_params}, deployment_params
    )

    assert dropped["system"] == [_SYSTEM_TEXT]
    assert dropped["messages"] == emptied["messages"] == _CLEAN_PROMPT
    assert emptied.get("system") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deployment_params",
    ({}, {"model": "anthropic/sonnet-prod", "api_base": "http://gateway.test"}),
    ids=("anthropics-own-api", "claude-gateway"),
)
async def test_anthropic_count_tokens_on_a_claude_deployment_sends_the_messages_inference_sends(
    monkeypatch: pytest.MonkeyPatch, deployment_params: Mapping[str, JsonValue]
) -> None:
    from litellm.llms.anthropic.messages.native.utils import normalize_native_messages

    history: Final[list[dict[str, JsonValue]]] = [
        {"role": "user", "content": "run the build"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "functions.Bash:0", "name": "Bash", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "functions.Bash:0", "content": "ok"}]},
    ]

    sent: Final = await _messages_sent_to_anthropic_count(monkeypatch, {"messages": history}, deployment_params)

    assert sent == normalize_native_messages(history)
    assert sent != history


@pytest.mark.asyncio
async def test_count_tokens_without_a_count_api_counts_a_non_string_user_text_as_dropped_under_drop_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dropped: Final = await _count_on_a_model_without_a_count_api(
        monkeypatch,
        {"model": "mistral-tool-result", "drop_params": True, "messages": _prompt_beside({"type": "text", "text": 5})},
    )
    clean: Final = await _count_on_a_model_without_a_count_api(
        monkeypatch, {"model": "mistral-tool-result", "messages": _CLEAN_PROMPT}
    )

    assert isinstance(dropped, dict) and isinstance(clean, dict)
    assert dropped["input_tokens"] == clean["input_tokens"]



@pytest.mark.asyncio
@pytest.mark.parametrize("text", ("", " \n"), ids=("empty", "blank"))
async def test_anthropic_count_tokens_counts_a_request_with_only_blank_user_text_as_sent(
    monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    messages: Final[list[dict[str, JsonValue]]] = [{"role": "user", "content": [{"type": "text", "text": text}]}]

    assert await _messages_sent_to_anthropic_count(monkeypatch, {"messages": messages}, {}) == messages


_EMPTIED_MESSAGES_ERROR: Final = "messages: no message has content left after dropping empty or non-string text blocks"


def _non_string_text_error(field: str) -> str:
    return f"{field}: Input should be a valid string"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("messages", "request_params", "error"),
    (
        (
            [{"role": "user", "content": [{"type": "text", "text": None}]}],
            {"drop_params": True},
            _EMPTIED_MESSAGES_ERROR,
        ),
        (
            _prompt_beside({"type": "text", "text": None}),
            {},
            _non_string_text_error("messages.0.content.1.text"),
        ),
        (_BLANK_BEFORE_NULL_PROMPT, {}, _non_string_text_error("messages.0.content.1.text")),
        (
            _CLEAN_PROMPT,
            {"system": [{"type": "text", "text": "Answer in one word"}, {"type": "text", "text": None}]},
            _non_string_text_error("system.1.text"),
        ),
        (
            _prompt_beside({"type": "text", "text": None}),
            {"system": [{"type": "text", "text": "Answer in one word"}, {"type": "text", "text": None}]},
            _non_string_text_error("messages.0.content.1.text"),
        ),
    ),
    ids=(
        "only-null-dropped",
        "null-beside-text",
        "blank-before-null",
        "null-system-text",
        "null-user-text-before-null-system-text",
    ),
)
@pytest.mark.parametrize(
    "model",
    (
        "anthropic/claude-sonnet-5-5",
        "openai/gpt-5.6",
        "gemini/gemini-3.8-flash",
        "mistral/mistral-small-latest",
    ),
)
async def test_anthropic_count_tokens_answers_the_inference_400_before_any_provider_call(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    messages: list[dict[str, JsonValue]],
    request_params: Mapping[str, JsonValue],
    error: str,
) -> None:
    import litellm.proxy.proxy_server as proxy_server

    router: Final = Router(
        model_list=[{"model_name": "emptied-count", "litellm_params": {"model": model, "api_key": "sk-test"}}]
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "proxy_config", proxy_server.ProxyConfig())
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    with respx.mock(assert_all_called=False) as upstream_mock:
        upstream: Final = upstream_mock.route().mock(
            return_value=httpx.Response(200, json={"input_tokens": 7, "totalTokens": 7})
        )
        response: Final = await anthropic_count_tokens(
            _json_request({"model": "emptied-count", "messages": messages, **request_params}, "python-httpx/0.28.1"),
            Response(),
            UserAPIKeyAuth(),
        )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert json.loads(bytes(response.body))["error"] == {"type": "invalid_request_error", "message": error}
    assert upstream.call_count == 0


def _weather_tool(description: str) -> dict[str, object]:
    return {
        "name": "get_weather",
        "description": description,
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }


@pytest.mark.asyncio
async def test_anthropic_count_tokens_falls_back_to_a_local_estimate_that_counts_mid_conversation_tool_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "claude-framed",
                    "litellm_params": {"model": "anthropic/claude-sonnet-5-5", "api_key": "sk-ant-test"},
                    "model_info": {"token_count_framing": 100},
                }
            ]
        ),
    )
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    brief: Final = _weather_tool("Weather.")
    detailed: Final = _weather_tool("Current weather, temperature and wind for a city in any country.")

    def conversation(tool: Mapping[str, object]) -> list[dict[str, object]]:
        return [
            {"role": "user", "content": "What is the weather in Paris?"},
            {"role": "system", "content": [{"type": "tool_addition", "tool": tool}]},
            {"role": "user", "content": "Never mind."},
            {
                "role": "system",
                "content": [{"type": "tool_removal", "tool": {"type": "tool_reference", "name": "get_weather"}}],
            },
        ]

    async def estimate(tool: Mapping[str, object], tools: list[dict[str, object]] | None) -> TokenCountResponse:
        return await count_request_tokens(
            request=TokenCountRequest(model="claude-framed", messages=conversation(tool), tools=tools),
            call_endpoint=True,
            messages_format="anthropic",
        )

    by_reference: Final = {"type": "tool_reference", "name": "get_weather"}
    with respx.mock:
        route: Final = respx.post("https://api.anthropic.com/v1/messages/count_tokens").mock(
            return_value=httpx.Response(
                404, json={"type": "error", "error": {"type": "not_found_error", "message": "count unavailable"}}
            )
        )
        inline_brief: Final = await estimate({"type": "tool_definition", "definition": brief}, None)
        inline_detailed: Final = await estimate({"type": "tool_definition", "definition": detailed}, None)
        declared_brief: Final = await estimate(by_reference, [brief])
        declared_detailed: Final = await estimate(by_reference, [detailed])

    inline_delta: Final = inline_detailed.total_tokens - inline_brief.total_tokens
    declared_delta: Final = declared_detailed.total_tokens - declared_brief.total_tokens
    assert route.called
    assert {inline_brief.count_source, inline_detailed.count_source} == {"local_estimate"}
    assert declared_delta > 0
    assert inline_delta == declared_delta


class _ScriptedCounter(BaseTokenCounter):
    def __init__(self, replies: dict[str, TokenCountResponse], burst: int = 1) -> None:
        self._replies = replies
        self._burst = burst
        self._burst_arrived = asyncio.Event()
        self.calls: list[tuple[str, object, object, object]] = []
        self.regions: list[object] = []

    def should_use_token_counting_api(self, custom_llm_provider: str | None = None) -> bool:
        return True

    async def count_tokens(
        self,
        model_to_use: str,
        messages: object,
        contents: object,
        deployment: object = None,
        request_model: str = "",
        tools: object = None,
        system: object = None,
        messages_format: TokenCountMessagesFormat = "openai",
        output_format: JsonValue = None,
        context_management: object = None,
        chat_params: object = None,
        count_options: object = None,
    ) -> TokenCountResponse | None:
        self.calls.append((model_to_use, messages, tools, system))
        self.regions.append(None if deployment is None else deployment["litellm_params"].get("aws_region_name"))
        if len(self.calls) >= self._burst:
            self._burst_arrived.set()
        await self._burst_arrived.wait()
        return self._replies[model_to_use]


def _provider_reply(
    model: str, total_tokens: int = 0, status_code: int | None = None, rejected_before_provider_call: bool = False
) -> TokenCountResponse:
    return TokenCountResponse(
        total_tokens=total_tokens,
        request_model="sonnet-5-5",
        model_used=model,
        tokenizer_type="bedrock_api",
        error=status_code is not None,
        error_message=None if status_code is None else f"scripted {status_code} for {model}",
        status_code=status_code,
        rejected_before_provider_call=rejected_before_provider_call,
    )


_COUNT_REQUEST = TokenCountRequest(
    model="sonnet-5-5",
    messages=[{"role": "user", "content": "hello"}],
    tools=[{"name": "get_quote", "input_schema": {"type": "object"}}],
    system="be brief",
)


_SONNET_DEPLOYMENT: Final = MappingProxyType(
    {"litellm_params": {"model": "bedrock/us.anthropic.claude-sonnet-5-5", "aws_region_name": "us-west-2"}}
)
_OPUS_IN_US_EAST_1: Final = TokenCountModel(model="us.anthropic.claude-opus-5-5", region="us-east-1")


async def _ladder(
    counter: _ScriptedCounter,
    settings: TokenCountSettings,
    local_estimate_allowed: bool = True,
    forwards_request_fields: bool = False,
) -> TokenCountResponse | None:
    return await _provider_token_count(
        count_with=partial(
            _try_provider_token_count,
            counter,
            "bedrock",
            messages=_COUNT_REQUEST.messages,
            contents=_COUNT_REQUEST.contents,
            deployment=_SONNET_DEPLOYMENT,
            request_model=_COUNT_REQUEST.model,
            tools=_COUNT_REQUEST.tools,
            system=_COUNT_REQUEST.system,
        ),
        model_to_use="us.anthropic.claude-sonnet-5-5",
        request_model=_COUNT_REQUEST.model,
        settings=settings,
        local_estimate_allowed=local_estimate_allowed,
        forwards_request_fields=forwards_request_fields,
    )


@pytest.mark.asyncio
async def test_model_without_token_count_model_counts_itself_in_the_deployment_region():
    counter: Final = _ScriptedCounter({"us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", total_tokens=447)})

    result: Final = await _ladder(counter, TokenCountSettings(framing=10, count_model=None))

    assert result is not None
    assert (result.total_tokens, result.count_source) == (447, "provider")
    assert ([call[0] for call in counter.calls], counter.regions) == (["us.anthropic.claude-sonnet-5-5"], ["us-west-2"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "expected_host", "expected_tokens"),
    (
        ("bedrock", "bedrock-runtime.us-east-1.amazonaws.com", 193),
        ("bedrock_mantle", "bedrock-mantle.us-east-1.api.aws", 271),
    ),
)
async def test_token_count_model_provider_selects_the_endpoint_and_preserves_the_public_model(
    provider: str, expected_host: str, expected_tokens: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    model: Final = "anthropic.claude-opus-5-5"

    def peer(request: httpx.Request) -> httpx.Response:
        assert request.url.host == expected_host
        if provider == "bedrock_mantle":
            assert json.loads(request.content)["model"] == model
            return httpx.Response(200, json={"input_tokens": expected_tokens})
        assert request.url.path == f"/model/{model}/count-tokens"
        return httpx.Response(200, json={"inputTokens": expected_tokens})

    settings: Final = resolve_token_count_settings(
        configured={},
        default={"token_count_model": {"model": model, "litellm_provider": provider, "region": "us-east-1"}},
        deployment_region=lambda: "us-west-2",
    )
    counter: Final = BedrockTokenCounter(client=AsyncHTTPHandler(transport=httpx.MockTransport(peer)))
    result: Final = await _provider_token_count(
        count_with=partial(
            _try_provider_token_count,
            counter,
            "bedrock",
            messages=_COUNT_REQUEST.messages,
            contents=_COUNT_REQUEST.contents,
            request_model=_COUNT_REQUEST.model,
            deployment={
                "litellm_params": {
                    "aws_access_key_id": "AKIATESTACCESSKEY",
                    "aws_secret_access_key": "test-secret",
                    "aws_region_name": "us-west-2",
                }
            },
            tools=_COUNT_REQUEST.tools,
            system=_COUNT_REQUEST.system,
        ),
        model_to_use="us.anthropic.claude-sonnet-5-5",
        request_model=_COUNT_REQUEST.model,
        settings=settings,
        local_estimate_allowed=False,
    )

    assert isinstance(result, TokenCountResponse)
    assert (result.total_tokens, result.model_used, result.count_source) == (
        expected_tokens,
        model,
        "token_count_model",
    )


@pytest.mark.parametrize(
    ("count_model", "region"),
    [
        (_OPUS_IN_US_EAST_1, "us-east-1"),
        (TokenCountModel(model="us.anthropic.claude-opus-5-5", region=None), "us-west-2"),
    ],
)
@pytest.mark.asyncio
async def test_token_count_model_counts_the_same_request_in_its_own_region_without_trying_the_model(
    count_model: TokenCountModel, region: str
):
    counter: Final = _ScriptedCounter({"us.anthropic.claude-opus-5-5": _provider_reply("opus", total_tokens=447)})

    result: Final = await _ladder(counter, TokenCountSettings(framing=None, count_model=count_model))

    assert result is not None
    assert (result.total_tokens, result.count_source, result.model_used) == (
        447,
        "token_count_model",
        "us.anthropic.claude-opus-5-5",
    )
    assert ([call[0] for call in counter.calls], counter.regions) == (["us.anthropic.claude-opus-5-5"], [region])
    assert counter.calls[0][1:] == (_COUNT_REQUEST.messages, _COUNT_REQUEST.tools, _COUNT_REQUEST.system)
    assert _SONNET_DEPLOYMENT["litellm_params"]["aws_region_name"] == "us-west-2"


@pytest.mark.parametrize(
    "count_model",
    [None, TokenCountModel(model="us.anthropic.claude-sonnet-5-5", region=None)],
    ids=["no_token_count_model", "shipped_token_count_model_in_the_deployment_region"],
)
@pytest.mark.parametrize("status_code", [401, 403, 429, 500])
@pytest.mark.asyncio
async def test_errors_other_than_cannot_count_raise_even_with_framing(
    count_model: TokenCountModel | None, status_code: int
):
    counter: Final = _ScriptedCounter(
        {"us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", status_code=status_code)}
    )

    with pytest.raises(ProxyException) as raised:
        await _ladder(counter, TokenCountSettings(framing=10, count_model=count_model))

    assert (raised.value.code, raised.value.message) == (str(status_code), f"scripted {status_code} for sonnet")


@pytest.mark.parametrize("status_code", [400, 403, 404, 500])
@pytest.mark.asyncio
async def test_token_count_model_failure_names_its_region_so_the_user_can_enable_it(status_code: int):
    counter: Final = _ScriptedCounter(
        {"us.anthropic.claude-opus-5-5": _provider_reply("opus", status_code=status_code)}
    )

    with pytest.raises(ProxyException) as raised:
        await _ladder(counter, TokenCountSettings(framing=None, count_model=_OPUS_IN_US_EAST_1))

    assert raised.value.code == str(status_code)
    assert raised.value.message.startswith(f"scripted {status_code} for opus")
    assert "us.anthropic.claude-opus-5-5 in us-east-1" in raised.value.message
    assert len(counter.calls) == 1


@pytest.mark.asyncio
async def test_token_count_model_without_a_region_answers_its_error_unchanged():
    counter: Final = _ScriptedCounter({"us.anthropic.claude-opus-5-5": _provider_reply("opus", status_code=404)})

    with pytest.raises(ProxyException) as raised:
        await _ladder(
            counter,
            TokenCountSettings(
                framing=None, count_model=TokenCountModel(model="us.anthropic.claude-opus-5-5", region=None)
            ),
        )

    assert (raised.value.code, raised.value.message) == ("404", "scripted 404 for opus")


@pytest.mark.asyncio
async def test_uncountable_model_without_framing_raises():
    counter: Final = _ScriptedCounter({"us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", status_code=404)})

    with pytest.raises(ProxyException) as raised:
        await _ladder(counter, TokenCountSettings(framing=None, count_model=None))

    assert raised.value.code == "404"


@pytest.mark.parametrize(
    "settings",
    [
        TokenCountSettings(framing=None, count_model=None, tool_framing=300),
        TokenCountSettings(framing=None, count_model=None, responses_framing=145),
        TokenCountSettings(framing=None, count_model=None, responses_tool_framing=100),
    ],
)
@pytest.mark.asyncio
async def test_uncountable_model_with_only_a_request_kind_framing_falls_through_to_the_local_estimate(
    settings: TokenCountSettings,
):
    counter: Final = _ScriptedCounter({"us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", status_code=404)})

    assert isinstance(await _ladder(counter, settings), ProviderCountFailure)


@pytest.mark.parametrize(
    ("count_model", "counted_as"),
    [
        (None, "us.anthropic.claude-sonnet-5-5"),
        (
            TokenCountModel(model="anthropic.claude-sonnet-5-5", region=None, litellm_provider="bedrock"),
            "bedrock/anthropic.claude-sonnet-5-5",
        ),
        (_OPUS_IN_US_EAST_1, "us.anthropic.claude-opus-5-5"),
    ],
    ids=[
        "no_token_count_model",
        "shipped_token_count_model_in_the_deployment_region",
        "shipped_token_count_model_pinned_to_a_region",
    ],
)
@pytest.mark.asyncio
async def test_uncountable_model_with_framing_falls_through_to_the_local_estimate(
    count_model: TokenCountModel | None, counted_as: str
):
    counter: Final = _ScriptedCounter({counted_as: _provider_reply("sonnet", status_code=400)})

    result: Final = await _ladder(counter, TokenCountSettings(framing=0, count_model=count_model))

    assert isinstance(result, ProviderCountFailure)
    assert [call[0] for call in counter.calls] == [counted_as]


@pytest.mark.parametrize("status_code", [401, 403])
@pytest.mark.asyncio
async def test_framing_stands_in_when_a_shipped_token_count_model_region_denies_the_account(status_code: int):
    counter: Final = _ScriptedCounter(
        {"us.anthropic.claude-opus-5-5": _provider_reply("opus", status_code=status_code)}
    )

    result: Final = await _ladder(counter, TokenCountSettings(framing=50, count_model=_OPUS_IN_US_EAST_1))

    assert isinstance(result, ProviderCountFailure)
    assert result.status_code == status_code


@pytest.mark.parametrize(
    ("count_model", "region_hints"),
    [
        (replace(_OPUS_IN_US_EAST_1, configured=True), 1),
        (replace(_OPUS_IN_US_EAST_1, region=None, configured=True), 0),
    ],
    ids=["pinned_to_a_region", "in_the_deployment_region"],
)
@pytest.mark.parametrize("status_code", [400, 401, 403, 404])
@pytest.mark.asyncio
async def test_configured_token_count_model_error_stands_even_with_framing(
    count_model: TokenCountModel, region_hints: int, status_code: int
):
    counter: Final = _ScriptedCounter(
        {"us.anthropic.claude-opus-5-5": _provider_reply("opus", status_code=status_code)}
    )

    with pytest.raises(ProxyException) as raised:
        await _ladder(counter, TokenCountSettings(framing=78, count_model=count_model))

    assert raised.value.code == str(status_code)
    assert raised.value.message.startswith(f"scripted {status_code} for opus")
    assert raised.value.message.count("counts tokens with us.anthropic.claude-opus-5-5 in us-east-1") == region_hints


@pytest.mark.parametrize(("status_code", "region_hints"), [(400, 0), (403, 1)])
@pytest.mark.asyncio
async def test_a_pinned_token_count_model_rejecting_forwarded_edits_answers_without_the_region_hint(
    status_code: int, region_hints: int
):
    counter: Final = _ScriptedCounter(
        {"us.anthropic.claude-opus-5-5": _provider_reply("opus", status_code=status_code)}
    )

    with pytest.raises(ProxyException) as raised:
        await _ladder(
            counter,
            TokenCountSettings(framing=None, count_model=replace(_OPUS_IN_US_EAST_1, configured=True)),
            forwards_request_fields=True,
        )

    assert raised.value.code == str(status_code)
    assert raised.value.message.count("counts tokens with us.anthropic.claude-opus-5-5 in us-east-1") == region_hints


@pytest.mark.parametrize(
    ("count_model", "region_hints"),
    [(_OPUS_IN_US_EAST_1, frozenset({True})), (replace(_OPUS_IN_US_EAST_1, region=None), frozenset())],
    ids=["pinned_to_a_region", "in_the_deployment_region"],
)
@pytest.mark.asyncio
async def test_framing_standing_in_for_a_pinned_token_count_model_warns_with_its_model_and_region(
    count_model: TokenCountModel, region_hints: frozenset[bool], caplog: pytest.LogCaptureFixture
):
    counter: Final = _ScriptedCounter({"us.anthropic.claude-opus-5-5": _provider_reply("opus", status_code=404)})
    verbose_proxy_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=verbose_proxy_logger.name):
            result: Final = await _ladder(counter, TokenCountSettings(framing=50, count_model=count_model))
    finally:
        verbose_proxy_logger.removeHandler(caplog.handler)

    assert isinstance(result, ProviderCountFailure)
    stand_in_warnings: Final = [
        r.getMessage() for r in caplog.records if "falls back to the framed local estimate" in r.getMessage()
    ]
    assert {
        f"for {_COUNT_REQUEST.model} answered 404 from us.anthropic.claude-opus-5-5 in us-east-1" in message
        for message in stand_in_warnings
    } == region_hints


@pytest.mark.asyncio
async def test_uncountable_model_with_framing_raises_when_local_counting_is_disabled():
    counter: Final = _ScriptedCounter({"us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", status_code=400)})

    with pytest.raises(ProxyException) as raised:
        await _ladder(counter, TokenCountSettings(framing=10, count_model=None), local_estimate_allowed=False)

    assert raised.value.code == "400"


@pytest.mark.asyncio
async def test_request_the_counter_rejects_before_calling_the_provider_answers_400_without_framing():
    counter: Final = _ScriptedCounter(
        {
            "us.anthropic.claude-sonnet-5-5": _provider_reply(
                "sonnet", status_code=400, rejected_before_provider_call=True
            )
        }
    )

    with pytest.raises(ProxyException) as raised:
        await _ladder(counter, TokenCountSettings(framing=100, count_model=None))

    assert (raised.value.code, raised.value.message) == ("400", "scripted 400 for sonnet")


@pytest.mark.parametrize("model", ["openai/gpt-5.4-mini", "gemini/gemini-3.8-flash"])
@pytest.mark.asyncio
async def test_framing_keeps_the_400_for_blocks_the_provider_counter_cannot_convert(
    monkeypatch: pytest.MonkeyPatch, model: str
):
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "framed",
                    "litellm_params": {"model": model, "api_key": "sk-unused-framed"},
                    "model_info": {"token_count_framing": 100},
                }
            ]
        ),
    )
    search_result: Final = {"type": "search_result", "source": "https://example.com", "title": "Notes", "content": []}

    with pytest.raises(ProxyException) as raised:
        await count_request_tokens(
            request=TokenCountRequest(
                model="framed",
                messages=[{"role": "user", "content": [{"type": "text", "text": "Summarize."}, search_result]}],
            ),
            call_endpoint=True,
            messages_format="anthropic",
        )

    assert raised.value.code == "400"
    assert "search_result" in raised.value.message


@pytest.mark.asyncio
async def test_framed_estimate_counts_server_tool_history_when_the_provider_counter_rejects_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _framed_kimi(monkeypatch)
    _available_kimi_k3(monkeypatch, tmp_path)

    async def count(title: str) -> TokenCountResponse:
        return await count_request_tokens(
            request=TokenCountRequest(
                model="kimi",
                messages=[
                    {"role": "user", "content": "What is new?"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "server_tool_use",
                                "id": "srvtoolu_01",
                                "name": "web_search",
                                "input": {"query": "new"},
                            },
                            {
                                "type": "web_search_tool_result",
                                "tool_use_id": "srvtoolu_01",
                                "content": [
                                    {"type": "web_search_result", "url": "https://example.invalid", "title": title}
                                ],
                            },
                        ],
                    },
                ],
            ),
            call_endpoint=True,
        )

    short: Final = await count("Release")
    long: Final = await count("Release notes " * 100)

    assert (short.count_source, long.count_source) == ("local_estimate", "local_estimate")
    assert long.total_tokens > short.total_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unreadable_body",
    [
        pytest.param(
            {"messages": [], "contents": [{"role": "user", "parts": [{"text": "What is new?"}]}]},
            id="google_contents",
        ),
        pytest.param(
            {
                "messages": [{"role": "user", "content": "Describe this"}],
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": "Describe this"},
                            {"fileData": {"mimeType": "image/png", "fileUri": "gs://b/c.png"}},
                        ],
                    }
                ],
            },
            id="google_contents_with_media_the_translation_dropped",
        ),
        pytest.param(
            {
                "messages": [
                    {"role": "user", "content": "What is new?"},
                    {"role": "assistant", "content": "Weighing the release notes"},
                ],
                "contents": [
                    {"role": "user", "parts": [{"text": "What is new?"}]},
                    {"role": "model", "parts": [{"text": "Weighing the release notes", "thought": True}]},
                ],
            },
            id="google_contents_with_a_thought_the_translation_counts_as_text",
        ),
    ],
)
async def test_framed_estimate_that_cannot_read_the_body_answers_the_provider_error(
    monkeypatch: pytest.MonkeyPatch, unreadable_body: dict[str, list[dict[str, object]]]
):
    counter: Final = _ScriptedCounter({"us.moonshotai.kimi-k3": _provider_reply("kimi", status_code=400)})
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(model_list=[{"model_name": "kimi", "litellm_params": {"model": "bedrock/us.moonshotai.kimi-k3"}}]),
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server, "_get_provider_token_counter", lambda deployment, model: (counter, model, "bedrock")
    )

    with pytest.raises(ProxyException) as raised:
        await count_request_tokens(request=TokenCountRequest(model="kimi", **unreadable_body), call_endpoint=True)
    readable: Final = await count_request_tokens(
        request=TokenCountRequest(model="kimi", messages=[{"role": "user", "content": "What is new?"}]),
        call_endpoint=True,
    )

    assert (raised.value.code, raised.value.message) == ("400", "scripted 400 for kimi")
    assert readable.count_source == "local_estimate"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("shorthand", "explicit"),
    [
        pytest.param({"type": "array"}, {"type": "array", "items": {}}, id="array_schema_without_items"),
        pytest.param(True, {}, id="boolean_subschema"),
    ],
)
async def test_framed_estimate_counts_a_json_schema_shorthand_like_its_explicit_form(
    monkeypatch: pytest.MonkeyPatch, shorthand: object, explicit: dict[str, object]
):
    counter: Final = _ScriptedCounter({"us.moonshotai.kimi-k3": _provider_reply("kimi", status_code=400)})
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(model_list=[{"model_name": "kimi", "litellm_params": {"model": "bedrock/us.moonshotai.kimi-k3"}}]),
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server, "_get_provider_token_counter", lambda deployment, model: (counter, model, "bedrock")
    )

    async def count(schema: object) -> TokenCountResponse:
        return await count_request_tokens(
            request=TokenCountRequest(
                model="kimi",
                messages=[{"role": "user", "content": "What is new?"}],
                tools=[{"name": "search", "input_schema": {"type": "object", "properties": {"tags": schema}}}],
            ),
            call_endpoint=True,
        )

    from_shorthand: Final = await count(shorthand)
    from_explicit: Final = await count(explicit)

    assert from_shorthand.count_source == "local_estimate"
    assert from_shorthand.total_tokens == from_explicit.total_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "contents",
    [
        pytest.param([{"role": "user", "parts": [{"text": "What is new in the release notes?"}]}], id="text"),
        pytest.param(
            [
                {"role": "user", "parts": [{"text": "Weather in Paris?"}]},
                {
                    "role": "model",
                    "parts": [
                        {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}, "thoughtSignature": "c2ln"}
                    ],
                },
                {
                    "role": "user",
                    "parts": [{"functionResponse": {"name": "get_weather", "response": {"forecast": "Sun"}}}],
                },
            ],
            id="sdk_agent_loop",
        ),
        pytest.param(
            [
                {"role": "user", "parts": [{"text": "Hi"}]},
                {"role": "model", "parts": [{"text": "Hello!", "thoughtSignature": "c2ln"}]},
                {"role": "user", "parts": [{"text": "What is new?"}]},
            ],
            id="sdk_chat_with_a_thought_signature",
        ),
        pytest.param([], id="empty"),
    ],
)
async def test_framed_estimate_counts_google_contents_like_the_same_conversation_sent_as_messages(
    monkeypatch: pytest.MonkeyPatch, contents: list[dict[str, object]]
):
    _framed_kimi(monkeypatch)
    google_request: Final = _google_count_tokens_request("kimi", {"contents": contents})

    from_google: Final = await count_request_tokens(request=google_request, call_endpoint=True)
    from_messages: Final = await count_request_tokens(
        request=TokenCountRequest(model="kimi", messages=google_request.messages), call_endpoint=True
    )

    assert from_google.count_source == "local_estimate"
    assert from_google.total_tokens == from_messages.total_tokens


def _local_luna(monkeypatch: pytest.MonkeyPatch) -> bool:
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(model_list=[{"model_name": "kimi", "litellm_params": {"model": "openai/gpt-6-luna"}}]),
    )
    return False


def _framed_kimi_count(monkeypatch: pytest.MonkeyPatch) -> bool:
    _framed_kimi(monkeypatch)
    return True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deployment",
    [pytest.param(_local_luna, id="local_count"), pytest.param(_framed_kimi_count, id="provider_count_fails")],
)
@pytest.mark.parametrize(
    "contents",
    [
        pytest.param([{"role": "user", "parts": [{"text": "Return exactly OK."}]}], id="text"),
        pytest.param([{"role": "user", "parts": {"text": "Return exactly OK."}}], id="parts_as_one_object"),
        pytest.param(
            [
                {"role": "user", "parts": [{"text": "Weather in Paris?"}]},
                {"role": "model", "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}]},
                {
                    "role": "user",
                    "parts": [{"functionResponse": {"name": "get_weather", "response": {"forecast": "Sun"}}}],
                },
            ],
            id="tool_loop",
        ),
    ],
)
async def test_utils_token_counter_counts_a_contents_only_body_like_the_google_count_tokens_route(
    monkeypatch: pytest.MonkeyPatch,
    deployment: Callable[[pytest.MonkeyPatch], bool],
    contents: Sequence[Mapping[str, object]],
):
    call_endpoint: Final = deployment(monkeypatch)

    from_utils: Final = await _token_counter_route(
        request=TokenCountRequest(model="kimi", contents=[dict(content) for content in contents]),
        call_endpoint=call_endpoint,
    )
    from_google: Final = await _token_counter_route(
        request=_google_count_tokens_request("kimi", {"contents": contents}), call_endpoint=call_endpoint
    )

    assert from_utils.count_source == "local_estimate"
    assert from_utils.total_tokens == from_google.total_tokens > 0


_WEATHER_CONTENTS: Final = ({"role": "user", "parts": [{"text": "What is the weather in Paris?"}]},)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deployment",
    [pytest.param(_local_luna, id="local_count"), pytest.param(_framed_kimi_count, id="provider_count_fails")],
)
async def test_utils_token_counter_counts_an_entry_without_parts_as_empty(
    monkeypatch: pytest.MonkeyPatch, deployment: Callable[[pytest.MonkeyPatch], bool]
):
    call_endpoint: Final = deployment(monkeypatch)

    with_partless_entries: Final = await _token_counter_route(
        request=TokenCountRequest(model="kimi", contents=[*_WEATHER_CONTENTS, {"role": "user"}, {}]),
        call_endpoint=call_endpoint,
    )
    without_them: Final = await _token_counter_route(
        request=TokenCountRequest(model="kimi", contents=list(_WEATHER_CONTENTS)), call_endpoint=call_endpoint
    )

    assert with_partless_entries.total_tokens == without_them.total_tokens > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "field", "message"),
    [
        pytest.param(
            {"contents": [{"role": "user", "parts": [{"text": "Hi"}]}, {"role": "user", "parts": 4}]},
            "contents",
            "Invalid contents[1].parts: expected a list",
            id="second_entry_parts_number",
        ),
        pytest.param(
            {"contents": list(_WEATHER_CONTENTS), "systemInstruction": {"parts": 4}},
            "systemInstruction",
            "Invalid systemInstruction.parts: expected a list",
            id="system_instruction_parts_number",
        ),
        pytest.param(
            {"contents": list(_WEATHER_CONTENTS), "systemInstruction": "Answer briefly."},
            "systemInstruction",
            "Invalid systemInstruction: expected an object",
            id="system_instruction_string",
        ),
        pytest.param(
            {"contents": list(_WEATHER_CONTENTS), "systemInstruction": {"parts": [{"text": 4}]}},
            "systemInstruction",
            "Invalid systemInstruction.parts[0].text: expected a string",
            id="system_instruction_text_number",
        ),
    ],
)
async def test_utils_token_counter_answers_400_for_a_google_body_it_cannot_read(
    monkeypatch: pytest.MonkeyPatch, body: dict[str, object], field: str, message: str
):
    _local_luna(monkeypatch)

    with pytest.raises(ProxyException) as rejected:
        await _token_counter_route(request=TokenCountRequest(model="kimi", **body), call_endpoint=False)

    assert rejected.value.code == "400"
    assert rejected.value.param == field
    assert rejected.value.message == message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extras",
    [
        pytest.param(
            {"systemInstruction": {"parts": [{"text": "Answer in one short sentence."}]}}, id="system_instruction"
        ),
        pytest.param(
            {"system_instruction": {"parts": [{"text": "Answer in one short sentence."}]}},
            id="snake_case_system_instruction",
        ),
        pytest.param(
            {
                "tools": [
                    {
                        "functionDeclarations": [
                            {
                                "name": "get_weather",
                                "description": "Look up the forecast for a city",
                                "parameters": {"type": "OBJECT", "properties": {"city": {"type": "STRING"}}},
                            }
                        ]
                    }
                ]
            },
            id="google_tools",
        ),
    ],
)
async def test_utils_token_counter_counts_system_instruction_and_google_tools_like_the_google_count_tokens_route(
    monkeypatch: pytest.MonkeyPatch, extras: dict[str, object]
):
    _local_luna(monkeypatch)
    contents: Final = list(_WEATHER_CONTENTS)

    from_utils: Final = await _token_counter_route(
        request=TokenCountRequest(model="kimi", contents=contents, **extras), call_endpoint=False
    )
    from_google: Final = await _token_counter_route(
        request=_google_count_tokens_request("kimi", {"contents": contents, **extras}), call_endpoint=False
    )
    contents_only: Final = await _token_counter_route(
        request=TokenCountRequest(model="kimi", contents=contents), call_endpoint=False
    )

    assert from_utils.total_tokens == from_google.total_tokens > contents_only.total_tokens


_SYSTEM_TEXT: Final = {"text": "Answer in one short sentence."}
_SYSTEM_IMAGE: Final = {"inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}}
_SYSTEM_PDF: Final = {"text": None, "fileData": {"mimeType": "application/pdf", "fileUri": "gs://bucket/rules.pdf"}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("parts", "countable_parts"),
    [
        pytest.param([_SYSTEM_TEXT, _SYSTEM_IMAGE], [_SYSTEM_TEXT], id="text_and_image"),
        pytest.param([_SYSTEM_PDF, _SYSTEM_TEXT], [_SYSTEM_TEXT], id="pdf_and_text"),
        pytest.param(_SYSTEM_IMAGE, [], id="image_alone"),
    ],
)
async def test_utils_token_counter_counts_the_system_text_beside_a_system_part_it_cannot_translate(
    monkeypatch: pytest.MonkeyPatch, parts: object, countable_parts: list[object]
):
    _local_luna(monkeypatch)
    contents: Final = list(_WEATHER_CONTENTS)

    with_media: Final = await _token_counter_route(
        request=TokenCountRequest(model="kimi", contents=contents, systemInstruction={"parts": parts}),
        call_endpoint=False,
    )
    countable: Final = await _token_counter_route(
        request=TokenCountRequest(model="kimi", contents=contents, systemInstruction={"parts": countable_parts}),
        call_endpoint=False,
    )

    assert with_media.total_tokens == countable.total_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool",
    [
        pytest.param(
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            },
            id="openai_tool",
        ),
        pytest.param(
            {
                "name": "get_weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
            id="anthropic_tool",
        ),
        pytest.param(
            {
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            },
            id="openai_tool_without_type",
        ),
    ],
)
async def test_utils_token_counter_keeps_its_own_system_and_tools_next_to_google_contents(
    monkeypatch: pytest.MonkeyPatch, tool: dict[str, object]
):
    _local_luna(monkeypatch)
    contents: Final = list(_WEATHER_CONTENTS)
    system: Final = "Answer in one short sentence."
    tools: Final = [tool]

    from_contents: Final = await _token_counter_route(
        request=TokenCountRequest(
            model="kimi",
            contents=contents,
            system=system,
            tools=tools,
            systemInstruction={"parts": [{"text": "A much longer instruction that the route's own system replaces."}]},
        ),
        call_endpoint=False,
    )
    from_messages: Final = await _token_counter_route(
        request=TokenCountRequest(
            model="kimi",
            messages=_google_count_tokens_request("kimi", {"contents": contents}).messages,
            system=system,
            tools=tools,
        ),
        call_endpoint=False,
    )

    assert from_contents.total_tokens == from_messages.total_tokens


@pytest.mark.asyncio
async def test_utils_token_counter_drops_a_prompt_sent_with_contents_like_one_sent_with_messages(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[{"model_name": "kimi", "litellm_params": {"model": "openai/gpt-6-luna", "drop_params": True}}]
        ),
    )
    prompt: Final = "one two three four five six seven eight nine ten eleven twelve"
    contents: Final = [{"role": "user", "parts": [{"text": "What is the weather in Paris and Rome?"}]}]

    from_contents: Final = await _token_counter_route(
        request=TokenCountRequest(model="kimi", prompt=prompt, contents=contents), call_endpoint=False
    )
    from_messages: Final = await _token_counter_route(
        request=TokenCountRequest(
            model="kimi", prompt=prompt, messages=_google_count_tokens_request("kimi", {"contents": contents}).messages
        ),
        call_endpoint=False,
    )
    prompt_only: Final = await _token_counter_route(
        request=TokenCountRequest(model="kimi", prompt=prompt), call_endpoint=False
    )

    assert from_contents.total_tokens == from_messages.total_tokens != prompt_only.total_tokens


@pytest.mark.asyncio
async def test_framed_deployment_whose_custom_tokenizer_cannot_load_answers_the_provider_error(
    monkeypatch: pytest.MonkeyPatch,
):
    counter: Final = _ScriptedCounter({"us.moonshotai.kimi-k3": _provider_reply("kimi", status_code=400)})
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "kimi",
                    "litellm_params": {"model": "bedrock/us.moonshotai.kimi-k3"},
                    "model_info": {"custom_tokenizer": {"identifier": "moonshotai/kimi-k3", "revision": "main"}},
                }
            ]
        ),
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server, "_get_provider_token_counter", lambda deployment, model: (counter, model, "bedrock")
    )

    with pytest.raises(ProxyException) as raised:
        await count_request_tokens(
            request=TokenCountRequest(model="kimi", messages=[{"role": "user", "content": "What is new?"}]),
            call_endpoint=True,
        )

    assert (raised.value.code, raised.value.message) == ("400", "scripted 400 for kimi")


def _framed_kimis_with_custom_tokenizers(
    monkeypatch: pytest.MonkeyPatch, tokenizers: Mapping[str, tuple[str, str]], burst: int = 1
) -> None:
    counter: Final = _ScriptedCounter(
        {
            "us.moonshotai.kimi-k3": _provider_reply("kimi", status_code=400),
            "us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", total_tokens=12),
        },
        burst=burst,
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "sonnet",
                    "litellm_params": {"model": "bedrock/us.anthropic.claude-sonnet-5-5"},
                    "model_info": {"token_count_model": {}},
                },
                *(
                    {
                        "model_name": name,
                        "litellm_params": {"model": "bedrock/us.moonshotai.kimi-k3"},
                        "model_info": {
                            "custom_tokenizer": {"identifier": identifier, "revision": revision, "auth_token": None}
                        },
                    }
                    for name, (identifier, revision) in tokenizers.items()
                ),
            ]
        ),
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server, "_get_provider_token_counter", lambda deployment, model: (counter, model, "bedrock")
    )


def _framed_kimi_with_custom_tokenizer(monkeypatch: pytest.MonkeyPatch, identifier: str) -> None:
    _framed_kimis_with_custom_tokenizers(monkeypatch, {"kimi": (identifier, "main")})


async def _count_status(model: str = "kimi") -> str:
    try:
        response: Final = await count_request_tokens(
            request=TokenCountRequest(model=model, messages=[{"role": "user", "content": "What is new?"}]),
            call_endpoint=True,
        )
    except ProxyException as e:
        return e.code
    return response.count_source or ""


class _SkippableClockLoop(asyncio.SelectorEventLoop):
    skipped_seconds: float = 0.0

    def time(self) -> float:
        return super().time() + self.skipped_seconds

    def skip(self, seconds: float) -> None:
        self.skipped_seconds += seconds


def _run_on_skippable_clock(
    scenario: Callable[[_SkippableClockLoop], Coroutine[object, object, tuple[str, ...]]],
) -> tuple[str, ...]:
    loop: Final = _SkippableClockLoop()
    try:
        return loop.run_until_complete(scenario(loop))
    finally:
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()


async def _until_finished(tasks: tuple[asyncio.Future[str], ...], count: int) -> None:
    for finished in itertools.islice(asyncio.as_completed(tasks), count):
        await finished


def _tokenizer_that_panics_while_encoding() -> SelectTokenizerResponse:
    template: Final = Tokenizer(WordLevel({"[UNK]": 0, "<s>": 1}, unk_token="[UNK]"))
    template.pre_tokenizer = Whitespace()
    template.post_processor = TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 1)])
    spec: Final = json.loads(template.to_str())
    return litellm.utils.create_tokenizer(
        json.dumps({**spec, "post_processor": {**spec["post_processor"], "special_tokens": {}}})
    )


def test_hanging_tokenizer_download_answers_the_provider_error_after_the_wait_with_one_download_while_it_runs(
    monkeypatch: pytest.MonkeyPatch,
):
    _framed_kimi_with_custom_tokenizer(monkeypatch, "lit-tests/hanging-tokenizer")
    download_started: Final = threading.Event()
    hub_released: Final = threading.Event()
    downloads: Final[list[str]] = []

    def hanging_download(model: str, custom_tokenizer: object = None) -> NoReturn:
        downloads.append(model)
        download_started.set()
        hub_released.wait(10)
        raise OSError("hub unreachable")

    monkeypatch.setattr(litellm.utils, "select_tokenizer", hanging_download)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_WAIT_SECONDS", 30)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_REUSE_SECONDS", 1e-9)

    async def burst_past_the_wait_then_one_more(loop: _SkippableClockLoop) -> tuple[str, ...]:
        try:
            burst: Final = asyncio.gather(*(_count_status() for _ in range(60)))
            await asyncio.to_thread(download_started.wait, 10)
            loop.skip(31)
            answered: Final = await asyncio.wait_for(burst, 8)
            later: Final = await asyncio.wait_for(_count_status(), 8)
        finally:
            hub_released.set()
        return (*answered, later)

    assert set(_run_on_skippable_clock(burst_past_the_wait_then_one_more)) == {"400"}
    assert len(downloads) == 1


@pytest.mark.asyncio
async def test_framed_count_waits_for_a_tokenizer_download_that_finishes_within_the_wait(
    monkeypatch: pytest.MonkeyPatch,
):
    _framed_kimi_with_custom_tokenizer(monkeypatch, "lit-tests/slow-tokenizer")
    real_select_tokenizer: Final = litellm.utils.select_tokenizer
    download_started: Final = threading.Event()
    hub_released: Final = threading.Event()

    def slow_download(model: str, custom_tokenizer: object = None):
        download_started.set()
        hub_released.wait(10)
        return real_select_tokenizer(model="gpt-5.4-mini")

    monkeypatch.setattr(litellm.utils, "select_tokenizer", slow_download)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_WAIT_SECONDS", 30)
    count: Final = asyncio.create_task(_count_status())
    try:
        await asyncio.to_thread(download_started.wait, 10)
    finally:
        hub_released.set()

    assert await asyncio.wait_for(count, 10) == "local_estimate"


@pytest.mark.asyncio
async def test_failed_tokenizer_download_is_reused_until_the_retry_window_passes(monkeypatch: pytest.MonkeyPatch):
    _framed_kimi_with_custom_tokenizer(monkeypatch, "lit-tests/flaky-tokenizer")
    real_select_tokenizer: Final = litellm.utils.select_tokenizer
    downloads: Final[list[str]] = []

    def hub_down_once(model: str, custom_tokenizer: object = None):
        downloads.append(model)
        if len(downloads) == 1:
            raise OSError("hub unreachable")
        return real_select_tokenizer(model="gpt-5.4-mini")

    monkeypatch.setattr(litellm.utils, "select_tokenizer", hub_down_once)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_REUSE_SECONDS", 1e9)
    within_window: Final = [await _count_status() for _ in range(3)]
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_REUSE_SECONDS", 1e-9)
    after_window: Final = await _count_status()

    assert within_window == ["400", "400", "400"]
    assert after_window == "local_estimate"
    assert len(downloads) == 2


def test_failed_tokenizer_load_is_reused_for_the_window_after_it_fails(monkeypatch: pytest.MonkeyPatch):
    _framed_kimi_with_custom_tokenizer(monkeypatch, "lit-tests/slow-failing-tokenizer")
    download_started: Final = threading.Event()
    hub_released: Final = threading.Event()
    downloads: Final[list[str]] = []

    def slow_failing_download(model: str, custom_tokenizer: object = None) -> NoReturn:
        downloads.append(model)
        download_started.set()
        hub_released.wait(10)
        raise OSError("hub unreachable")

    monkeypatch.setattr(litellm.utils, "select_tokenizer", slow_failing_download)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_WAIT_SECONDS", 1000)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_REUSE_SECONDS", 60)

    async def fail_after_longer_than_the_window_then_retry(loop: _SkippableClockLoop) -> tuple[str, ...]:
        first: Final = asyncio.ensure_future(_count_status())
        try:
            await asyncio.to_thread(download_started.wait, 10)
            loop.skip(120)
        finally:
            hub_released.set()
        return await asyncio.wait_for(first, 8), await asyncio.wait_for(_count_status(), 8)

    assert _run_on_skippable_clock(fail_after_longer_than_the_window_then_retry) == ("400", "400")
    assert len(downloads) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "second_tokenizer",
    [("lit-tests/tokenizer-b", "main"), ("lit-tests/tokenizer-a", "v2")],
    ids=["identifier", "revision"],
)
async def test_framed_deployments_with_different_custom_tokenizers_each_load_their_own(
    monkeypatch: pytest.MonkeyPatch, second_tokenizer: tuple[str, str]
):
    _framed_kimis_with_custom_tokenizers(
        monkeypatch, {"kimi-a": ("lit-tests/tokenizer-a", "main"), "kimi-b": second_tokenizer}
    )
    real_select_tokenizer: Final = litellm.utils.select_tokenizer
    loaded: Final[list[object]] = []

    def load(model: str, custom_tokenizer: object = None) -> SelectTokenizerResponse:
        loaded.append(custom_tokenizer)
        return real_select_tokenizer(model="gpt-5.4-mini")

    monkeypatch.setattr(litellm.utils, "select_tokenizer", load)

    statuses: Final = [await _count_status(name) for name in ("kimi-a", "kimi-b")]

    assert statuses == ["local_estimate", "local_estimate"]
    assert loaded == [
        {"identifier": "lit-tests/tokenizer-a", "revision": "main", "auth_token": None},
        {"identifier": second_tokenizer[0], "revision": second_tokenizer[1], "auth_token": None},
    ]


@pytest.mark.asyncio
async def test_tokenizer_loads_past_the_cap_answer_the_provider_error_without_stalling_other_counts(
    monkeypatch: pytest.MonkeyPatch,
):
    tokenizers: Final = {f"kimi-{n}": (f"lit-tests/unreachable-tokenizer-{n}", "main") for n in range(5)}
    _framed_kimis_with_custom_tokenizers(monkeypatch, tokenizers, burst=len(tokenizers))
    real_select_tokenizer: Final = litellm.utils.select_tokenizer
    hub_released: Final = threading.Event()
    downloads: Final[list[object]] = []

    def held_download(model: str, custom_tokenizer: object = None) -> SelectTokenizerResponse:
        downloads.append(custom_tokenizer)
        hub_released.wait(10)
        return real_select_tokenizer(model="gpt-5.4-mini")

    monkeypatch.setattr(litellm.utils, "select_tokenizer", held_download)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_MAX_RUNNING", 2)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_WAIT_SECONDS", 30)
    monkeypatch.setattr(anyio.to_thread.current_default_thread_limiter(), "total_tokens", 2)
    framed: Final = tuple(asyncio.ensure_future(_count_status(name)) for name in tokenizers)
    try:
        await asyncio.wait_for(_until_finished(framed, 3), 8)
        unrelated: Final = await asyncio.wait_for(_count_status("sonnet"), 5)
    finally:
        hub_released.set()
    statuses: Final = await asyncio.wait_for(asyncio.gather(*framed), 8)

    assert sorted(statuses) == ["400"] * 3 + ["local_estimate"] * 2
    assert unrelated == "provider"
    assert len(downloads) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("panics_while", ["loading", "counting"])
async def test_framed_count_answers_the_provider_error_when_the_tokenizer_panics(
    monkeypatch: pytest.MonkeyPatch, panics_while: str
):
    _framed_kimi_with_custom_tokenizer(monkeypatch, f"lit-tests/panicking-tokenizer-{panics_while}")
    panicking: Final = _tokenizer_that_panics_while_encoding()

    def load(model: str, custom_tokenizer: object = None) -> SelectTokenizerResponse:
        if panics_while == "loading":
            panicking["tokenizer"].encode("hello world")
        return panicking

    monkeypatch.setattr(litellm.utils, "select_tokenizer", load)

    assert await _count_status() == "400"


def _framed_kimi(monkeypatch: pytest.MonkeyPatch) -> None:
    counter: Final = _ScriptedCounter({"us.moonshotai.kimi-k3": _provider_reply("kimi", status_code=400)})
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(model_list=[{"model_name": "kimi", "litellm_params": {"model": "bedrock/us.moonshotai.kimi-k3"}}]),
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server, "_get_provider_token_counter", lambda deployment, model: (counter, model, "bedrock")
    )


@pytest.mark.asyncio
async def test_local_count_loads_the_tokenizer_again_after_a_failed_load(monkeypatch: pytest.MonkeyPatch):
    _framed_kimi_with_custom_tokenizer(monkeypatch, "lit-tests/local-route-tokenizer")
    real_select_tokenizer: Final = litellm.utils.select_tokenizer
    downloads: Final[list[str]] = []

    def hub_down_once(model: str, custom_tokenizer: object = None):
        downloads.append(model)
        if len(downloads) == 1:
            raise OSError("hub unreachable")
        return real_select_tokenizer(model="gpt-5.4-mini")

    monkeypatch.setattr(litellm.utils, "select_tokenizer", hub_down_once)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_REUSE_SECONDS", 1e9)
    request: Final = TokenCountRequest(model="kimi", messages=[{"role": "user", "content": "What is new?"}])

    with pytest.raises(OSError, match="hub unreachable"):
        await count_request_tokens(request=request)
    retried: Final = await count_request_tokens(request=request)

    assert retried.count_source == "local_estimate" and retried.total_tokens > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reuse_seconds", "tokenizer_types"),
    [(1e9, ("openai_tokenizer", "openai_tokenizer")), (1e-9, ("openai_tokenizer", "huggingface_tokenizer"))],
    ids=["within_the_window", "after_the_window"],
)
async def test_framed_count_loads_the_tokenizer_again_only_after_the_window_of_a_load_that_fell_back(
    monkeypatch: pytest.MonkeyPatch, reuse_seconds: float, tokenizer_types: tuple[str, str]
):
    _framed_kimi(monkeypatch)
    real_select_tokenizer: Final = litellm.utils.select_tokenizer
    hub_tokenizer: Final = litellm.utils.create_tokenizer(
        Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")).to_str()
    )
    loads: Final[list[str]] = []

    def hub_up_on_the_second_load(model: str, custom_tokenizer: object = None):
        loads.append(model)
        return real_select_tokenizer(model="gpt-5.4-mini") if len(loads) == 1 else hub_tokenizer

    monkeypatch.setattr(litellm.utils, "select_tokenizer", hub_up_on_the_second_load)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_REUSE_SECONDS", reuse_seconds)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_MAX_RUNNING", 1)
    request: Final = TokenCountRequest(model="kimi", messages=[{"role": "user", "content": "What is new?"}])

    first: Final = await count_request_tokens(request=request, call_endpoint=True)
    second: Final = await count_request_tokens(request=request, call_endpoint=True)

    assert (first.tokenizer_type, second.tokenizer_type) == tokenizer_types


async def _fork_guard_traceback_frames() -> int:
    with pytest.raises(ForkedAfterNativeRuntimeStarted) as raised:
        await count_request_tokens(
            request=TokenCountRequest(model="kimi", messages=[{"role": "user", "content": "What is new?"}]),
            call_endpoint=True,
        )
    return len(traceback.extract_tb(raised.value.__traceback__))


@pytest.mark.asyncio
async def test_fork_guard_error_while_loading_the_tokenizer_is_raised_to_every_count_sharing_the_load(
    monkeypatch: pytest.MonkeyPatch,
):
    _framed_kimi(monkeypatch)

    def forked_load(model: str, custom_tokenizer: object = None) -> NoReturn:
        raise ForkedAfterNativeRuntimeStarted("native runtime started before fork")

    monkeypatch.setattr(litellm.utils, "select_tokenizer", forked_load)

    frames: Final = [await _fork_guard_traceback_frames() for _ in range(3)]

    assert len(set(frames)) == 1


@pytest.mark.asyncio
async def test_swallowed_estimate_error_is_logged_by_type_without_the_request_content(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    counter: Final = _ScriptedCounter({"us.moonshotai.kimi-k3": _provider_reply("kimi", status_code=400)})
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(model_list=[{"model_name": "kimi", "litellm_params": {"model": "bedrock/us.moonshotai.kimi-k3"}}]),
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server, "_get_provider_token_counter", lambda deployment, model: (counter, model, "bedrock")
    )
    verbose_proxy_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=verbose_proxy_logger.name), pytest.raises(ProxyException):
            await count_request_tokens(
                request=TokenCountRequest(
                    model="kimi",
                    messages=[
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "ok", "tool_calls": "private-marker-in-the-prompt"},
                    ],
                ),
                call_endpoint=True,
            )
    finally:
        verbose_proxy_logger.removeHandler(caplog.handler)

    warnings: Final = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("count error 400" in message and "ValueError" in message for message in warnings)
    assert not any("private-marker-in-the-prompt" in message for message in warnings)


@pytest.mark.asyncio
async def test_fork_guard_error_in_the_estimate_is_never_answered_as_the_provider_error():
    async def forked_estimate() -> NoReturn:
        raise ForkedAfterNativeRuntimeStarted("native runtime started before fork")

    with pytest.raises(ForkedAfterNativeRuntimeStarted):
        await _estimate_or_provider_error(
            forked_estimate(), provider_failure=ProviderCountFailure(status_code=400, message="scripted 400")
        )


@pytest.mark.asyncio
async def test_local_estimate_adds_the_configured_framing_once_per_request(monkeypatch: pytest.MonkeyPatch):
    def deployment(name: str, model_info: dict[str, object]) -> dict[str, object]:
        return {
            "model_name": name,
            "litellm_params": {"model": "openai/gpt-5.4-mini", "api_key": f"sk-unused-{name}"},
            "model_info": model_info,
        }

    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(model_list=[deployment("framed", {"token_count_framing": 100}), deployment("plain", {})]),
    )
    messages: Final = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello there"}]

    plain: Final = await count_request_tokens(request=TokenCountRequest(model="plain", messages=messages))
    framed: Final = await count_request_tokens(request=TokenCountRequest(model="framed", messages=messages))
    framed_prompt: Final = await count_request_tokens(request=TokenCountRequest(model="framed", prompt="hello there"))

    assert plain.total_tokens == litellm.token_counter(model="gpt-5.4-mini", messages=messages)
    assert framed.total_tokens == plain.total_tokens + 100
    assert (plain.count_source, framed.count_source) == ("local_estimate", "local_estimate")
    assert framed_prompt.total_tokens == litellm.token_counter(model="gpt-5.4-mini", text="hello there")


@pytest.mark.asyncio
async def test_framing_comes_from_the_row_the_deployment_provider_or_base_model_names(monkeypatch: pytest.MonkeyPatch):
    def deployment(name: str, litellm_params: dict[str, object], model_info: dict[str, object]) -> dict[str, object]:
        return {
            "model_name": name,
            "litellm_params": {**litellm_params, "api_key": "unused", "api_base": "https://unused.invalid"},
            "model_info": model_info,
        }

    kimi: Final = "databricks/system.ai.kimi-k3"
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                deployment("prefixed", {"model": kimi}, {}),
                deployment("split", {"model": "system.ai.kimi-k3", "custom_llm_provider": "databricks"}, {}),
                deployment("aliased", {"model": "gpt-5.4-mini", "custom_llm_provider": "openai"}, {"base_model": kimi}),
            ]
        ),
    )
    messages: Final = [{"role": "user", "content": "hello there"}]
    framing: Final = cost_map_entry(kimi)["token_count_framing"]

    prefixed: Final = await count_request_tokens(request=TokenCountRequest(model="prefixed", messages=messages))
    split: Final = await count_request_tokens(request=TokenCountRequest(model="split", messages=messages))
    aliased: Final = await count_request_tokens(request=TokenCountRequest(model="aliased", messages=messages))

    assert prefixed.total_tokens == litellm.token_counter(model="system.ai.kimi-k3", messages=messages) + framing
    assert split.total_tokens == prefixed.total_tokens
    assert aliased.total_tokens == litellm.token_counter(model="gpt-5.4-mini", messages=messages) + framing


@pytest.mark.asyncio
async def test_a_tool_request_counts_its_framing_in_place_of_the_plain_one_per_api(monkeypatch: pytest.MonkeyPatch):
    def deployment(name: str, model_info: dict[str, object]) -> dict[str, object]:
        return {
            "model_name": name,
            "litellm_params": {"model": "openai/gpt-5.4-mini", "api_key": f"sk-unused-{name}"},
            "model_info": model_info,
        }

    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                deployment("plain", {}),
                deployment("chat", {"token_count_framing": 7, "token_count_tool_framing": 50}),
                deployment(
                    "per-api",
                    {
                        "token_count_framing": 7,
                        "token_count_tool_framing": 50,
                        "token_count_responses_framing": 5,
                        "token_count_responses_tool_framing": 2,
                    },
                ),
            ]
        ),
    )
    messages: Final = [{"role": "user", "content": "What is the weather in Paris?"}]
    tools: Final = [_KIMI_WEATHER]

    async def count(model: str, request_tools: list[dict[str, object]] | None, responses: bool) -> int:
        response: Final = await count_request_tokens(
            request=TokenCountRequest(model=model, messages=messages, tools=request_tools),
            responses_request={"model": model, "input": "What is the weather in Paris?"} if responses else None,
        )
        return response.total_tokens

    plain_with_tools: Final = await count("plain", tools, responses=False)
    plain_without_tools: Final = await count("plain", None, responses=False)

    assert await count("plain", tools, responses=True) == plain_with_tools
    assert await count("chat", tools, responses=False) == plain_with_tools + 50
    assert await count("chat", None, responses=False) == plain_without_tools + 7
    assert await count("chat", [], responses=False) == plain_without_tools + 7
    assert await count("chat", tools, responses=True) == plain_with_tools + 50
    assert await count("chat", None, responses=True) == plain_without_tools + 7
    assert await count("per-api", tools, responses=False) == plain_with_tools + 50
    assert await count("per-api", tools, responses=True) == plain_with_tools + 2
    assert await count("per-api", None, responses=True) == plain_without_tools + 5


_GET_WEATHER: Final = {
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
}
_THREE_TOOLS: Final = (
    _GET_WEATHER,
    {
        "name": "get_time",
        "description": "Get the local time in a timezone",
        "input_schema": {
            "type": "object",
            "properties": {"timezone": {"type": "string", "description": "IANA timezone"}},
            "required": ["timezone"],
        },
    },
    {
        "name": "search_docs",
        "description": "Search internal documents and return the top matches",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "description": "Max results"}},
            "required": ["query"],
        },
    },
)
_FORM_FIELDS: Final = tuple(f"field_{index}" for index in range(10))
_FILL_FORM: Final = {
    "name": "fill_form",
    "description": "Fill a form",
    "input_schema": {
        "type": "object",
        "properties": {
            field: {"type": "string", "description": f"Value for field number {index}"}
            for index, field in enumerate(_FORM_FIELDS)
        },
        "required": list(_FORM_FIELDS),
    },
}
_DEPLOYMENT_TOOLS: Final = (
    {
        "name": "configure_deployment",
        "description": (
            "Configure a model deployment with routing, limits, retries, fallbacks and observability settings for a "
            "production gateway"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Deployment name"},
                "provider": {
                    "type": "string",
                    "enum": ["openai", "azure", "anthropic", "bedrock", "vertex_ai", "xai"],
                    "description": "Upstream provider",
                },
                "limits": {
                    "type": "object",
                    "description": "Rate limits",
                    "properties": {
                        "rpm": {"type": "integer", "description": "Requests per minute"},
                        "tpm": {"type": "integer", "description": "Tokens per minute"},
                        "max_parallel": {"type": "integer", "description": "Max parallel requests"},
                    },
                },
                "retries": {
                    "type": "object",
                    "description": "Retry policy",
                    "properties": {
                        "count": {"type": "integer"},
                        "backoff_seconds": {"type": "number"},
                        "retry_on": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Error classes to retry",
                        },
                    },
                },
                "fallbacks": {
                    "type": "array",
                    "description": "Ordered fallback deployments",
                    "items": {
                        "type": "object",
                        "properties": {"model": {"type": "string"}, "weight": {"type": "number"}},
                    },
                },
                "observability": {
                    "type": "object",
                    "properties": {
                        "log_requests": {"type": "boolean"},
                        "callbacks": {"type": "array", "items": {"type": "string"}},
                        "sample_rate": {"type": "number", "description": "Fraction of requests to trace"},
                    },
                },
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Free-form tags"},
            },
            "required": ["name", "provider"],
        },
    },
    {
        "name": "search_docs",
        "description": "Search the internal documentation",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient address"},
                "subject": {"type": "string", "description": "Subject line"},
                "body": {"type": "string", "description": "Plain text body"},
            },
            "required": ["to", "subject", "body"],
        },
    },
)
_WEATHER_SYSTEM: Final = "You are a helpful assistant that answers weather questions briefly."
_AZURE_GROK_4_1: Final = "azure_ai/grok-4-1-fast-reasoning"
_FIREWORKS_DEEPSEEK_V4_1: Final = "fireworks_ai/accounts/fireworks/models/deepseek-v4p1-flash"
_XAI_GROK_4_3: Final = "xai/grok-4.3"
_AZURE_GROK_4_1_NON_REASONING: Final = "azure_ai/grok-4-1-fast-non-reasoning"
_AZURE_GROK_4_3: Final = "azure_ai/grok-4.3"
_VERTEX_GROK_4_3: Final = "vertex_ai/xai/grok-4.3"
_MANTLE_GROK_4_3: Final = "bedrock_mantle/xai.grok-4.3"
_PERPLEXITY_GROK_4_3: Final = "perplexity/xai/grok-4.3"
_NATIVE_RESPONSES_BASE: Final = {
    _AZURE_GROK_4_1: "https://unused.services.ai.azure.com/models",
    _AZURE_GROK_4_1_NON_REASONING: "https://unused.services.ai.azure.com/models",
    _FIREWORKS_DEEPSEEK_V4_1: "https://unused.invalid",
    _XAI_GROK_4_3: "https://unused.invalid",
    _PERPLEXITY_GROK_4_3: "https://unused.invalid",
    _VERTEX_GROK_4_3: "https://unused.invalid",
}


def _responses_function(tool: Mapping[str, object], **strict: bool | None) -> dict[str, object]:
    return {
        "type": "function",
        "name": tool["name"],
        "description": tool["description"],
        "parameters": tool["input_schema"],
        **strict,
    }


def _get_weather_copies(count: int) -> tuple[dict[str, object], ...]:
    return tuple({**_GET_WEATHER, "name": f"get_weather_{index}"} for index in range(count))


_MAX_TOKENS_OVER_A_FLAT_HOST_BILL: Final = 4


@pytest.fixture
def checked_in_cost_map() -> None:
    assert isinstance(
        install_model_cost_map(GetModelCostMap.read_local_model_cost_map_bytes()), ModelCostMapReloaded
    )


# usage.input_tokens plus cache_read_input_tokens that POST /v1/messages billed for these requests with the user turn
# "What is the weather in Paris?" through Azure AI Foundry grok-4-1-fast-reasoning, Fireworks deepseek-v4p1-flash and
# xAI grok-4.3 on 2026-10-29, usage.prompt_tokens that POST /v1/chat/completions billed through DeepSeek's API on
# 2026-10-30, and usage.input_tokens that POST /v1/messages billed through xAI grok-4.3-latest, Kyma grok-4.3, Azure AI
# Foundry grok-4-1-fast-non-reasoning and grok-4.3, Vertex AI xai/grok-4.3 and Bedrock Mantle xai.grok-4.3 on 2026-11-17,
# except the request with no tools through Azure AI Foundry grok-4.3, billed on 2026-11-21
@pytest.mark.usefixtures("checked_in_cost_map")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "system", "tools", "billed_prompt_tokens"),
    [
        (_AZURE_GROK_4_1, None, (), 163),
        (_AZURE_GROK_4_1, None, (_GET_WEATHER,), 357),
        (_AZURE_GROK_4_1, None, _THREE_TOOLS, 445),
        (_AZURE_GROK_4_1, None, (_FILL_FORM,), 550),
        (_AZURE_GROK_4_1, _WEATHER_SYSTEM, (_GET_WEATHER,), 368),
        (_FIREWORKS_DEEPSEEK_V4_1, None, (), 37),
        (_FIREWORKS_DEEPSEEK_V4_1, None, (_GET_WEATHER,), 304),
        (_FIREWORKS_DEEPSEEK_V4_1, None, _THREE_TOOLS, 430),
        (_FIREWORKS_DEEPSEEK_V4_1, None, (_FILL_FORM,), 550),
        (_FIREWORKS_DEEPSEEK_V4_1, _WEATHER_SYSTEM, (_GET_WEATHER,), 315),
        (_XAI_GROK_4_3, None, (), 199),
        (_XAI_GROK_4_3, None, (_GET_WEATHER,), 283),
        (_XAI_GROK_4_3, None, _THREE_TOOLS, 402),
        (_XAI_GROK_4_3, None, (_FILL_FORM,), 530),
        (_XAI_GROK_4_3, _WEATHER_SYSTEM, (_GET_WEATHER,), 297),
        ("deepseek/deepseek-v4-flash", None, (), 37),
        ("deepseek/deepseek-v4-flash", None, (_GET_WEATHER,), 304),
        ("deepseek/deepseek-v4-flash", None, _THREE_TOOLS, 430),
        ("deepseek/deepseek-v4-flash", _WEATHER_SYSTEM, (_GET_WEATHER,), 315),
        ("deepseek/deepseek-v4-pro", None, (), 90),
        ("deepseek/deepseek-v4-pro", None, (_GET_WEATHER,), 367),
        ("deepseek/deepseek-v4-pro", None, _THREE_TOOLS, 493),
        ("deepseek/deepseek-v4-pro", _WEATHER_SYSTEM, (_GET_WEATHER,), 378),
        ("xai/grok-4.3-latest", None, (), 199),
        ("xai/grok-4.3-latest", None, (_GET_WEATHER,), 283),
        ("xai/grok-4.3-latest", None, _THREE_TOOLS, 402),
        ("xai/grok-4.3-latest", None, (_FILL_FORM,), 530),
        ("xai/grok-4.3-latest", _WEATHER_SYSTEM, (_GET_WEATHER,), 297),
        ("kyma/grok-4.3", None, (), 199),
        ("kyma/grok-4.3", None, (_GET_WEATHER,), 283),
        ("kyma/grok-4.3", None, _THREE_TOOLS, 402),
        ("kyma/grok-4.3", None, (_FILL_FORM,), 530),
        ("kyma/grok-4.3", _WEATHER_SYSTEM, (_GET_WEATHER,), 297),
        (_AZURE_GROK_4_1_NON_REASONING, None, (), 175),
        (_AZURE_GROK_4_1_NON_REASONING, None, (_GET_WEATHER,), 369),
        (_AZURE_GROK_4_1_NON_REASONING, None, _THREE_TOOLS, 457),
        (_AZURE_GROK_4_1_NON_REASONING, None, (_FILL_FORM,), 562),
        (_AZURE_GROK_4_1_NON_REASONING, _WEATHER_SYSTEM, (_GET_WEATHER,), 380),
        (_AZURE_GROK_4_3, None, (), 11),
        (_AZURE_GROK_4_3, None, (_GET_WEATHER,), 96),
        (_AZURE_GROK_4_3, None, _THREE_TOOLS, 215),
        (_AZURE_GROK_4_3, None, (_FILL_FORM,), 343),
        (_AZURE_GROK_4_3, _WEATHER_SYSTEM, (_GET_WEATHER,), 110),
        (_VERTEX_GROK_4_3, None, (), 342),
        (_VERTEX_GROK_4_3, None, (_GET_WEATHER,), 441),
        (_VERTEX_GROK_4_3, None, _THREE_TOOLS, 560),
        (_VERTEX_GROK_4_3, None, (_FILL_FORM,), 688),
        (_VERTEX_GROK_4_3, _WEATHER_SYSTEM, (_GET_WEATHER,), 455),
        (_MANTLE_GROK_4_3, None, (), 37),
    ],
)
async def test_local_estimate_matches_the_billed_prompt_with_the_shipped_framing(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    system: str | None,
    tools: tuple[dict[str, object], ...],
    billed_prompt_tokens: int,
):
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "billed",
                    "litellm_params": {"model": model, "api_key": "unused", "api_base": "https://unused.invalid"},
                }
            ]
        ),
    )

    counted: Final = await count_request_tokens(
        request=TokenCountRequest(
            model="billed",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=list(tools) or None,
            system=system,
        )
    )

    assert counted.count_source == "local_estimate"
    assert 0 <= counted.total_tokens - billed_prompt_tokens <= _MAX_TOKENS_OVER_A_FLAT_HOST_BILL


# usage.input_tokens that POST /v1/responses billed for these requests with the input "What is the weather in Paris?"
# through Azure AI Foundry grok-4-1-fast-reasoning, Fireworks deepseek-v4p1-flash and xAI grok-4.3 on 2026-10-29, and
# through Azure AI Foundry grok-4-1-fast-non-reasoning, Perplexity xai/grok-4.3 and Vertex AI xai/grok-4.3 on 2026-11-17
@pytest.mark.usefixtures("checked_in_cost_map")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "system", "tools", "billed_prompt_tokens"),
    [
        (_AZURE_GROK_4_1, None, (), 159),
        (_AZURE_GROK_4_1, None, (_GET_WEATHER,), 159),
        (_AZURE_GROK_4_1, None, _THREE_TOOLS, 257),
        (_AZURE_GROK_4_1, None, (_FILL_FORM,), 352),
        (_AZURE_GROK_4_1, _WEATHER_SYSTEM, (_GET_WEATHER,), 170),
        (_FIREWORKS_DEEPSEEK_V4_1, None, (_GET_WEATHER,), 304),
        (_XAI_GROK_4_3, None, (_GET_WEATHER,), 283),
        (_AZURE_GROK_4_1_NON_REASONING, None, (), 181),
        (_PERPLEXITY_GROK_4_3, None, (), 218),
        (_VERTEX_GROK_4_3, None, (), 342),
        (_VERTEX_GROK_4_3, None, (_GET_WEATHER,), 441),
        (_VERTEX_GROK_4_3, None, _THREE_TOOLS, 560),
    ],
)
async def test_responses_estimate_matches_the_billed_responses_prompt_with_the_shipped_framing(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    system: str | None,
    tools: tuple[dict[str, object], ...],
    billed_prompt_tokens: int,
):
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "billed",
                    "litellm_params": {"model": model, "api_key": "unused", "api_base": _NATIVE_RESPONSES_BASE[model]},
                }
            ]
        ),
    )

    counted: Final = await count_request_tokens(
        request=TokenCountRequest(
            model="billed",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[_responses_function(tool) for tool in tools] or None,
            system=system,
        ),
        responses_request={"model": "billed", "input": "What is the weather in Paris?"},
    )

    assert counted.count_source == "local_estimate"
    assert 0 <= counted.total_tokens - billed_prompt_tokens <= _MAX_TOKENS_OVER_A_FLAT_HOST_BILL


def _single_deployment_router(litellm_params: dict[str, object], model_info: dict[str, object]) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "counted",
                "litellm_params": {"api_key": "unused", **litellm_params},
                "model_info": model_info,
            }
        ]
    )


async def _count_with_router(
    monkeypatch: pytest.MonkeyPatch, router: Router, tools: tuple[dict[str, object], ...], responses: bool
) -> int:
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
    counted: Final = await count_request_tokens(
        request=TokenCountRequest(
            model="counted",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=list(tools) or None,
        ),
        responses_request={"model": "counted", "input": "What is the weather in Paris?"} if responses else None,
    )
    return counted.total_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize("tools", [(), (_GET_WEATHER,), _THREE_TOOLS, (_FILL_FORM,)])
@pytest.mark.parametrize("responses", [False, True])
async def test_a_custom_deployment_name_counts_like_the_base_model_it_names(
    monkeypatch: pytest.MonkeyPatch, tools: tuple[dict[str, object], ...], responses: bool
):
    api_base: Final = _NATIVE_RESPONSES_BASE[_AZURE_GROK_4_1]
    custom_name: Final = _single_deployment_router(
        {"model": "azure_ai/prod-grok", "api_base": api_base}, {"base_model": _AZURE_GROK_4_1}
    )
    row_name: Final = _single_deployment_router({"model": _AZURE_GROK_4_1, "api_base": api_base}, {})

    assert await _count_with_router(monkeypatch, custom_name, tools, responses) == await _count_with_router(
        monkeypatch, row_name, tools, responses
    )


# usage.input_tokens that POST /v1/responses billed through Azure AI Foundry grok-4-1-fast-reasoning on 2026-10-29 for
# the input "What is the weather in Paris?" with these function tools, strict left unset or sent as false or null,
# the same on the .services.ai.azure.com and .openai.azure.com hosts
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tools", "strict", "billed_prompt_tokens"),
    [
        ((_GET_WEATHER,), {}, 159),
        ((_GET_WEATHER,), {"strict": False}, 155),
        ((_GET_WEATHER,), {"strict": None}, 155),
        (_THREE_TOOLS, {}, 257),
        (_get_weather_copies(5), {}, 345),
        (_get_weather_copies(10), {}, 575),
        (_get_weather_copies(10), {"strict": False}, 535),
        (_get_weather_copies(10), {"strict": None}, 535),
    ],
)
@pytest.mark.parametrize(
    ("litellm_params", "model_info"),
    [
        ({"model": _AZURE_GROK_4_1, "api_base": _NATIVE_RESPONSES_BASE[_AZURE_GROK_4_1]}, {}),
        (
            {"model": "azure/grok-4-1-fast-reasoning", "api_base": "https://unused.openai.azure.com/openai/v1"},
            {"base_model": _AZURE_GROK_4_1},
        ),
    ],
)
async def test_azure_responses_estimate_stays_at_or_just_over_the_billed_prompt_as_function_tools_are_added(
    monkeypatch: pytest.MonkeyPatch,
    tools: tuple[dict[str, object], ...],
    strict: dict[str, bool | None],
    billed_prompt_tokens: int,
    litellm_params: dict[str, object],
    model_info: dict[str, object],
):
    router: Final = _single_deployment_router(litellm_params, model_info)

    counted: Final = await _count_with_router(
        monkeypatch, router, tuple(_responses_function(tool, **strict) for tool in tools), responses=True
    )

    assert billed_prompt_tokens <= counted <= billed_prompt_tokens * 1.02


# usage.input_tokens that POST /v1/messages billed through Bedrock Mantle xai.grok-4.3, and POST /v1/responses through
# Perplexity xai/grok-4.3 and Azure AI Foundry grok-4-1-fast-non-reasoning, on 2026-11-17 for the input "What is the
# weather in Paris?" with these tools. Each host bills more hidden tokens per tool as the schema grows, so one flat
# tool framing per host lands from 0 to 175 tokens over these bills on Mantle, 102 on Perplexity and 74 on Azure
@pytest.mark.usefixtures("checked_in_cost_map")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "responses", "tools", "billed_prompt_tokens", "most_over"),
    [
        (_MANTLE_GROK_4_3, False, (_GET_WEATHER,), 114, 180),
        (_MANTLE_GROK_4_3, False, _THREE_TOOLS, 256, 180),
        (_MANTLE_GROK_4_3, False, (_FILL_FORM,), 424, 180),
        (_MANTLE_GROK_4_3, False, _DEPLOYMENT_TOOLS, 789, 180),
        (_PERPLEXITY_GROK_4_3, True, (_GET_WEATHER,), 308, 110),
        (_PERPLEXITY_GROK_4_3, True, _THREE_TOOLS, 442, 110),
        (_PERPLEXITY_GROK_4_3, True, (_FILL_FORM,), 555, 110),
        (_PERPLEXITY_GROK_4_3, True, _DEPLOYMENT_TOOLS, 913, 110),
        (_AZURE_GROK_4_1_NON_REASONING, True, (_GET_WEATHER,), 178, 80),
        (_AZURE_GROK_4_1_NON_REASONING, True, _THREE_TOOLS, 276, 80),
        (_AZURE_GROK_4_1_NON_REASONING, True, (_FILL_FORM,), 371, 80),
        (_AZURE_GROK_4_1_NON_REASONING, True, _DEPLOYMENT_TOOLS, 614, 80),
    ],
)
async def test_a_host_whose_tool_prompt_grows_with_the_schema_counts_at_or_a_bounded_amount_over_the_bill(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    responses: bool,
    tools: tuple[dict[str, object], ...],
    billed_prompt_tokens: int,
    most_over: int,
):
    router: Final = _single_deployment_router(
        {"model": model, "api_base": _NATIVE_RESPONSES_BASE.get(model, "https://unused.invalid")}, {}
    )
    sent_tools: Final = tuple(_responses_function(tool) for tool in tools) if responses else tools

    counted: Final = await _count_with_router(monkeypatch, router, sent_tools, responses)

    assert (
        billed_prompt_tokens <= counted <= billed_prompt_tokens + most_over
    ), f"{model} counted {counted} for a request billed {billed_prompt_tokens}"


@pytest.mark.usefixtures("checked_in_cost_map")
@pytest.mark.asyncio
@pytest.mark.parametrize("tools", [(), (_GET_WEATHER,), _THREE_TOOLS])
async def test_the_govcloud_mantle_grok_row_counts_like_the_commercial_row(
    monkeypatch: pytest.MonkeyPatch, tools: tuple[dict[str, object], ...]
):
    govcloud: Final = _single_deployment_router(
        {"model": "bedrock_mantle/us-gov-west-1/xai.grok-4.3", "api_base": "https://unused.invalid"}, {}
    )
    commercial: Final = _single_deployment_router({"model": _MANTLE_GROK_4_3, "api_base": "https://unused.invalid"}, {})

    assert await _count_with_router(monkeypatch, govcloud, tools, False) == await _count_with_router(
        monkeypatch, commercial, tools, False
    )


# POST /v1/responses billed the same usage.input_tokens with strict unset and false on 2026-10-29, 283 for one tool
# through xAI grok-4.3
@pytest.mark.asyncio
async def test_a_responses_host_that_bills_an_unset_strict_like_false_counts_them_alike(
    monkeypatch: pytest.MonkeyPatch,
):
    router: Final = _single_deployment_router(
        {"model": _XAI_GROK_4_3, "api_base": _NATIVE_RESPONSES_BASE[_XAI_GROK_4_3]}, {}
    )

    unset: Final = await _count_with_router(monkeypatch, router, (_responses_function(_GET_WEATHER),), responses=True)

    assert unset == await _count_with_router(
        monkeypatch, router, (_responses_function(_GET_WEATHER, strict=False),), responses=True
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("built_in_tool", "responses"),
    [({"type": "web_search"}, True), ({"type": "web_search_20250305", "name": "web_search"}, False)],
)
async def test_a_request_whose_only_tools_are_built_in_counts_like_one_without_tools(
    monkeypatch: pytest.MonkeyPatch, built_in_tool: dict[str, object], responses: bool
):
    router: Final = _single_deployment_router(
        {"model": _AZURE_GROK_4_1, "api_base": _NATIVE_RESPONSES_BASE[_AZURE_GROK_4_1]}, {}
    )

    assert await _count_with_router(monkeypatch, router, (built_in_tool,), responses) == await _count_with_router(
        monkeypatch, router, (), responses
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("tools", [(), (_GET_WEATHER,), _THREE_TOOLS])
async def test_a_responses_count_the_proxy_bridges_to_chat_completions_frames_like_the_chat_count(
    monkeypatch: pytest.MonkeyPatch, tools: tuple[dict[str, object], ...]
):
    bridged: Final = _single_deployment_router(
        {"model": _AZURE_GROK_4_1, "api_base": "https://unused.cognitiveservices.azure.com/models"}, {}
    )
    native: Final = _single_deployment_router(
        {"model": _AZURE_GROK_4_1, "api_base": _NATIVE_RESPONSES_BASE[_AZURE_GROK_4_1]}, {}
    )
    chat_count: Final = await _count_with_router(monkeypatch, bridged, tools, responses=False)

    assert await _count_with_router(monkeypatch, bridged, tools, responses=True) == chat_count
    assert await _count_with_router(monkeypatch, native, tools, responses=True) != chat_count


@pytest.mark.asyncio
@pytest.mark.parametrize("from_router_defaults", [False, True])
async def test_a_responses_count_that_asks_for_the_chat_completions_bridge_frames_like_the_chat_count(
    monkeypatch: pytest.MonkeyPatch, from_router_defaults: bool
):
    router: Final = Router(
        model_list=[
            {
                "model_name": "counted",
                "litellm_params": {
                    "api_key": "unused",
                    "model": _AZURE_GROK_4_1,
                    "api_base": _NATIVE_RESPONSES_BASE[_AZURE_GROK_4_1],
                },
                "model_info": {},
            }
        ],
        default_litellm_params={"use_chat_completions_api": True} if from_router_defaults else None,
    )
    chat_count: Final = await _count_with_router(monkeypatch, router, (_GET_WEATHER,), responses=False)

    counted: Final = await count_request_tokens(
        request=TokenCountRequest(
            model="counted",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[_GET_WEATHER],
        ),
        responses_request={
            "model": "counted",
            "input": "What is the weather in Paris?",
            **({} if from_router_defaults else {"use_chat_completions_api": True}),
        },
    )

    assert counted.total_tokens == chat_count


@pytest.mark.asyncio
async def test_a_responses_count_frames_for_the_api_base_router_defaults_send_inference_to(
    monkeypatch: pytest.MonkeyPatch,
):
    router: Final = Router(
        model_list=[
            {
                "model_name": "counted",
                "litellm_params": {
                    "api_key": "unused",
                    "model": _AZURE_GROK_4_1,
                    "api_base": _NATIVE_RESPONSES_BASE[_AZURE_GROK_4_1],
                },
                "model_info": {},
            }
        ],
        default_litellm_params={"api_base": "https://unused.cognitiveservices.azure.com/models"},
    )
    chat_count: Final = await _count_with_router(monkeypatch, router, (_GET_WEATHER,), responses=False)

    assert await _count_with_router(monkeypatch, router, (_GET_WEATHER,), responses=True) == chat_count


@pytest.mark.asyncio
@pytest.mark.parametrize("use_chat_completions_api", ["maybe", {"a": 1}])
async def test_a_responses_count_with_a_malformed_bridge_flag_reads_it_the_way_the_responses_route_does(
    monkeypatch: pytest.MonkeyPatch, use_chat_completions_api: object
):
    router: Final = _single_deployment_router(
        {"model": _AZURE_GROK_4_1, "api_base": _NATIVE_RESPONSES_BASE[_AZURE_GROK_4_1]}, {}
    )
    chat_count: Final = await _count_with_router(monkeypatch, router, (_GET_WEATHER,), responses=False)

    counted: Final = await count_request_tokens(
        request=TokenCountRequest(
            model="counted",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[_GET_WEATHER],
        ),
        responses_request={
            "model": "counted",
            "input": "What is the weather in Paris?",
            "use_chat_completions_api": use_chat_completions_api,
        },
    )

    assert counted.total_tokens == chat_count


@pytest.mark.asyncio
async def test_an_azure_responses_count_leaves_strict_unset_for_templates_not_measured_there(
    monkeypatch: pytest.MonkeyPatch,
):
    router: Final = _single_deployment_router(
        {"model": "azure_ai/deepseek-v4.1-flash", "api_base": "https://unused.services.ai.azure.com/models"}, {}
    )

    async def counted(**strict: bool) -> int:
        return await _count_with_router(monkeypatch, router, (_responses_function(_GET_WEATHER, **strict),), True)

    assert await counted() < await counted(strict=True)


_KIMI_WEATHER: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def _byte_level_kimi_k3(path: Path) -> KimiK3TokenizerSource:
    path.write_bytes(b"\n".join(base64.b64encode(bytes((value,))) + b" %d" % value for value in range(256)))
    return KimiK3TokenizerSource(ranks_path=lambda: str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _available_kimi_k3(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tiktoken.Encoding:
    source: Final = _byte_level_kimi_k3(tmp_path / "tiktoken.model")
    monkeypatch.setattr(
        litellm.proxy.proxy_server, "_KIMI_K3_ENCODINGS", KimiK3Encodings(source=source, retry_after_seconds=60)
    )
    encoding: Final = load_kimi_k3_encoding(source)
    assert encoding is not None
    return encoding


def _shipped_kimi_k3_framing() -> int:
    framing: Final = cost_map_entry("bedrock/us.moonshotai.kimi-k3")["token_count_framing"]
    assert isinstance(framing, int)
    return framing


@pytest.mark.asyncio
@pytest.mark.parametrize("as_blocks", [False, True])
@pytest.mark.parametrize(
    "model", ["azure/responses/gpt-6.2-sol", "bedrock_mantle/openai.gpt-6.2-sol", "openai/gpt-6.2-sol"]
)
async def test_bridged_messages_local_count_ignores_leading_claude_code_billing_header(
    monkeypatch: pytest.MonkeyPatch, as_blocks: bool, model: str
) -> None:
    router: Final = _single_deployment_router({"model": model}, {})
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
    billing: Final = "x-anthropic-billing-header: cc_version=2.1.263.abc; cc_entrypoint=cli; cch=781fd;"
    system: Final = [{"type": "text", "text": "Be concise."}] if as_blocks else "Be concise."
    with_billing: Final = (
        [{"type": "text", "text": billing}, *system] if isinstance(system, list) else billing + "\n" + system
    )
    request: Final = TokenCountRequest(model="counted", messages=[{"role": "user", "content": "Hello"}], system=system)

    baseline: Final = await count_request_tokens(request=request, messages_format="anthropic")
    counted: Final = await count_request_tokens(
        request=request.model_copy(update={"system": with_billing}), messages_format="anthropic"
    )

    assert counted.total_tokens == baseline.total_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_fallback", [False, True])
@pytest.mark.parametrize("with_edits", [False, True])
@pytest.mark.parametrize("as_blocks", [False, True])
async def test_provider_count_removes_only_the_first_adjacent_billing_block(
    monkeypatch: pytest.MonkeyPatch, chat_fallback: bool, with_edits: bool, as_blocks: bool
) -> None:
    monkeypatch.setattr(litellm, "use_chat_completions_url_for_anthropic_messages", chat_fallback)
    router: Final = _single_deployment_router(
        {"model": "openai/gpt-6.2-sol", "api_key": "test-key", "store": False}, {}
    )
    billing_a: Final = "x-anthropic-billing-header: cc_version=2; cc_entrypoint=cli; cch=A;"
    billing_b: Final = "x-anthropic-billing-header: cc_version=2; cc_entrypoint=cli; cch=B;"
    bodies: Final[list[dict[str, object]]] = []  # mutable-ok: records the HTTP peer's input

    def peer(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 47}, request=request)

    with patch.object(
        openai_count_tokens_http,
        "get_async_httpx_client",
        return_value=AsyncHTTPHandler(transport=httpx.MockTransport(peer)),
    ):
        assert (
            await _input_tokens(
                router,
                {
                    "model": "counted",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "system": [
                        {"type": "text", "text": billing_a},
                        {"type": "text", "text": billing_b},
                        {"type": "text", "text": "Be concise."},
                    ]
                    if as_blocks
                    else billing_a + "\n" + billing_b + "\nBe concise.",
                    **({"context_management": _CLEAR_TOOL_USES} if with_edits else {}),
                },
            )
            == 47
        )

    assert bodies[0]["input"] == [
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": billing_b}, {"type": "input_text", "text": "Be concise."}]
            if as_blocks
            else [{"type": "input_text", "text": billing_b + "\nBe concise."}],
        },
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
    ]


@pytest.mark.parametrize("as_blocks", [False, True])
def test_generation_keeps_the_second_adjacent_billing_block(as_blocks: bool) -> None:
    billing_a: Final = "x-anthropic-billing-header: cc_version=2; cc_entrypoint=cli; cch=A;"
    billing_b: Final = "x-anthropic-billing-header: cc_version=2; cc_entrypoint=cli; cch=B;"
    request: Final[AnthropicMessagesRequest] = {
        "model": "openai/gpt-6.2-sol",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "Hello"}],
        "system": [
            {"type": "text", "text": billing_a},
            {"type": "text", "text": billing_b},
            {"type": "text", "text": "Be concise."},
        ]
        if as_blocks
        else billing_a + "\n" + billing_b + "\nBe concise.",
    }
    chat, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(request)
    responses: Final = LiteLLMAnthropicToResponsesAPIAdapter().translate_request(request)
    assert chat["messages"][0] == {
        "role": "system",
        "content": [{"type": "text", "text": billing_b}, {"type": "text", "text": "Be concise."}]
        if as_blocks
        else billing_b + "\nBe concise.",
    }
    assert responses["input"][0] == {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": billing_b}, {"type": "input_text", "text": "Be concise."}]
        if as_blocks
        else [{"type": "input_text", "text": billing_b + "\nBe concise."}],
    }


@pytest.mark.asyncio
async def test_bedrock_kimi_k3_is_counted_with_its_tokenizer_plus_the_shipped_framing_in_both_request_formats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _framed_kimi(monkeypatch)
    encoding: Final = _available_kimi_k3(monkeypatch, tmp_path)
    question: Final = {"role": "user", "content": "Weather in Paris?"}
    openai_messages: Final = [
        {"role": "system", "content": "Be terse."},
        question,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "Sun"},
    ]

    openai: Final = await count_request_tokens(
        request=TokenCountRequest(model="kimi", messages=openai_messages, tools=[_KIMI_WEATHER]), call_endpoint=True
    )
    anthropic: Final = await count_request_tokens(
        request=TokenCountRequest(
            model="kimi",
            system="Be terse.",
            messages=[
                question,
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "c1", "name": "get_weather", "input": {"city": "Paris"}}],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "Sun"}]},
            ],
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "input_schema": _KIMI_WEATHER["function"]["parameters"],
                }
            ],
        ),
        call_endpoint=True,
        messages_format="anthropic",
    )
    exact: Final = kimi_k3_tokens(encoding, "us.moonshotai.kimi-k3", openai_messages, [_KIMI_WEATHER])

    assert exact is not None
    assert (openai.total_tokens, openai.tokenizer_type, openai.count_source) == (
        exact + _shipped_kimi_k3_framing(),
        "huggingface_tokenizer",
        "local_estimate",
    )
    assert (anthropic.total_tokens, anthropic.tokenizer_type) == (openai.total_tokens, "huggingface_tokenizer")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("messages_format", "tools", "forced", "unforced"),
    [
        pytest.param(
            "openai", [_KIMI_WEATHER], {"type": "function", "function": {"name": "get_weather"}}, "auto", id="openai"
        ),
        pytest.param(
            "anthropic",
            [
                {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "input_schema": _KIMI_WEATHER["function"]["parameters"],
                }
            ],
            {"type": "tool", "name": "get_weather"},
            {"type": "auto"},
            id="anthropic",
        ),
    ],
)
@pytest.mark.parametrize("drops_tool_choice", [False, True])
async def test_bedrock_kimi_k3_count_adds_the_template_message_of_a_forced_tool_choice_inference_sends(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    messages_format: TokenCountMessagesFormat,
    tools: list[dict[str, object]],
    forced: dict[str, object],
    unforced: str | dict[str, object],
    drops_tool_choice: bool,
):
    _framed_kimi(monkeypatch)
    if drops_tool_choice:
        monkeypatch.setattr(
            litellm.proxy.proxy_server,
            "llm_router",
            Router(
                model_list=[
                    {
                        "model_name": "kimi",
                        "litellm_params": {
                            "model": "bedrock/us.moonshotai.kimi-k3",
                            "additional_drop_params": ["tool_choice"],
                        },
                    }
                ]
            ),
        )
    encoding: Final = _available_kimi_k3(monkeypatch, tmp_path)
    question: Final = [{"role": "user", "content": "Weather in Paris?"}]

    async def count(tool_choice: str | dict[str, object] | None) -> int:
        counted: Final = await count_request_tokens(
            request=TokenCountRequest(model="kimi", messages=question, tools=tools, tool_choice=tool_choice),
            call_endpoint=True,
            messages_format=messages_format,
        )
        assert counted.tokenizer_type == "huggingface_tokenizer"
        return counted.total_tokens

    plain: Final = kimi_k3_tokens(encoding, "us.moonshotai.kimi-k3", question, [_KIMI_WEATHER])
    required: Final = kimi_k3_tokens(encoding, "us.moonshotai.kimi-k3", question, [_KIMI_WEATHER], "required")
    unset: Final = await count(None)

    assert plain is not None and required is not None and required > plain
    assert (await count(unforced), await count(forced)) == (
        unset,
        unset if drops_tool_choice else unset + required - plain,
    )


@pytest.mark.asyncio
async def test_bedrock_kimi_k3_deployment_with_a_custom_tokenizer_keeps_counting_with_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _framed_kimi_with_custom_tokenizer(monkeypatch, "lit-tests/kimi-custom-tokenizer")
    _available_kimi_k3(monkeypatch, tmp_path)
    custom: Final = litellm.utils.select_tokenizer(model="gpt-5.4-mini")
    monkeypatch.setattr(litellm.utils, "select_tokenizer", lambda model, custom_tokenizer=None: custom)
    messages: Final = [{"role": "user", "content": "What is new?"}]

    counted: Final = await count_request_tokens(
        request=TokenCountRequest(model="kimi", messages=messages), call_endpoint=True
    )

    assert counted.total_tokens == (
        litellm.token_counter(model=counted.model_used, messages=messages, custom_tokenizer=custom)
        + kimi_k3_template_tokens(len(messages), False)
        + _shipped_kimi_k3_framing()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("messages", "tools", "template_tokens"),
    [
        pytest.param([{"role": "user", "content": "What is new?"}], None, 14, id="one_message"),
        pytest.param(
            [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "content": "Sunny."},
                {"role": "user", "content": "And Oslo?"},
            ],
            [_KIMI_WEATHER],
            14 + 3 * 13 + 40,
            id="turns_and_tools",
        ),
    ],
)
async def test_bedrock_kimi_k3_without_its_tokenizer_adds_the_template_offsets_to_the_local_estimate(
    monkeypatch: pytest.MonkeyPatch,
    messages: list[dict[str, object]],
    tools: list[dict[str, object]] | None,
    template_tokens: int,
):
    _framed_kimi(monkeypatch)

    counted: Final = await count_request_tokens(
        request=TokenCountRequest(model="kimi", messages=messages, tools=tools), call_endpoint=True
    )
    estimate: Final = litellm.token_counter(
        model=counted.model_used,
        messages=messages,
        tools=tools,
        custom_tokenizer=litellm.utils.select_tokenizer(model=counted.model_used),
    )

    assert counted.total_tokens == estimate + template_tokens + _shipped_kimi_k3_framing()


@pytest.mark.asyncio
async def test_slow_cost_map_lookup_does_not_stall_other_requests_on_the_worker(monkeypatch: pytest.MonkeyPatch):
    real_get_model_info: Final = litellm.get_model_info
    lookup_started: Final = threading.Event()
    other_request_ran: Final = threading.Event()
    released_by_the_other_request: Final[list[bool]] = []

    def slow_get_model_info(model: str, *args: object, **kwargs: object):
        if model == "slow-lookup-model":
            lookup_started.set()
            released_by_the_other_request.append(other_request_ran.wait(5))
        return real_get_model_info(model, *args, **kwargs)

    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", None)
    monkeypatch.setattr(litellm, "get_model_info", slow_get_model_info)
    counting: Final = asyncio.create_task(
        count_request_tokens(
            request=TokenCountRequest(model="slow-lookup-model", messages=[{"role": "user", "content": "hi"}])
        )
    )
    assert await asyncio.to_thread(lookup_started.wait, 5)
    other_request_ran.set()
    await counting

    assert set(released_by_the_other_request) == {True}


@pytest.mark.asyncio
async def test_uncountable_model_keeps_the_provider_error_when_local_counting_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    counter: Final = _ScriptedCounter({"us.moonshotai.kimi-k3": _provider_reply("kimi", status_code=400)})
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "kimi",
                    "litellm_params": {"model": "bedrock/us.moonshotai.kimi-k3"},
                    "model_info": {"token_count_framing": 78},
                }
            ]
        ),
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server, "_get_provider_token_counter", lambda deployment, model: (counter, model, "bedrock")
    )
    monkeypatch.setattr(litellm, "disable_token_counter", True)

    with pytest.raises(ProxyException) as raised:
        await count_request_tokens(
            request=TokenCountRequest(model="kimi", messages=[{"role": "user", "content": "hi"}]), call_endpoint=True
        )

    assert raised.value.code == "400"
    assert "scripted 400 for kimi" in raised.value.message


@pytest.mark.parametrize(
    ("model_info", "counted_with"),
    [
        ({}, ("bedrock_mantle/us.anthropic.claude-opus-5-5", "us-east-1", 447)),
        (
            {"token_count_model": {"model": "us.anthropic.claude-opus-5-5", "region": "ap-southeast-4"}},
            ("us.anthropic.claude-opus-5-5", "ap-southeast-4", 447),
        ),
        ({"token_count_model": {}}, ("us.anthropic.claude-sonnet-5-5", "us-west-2", 300)),
    ],
)
@pytest.mark.asyncio
async def test_configured_token_count_model_overrides_the_shipped_one_and_an_empty_one_counts_the_model_itself(
    monkeypatch: pytest.MonkeyPatch, model_info: dict[str, object], counted_with: tuple[str, str, int]
):
    counter: Final = _ScriptedCounter(
        {
            "us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", total_tokens=300),
            "us.anthropic.claude-opus-5-5": _provider_reply("opus", total_tokens=447),
            "bedrock_mantle/us.anthropic.claude-opus-5-5": _provider_reply("opus", total_tokens=447),
        }
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "sonnet",
                    "litellm_params": {
                        "model": "bedrock/us.anthropic.claude-sonnet-5-5",
                        "aws_region_name": "us-west-2",
                    },
                    "model_info": model_info,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server, "_get_provider_token_counter", lambda deployment, model: (counter, model, "bedrock")
    )

    counted: Final = await count_request_tokens(
        request=TokenCountRequest(model="sonnet", messages=[{"role": "user", "content": "hi"}]), call_endpoint=True
    )

    assert (counter.calls[0][0], counter.regions[0], counted.total_tokens) == counted_with
    assert len(counter.calls) == 1


@pytest.mark.parametrize(
    ("aws_region_name", "counted_in"),
    [("eu-central-1", "eu-central-1"), ("us-west-2", "us-east-1")],
)
@pytest.mark.asyncio
async def test_shipped_token_count_model_keeps_the_prompt_in_the_deployment_geography(
    monkeypatch: pytest.MonkeyPatch, aws_region_name: str, counted_in: str
):
    counter: Final = _ScriptedCounter(
        {"bedrock_mantle/anthropic.claude-opus-5-5": _provider_reply("opus", total_tokens=447)}
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "sonnet",
                    "litellm_params": {
                        "model": "bedrock/anthropic.claude-sonnet-5-5",
                        "aws_region_name": aws_region_name,
                    },
                }
            ]
        ),
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server, "_get_provider_token_counter", lambda deployment, model: (counter, model, "bedrock")
    )

    await count_request_tokens(
        request=TokenCountRequest(model="sonnet", messages=[{"role": "user", "content": "hi"}]), call_endpoint=True
    )

    assert (counter.calls[0][0], counter.regions[0]) == ("bedrock_mantle/anthropic.claude-opus-5-5", counted_in)


_LOOP_PROGRESS_TIMEOUT_SECONDS: Final = 5


class _VaultWaitingForLoopProgress:
    def __init__(
        self,
        lookup_started: threading.Event,
        loop_progressed: threading.Event,
        loop_progressed_during_lookup: threading.Event,
    ) -> None:
        self._lookup_started = lookup_started
        self._loop_progressed = loop_progressed
        self._loop_progressed_during_lookup = loop_progressed_during_lookup

    def sync_read_secret(self, secret_name: str) -> None:
        self._lookup_started.set()
        if self._loop_progressed.wait(_LOOP_PROGRESS_TIMEOUT_SECONDS):
            self._loop_progressed_during_lookup.set()


@pytest.mark.asyncio
async def test_region_lookup_for_a_shipped_count_model_leaves_the_event_loop_free(monkeypatch: pytest.MonkeyPatch):
    from litellm.proxy import proxy_server

    counter: Final = _ScriptedCounter(
        {"bedrock_mantle/anthropic.claude-opus-5-5": _provider_reply("opus", total_tokens=447)}
    )

    def bedrock_counter(deployment: object, model: str) -> tuple[_ScriptedCounter, str, str]:
        return counter, model, "bedrock"

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        Router(
            model_list=[{"model_name": "sonnet", "litellm_params": {"model": "bedrock/anthropic.claude-sonnet-5-5"}}]
        ),
    )
    monkeypatch.setattr(proxy_server, "_get_provider_token_counter", bedrock_counter)
    lookup_started: Final = threading.Event()
    loop_progressed: Final = threading.Event()
    loop_progressed_during_lookup: Final = threading.Event()
    monkeypatch.setattr(
        litellm,
        "secret_manager_client",
        _VaultWaitingForLoopProgress(lookup_started, loop_progressed, loop_progressed_during_lookup),
    )
    monkeypatch.setattr(litellm, "_key_management_system", KeyManagementSystem.HASHICORP_VAULT)
    monkeypatch.setattr(litellm, "_key_management_settings", KeyManagementSettings(access_mode="read_only"))

    async def count() -> TokenCountResponse:
        return await count_request_tokens(
            request=TokenCountRequest(model="sonnet", messages=[{"role": "user", "content": "hi"}]),
            call_endpoint=True,
        )

    async def prove_loop_progress() -> None:
        assert await asyncio.to_thread(lookup_started.wait, _LOOP_PROGRESS_TIMEOUT_SECONDS)
        loop_progressed.set()

    response, _ = await asyncio.gather(count(), prove_loop_progress())

    assert loop_progressed_during_lookup.is_set()
    assert response.total_tokens == 447


@pytest.mark.asyncio
async def test_framing_still_applies_while_the_deployment_is_cooling_down(monkeypatch: pytest.MonkeyPatch):
    router: Final = Router(
        model_list=[
            {
                "model_name": "framed",
                "litellm_params": {"model": "openai/gpt-5.4-mini", "api_key": "sk-unused-framed"},
                "model_info": {"id": "framed-deployment", "token_count_framing": 100},
            }
        ]
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
    messages: Final = [{"role": "user", "content": "hello there"}]
    warm: Final = await count_request_tokens(request=TokenCountRequest(model="framed", messages=messages))

    router.cooldown_cache.add_deployment_to_cooldown(
        model_id="framed-deployment",
        original_exception=Exception("rate limited"),
        exception_status=429,
        cooldown_time=60,
    )
    cooling: Final = await count_request_tokens(request=TokenCountRequest(model="framed", messages=messages))

    assert warm.total_tokens == litellm.token_counter(model="gpt-5.4-mini", messages=messages) + 100
    assert cooling.total_tokens - 100 == litellm.token_counter(model="framed", messages=messages)
    assert cooling.count_source == "local_estimate"


@pytest.mark.asyncio
async def test_a_cooling_team_deployment_frames_with_its_own_settings_not_a_same_named_global_one(
    monkeypatch: pytest.MonkeyPatch,
):
    router: Final = Router(
        model_list=[
            {
                "model_name": "shared-framed",
                "litellm_params": {"model": "openai/gpt-5.4-mini", "api_key": "sk-unused-global"},
                "model_info": {"id": "global-framed", "token_count_framing": 7},
            },
            {
                "model_name": "model_name_team-a_1",
                "litellm_params": {"model": "openai/gpt-5.4-mini", "api_key": "sk-unused-team"},
                "model_info": {
                    "id": "team-a-framed",
                    "team_id": "team-a",
                    "team_public_model_name": "shared-framed",
                    "token_count_framing": 100,
                },
            },
        ]
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
    router.cooldown_cache.add_deployment_to_cooldown(
        model_id="team-a-framed",
        original_exception=Exception("rate limited"),
        exception_status=429,
        cooldown_time=60,
    )
    messages: Final = [{"role": "user", "content": "hello there"}]

    cooling: Final = await count_request_tokens(
        request=TokenCountRequest(model="shared-framed", messages=messages), caller=UserAPIKeyAuth(team_id="team-a")
    )

    assert cooling.count_source == "local_estimate"
    assert cooling.total_tokens - 100 == litellm.token_counter(model="shared-framed", messages=messages)


_GEMINI_TOOL_OUTPUT: Final = "line of a long file that the agent read earlier. " * 180


class _GeminiCountTokensPeer:
    def __init__(self) -> None:
        self.request_bodies: tuple[bytes, ...] = ()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.request_bodies = (*self.request_bodies, request.content)
        return httpx.Response(200, json={"totalTokens": 41}, request=request)


def _agent_turns_with_tool_results(rounds: int) -> list[dict[str, JsonValue]]:
    return [
        *(
            turn
            for index in range(rounds)
            for turn in (
                {"role": "user", "content": f"Read file {index}."},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": f"toolu_{index}", "name": "read_file", "input": {"path": f"{index}"}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": f"toolu_{index}", "content": _GEMINI_TOOL_OUTPUT}
                    ],
                },
            )
        ),  # comprehension-ok: flattens each round into its three turns
        {"role": "user", "content": "Summarize what you read."},
    ]


_READ_FILE_TOOL: Final[dict[str, JsonValue]] = {
    "name": "read_file",
    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
}
_CLEAR_TOOL_USES_EDITS: Final[list[JsonValue]] = [
    {
        "type": "clear_tool_uses_20250919",
        "trigger": {"type": "input_tokens", "value": 500},
        "keep": {"type": "tool_uses", "value": 1},
    }
]
_CLEAR_TOOL_USES: Final[dict[str, JsonValue]] = {"edits": _CLEAR_TOOL_USES_EDITS}
_INVALID_COMPACT_EDIT: Final[dict[str, JsonValue]] = {
    "edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 1000}}]
}


def _gemini_deployment(model_name: str, litellm_params: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "model_name": model_name,
        "litellm_params": {"model": "gemini/gemini-2.5-flash", "api_key": "test-key", **litellm_params},
    }


async def _count_through_the_messages_endpoint(
    router: Router, request_options: Mapping[str, JsonValue], caller: UserAPIKeyAuth | None = None
) -> tuple[object, tuple[bytes, ...]]:
    import litellm.proxy.anthropic_endpoints.endpoints as anthropic_endpoints
    import litellm.proxy.proxy_server as proxy_server
    from litellm.llms.gemini.count_tokens import handler as gemini_count_tokens_handler

    peer: Final = _GeminiCountTokensPeer()
    request_body: Final[dict[str, JsonValue]] = {
        "messages": _agent_turns_with_tool_results(3),
        "tools": [_READ_FILE_TOOL],
        "context_management": _CLEAR_TOOL_USES,
        **request_options,
    }
    mock_request: Final = MagicMock(spec=Request)
    mock_request.headers = {}

    with (
        patch.object(proxy_server, "llm_router", router),
        patch.object(anthropic_endpoints, "read_request_body", AsyncMock(return_value=request_body)),
        patch.object(
            gemini_count_tokens_handler,
            "get_async_httpx_client",
            return_value=AsyncHTTPHandler(transport=httpx.MockTransport(peer)),
        ),
    ):
        response: Final = await anthropic_count_tokens(mock_request, Response(), caller or MagicMock())
    return response, peer.request_bodies


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "requested_model",
        "additional_drop_params",
        "router_additional_drop_params",
        "request_options",
        "expected_tool_outputs_sent",
    ),
    (
        ("gemini-agent", None, None, {}, 1),
        ("gemini-agent", ["context_management"], None, {}, 3),
        ("gemini-agent", None, None, {"additional_drop_params": ["context_management"]}, 3),
        ("gemini-agent", ["context_management"], None, {"additional_drop_params": []}, 1),
        ("gemini-agent", None, ["context_management"], {}, 3),
        ("gemini-agent", None, ["context_management"], {"additional_drop_params": []}, 1),
        ("gemini/gemini-2.5-flash", None, None, {}, 1),
        ("gemini/gemini-2.5-flash", ["context_management"], None, {}, 3),
        ("gemini/gemini-2.5-flash", None, ["context_management"], {}, 3),
        ("gemini/gemini-2.5-flash", None, None, {"additional_drop_params": ["context_management"]}, 3),
        ("gemini/gemini-2.5-flash", ["context_management"], None, {"additional_drop_params": []}, 1),
    ),
    ids=(
        "edits-applied",
        "deployment-drops-context-management",
        "request-drops-context-management",
        "request-overrides-the-deployment-drop",
        "router-default-drops-context-management",
        "request-overrides-the-router-default-drop",
        "litellm-model-name-edits-applied",
        "litellm-model-name-deployment-drops-context-management",
        "litellm-model-name-router-default-drops-context-management",
        "litellm-model-name-request-drops-context-management",
        "litellm-model-name-request-overrides-the-deployment-drop",
    ),
)
async def test_gemini_messages_count_tokens_applies_the_context_edits_inference_applies(
    requested_model: str,
    additional_drop_params: list[str] | None,
    router_additional_drop_params: list[str] | None,
    request_options: dict[str, JsonValue],
    expected_tool_outputs_sent: int,
) -> None:
    router: Final = Router(
        model_list=[_gemini_deployment("gemini-agent", {"additional_drop_params": additional_drop_params})],
        default_litellm_params={"additional_drop_params": router_additional_drop_params},
    )
    response, sent_bodies = await _count_through_the_messages_endpoint(
        router, {**request_options, "model": requested_model}
    )

    assert response == {"input_tokens": 41, "litellm_count_source": "provider"}
    assert len(sent_bodies) == 1
    assert sent_bodies[0].decode().count(_GEMINI_TOOL_OUTPUT) == expected_tool_outputs_sent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested_model", "model_list"),
    (
        (
            "gemini-agent",
            [
                {
                    "model_name": "gemini-agent",
                    "litellm_params": {"model": "gemini/gemini-2.5-flash", "api_key": "test-key", "order": 1},
                },
                {
                    "model_name": "gemini-agent",
                    "litellm_params": {
                        "model": "gemini/gemini-2.5-flash",
                        "api_key": "test-key",
                        "order": 2,
                        "additional_drop_params": ["context_management"],
                    },
                },
            ],
        ),
        (
            "gemini-agent-deployment-id",
            [
                {
                    "model_name": "gemini-agent",
                    "litellm_params": {
                        "model": "gemini/gemini-2.5-flash",
                        "api_key": "test-key",
                        "additional_drop_params": ["context_management"],
                    },
                    "model_info": {"id": "gemini-agent-deployment-id"},
                },
            ],
        ),
        (
            "gemini-agent-deployment-id",
            [
                {"model_name": "*", "litellm_params": {"model": "gemini/*", "api_key": "test-key"}},
                {
                    "model_name": "gemini-agent",
                    "litellm_params": {
                        "model": "gemini/gemini-2.5-flash",
                        "api_key": "test-key",
                        "additional_drop_params": ["context_management"],
                    },
                    "model_info": {"id": "gemini-agent-deployment-id"},
                },
            ],
        ),
    ),
    ids=("group-member-drops", "deployment-id-drops", "deployment-id-drops-beside-a-wildcard"),
)
async def test_gemini_messages_count_tokens_keeps_tool_results_when_a_routable_deployment_drops_context_management(
    requested_model: str, model_list: list[dict[str, JsonValue]]
) -> None:
    response, sent_bodies = await _count_through_the_messages_endpoint(
        Router(model_list=model_list), {"model": requested_model}
    )

    assert response == {"input_tokens": 41, "litellm_count_source": "provider"}
    assert sent_bodies[0].decode().count(_GEMINI_TOOL_OUTPUT) == 3


def _team_and_global_gemini_router(team_drops_context_management: bool) -> Router:
    drops: Final[dict[str, JsonValue]] = {"additional_drop_params": ["context_management"]}
    return Router(
        model_list=[
            _gemini_deployment("gemini-agent", {} if team_drops_context_management else drops),
            {
                **_gemini_deployment("model_name_team-a_1", drops if team_drops_context_management else {}),
                "model_info": {"id": "team-a-gemini", "team_id": "team-a", "team_public_model_name": "gemini-agent"},
            },
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("team_drops_context_management", "team_id", "expected_tool_outputs_sent"),
    (
        (True, "team-a", 3),
        (True, "team-b", 1),
        (False, "team-a", 1),
        (False, "team-b", 3),
    ),
    ids=(
        "team-deployment-drops-for-its-team",
        "team-deployment-drop-skips-other-teams",
        "global-drop-skips-the-team-with-its-own-deployment",
        "global-drop-applies-to-teams-on-the-global-deployment",
    ),
)
async def test_gemini_messages_count_tokens_reads_the_context_management_opt_out_from_the_callers_deployments(
    team_drops_context_management: bool, team_id: str, expected_tool_outputs_sent: int
) -> None:
    response, sent_bodies = await _count_through_the_messages_endpoint(
        _team_and_global_gemini_router(team_drops_context_management),
        {"model": "gemini-agent"},
        UserAPIKeyAuth(team_id=team_id),
    )

    assert response == {"input_tokens": 41, "litellm_count_source": "provider"}
    assert sent_bodies[0].decode().count(_GEMINI_TOOL_OUTPUT) == expected_tool_outputs_sent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested_model", "make_router", "request_options", "expected_tool_outputs_sent"),
    (
        (
            "unrouted-model",
            lambda: Router(model_list=[_gemini_deployment("gemini-agent", {})], default_fallbacks=["gemini-agent"]),
            {},
            1,
        ),
        (
            "unrouted-model",
            lambda: Router(
                model_list=[_gemini_deployment("gemini-agent", {"additional_drop_params": ["context_management"]})],
                default_fallbacks=["gemini-agent"],
            ),
            {},
            3,
        ),
        (
            "unrouted-model",
            lambda: Router(
                model_list=[_gemini_deployment("gemini-agent", {})],
                default_fallbacks=["gemini-agent"],
                default_litellm_params={"additional_drop_params": ["context_management"]},
            ),
            {},
            3,
        ),
        (
            "unrouted-model",
            lambda: Router(model_list=[_gemini_deployment("gemini-agent", {})], default_fallbacks=["gemini-agent"]),
            {"additional_drop_params": ["context_management"]},
            3,
        ),
        (
            "flash-alias",
            lambda: Router(
                model_list=[_gemini_deployment("gemini-agent", {"additional_drop_params": ["context_management"]})],
                model_group_alias={"flash-alias": "gemini/gemini-2.5-flash"},
            ),
            {},
            3,
        ),
    ),
    ids=(
        "default-fallback-edits-applied",
        "default-fallback-deployment-drops-context-management",
        "default-fallback-router-default-drops-context-management",
        "default-fallback-request-drops-context-management",
        "alias-to-a-litellm-model-deployment-drops-context-management",
    ),
)
async def test_gemini_messages_count_tokens_reads_the_opt_out_of_the_deployment_the_router_falls_back_to(
    requested_model: str,
    make_router: Callable[[], Router],
    request_options: dict[str, JsonValue],
    expected_tool_outputs_sent: int,
) -> None:
    response, sent_bodies = await _count_through_the_messages_endpoint(
        make_router(), {**request_options, "model": requested_model}
    )

    assert response == {"input_tokens": 41, "litellm_count_source": "provider"}
    assert sent_bodies[0].decode().count(_GEMINI_TOOL_OUTPUT) == expected_tool_outputs_sent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested_model", "completion_model", "expected_tool_outputs_sent"),
    (("gemini-agent", "gemini-dropping", 3), ("gemini-dropping", "gemini-agent", 1)),
    ids=("completion-model-drops-context-management", "completion-model-keeps-context-management"),
)
async def test_gemini_messages_count_tokens_reads_the_opt_out_of_the_completion_model_inference_sends_to(
    requested_model: str, completion_model: str, expected_tool_outputs_sent: int
) -> None:
    import litellm.proxy.proxy_server as proxy_server

    router: Final = Router(
        model_list=[
            _gemini_deployment("gemini-agent", {}),
            _gemini_deployment("gemini-dropping", {"additional_drop_params": ["context_management"]}),
        ]
    )
    with patch.object(proxy_server, "general_settings", {"completion_model": completion_model}):
        response, sent_bodies = await _count_through_the_messages_endpoint(router, {"model": requested_model})

    assert response == {"input_tokens": 41, "litellm_count_source": "provider"}
    assert sent_bodies[0].decode().count(_GEMINI_TOOL_OUTPUT) == expected_tool_outputs_sent


class _GeminiPart(BaseModel):
    model_config = ConfigDict(frozen=True)
    text: str = ""


class _GeminiContent(BaseModel):
    model_config = ConfigDict(frozen=True)
    parts: tuple[_GeminiPart, ...] = ()


class _GeminiCountTokensBody(BaseModel):
    model_config = ConfigDict(frozen=True)
    contents: tuple[_GeminiContent, ...] = ()


def _word_count(content: _GeminiContent) -> int:
    return sum(len(part.text.split()) for part in content.parts)


def _gemini_word_counting_peer(request: httpx.Request) -> httpx.Response:
    body: Final = _GeminiCountTokensBody.model_validate_json(request.content)
    total: Final = sum(_word_count(content) for content in body.contents)
    return httpx.Response(200, json={"totalTokens": total}, request=request)


def _unavailable_peer(request: httpx.Request) -> httpx.Response:
    return httpx.Response(503, json={"error": {"message": "unavailable"}}, request=request)


def _gemini_router(drop_params: bool = False) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "gemini-counted",
                "litellm_params": {
                    "model": "gemini/gemini-3.8-flash",
                    "api_key": "test-key",
                    "api_base": "https://count-tokens.test",
                    "drop_params": drop_params,
                },
            }
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("deployment_params", "router_defaults", "request_options", "litellm_drop_params"),
    (
        ({"drop_params": True}, {}, {}, False),
        ({}, {"drop_params": True}, {}, False),
        ({}, {}, {"drop_params": True}, False),
        ({}, {}, {"drop_params": "true"}, False),
        ({}, {}, {}, True),
    ),
    ids=(
        "deployment-drop-params",
        "router-default-drop-params",
        "request-drop-params",
        "request-drop-params-string",
        "litellm-drop-params",
    ),
)
async def test_gemini_messages_count_tokens_counts_the_unedited_body_when_drop_params_drops_an_invalid_edit(
    deployment_params: dict[str, JsonValue],
    router_defaults: dict[str, JsonValue],
    request_options: dict[str, JsonValue],
    litellm_drop_params: bool,
) -> None:
    router: Final = Router(
        model_list=[_gemini_deployment("gemini-agent", deployment_params)], default_litellm_params=router_defaults
    )
    with patch.object(litellm, "drop_params", litellm_drop_params):
        response, sent_bodies = await _count_through_the_messages_endpoint(
            router, {**request_options, "model": "gemini-agent", "context_management": _INVALID_COMPACT_EDIT}
        )

    assert response == {"input_tokens": 41, "litellm_count_source": "provider"}
    assert sent_bodies[0].decode().count(_GEMINI_TOOL_OUTPUT) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("router_defaults", "request_options", "expected_status"),
    (({}, {"drop_params": True}, None), ({"drop_params": True}, {}, None), ({}, {}, 400)),
    ids=("request-drop-params", "router-default-drop-params", "no-drop-params"),
)
async def test_messages_count_tokens_for_a_model_with_no_deployment_drops_an_invalid_edit_only_under_drop_params(
    monkeypatch: pytest.MonkeyPatch,
    router_defaults: dict[str, JsonValue],
    request_options: dict[str, JsonValue],
    expected_status: int | None,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    with patch.object(litellm, "drop_params", False):
        response, _ = await _count_through_the_messages_endpoint(
            Router(model_list=[_gemini_deployment("gemini-agent", {})], default_litellm_params=router_defaults),
            {**request_options, "model": "gemini/gemini-3.8-pro", "context_management": _INVALID_COMPACT_EDIT},
        )

    assert (response.status_code if isinstance(response, JSONResponse) else None) == expected_status


@pytest.mark.asyncio
async def test_gemini_messages_count_tokens_rejects_an_invalid_edit_without_drop_params() -> None:
    with patch.object(litellm, "drop_params", False):
        response, sent_bodies = await _count_through_the_messages_endpoint(
            Router(model_list=[_gemini_deployment("gemini-agent", {})]),
            {"model": "gemini-agent", "context_management": _INVALID_COMPACT_EDIT},
        )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert sent_bodies == ()


_GROQ_MODEL: Final = "groq/llama-3.3-70b-versatile"


def _groq_router(
    litellm_params: Mapping[str, JsonValue] = MappingProxyType({}),
    router_defaults: Mapping[str, JsonValue] = MappingProxyType({}),
) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "groq-agent",
                "litellm_params": {"model": _GROQ_MODEL, "api_key": "test-key", **litellm_params},
            }
        ],
        default_litellm_params=dict(router_defaults),
    )


async def _messages_endpoint_count(router: Router, request_body: Mapping[str, JsonValue]) -> object:
    import litellm.proxy.anthropic_endpoints.endpoints as anthropic_endpoints
    import litellm.proxy.proxy_server as proxy_server

    mock_request: Final = MagicMock(spec=Request)
    mock_request.headers = {}
    with (
        patch.object(proxy_server, "llm_router", router),
        patch.object(anthropic_endpoints, "read_request_body", AsyncMock(return_value=dict(request_body))),
    ):
        return await anthropic_count_tokens(mock_request, Response(), MagicMock())


class _MessagesCountReply(BaseModel):
    model_config = ConfigDict(frozen=True)
    input_tokens: int


async def _input_tokens(router: Router, request_body: Mapping[str, JsonValue]) -> int:
    return _MessagesCountReply.model_validate(await _messages_endpoint_count(router, request_body)).input_tokens


async def _messages_the_inference_polyfill_sends(model: str) -> list[dict[str, JsonValue]]:
    from litellm.llms.anthropic.messages.adapters.handler import prepare_context_managed_request

    prepared: Final = await prepare_context_managed_request(
        model=model,
        messages=[dict(message) for message in _agent_turns_with_tool_results(3)],
        tools=[dict(_READ_FILE_TOOL)],
        system=None,
        context_management_spec=dict(_CLEAR_TOOL_USES),
        litellm_metadata=None,
        additional_drop_params=None,
        llm_router=None,
    )
    assert prepared is not None
    return TypeAdapter(list[dict[str, JsonValue]]).validate_python(prepared.messages)


@pytest.mark.asyncio
async def test_messages_count_tokens_counts_the_body_the_inference_polyfill_sends_on_a_bridged_provider() -> None:
    router: Final = _groq_router()
    unedited_body: Final[dict[str, JsonValue]] = {
        "model": "groq-agent",
        "messages": _agent_turns_with_tool_results(3),
        "tools": [_READ_FILE_TOOL],
    }

    edited: Final = await _input_tokens(router, {**unedited_body, "context_management": _CLEAR_TOOL_USES})
    unedited: Final = await _input_tokens(router, unedited_body)
    polyfilled: Final = await _input_tokens(
        router, {**unedited_body, "messages": await _messages_the_inference_polyfill_sends(_GROQ_MODEL)}
    )

    assert edited < unedited
    assert edited == polyfilled


class _MistralCountTokensPeer:
    def __init__(self) -> None:
        self.request_bodies: tuple[bytes, ...] = ()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.request_bodies = (*self.request_bodies, request.content)
        return httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 57}, request=request)


@pytest.mark.asyncio
@pytest.mark.parametrize("as_blocks", [False, True])
@pytest.mark.parametrize("prefix", ["ordinary", "single", "adjacent", "empty_first", "literal", "header_only", "style"])
async def test_mistral_count_system_matches_the_generation_http_payload(
    monkeypatch: pytest.MonkeyPatch, as_blocks: bool, prefix: str
) -> None:
    billing_a: Final = "x-anthropic-billing-header: cc_version=2; cc_entrypoint=cli; cch=A;"
    billing_b: Final = "x-anthropic-billing-header: cc_version=2; cc_entrypoint=cli; cch=B;"
    before, after = {
        "ordinary": (["answer=OK"], ["answer=OK"]),
        "single": ([billing_a, "Be concise."], ["Be concise."]),
        "adjacent": ([billing_a, billing_b, "answer=OK"], [billing_b, "answer=OK"]),
        "empty_first": (["", billing_b, "Be concise."], ["", billing_b, "Be concise."]),
        "literal": (
            ["x-anthropic-billing-header: not metadata", "answer=OK"],
            ["x-anthropic-billing-header: not metadata", "answer=OK"],
        ),
        "header_only": ([billing_a], []),
        "style": ([billing_a, "style=concise"], ["style=concise"]),
    }[prefix]
    system: Final = [{"type": "text", "text": text} for text in before] if as_blocks else "\n".join(before)
    expected: Final = ("" if as_blocks else "\n").join(after)
    router: Final = Router(
        model_list=[
            {
                "model_name": "mistral-agent",
                "litellm_params": {"model": "mistral/mistral-large-4", "api_key": "test-key"},
            }
        ]
    )
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    with respx.mock:
        counted: Final = respx.post("https://api.mistral.ai/v1/responses/input_tokens").mock(
            return_value=httpx.Response(200, json={"input_tokens": 57})
        )
        generated: Final = respx.post("https://api.mistral.ai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "mistral-large-4",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 57, "completion_tokens": 1, "total_tokens": 58},
                },
            )
        )
        assert (
            await _input_tokens(
                router, {"model": "mistral-agent", "messages": [{"role": "user", "content": "Hello"}], "system": system}
            )
            == 57
        )
        await litellm.anthropic.messages.acreate(
            model="mistral/mistral-large-4",
            api_key="test-key",
            messages=[{"role": "user", "content": "Hello"}],
            system=system,
            max_tokens=32,
        )
    sent: Final = json.loads(generated.calls.last.request.content)
    assert sent["messages"] == ([{"role": "system", "content": expected}] if expected else []) + [
        {"role": "user", "content": "Hello"}
    ]
    count_body: Final = json.loads(counted.calls.last.request.content)
    assert count_body == {
        "model": "mistral-large-4",
        "input": (
            [{"type": "message", "role": "developer", "content": [{"type": "input_text", "text": expected}]}]
            if expected
            else []
        )
        + [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
    }


@pytest.mark.asyncio
async def test_messages_count_tokens_sends_mistral_the_messages_the_inference_polyfill_edits() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "mistral-agent",
                "litellm_params": {"model": "mistral/mistral-large-4", "api_key": "test-mistral-key"},
            }
        ]
    )
    unedited_body: Final[dict[str, JsonValue]] = {
        "model": "mistral-agent",
        "messages": _agent_turns_with_tool_results(3),
        "tools": [_READ_FILE_TOOL],
    }
    polyfilled_messages: Final = await _messages_the_inference_polyfill_sends("mistral/mistral-large-4")
    peer: Final = _MistralCountTokensPeer()

    with patch.object(
        mistral_count_tokens_http,
        "get_async_httpx_client",
        return_value=AsyncHTTPHandler(transport=httpx.MockTransport(peer)),
    ):
        for request_body in (
            {**unedited_body, "context_management": _CLEAR_TOOL_USES},
            {**unedited_body, "messages": polyfilled_messages},
            unedited_body,
        ):
            assert await _input_tokens(router, request_body) == 57

    edited, polyfilled, unedited = (json.loads(body)["input"] for body in peer.request_bodies)
    assert edited == polyfilled
    assert edited != unedited


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("deployment_params", "router_defaults", "request_options"),
    (
        ({"additional_drop_params": ["context_management"]}, {}, {}),
        ({}, {"additional_drop_params": ["context_management"]}, {}),
        ({}, {}, {"additional_drop_params": ["context_management"]}),
    ),
    ids=("deployment-drops", "router-default-drops", "request-drops"),
)
async def test_messages_count_tokens_on_a_bridged_provider_keeps_the_unedited_count_when_context_management_is_dropped(
    deployment_params: dict[str, JsonValue],
    router_defaults: dict[str, JsonValue],
    request_options: dict[str, JsonValue],
) -> None:
    router: Final = _groq_router(deployment_params, router_defaults)
    unedited_body: Final[dict[str, JsonValue]] = {
        "model": "groq-agent",
        "messages": _agent_turns_with_tool_results(3),
        "tools": [_READ_FILE_TOOL],
        **request_options,
    }

    edited: Final = await _input_tokens(router, {**unedited_body, "context_management": _CLEAR_TOOL_USES})

    assert edited == await _input_tokens(router, unedited_body)


@pytest.mark.asyncio
async def test_messages_count_tokens_on_a_bridged_provider_rejects_an_invalid_edit_without_drop_params() -> None:
    with patch.object(litellm, "drop_params", False):
        response: Final = await _messages_endpoint_count(
            _groq_router(),
            {
                "model": "groq-agent",
                "messages": _agent_turns_with_tool_results(3),
                "tools": [_READ_FILE_TOOL],
                "context_management": _INVALID_COMPACT_EDIT,
            },
        )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_messages_count_tokens_on_a_bridged_provider_counts_without_an_invalid_edit_under_drop_params() -> None:
    router: Final = _groq_router()
    unedited_body: Final[dict[str, JsonValue]] = {
        "model": "groq-agent",
        "messages": _agent_turns_with_tool_results(3),
        "tools": [_READ_FILE_TOOL],
        "drop_params": True,
    }

    with patch.object(litellm, "drop_params", False):
        edited: Final = await _input_tokens(router, {**unedited_body, "context_management": _INVALID_COMPACT_EDIT})
        unedited: Final = await _input_tokens(router, unedited_body)

    assert edited == unedited


def _turns_compacted_by_the_client() -> list[dict[str, JsonValue]]:
    return [
        *_agent_turns_with_tool_results(3)[:-1],
        {"role": "assistant", "content": [{"type": "compaction", "content": "The agent read files 0, 1 and 2."}]},
        {"role": "user", "content": "Summarize what you read."},
    ]


@pytest.mark.asyncio
async def test_messages_count_tokens_on_a_bridged_provider_counts_only_what_follows_a_client_compaction_block() -> None:
    from litellm.llms.anthropic.messages.adapters.handler import prepare_context_managed_request

    router: Final = _groq_router()
    sent: Final = await prepare_context_managed_request(
        model=_GROQ_MODEL,
        messages=[dict(message) for message in _turns_compacted_by_the_client()],
        tools=[dict(_READ_FILE_TOOL)],
        system=None,
        context_management_spec=None,
        litellm_metadata=None,
        additional_drop_params=None,
        llm_router=None,
    )
    assert sent is not None

    compacted: Final = await _input_tokens(
        router, {"model": "groq-agent", "messages": _turns_compacted_by_the_client(), "tools": [_READ_FILE_TOOL]}
    )
    uncompacted: Final = await _input_tokens(
        router, {"model": "groq-agent", "messages": _agent_turns_with_tool_results(3), "tools": [_READ_FILE_TOOL]}
    )
    sent_body_count: Final = await _input_tokens(
        router,
        {
            "model": "groq-agent",
            "messages": TypeAdapter(list[dict[str, JsonValue]]).validate_python(sent.messages),
            "system": TypeAdapter(JsonValue).validate_python(sent.system),
            "tools": [_READ_FILE_TOOL],
        },
    )

    assert compacted < uncompacted
    assert compacted == sent_body_count


@pytest.mark.asyncio
async def test_messages_count_tokens_applies_the_edits_a_responses_api_bridge_polyfills_for_a_non_openai_provider() -> (
    None
):
    perplexity_model: Final = "perplexity/perplexity/sonar"
    router: Final = Router(
        model_list=[{"model_name": "sonar-agent", "litellm_params": {"model": perplexity_model, "api_key": "test-key"}}]
    )
    unedited_body: Final[dict[str, JsonValue]] = {
        "model": "sonar-agent",
        "messages": _agent_turns_with_tool_results(3),
        "tools": [_READ_FILE_TOOL],
    }

    edited: Final = await _input_tokens(router, {**unedited_body, "context_management": _CLEAR_TOOL_USES})
    unedited: Final = await _input_tokens(router, unedited_body)
    polyfilled: Final = await _input_tokens(
        router, {**unedited_body, "messages": await _messages_the_inference_polyfill_sends(perplexity_model)}
    )

    assert edited < unedited
    assert edited == polyfilled


@pytest.mark.asyncio
async def test_messages_count_tokens_on_a_responses_api_bridge_counts_the_unedited_body_when_an_edit_fails() -> None:
    from litellm.llms.anthropic.messages.responses_adapters import handler as responses_adapters_handler

    router: Final = Router(
        model_list=[
            {
                "model_name": "sonar-agent",
                "litellm_params": {"model": "perplexity/perplexity/sonar", "api_key": "test-key"},
            }
        ]
    )
    unedited_body: Final[dict[str, JsonValue]] = {
        "model": "sonar-agent",
        "messages": _agent_turns_with_tool_results(3),
        "tools": [_READ_FILE_TOOL],
    }

    with patch.object(
        responses_adapters_handler, "apply_context_management", AsyncMock(side_effect=RuntimeError("editor bug"))
    ):
        edited: Final = await _input_tokens(router, {**unedited_body, "context_management": _CLEAR_TOOL_USES})

    assert edited == await _input_tokens(router, unedited_body)


@pytest.mark.asyncio
async def test_messages_count_tokens_on_a_responses_api_bridge_rejects_an_invalid_edit_without_drop_params() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "sonar-agent",
                "litellm_params": {"model": "perplexity/perplexity/sonar", "api_key": "test-key"},
            }
        ]
    )
    invalid_clear_edit: Final[dict[str, JsonValue]] = {
        "edits": [{"type": "clear_tool_uses_20250919", "clear_at_least": "all of them"}]
    }

    with patch.object(litellm, "drop_params", False):
        response: Final = await _messages_endpoint_count(
            router,
            {
                "model": "sonar-agent",
                "messages": _agent_turns_with_tool_results(3),
                "tools": [_READ_FILE_TOOL],
                "context_management": invalid_clear_edit,
            },
        )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    message: Final = json.loads(bytes(response.body))["error"]["message"]
    assert "context_management.edits.0.clear_tool_uses_20250919.clear_at_least" in message


class _CountTokensPeer:
    def __init__(self, reply: Mapping[str, JsonValue], status_code: int = 200) -> None:
        self._reply: Final = reply
        self._status_code: Final = status_code
        self.requests: tuple[httpx.Request, ...] = ()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests = (*self.requests, request)
        return httpx.Response(self._status_code, json=dict(self._reply), request=request)


class _BedrockInvokeModel(BaseModel):
    model_config = ConfigDict(frozen=True)
    body: str


class _BedrockCountTokensInput(BaseModel):
    model_config = ConfigDict(frozen=True)
    invoke_model: _BedrockInvokeModel = Field(alias="invokeModel")


class _BedrockCountTokensBody(BaseModel):
    model_config = ConfigDict(frozen=True)
    input: _BedrockCountTokensInput


def _bedrock_counted_request(request_body: bytes) -> str:
    invoke_model: Final = _BedrockCountTokensBody.model_validate_json(request_body).input.invoke_model
    return base64.b64decode(invoke_model.body).decode()


@pytest.mark.asyncio
@pytest.mark.parametrize("as_blocks", [False, True])
@pytest.mark.parametrize("provider", ["gemini", "bedrock"])
async def test_nontranslating_provider_count_removes_only_the_first_billing_block(
    monkeypatch: pytest.MonkeyPatch, as_blocks: bool, provider: str
) -> None:
    billing_a: Final = "x-anthropic-billing-header: cc_version=2; cc_entrypoint=cli; cch=A;"
    billing_b: Final = "x-anthropic-billing-header: cc_version=2; cc_entrypoint=cli; cch=B;"
    body: Final[dict[str, JsonValue]] = {
        "messages": [{"role": "user", "content": "Hello"}],
        "system": [{"type": "text", "text": text} for text in (billing_a, billing_b, "Be concise.")]
        if as_blocks
        else billing_a + "\n" + billing_b + "\nBe concise.",
    }
    expected: Final = (
        [{"text": text} for text in (billing_b, "Be concise.")]
        if as_blocks
        else [{"text": billing_b + "\nBe concise."}]
    )
    if provider == "bedrock":
        response, sent = await _count_on_a_bridged_provider_count_api(body, monkeypatch)
        assert response == {"input_tokens": 57, "litellm_count_source": "provider"}
        assert json.loads(sent[0].content)["input"]["converse"]["system"] == expected
        return
    router: Final = Router(model_list=[_gemini_deployment("gemini-agent", {})])
    response, sent_bodies = await _count_through_the_messages_endpoint(router, {**body, "model": "gemini-agent"})
    assert response == {"input_tokens": 41, "litellm_count_source": "provider"}
    assert json.loads(sent_bodies[0])["generateContentRequest"]["systemInstruction"] == {"parts": expected}


async def _count_on_a_bridged_provider_count_api(
    request_body: Mapping[str, JsonValue],
    monkeypatch: pytest.MonkeyPatch,
    bedrock_model: str = "bedrock/us.amazon.nova-2-lite-v1:0",
    peer_reply: Mapping[str, JsonValue] = MappingProxyType({"inputTokens": 57}),
    model_info: Mapping[str, JsonValue] = MappingProxyType({}),
    peer_status: int = 200,
) -> tuple[object, tuple[httpx.Request, ...]]:
    from litellm.llms.bedrock.count_tokens import handler as bedrock_count_tokens_handler
    from litellm.llms.bedrock.count_tokens import mantle_handler as bedrock_mantle_count_tokens_handler

    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    peer: Final = _CountTokensPeer(peer_reply, peer_status)
    router: Final = Router(
        model_list=[
            {
                "model_name": "nova-agent",
                "litellm_params": {
                    "model": bedrock_model,
                    "aws_access_key_id": "AKIATESTACCESSKEY",
                    "aws_secret_access_key": "test-secret",
                    "aws_region_name": "us-east-1",
                },
                "model_info": dict(model_info),
            }
        ]
    )
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(peer))
    with (
        patch.object(bedrock_count_tokens_handler, "get_async_httpx_client", return_value=client),
        patch.object(bedrock_mantle_count_tokens_handler, "get_async_httpx_client", return_value=client),
    ):
        response: Final = await _messages_endpoint_count(router, {**request_body, "model": "nova-agent"})
    return response, peer.requests


@pytest.mark.asyncio
async def test_messages_count_tokens_sends_the_edited_messages_to_a_bridged_provider_count_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response, sent_bodies = await _count_on_a_bridged_provider_count_api(
        {
            "messages": _agent_turns_with_tool_results(3),
            "tools": [_READ_FILE_TOOL],
            "context_management": _CLEAR_TOOL_USES,
        },
        monkeypatch,
    )

    assert response == {"input_tokens": 57, "litellm_count_source": "provider"}
    assert len(sent_bodies) == 1
    assert _bedrock_counted_request(sent_bodies[0].content).count(_GEMINI_TOOL_OUTPUT) == 1


@pytest.mark.asyncio
async def test_messages_count_tokens_counts_only_empty_text_on_a_bridged_provider_count_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_turn: Final[dict[str, JsonValue]] = {"role": "user", "content": [{"type": "text", "text": ""}]}

    response, sent_bodies = await _count_on_a_bridged_provider_count_api({"messages": [empty_turn]}, monkeypatch)

    assert response == {"input_tokens": 57, "litellm_count_source": "provider"}
    assert len(sent_bodies) == 1


_BEDROCK_CLAUDE_MODEL: Final = "bedrock/us.anthropic.claude-sonnet-5-5"
_CLEAR_THINKING_EDIT: Final[dict[str, JsonValue]] = {"type": "clear_thinking_20251015"}
_COUNTED_ON_MANTLE: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"token_count_model": {"model": "us.anthropic.claude-sonnet-5-5", "litellm_provider": "bedrock_mantle"}}
)
_COUNTED_ON_RUNTIME: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"token_count_model": {"model": "us.anthropic.claude-sonnet-5-5", "litellm_provider": "bedrock"}}
)


def _bedrock_claude_count_request(edits: list[JsonValue]) -> Mapping[str, JsonValue]:
    return MappingProxyType(
        {
            "messages": _agent_turns_with_tool_results(3),
            "tools": [_READ_FILE_TOOL],
            "context_management": {"edits": edits},
        }
    )


@pytest.mark.asyncio
async def test_messages_count_tokens_forwards_bedrock_supported_edits_and_their_beta_to_the_mantle_count_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response, sent = await _count_on_a_bridged_provider_count_api(
        _bedrock_claude_count_request([*_CLEAR_TOOL_USES_EDITS, _CLEAR_THINKING_EDIT]),
        monkeypatch,
        _BEDROCK_CLAUDE_MODEL,
        MappingProxyType({"input_tokens": 57, "context_management": {"original_input_tokens": 4100}}),
        _COUNTED_ON_MANTLE,
    )

    assert isinstance(response, dict)
    assert (response["input_tokens"], response["context_management"]) == (57, {"original_input_tokens": 4100})
    assert len(sent) == 1
    assert json.loads(sent[0].content)["context_management"] == _CLEAR_TOOL_USES
    assert sent[0].headers["anthropic-beta"] == ANTHROPIC_BETA_HEADER_VALUES.CONTEXT_MANAGEMENT_2025_06_27.value


@pytest.mark.asyncio
async def test_messages_count_tokens_forwards_bedrock_supported_edits_and_their_beta_to_the_runtime_count_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response, sent = await _count_on_a_bridged_provider_count_api(
        _bedrock_claude_count_request([*_CLEAR_TOOL_USES_EDITS, _CLEAR_THINKING_EDIT]),
        monkeypatch,
        _BEDROCK_CLAUDE_MODEL,
        model_info=_COUNTED_ON_RUNTIME,
    )

    assert isinstance(response, dict)
    assert response["input_tokens"] == 57
    assert len(sent) == 1
    invoke_body: Final = json.loads(_bedrock_counted_request(sent[0].content))
    assert (invoke_body["context_management"], invoke_body["anthropic_beta"]) == (
        _CLEAR_TOOL_USES,
        [ANTHROPIC_BETA_HEADER_VALUES.CONTEXT_MANAGEMENT_2025_06_27.value],
    )


@pytest.mark.parametrize(
    ("count_options", "region_hints"),
    [({"thinking": {"type": "enabled", "budget_tokens": 1024}}, 0), ({}, 1)],
    ids=["count_options_forwarded", "no_count_options"],
)
@pytest.mark.asyncio
async def test_a_mantle_count_rejecting_the_forwarded_count_options_answers_without_the_region_hint(
    count_options: Mapping[str, JsonValue], region_hints: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    response, sent = await _count_on_a_bridged_provider_count_api(
        {"messages": [{"role": "user", "content": "Read the README"}], **count_options},
        monkeypatch,
        _BEDROCK_CLAUDE_MODEL,
        MappingProxyType(
            {"type": "error", "error": {"type": "invalid_request_error", "message": "thinking: invalid value"}}
        ),
        MappingProxyType(
            {
                "token_count_model": {
                    "model": "us.anthropic.claude-sonnet-5-5",
                    "litellm_provider": "bedrock_mantle",
                    "region": "us-west-2",
                }
            }
        ),
        peer_status=400,
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert len(sent) == 1
    assert ("thinking" in json.loads(sent[0].content)) == bool(count_options)
    message: Final = json.loads(bytes(response.body))["error"]["message"]
    assert message.count("counts tokens with us.anthropic.claude-sonnet-5-5 in us-west-2") == region_hints


@pytest.mark.asyncio
async def test_messages_count_tokens_sends_no_context_management_to_bedrock_when_no_edit_is_bedrock_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response, sent = await _count_on_a_bridged_provider_count_api(
        _bedrock_claude_count_request([_CLEAR_THINKING_EDIT]),
        monkeypatch,
        _BEDROCK_CLAUDE_MODEL,
        MappingProxyType({"input_tokens": 57}),
        _COUNTED_ON_MANTLE,
    )

    assert isinstance(response, dict)
    assert response["input_tokens"] == 57
    assert len(sent) == 1
    assert "context_management" not in json.loads(sent[0].content)
    assert "anthropic-beta" not in sent[0].headers


@pytest.mark.asyncio
async def test_messages_count_tokens_rejects_a_non_string_tool_result_text_before_a_bridged_provider_applies_edits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response, sent_bodies = await _count_on_a_bridged_provider_count_api(
        {"messages": _build_log_messages(5), "context_management": _CLEAR_TOOL_USES}, monkeypatch
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert "tool_result toolu_log content.1.text" in json.loads(bytes(response.body))["error"]["message"]
    assert sent_bodies == ()


@pytest.mark.asyncio
async def test_messages_count_tokens_applies_a_responses_api_deployment_edits_once() -> None:
    peer: Final = _CountTokensPeer({"object": "response.input_tokens", "input_tokens": 71})
    router: Final = Router(
        model_list=[{"model_name": "gpt-agent", "litellm_params": {"model": "openai/gpt-5.5", "api_key": "test-key"}}]
    )

    with patch.object(
        openai_count_tokens_http,
        "get_async_httpx_client",
        return_value=AsyncHTTPHandler(transport=httpx.MockTransport(peer)),
    ):
        response: Final = await _messages_endpoint_count(
            router,
            {
                "model": "gpt-agent",
                "messages": _agent_turns_with_tool_results(3),
                "tools": [_READ_FILE_TOOL],
                "context_management": _CLEAR_TOOL_USES,
            },
        )

    assert response == {"input_tokens": 71, "litellm_count_source": "provider"}
    assert len(peer.requests) == 1
    assert peer.requests[0].content.decode().count(_GEMINI_TOOL_OUTPUT) == 1


_DATABRICKS_CLAUDE_MODEL: Final = "databricks/databricks-claude-sonnet-4-6"


def _databricks_claude_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "databricks-claude-agent",
                "litellm_params": {
                    "model": _DATABRICKS_CLAUDE_MODEL,
                    "api_key": "test-key",
                    "api_base": "https://example.cloud.databricks.com/serving-endpoints",
                },
            }
        ]
    )


@pytest.mark.asyncio
async def test_messages_count_tokens_counts_the_edited_body_on_a_native_messages_deployment_without_a_count_api() -> (
    None
):
    router: Final = _databricks_claude_router()
    unedited_body: Final[dict[str, JsonValue]] = {
        "model": "databricks-claude-agent",
        "messages": _agent_turns_with_tool_results(3),
        "tools": [_READ_FILE_TOOL],
    }

    edited: Final = await _input_tokens(router, {**unedited_body, "context_management": _CLEAR_TOOL_USES})
    unedited: Final = await _input_tokens(router, unedited_body)
    polyfilled: Final = await _input_tokens(
        router, {**unedited_body, "messages": await _messages_the_inference_polyfill_sends(_DATABRICKS_CLAUDE_MODEL)}
    )

    assert edited < unedited
    assert edited == polyfilled


@pytest.mark.asyncio
@pytest.mark.parametrize("drop_params", [False, True])
async def test_messages_count_tokens_on_a_native_messages_deployment_without_a_count_api_rejects_an_invalid_edit(
    drop_params: bool,
) -> None:
    with patch.object(litellm, "drop_params", drop_params):
        response: Final = await _messages_endpoint_count(
            _databricks_claude_router(),
            {
                "model": "databricks-claude-agent",
                "messages": _agent_turns_with_tool_results(3),
                "tools": [_READ_FILE_TOOL],
                "context_management": _INVALID_COMPACT_EDIT,
            },
        )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_messages_count_tokens_leaves_the_edits_to_a_native_messages_deployment() -> None:
    peer: Final = _CountTokensPeer({"input_tokens": 63})
    router: Final = Router(
        model_list=[
            {
                "model_name": "claude-agent",
                "litellm_params": {"model": "anthropic/claude-sonnet-5-5", "api_key": "test-key"},
            }
        ]
    )

    with patch.object(
        anthropic_count_tokens_handler,
        "get_async_httpx_client",
        return_value=AsyncHTTPHandler(transport=httpx.MockTransport(peer)),
    ):
        response: Final = await _messages_endpoint_count(
            router,
            {
                "model": "claude-agent",
                "messages": _agent_turns_with_tool_results(3),
                "tools": [_READ_FILE_TOOL],
                "context_management": _CLEAR_TOOL_USES,
            },
        )

    assert response == {"input_tokens": 63, "litellm_count_source": "provider"}
    assert len(peer.requests) == 1
    assert peer.requests[0].content.decode().count(_GEMINI_TOOL_OUTPUT) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text", ["hello there", "The quick brown fox jumps over the lazy dog. " * 50], ids=["short", "long"]
)
async def test_call_endpoint_counts_a_prompt_like_the_same_text_sent_as_a_user_message(
    text: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", _gemini_router())
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(_gemini_word_counting_peer))

    with patch.object(gemini_count_tokens_handler, "get_async_httpx_client", return_value=client):
        from_prompt: Final = await count_request_tokens(
            request=TokenCountRequest(model="gemini-counted", prompt=text), call_endpoint=True
        )
        from_message: Final = await count_request_tokens(
            request=TokenCountRequest(model="gemini-counted", messages=[{"role": "user", "content": text}]),
            call_endpoint=True,
        )

    assert from_prompt.error is False
    assert from_prompt.total_tokens == from_message.total_tokens == len(text.split())


class _PartnerMessage(BaseModel):
    model_config = ConfigDict(frozen=True)
    content: str = ""


class _PartnerCountRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    messages: tuple[_PartnerMessage, ...] = ()


async def _vertex_partner_word_count(
    model: str, request_data: JsonValue, litellm_params: JsonValue
) -> dict[str, JsonValue]:
    body: Final = _PartnerCountRequest.model_validate(request_data)
    return {"input_tokens": sum(len(message.content.split()) for message in body.messages)}


@pytest.mark.asyncio
async def test_call_endpoint_counts_a_prompt_on_a_vertex_ai_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "vertex-counted",
                "litellm_params": {
                    "model": "vertex_ai/claude-sonnet-5-5",
                    "vertex_project": "test-project",
                    "vertex_location": "us-east5",
                },
            }
        ]
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
    text: Final = "The quick brown fox jumps over the lazy dog."

    with patch.object(
        VertexAIPartnerModelsTokenCounter, "handle_count_tokens_request", side_effect=_vertex_partner_word_count
    ):
        from_prompt: Final = await count_request_tokens(
            request=TokenCountRequest(model="vertex-counted", prompt=text), call_endpoint=True
        )

    assert (from_prompt.error, from_prompt.total_tokens) == (False, 9)


_FOX: Final = "The quick brown fox jumps over the lazy dog."


_CONVERSATIONS: Final = (
    {"messages": [{"role": "user", "content": _FOX}]},
    {"contents": [{"role": "user", "parts": [{"text": _FOX}]}]},
)


@pytest.mark.asyncio
@pytest.mark.parametrize("conversation", _CONVERSATIONS, ids=["messages", "contents"])
@pytest.mark.parametrize("call_endpoint", [True, False], ids=["provider", "local"])
async def test_prompt_next_to_a_conversation_answers_400_without_drop_params(
    conversation: dict[str, JsonValue], call_endpoint: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", _gemini_router())
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(_gemini_word_counting_peer))

    with (
        patch.object(gemini_count_tokens_handler, "get_async_httpx_client", return_value=client),
        pytest.raises(ProxyException) as rejected,
    ):
        await count_request_tokens(
            request=TokenCountRequest(model="gemini-counted", prompt="hi", **conversation), call_endpoint=call_endpoint
        )

    assert (rejected.value.code, rejected.value.param) == ("400", "prompt")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("conversation", "call_endpoint", "global_drop_params", "deployment_drop_params"),
    [
        (_CONVERSATIONS[0], True, False, True),
        (_CONVERSATIONS[1], True, False, True),
        (_CONVERSATIONS[0], False, False, True),
        (_CONVERSATIONS[0], False, True, False),
    ],
    ids=[
        "provider_messages",
        "provider_contents",
        "local_messages",
        "local_messages_global_setting",
    ],
)
async def test_drop_params_counts_a_conversation_without_the_prompt(
    conversation: dict[str, JsonValue],
    call_endpoint: bool,
    global_drop_params: bool,
    deployment_drop_params: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", _gemini_router(drop_params=deployment_drop_params))
    monkeypatch.setattr(litellm, "drop_params", global_drop_params)
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(_gemini_word_counting_peer))

    with patch.object(gemini_count_tokens_handler, "get_async_httpx_client", return_value=client):
        with_prompt: Final = await count_request_tokens(
            request=TokenCountRequest(model="gemini-counted", prompt="hi", **conversation), call_endpoint=call_endpoint
        )
        conversation_only: Final = await count_request_tokens(
            request=TokenCountRequest(model="gemini-counted", **conversation), call_endpoint=call_endpoint
        )

    assert with_prompt.error is False
    assert with_prompt.total_tokens == conversation_only.total_tokens > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_conversation", [{"messages": []}, {"contents": []}], ids=["messages", "contents"])
async def test_call_endpoint_prompt_next_to_an_empty_conversation_counts_the_prompt(
    empty_conversation: dict[str, JsonValue], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", _gemini_router())
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(_gemini_word_counting_peer))

    with patch.object(gemini_count_tokens_handler, "get_async_httpx_client", return_value=client):
        counted: Final = await count_request_tokens(
            request=TokenCountRequest(model="gemini-counted", prompt=_FOX, **empty_conversation), call_endpoint=True
        )

    assert (counted.error, counted.total_tokens) == (False, 9)


@pytest.mark.asyncio
async def test_local_count_of_a_prompt_next_to_empty_messages_counts_the_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", _gemini_router())

    with_empty: Final = await count_request_tokens(
        request=TokenCountRequest(model="gemini-counted", prompt=_FOX, messages=[])
    )
    alone: Final = await count_request_tokens(request=TokenCountRequest(model="gemini-counted", prompt=_FOX))

    assert with_empty.error is False
    assert with_empty.total_tokens == alone.total_tokens > 0


_WEATHER_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chat_fields",
    [{"system": "You answer in one short sentence. " * 20}, {"tools": [_WEATHER_TOOL]}],
    ids=["system", "tools"],
)
@pytest.mark.parametrize("empty_conversation", [{}, {"messages": []}], ids=["no_messages", "empty_messages"])
async def test_local_count_of_a_prompt_with_system_or_tools_counts_them_like_the_message_form(
    chat_fields: dict[str, JsonValue], empty_conversation: dict[str, JsonValue], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", _gemini_router())

    from_prompt: Final = await count_request_tokens(
        request=TokenCountRequest(model="gemini-counted", prompt=_FOX, **empty_conversation, **chat_fields)
    )
    from_message: Final = await count_request_tokens(
        request=TokenCountRequest(model="gemini-counted", messages=[{"role": "user", "content": _FOX}], **chat_fields)
    )
    prompt_alone: Final = await count_request_tokens(request=TokenCountRequest(model="gemini-counted", prompt=_FOX))

    assert from_prompt.error is False
    assert from_prompt.total_tokens == from_message.total_tokens > prompt_alone.total_tokens


_PROVIDER_COUNTED_DEPLOYMENTS: Final = (
    pytest.param(
        {"model": "anthropic/claude-sonnet-5-5", "api_key": "test-key", "api_base": "https://count-tokens.test"},
        id="anthropic",
    ),
    pytest.param(
        {"model": "azure_ai/claude-sonnet-5-5", "api_key": "test-key", "api_base": "https://count-tokens.test"},
        id="azure_ai",
    ),
    pytest.param(
        {"model": "openai/gpt-5.6", "api_key": "test-key", "api_base": "https://count-tokens.test/v1"}, id="openai"
    ),
    pytest.param(
        {
            "model": "bedrock/us.anthropic.claude-sonnet-5-5",
            "aws_access_key_id": "AKIATESTACCESSKEY",
            "aws_secret_access_key": "test-secret",
            "aws_region_name": "us-west-2",
        },
        id="bedrock",
    ),
)


@contextmanager
def _count_apis_answering(peer: Callable[[httpx.Request], httpx.Response]) -> Iterator[None]:
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(peer))
    with (
        patch.object(anthropic_count_tokens_handler, "get_async_httpx_client", return_value=client),
        patch.object(azure_ai_count_tokens_http, "get_async_httpx_client", return_value=client),
        patch.object(bedrock_count_tokens_http, "get_async_httpx_client", return_value=client),
        patch.object(bedrock_mantle_count_tokens_http, "get_async_httpx_client", return_value=client),
        patch.object(openai_count_tokens_http, "get_async_httpx_client", return_value=client),
    ):
        yield


def _route_to_one_deployment(litellm_params: dict[str, JsonValue], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(model_list=[{"model_name": "provider-counted", "litellm_params": litellm_params}]),
    )


_FOX_AS_PROMPT: Final = TokenCountRequest(model="provider-counted", prompt=_FOX)
_FOX_AS_MESSAGE: Final = TokenCountRequest(model="provider-counted", messages=[{"role": "user", "content": _FOX}])


@pytest.mark.asyncio
@pytest.mark.parametrize("litellm_params", _PROVIDER_COUNTED_DEPLOYMENTS)
async def test_call_endpoint_sends_a_prompt_to_the_provider_as_one_user_message(
    litellm_params: dict[str, JsonValue], monkeypatch: pytest.MonkeyPatch
) -> None:
    _route_to_one_deployment(litellm_params, monkeypatch)
    sent_bodies: Final[list[JsonValue]] = []  # mutable-ok: collects what the mock peer receives

    def count_api(request: httpx.Request) -> httpx.Response:
        sent_bodies.append(json.loads(request.content))
        return httpx.Response(
            200, json={"object": "response.input_tokens", "input_tokens": 41, "inputTokens": 41}, request=request
        )

    with _count_apis_answering(count_api):
        from_prompt: Final = await count_request_tokens(request=_FOX_AS_PROMPT, call_endpoint=True)
        from_message: Final = await count_request_tokens(request=_FOX_AS_MESSAGE, call_endpoint=True)

    assert (from_prompt.total_tokens, from_prompt.error) == (41, False)
    assert from_prompt.count_source != "local_estimate"
    assert from_prompt == from_message
    assert len(sent_bodies) == 2 and sent_bodies[0] == sent_bodies[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 503])
@pytest.mark.parametrize("litellm_params", _PROVIDER_COUNTED_DEPLOYMENTS)
async def test_call_endpoint_answers_a_provider_error_on_a_prompt_like_on_the_same_text_sent_as_a_user_message(
    litellm_params: dict[str, JsonValue], status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _route_to_one_deployment(litellm_params, monkeypatch)

    def count_api(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": f"scripted {status}"}}, request=request)

    with _count_apis_answering(count_api):
        with pytest.raises(ProxyException) as from_prompt:
            await count_request_tokens(request=_FOX_AS_PROMPT, call_endpoint=True)
        with pytest.raises(ProxyException) as from_message:
            await count_request_tokens(request=_FOX_AS_MESSAGE, call_endpoint=True)

    assert from_prompt.value.code == str(status)
    assert f"scripted {status}" in from_prompt.value.message
    assert (from_prompt.value.code, from_prompt.value.type, from_prompt.value.message) == (
        from_message.value.code,
        from_message.value.type,
        from_message.value.message,
    )


@pytest.mark.asyncio
async def test_framed_stand_in_counts_a_prompt_like_the_same_text_sent_as_a_user_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _framed_kimi(monkeypatch)

    from_prompt: Final = await count_request_tokens(request=TokenCountRequest(model="kimi", prompt=_FOX), call_endpoint=True)
    from_message: Final = await count_request_tokens(
        request=TokenCountRequest(model="kimi", messages=[{"role": "user", "content": _FOX}]), call_endpoint=True
    )
    local_prompt: Final = await count_request_tokens(request=TokenCountRequest(model="kimi", prompt=_FOX))

    assert (from_prompt.count_source, from_prompt.error) == ("local_estimate", False)
    assert from_prompt.total_tokens == from_message.total_tokens > local_prompt.total_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    ["fireworks_ai/accounts/fireworks/models/glm-5", "fireworks_ai/accounts/fireworks/models/deepseek-v4p1-flash"],
    ids=["plain", "framed"],
)
async def test_call_endpoint_counts_a_prompt_on_a_provider_without_a_count_api_as_raw_text(
    model: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _route_to_one_deployment({"model": model, "api_key": "test-key"}, monkeypatch)
    sent: Final[list[httpx.Request]] = []  # mutable-ok: collects what the mock peer receives

    def count_api(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"input_tokens": 41}, request=request)

    with _count_apis_answering(count_api):
        from_prompt: Final = await count_request_tokens(request=_FOX_AS_PROMPT, call_endpoint=True)
        from_message: Final = await count_request_tokens(request=_FOX_AS_MESSAGE, call_endpoint=True)
    local_prompt: Final = await count_request_tokens(request=_FOX_AS_PROMPT, call_endpoint=False)

    assert sent == []
    assert (from_prompt.count_source, from_prompt.error) == ("local_estimate", False)
    assert from_prompt.total_tokens == local_prompt.total_tokens < from_message.total_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deployment",
    [
        pytest.param(
            {"litellm_params": {"model": "openai/text-embedding-3-small", "api_key": "test-key"}}, id="cost map mode"
        ),
        pytest.param(
            {
                "litellm_params": {"model": "openai/in-house-embedder", "api_key": "test-key"},
                "model_info": {"mode": "embedding"},
            },
            id="configured mode",
        ),
    ],
)
async def test_call_endpoint_counts_a_prompt_on_a_non_chat_deployment_as_raw_text_without_the_provider(
    deployment: dict[str, JsonValue], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        litellm.proxy.proxy_server, "llm_router", Router(model_list=[{"model_name": "provider-counted", **deployment}])
    )
    sent: Final[list[httpx.Request]] = []  # mutable-ok: collects what the mock peer receives

    def count_api(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"input_tokens": 41}, request=request)

    with _count_apis_answering(count_api):
        from_endpoint: Final = await count_request_tokens(request=_FOX_AS_PROMPT, call_endpoint=True)
    local: Final = await count_request_tokens(request=_FOX_AS_PROMPT, call_endpoint=False)

    assert sent == []
    assert (from_endpoint.total_tokens, from_endpoint.error) == (local.total_tokens, False)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["chat", "responses"])
async def test_call_endpoint_sends_a_prompt_on_a_chat_mode_deployment_to_the_provider(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "provider-counted",
                    "litellm_params": {
                        "model": "openai/gpt-5.6",
                        "api_key": "test-key",
                        "api_base": "https://count-tokens.test/v1",
                    },
                    "model_info": {"mode": mode},
                }
            ]
        ),
    )
    sent: Final[list[httpx.Request]] = []  # mutable-ok: collects what the mock peer receives

    def count_api(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 41}, request=request)

    with _count_apis_answering(count_api):
        from_prompt: Final = await count_request_tokens(request=_FOX_AS_PROMPT, call_endpoint=True)

    assert (from_prompt.total_tokens, from_prompt.error) == (41, False)
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_call_endpoint_counts_a_prompt_for_a_model_without_a_deployment_as_raw_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(model_list=[{"model_name": "other", "litellm_params": {"model": "openai/gpt-5.6"}}]),
    )
    request: Final = TokenCountRequest(model="deepseek/deepseek-v4-flash", prompt=_FOX)

    from_endpoint: Final = await count_request_tokens(request=request, call_endpoint=True)
    local: Final = await count_request_tokens(request=request, call_endpoint=False)

    assert (from_endpoint.total_tokens, from_endpoint.error) == (local.total_tokens, False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "path"),
    [("anthropic/claude-sonnet-5-5", ""), ("azure_ai/claude-sonnet-5-5", ""), ("openai/gpt-5.6", "/v1")],
    ids=["anthropic", "azure_ai", "openai"],
)
@pytest.mark.parametrize(
    ("row_timeout", "router_timeout"),
    [({"timeout": 0.5}, None), ({"request_timeout": 0.5}, 30.0), ({}, 0.5)],
    ids=["deployment timeout", "deployment request_timeout", "router_settings timeout"],
)
async def test_call_endpoint_answers_a_timeout_when_a_stalled_count_api_outlasts_the_chat_timeout(
    model: str,
    path: str,
    row_timeout: dict[str, JsonValue],
    router_timeout: float | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with socket.create_server(("127.0.0.1", 0)) as never_answers:
        port: Final = never_answers.getsockname()[1]
        litellm_params: Final[dict[str, JsonValue]] = {
            "model": model,
            "api_key": "test-key",
            "api_base": f"http://127.0.0.1:{port}{path}",
        }
        deployment: Final[dict[str, JsonValue]] = {
            "model_name": "provider-counted",
            "litellm_params": {**litellm_params, **row_timeout},
        }
        monkeypatch.setattr(
            litellm.proxy.proxy_server, "llm_router", Router(model_list=[deployment], timeout=router_timeout)
        )

        with pytest.raises(ProxyException) as error:
            async with asyncio.timeout(10):
                await count_request_tokens(request=_FOX_AS_PROMPT, call_endpoint=True)

    assert error.value.code == "408"


@pytest.mark.asyncio
async def test_responses_count_answers_a_timeout_when_a_stalled_count_api_outlasts_the_deployment_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with socket.create_server(("127.0.0.1", 0)) as never_answers:
        port: Final = never_answers.getsockname()[1]
        _route_to_one_deployment(
            {
                "model": "openai/gpt-5.6",
                "api_key": "test-key",
                "api_base": f"http://127.0.0.1:{port}/v1",
                "timeout": 0.5,
            },
            monkeypatch,
        )

        with pytest.raises(ProxyException) as error:
            async with asyncio.timeout(10):
                await count_request_tokens(
                    request=_FOX_AS_MESSAGE,
                    call_endpoint=True,
                    responses_request={"model": "provider-counted", "input": _FOX},
                )

    assert error.value.code == "408"


@pytest.mark.asyncio
async def test_call_endpoint_names_the_status_of_a_count_api_error_with_an_empty_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _route_to_one_deployment(
        {"model": "openai/gpt-5.6", "api_key": "test-key", "api_base": "https://count-tokens.test/v1"}, monkeypatch
    )

    with _count_apis_answering(lambda request: httpx.Response(404, request=request)):
        with pytest.raises(ProxyException) as error:
            await count_request_tokens(request=_FOX_AS_PROMPT, call_endpoint=True)

    assert error.value.code == "404"
    assert "404" in error.value.message


_PINNED_COUNTS: Final = MappingProxyType({"deployment-a.test": 11, "deployment-b.test": 22})


def _pinned_count_peer(request: httpx.Request) -> httpx.Response:
    if json.loads(request.content).get("previous_response_id") != "resp_original":
        return httpx.Response(400, json={"error": {"message": "previous_response_id was not decoded"}}, request=request)
    return httpx.Response(200, json={"input_tokens": _PINNED_COUNTS[request.url.host]}, request=request)


@pytest.mark.asyncio
@pytest.mark.parametrize("pinned", ["deployment-a", "deployment-b"])
async def test_responses_count_goes_to_the_deployment_that_served_the_previous_response(
    monkeypatch: pytest.MonkeyPatch, pinned: str
) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "gpt-group",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "test-key",
                    "api_base": f"https://{identity}.test/v1",
                },
                "model_info": {"id": identity},
            }
            for identity in ("deployment-a", "deployment-b")
        ],
        optional_pre_call_checks=["responses_api_deployment_check"],
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(_pinned_count_peer))
    previous_response_id: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        "openai", pinned, "resp_original"
    )

    with patch.object(openai_count_tokens_http, "get_async_httpx_client", return_value=client):
        counts: Final = [
            (
                await count_request_tokens(
                    request=TokenCountRequest(model="gpt-group", messages=[{"role": "user", "content": "again"}]),
                    call_endpoint=True,
                    responses_request={
                        "model": "gpt-group",
                        "input": "again",
                        "previous_response_id": previous_response_id,
                    },
                )
            ).total_tokens
            for _ in range(8)
        ]

    assert counts == [_PINNED_COUNTS[f"{pinned}.test"]] * 8


def _managed_file_id(provider_file_id: str, model_id: str) -> str:
    unified: Final = SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
        "application/pdf", "3f0c9a", "gpt-files", provider_file_id, model_id
    )
    return base64.urlsafe_b64encode(unified.encode()).decode().rstrip("=")


@pytest.mark.asyncio
@pytest.mark.parametrize(("with_mapping", "expected_file_id"), [(False, "file-uploaded"), (True, "file-on-a")])
async def test_responses_count_sends_openai_the_provider_file_id_inference_sends(
    monkeypatch: pytest.MonkeyPatch, with_mapping: bool, expected_file_id: str
) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "gpt-files",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "test-key",
                    "api_base": "https://count.test/v1",
                },
                "model_info": {"id": "deployment-a"},
            }
        ]
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
    managed_id: Final = _managed_file_id("file-uploaded", "deployment-a")
    bodies: Final[list[dict[str, JsonValue]]] = []  # mutable-ok: collects what the mock peer receives

    def count_peer(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 49}, request=request)

    with patch.object(
        openai_count_tokens_http,
        "get_async_httpx_client",
        return_value=AsyncHTTPHandler(transport=httpx.MockTransport(count_peer)),
    ):
        result: Final = await count_request_tokens(
            request=TokenCountRequest(model="gpt-files", messages=[{"role": "user", "content": "Summarize"}]),
            call_endpoint=True,
            responses_request={
                "model": "gpt-files",
                "input": [{"role": "user", "content": [{"type": "input_file", "file_id": managed_id}]}],
                **({"model_file_id_mapping": {managed_id: {"deployment-a": "file-on-a"}}} if with_mapping else {}),
            },
        )

    assert (result.error, result.total_tokens) == (False, 49)
    (body,) = bodies
    assert body["input"] == [{"role": "user", "content": [{"type": "input_file", "file_id": expected_file_id}]}]
    assert "model_file_id_mapping" not in body


_ANSWER_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


@pytest.mark.asyncio
async def test_responses_count_on_claude_counts_the_chat_request_responses_inference_sends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "claude-counted",
                "litellm_params": {
                    "model": "anthropic/claude-opus-5-5",
                    "api_key": "test-key",
                    "api_base": "https://count-tokens.test",
                },
            }
        ]
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
    bodies: Final[list[dict[str, JsonValue]]] = []  # mutable-ok: collects what the mock peer receives

    def count_peer(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"input_tokens": 77}, request=request)

    responses_tools: Final[list[dict[str, JsonValue]]] = [
        {"type": "function", "name": "get_weather", "parameters": {"type": "object", "properties": {}}},
        {"type": "web_search"},
        {"type": "file_search", "vector_store_ids": ["vs_docs"]},
    ]
    with patch.object(
        anthropic_count_tokens_handler,
        "get_async_httpx_client",
        return_value=AsyncHTTPHandler(transport=httpx.MockTransport(count_peer)),
    ):
        result: Final = await count_request_tokens(
            request=TokenCountRequest(
                model="claude-counted", messages=[{"role": "user", "content": "Weather?"}], tools=responses_tools
            ),
            call_endpoint=True,
            responses_request={
                "model": "claude-counted",
                "input": "Weather?",
                "tools": responses_tools,
                "tool_choice": "auto",
                "text": {"format": {"type": "json_schema", "name": "answer", "schema": _ANSWER_SCHEMA}},
                "reasoning": {"effort": "low"},
            },
        )

    assert (result.error, result.total_tokens) == (False, 77)
    (body,) = bodies
    tools: Final = body["tools"]
    assert isinstance(tools, list)
    tool_names: Final = [tool["name"] for tool in tools if isinstance(tool, dict)]
    assert sorted(tool_names) == ["get_weather", "litellm_file_search", "web_search"]
    assert any(isinstance(tool, dict) and str(tool.get("type")).startswith("web_search") for tool in tools)
    assert body["tool_choice"] == {"type": "auto"}
    output_config: Final = body["output_config"]
    assert isinstance(output_config, dict)
    assert output_config["format"] == {"type": "json_schema", "schema": _ANSWER_SCHEMA}
    assert "thinking" in body


_WEATHER_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {}},
}


async def _claude_count(
    monkeypatch: pytest.MonkeyPatch,
    deployment_params: Mapping[str, JsonValue],
    responses_request: Mapping[str, object],
) -> tuple[TokenCountResponse, list[dict[str, JsonValue]]]:
    router: Final = Router(
        model_list=[
            {
                "model_name": "claude-counted",
                "litellm_params": {
                    "model": "anthropic/claude-zeta-9",
                    "api_key": "test-key",
                    "api_base": "https://count-tokens.test",
                    **deployment_params,
                },
            }
        ]
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
    bodies: Final[list[dict[str, JsonValue]]] = []  # mutable-ok: collects what the mock peer receives

    def count_peer(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"input_tokens": 77}, request=request)

    with patch.object(
        anthropic_count_tokens_handler,
        "get_async_httpx_client",
        return_value=AsyncHTTPHandler(transport=httpx.MockTransport(count_peer)),
    ):
        result: Final = await count_request_tokens(
            request=TokenCountRequest(model="claude-counted", messages=[{"role": "user", "content": "Weather?"}]),
            call_endpoint=True,
            responses_request={"model": "claude-counted", **responses_request},
        )
    return result, bodies


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("deployment_params", "request_params"),
    [({"drop_params": True}, {}), ({}, {"drop_params": True})],
    ids=["deployment", "request"],
)
async def test_responses_count_on_claude_downgrades_forced_tool_choice_under_drop_params(
    monkeypatch: pytest.MonkeyPatch,
    deployment_params: Mapping[str, JsonValue],
    request_params: Mapping[str, JsonValue],
) -> None:
    monkeypatch.setitem(litellm.model_cost, "claude-zeta-9", {"supports_forced_tool_use": False})

    result, bodies = await _claude_count(
        monkeypatch,
        deployment_params,
        {"input": "Weather?", "tools": [_WEATHER_TOOL], "tool_choice": "required", **request_params},
    )

    assert (result.error, result.total_tokens) == (False, 77)
    (body,) = bodies
    assert body["tool_choice"] == {"type": "auto"}


@pytest.mark.asyncio
async def test_responses_count_on_claude_rejects_forced_tool_choice_without_drop_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(litellm.model_cost, "claude-zeta-9", {"supports_forced_tool_use": False})

    with pytest.raises(ProxyException) as rejected:
        await _claude_count(monkeypatch, {}, {"input": "Weather?", "tools": [_WEATHER_TOOL], "tool_choice": "required"})

    assert (rejected.value.code, "forced tool use" in rejected.value.message) == ("400", True)


def _texts(message: JsonValue) -> list[str]:
    assert isinstance(message, dict)
    content: Final = message["content"]
    if isinstance(content, str):
        return [content]
    assert isinstance(content, list)
    return [str(block["text"]) for block in content if isinstance(block, dict) and "text" in block]


@pytest.mark.asyncio
async def test_responses_count_on_claude_counts_the_turns_previous_response_id_replays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored_turn: Final = {
        "request_id": "chatcmpl-stored",
        "call_type": "aresponses",
        "session_id": "session-1",
        "proxy_server_request": {"input": "Who won in 1998?", "model": "claude-counted"},
        "response": {
            "id": "chatcmpl-stored",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "The Bulls.",
                        "reasoning_content": "Chicago beat Utah in six games.",
                    },
                    "finish_reason": "stop",
                }
            ],
        },
        "status": "success",
    }

    with patch.object(
        ResponsesSessionHandler,
        "get_all_spend_logs_for_previous_response_id",
        new_callable=AsyncMock,
        return_value=[stored_turn],
    ) as spend_logs:
        result, bodies = await _claude_count(
            monkeypatch,
            {},
            {
                "input": "And in 1999?",
                "previous_response_id": "chatcmpl-stored",
                ADDRESSED_RESPONSE_ID_FIELD: "resp_client_form",
            },
        )

    assert (result.error, result.total_tokens) == (False, 77)
    (body,) = bodies
    messages: Final = body["messages"]
    assert isinstance(messages, list)
    assert list(itertools.chain.from_iterable(map(_texts, messages))) == [
        "Who won in 1998?",
        "The Bulls.",
        "And in 1999?",
    ]
    assert [sorted(message) for message in messages if isinstance(message, dict)] == [["content", "role"]] * 3
    spend_logs.assert_awaited_once_with("chatcmpl-stored", ("chatcmpl-stored", "resp_client_form"))


@pytest.mark.asyncio
async def test_responses_count_on_claude_rejects_a_previous_response_id_with_no_stored_turns_on_that_param(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm.proxy.proxy_server, "prisma_client", SimpleNamespace(db=_SpendLogsTable(())))
    with pytest.raises(ProxyException) as rejected:
        await _claude_count(
            monkeypatch,
            {},
            {
                "input": "And in 1999?",
                "previous_response_id": "chatcmpl-lost",
                ADDRESSED_RESPONSE_ID_FIELD: "resp_client_form",
            },
        )

    assert (rejected.value.code, rejected.value.type, rejected.value.param) == (
        "400",
        "invalid_request_error",
        "previous_response_id",
    )
    assert "'resp_client_form' not found" in rejected.value.message


_MULTILINGUAL_TEXT: Final = "日本語のテキストとहिन्दी पाठ और русский текст. " * 20


@pytest.mark.asyncio
async def test_token_counter_counts_a_custom_azure_deployment_name_with_its_base_models_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
):
    azure_params: Final = {"api_key": "sk-test", "api_base": "https://example.openai.azure.com"}
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "named",
                    "litellm_params": {"model": "azure/my-gpt-6", "base_model": "azure/gpt-6.1-sol", **azure_params},
                },
                {"model_name": "served", "litellm_params": {"model": "azure/gpt-6.1-sol", **azure_params}},
            ]
        ),
    )
    messages: Final = [{"role": "user", "content": _MULTILINGUAL_TEXT}]

    named: Final = await count_request_tokens(request=TokenCountRequest(model="named", messages=messages))
    served: Final = await count_request_tokens(request=TokenCountRequest(model="served", messages=messages))

    assert named.total_tokens == served.total_tokens


_LOOKUP_TOOL: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order by its id",
            "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
        },
    }
)


_FLASH_BASE_MODEL: Final = "azure_ai/deepseek-v4-flash"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("served_model", "named_params", "named_model_info", "tools"),
    [
        pytest.param(
            "azure_ai/DeepSeek-V4-Flash-0731",
            {"model": "azure_ai/my-flash"},
            {"base_model": _FLASH_BASE_MODEL},
            [dict(_LOOKUP_TOOL)],
            id="model_info base_model, one tool",
        ),
        pytest.param(
            "azure_ai/DeepSeek-V4-Flash-0731",
            {"model": "azure_ai/my-flash"},
            {"base_model": _FLASH_BASE_MODEL},
            None,
            id="model_info base_model, text only",
        ),
        pytest.param(
            "azure_ai/DeepSeek-V4-Flash-0731",
            {"model": "azure_ai/my-flash", "base_model": _FLASH_BASE_MODEL},
            {},
            [dict(_LOOKUP_TOOL)],
            id="litellm_params base_model",
        ),
        pytest.param(
            "anthropic/claude-opus-5-5",
            {"model": "anthropic/my-opus"},
            {"base_model": "anthropic/claude-opus-5-5"},
            None,
            id="provider-prefixed claude base_model",
        ),
    ],
)
async def test_token_counter_counts_a_custom_named_deployment_like_the_base_model_it_declares(
    monkeypatch: pytest.MonkeyPatch,
    served_model: str,
    named_params: Mapping[str, str],
    named_model_info: Mapping[str, str],
    tools: list[dict[str, JsonValue]] | None,
) -> None:
    credentials: Final = {"api_key": "sk-test", "api_base": "https://example.services.ai.azure.com"}
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.llm_router",
        Router(
            model_list=[
                {
                    "model_name": "named",
                    "litellm_params": {**named_params, **credentials},
                    "model_info": dict(named_model_info),
                },
                {"model_name": "served", "litellm_params": {"model": served_model, **credentials}},
            ]
        ),
    )
    messages: Final = [{"role": "user", "content": _MULTILINGUAL_TEXT}]

    named: Final = await _token_counter_route(TokenCountRequest(model="named", messages=messages, tools=tools))
    served: Final = await _token_counter_route(TokenCountRequest(model="served", messages=messages, tools=tools))

    assert named.total_tokens == served.total_tokens


@pytest.fixture
def no_huggingface_tokenizer_loaded() -> Iterator[None]:
    _HUGGINGFACE_TOKENIZER_LOAD_OUTCOMES.flush_cache()
    yield
    _HUGGINGFACE_TOKENIZER_LOAD_OUTCOMES.flush_cache()


_LLAMA_3_HUB_TOKENIZER: Final = "Xenova/llama-3-tokenizer"


async def _token_counter_route(request: TokenCountRequest, call_endpoint: bool = False) -> TokenCountResponse:
    return await token_counter(
        request=request,
        http_request=_json_request({}, "python-httpx/0.28.1"),
        fastapi_response=Response(),
        user_api_key_dict=UserAPIKeyAuth(),
        call_endpoint=call_endpoint,
    )


def _stub_hub_downloads(monkeypatch: pytest.MonkeyPatch, download: Callable[[str], None]) -> None:
    downloaded: Final = litellm.utils._load_huggingface_tokenizer("deepseek")

    def from_pretrained(identifier: str, revision: str = "main", token: str | None = None) -> HuggingFace:
        download(identifier)
        return downloaded

    monkeypatch.setattr(litellm.utils.tokenizer_dispatch, "from_pretrained", from_pretrained)


def _hanging_llama_download(
    monkeypatch: pytest.MonkeyPatch, download_started: threading.Event, hub_released: threading.Event
) -> None:
    def llama_download_hangs(identifier: str) -> None:
        if identifier == _LLAMA_3_HUB_TOKENIZER:
            download_started.set()
            hub_released.wait(10)

    _stub_hub_downloads(monkeypatch, llama_download_hangs)


@pytest.mark.usefixtures("no_huggingface_tokenizer_loaded")
def test_custom_named_llama_deployment_counts_with_tiktoken_while_its_tokenizer_download_hangs(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.llm_router",
        Router(
            model_list=[
                {
                    "model_name": "my-llama",
                    "litellm_params": {"model": "hosted_vllm/my-llama", "api_base": "http://127.0.0.1:9/v1"},
                    "model_info": {"base_model": "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct"},
                }
            ]
        ),
    )
    download_started: Final = threading.Event()
    hub_released: Final = threading.Event()
    downloads: Final[list[str]] = []

    def hanging_download(identifier: str) -> None:
        downloads.append(identifier)
        download_started.set()
        hub_released.wait(10)

    _stub_hub_downloads(monkeypatch, hanging_download)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_WAIT_SECONDS", 30)

    async def tokenizer_type() -> str:
        response: Final = await _token_counter_route(
            TokenCountRequest(model="my-llama", messages=[{"role": "user", "content": "What is new?"}])
        )
        return response.tokenizer_type or ""

    async def burst_past_the_wait(loop: _SkippableClockLoop) -> tuple[str, ...]:
        try:
            burst: Final = asyncio.gather(*(tokenizer_type() for _ in range(20)))
            await asyncio.to_thread(download_started.wait, 10)
            loop.skip(31)
            return tuple(await asyncio.wait_for(burst, 8))
        finally:
            hub_released.set()

    assert set(_run_on_skippable_clock(burst_past_the_wait)) == {"openai_tokenizer"}
    assert downloads == [_LLAMA_3_HUB_TOKENIZER]


@pytest.mark.usefixtures("no_huggingface_tokenizer_loaded")
def test_custom_named_deepseek_deployment_keeps_its_bundled_tokenizer_while_a_hub_download_fills_the_load_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.llm_router",
        Router(
            model_list=[
                {
                    "model_name": "my-llama",
                    "litellm_params": {"model": "hosted_vllm/my-llama", "api_base": "http://127.0.0.1:9/v1"},
                    "model_info": {"base_model": "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct"},
                },
                {
                    "model_name": "my-deepseek",
                    "litellm_params": {"model": "hosted_vllm/my-deepseek", "api_base": "http://127.0.0.1:9/v1"},
                    "model_info": {"base_model": "hosted_vllm/deepseek-ai/DeepSeek-V4-Flash"},
                },
            ]
        ),
    )
    download_started: Final = threading.Event()
    hub_released: Final = threading.Event()
    _hanging_llama_download(monkeypatch, download_started, hub_released)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_MAX_RUNNING", 1)

    async def tokenizer_type(model: str) -> str:
        response: Final = await _token_counter_route(
            TokenCountRequest(model=model, messages=[{"role": "user", "content": "What is new?"}])
        )
        return response.tokenizer_type or ""

    async def deepseek_while_the_llama_download_hangs() -> str:
        llama: Final = asyncio.ensure_future(tokenizer_type("my-llama"))
        try:
            await asyncio.to_thread(download_started.wait, 10)
            return await asyncio.wait_for(tokenizer_type("my-deepseek"), 5)
        finally:
            hub_released.set()
            await llama

    assert asyncio.run(deepseek_while_the_llama_download_hangs()) == "huggingface_tokenizer"


@pytest.mark.usefixtures("no_huggingface_tokenizer_loaded")
@pytest.mark.parametrize(
    "model_info",
    [{}, {"base_model": "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct"}],
    ids=["no base_model", "base_model naming its own model"],
)
def test_named_llama_deployment_keeps_waiting_for_its_slow_tokenizer_download(
    monkeypatch: pytest.MonkeyPatch, model_info: dict[str, str]
) -> None:
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.llm_router",
        Router(
            model_list=[
                {
                    "model_name": "llama-3.3-70b",
                    "litellm_params": {
                        "model": "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
                        "api_base": "http://127.0.0.1:9/v1",
                    },
                    "model_info": model_info,
                }
            ]
        ),
    )
    download_started: Final = threading.Event()
    hub_released: Final = threading.Event()
    _hanging_llama_download(monkeypatch, download_started, hub_released)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_WAIT_SECONDS", 30)

    async def tokenizer_type() -> str:
        response: Final = await _token_counter_route(
            TokenCountRequest(model="llama-3.3-70b", messages=[{"role": "user", "content": "What is new?"}])
        )
        return response.tokenizer_type or ""

    async def count_past_the_wait(loop: _SkippableClockLoop) -> tuple[str, ...]:
        try:
            count: Final = asyncio.ensure_future(tokenizer_type())
            await asyncio.to_thread(download_started.wait, 10)
            loop.skip(31)
            await asyncio.wait({count}, timeout=1)
        finally:
            hub_released.set()
        return (await asyncio.wait_for(count, 8),)

    assert _run_on_skippable_clock(count_past_the_wait) == ("huggingface_tokenizer",)


def _custom_named_llama_router(model_info: Mapping[str, object]) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "my-llama",
                "litellm_params": {"model": "hosted_vllm/my-llama", "api_base": "http://127.0.0.1:9/v1"},
                "model_info": {"base_model": "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct", **model_info},
            }
        ]
    )


async def _my_llama_tokenizer_type() -> str:
    response: Final = await _token_counter_route(
        TokenCountRequest(model="my-llama", messages=[{"role": "user", "content": "What is new?"}])
    )
    return response.tokenizer_type or ""


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_huggingface_tokenizer_loaded")
async def test_custom_named_llama_deployment_counts_with_its_base_models_tokenizer_once_downloaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", _custom_named_llama_router({}))
    _stub_hub_downloads(monkeypatch, lambda identifier: None)

    assert await _my_llama_tokenizer_type() == "huggingface_tokenizer"


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_huggingface_tokenizer_loaded")
@pytest.mark.parametrize(
    ("model", "base_model"),
    [
        pytest.param(
            "anthropic/claude-sonnet-5-5", "bedrock/us.anthropic.claude-sonnet-5-5", id="bundled claude tokenizer"
        ),
        pytest.param(
            "databricks/databricks-meta-llama-3-3-70b-instruct",
            "bedrock/meta.llama3-3-70b-instruct-v1:0",
            id="hub llama tokenizer",
        ),
    ],
)
async def test_base_model_without_a_tokenizer_leaves_the_deployment_on_its_own_models_tokenizer(
    monkeypatch: pytest.MonkeyPatch, model: str, base_model: str
) -> None:
    credentials: Final = {"api_key": "sk-test", "api_base": "http://127.0.0.1:9/v1"}
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.llm_router",
        Router(
            model_list=[
                {
                    "model_name": "based",
                    "litellm_params": {"model": model, **credentials},
                    "model_info": {"base_model": base_model},
                },
                {"model_name": "own", "litellm_params": {"model": model, **credentials}},
            ]
        ),
    )
    _stub_hub_downloads(monkeypatch, lambda identifier: None)
    messages: Final = [{"role": "user", "content": _MULTILINGUAL_TEXT}]

    based: Final = await _token_counter_route(TokenCountRequest(model="based", messages=messages))
    own: Final = await _token_counter_route(TokenCountRequest(model="own", messages=messages))

    assert (based.tokenizer_type, based.total_tokens) == ("huggingface_tokenizer", own.total_tokens)


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_huggingface_tokenizer_loaded")
async def test_deployment_with_a_custom_tokenizer_never_waits_for_its_base_models_tokenizer_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        _custom_named_llama_router(
            {"custom_tokenizer": {"identifier": "lit-tests/my-llama", "revision": "main", "auth_token": None}}
        ),
    )
    hub_released: Final = threading.Event()
    downloads: Final[list[str]] = []

    def llama_download_hangs(identifier: str) -> None:
        downloads.append(identifier)
        if identifier == _LLAMA_3_HUB_TOKENIZER:
            hub_released.wait(10)

    _stub_hub_downloads(monkeypatch, llama_download_hangs)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_WAIT_SECONDS", 30)
    try:
        tokenizer_type: Final = await asyncio.wait_for(_my_llama_tokenizer_type(), 5)
    finally:
        hub_released.set()

    assert tokenizer_type == "huggingface_tokenizer"
    assert downloads == ["lit-tests/my-llama"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_huggingface_tokenizer_loaded")
async def test_cooling_down_deployment_with_a_custom_tokenizer_never_downloads_its_base_models_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router: Final = _custom_named_llama_router(
        {
            "id": "my-llama-deployment",
            "custom_tokenizer": {"identifier": "lit-tests/my-llama", "revision": "main", "auth_token": None},
        }
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
    downloads: Final[list[str]] = []
    _stub_hub_downloads(monkeypatch, downloads.append)
    router.health().cooldowns.add_deployment_to_cooldown(
        model_id="my-llama-deployment",
        original_exception=Exception("rate limited"),
        exception_status=429,
        cooldown_time=60,
    )

    assert await _my_llama_tokenizer_type() == "openai_tokenizer"
    assert downloads == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_huggingface_tokenizer_loaded")
async def test_custom_named_deployments_whose_base_models_share_a_tokenizer_download_it_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": name,
                    "litellm_params": {"model": f"hosted_vllm/{name}", "api_base": "http://127.0.0.1:9/v1"},
                    "model_info": {"base_model": f"hosted_vllm/meta-llama/{base_model}"},
                }
                for name, base_model in (
                    ("my-llama-70b", "Llama-3.3-70B-Instruct"),
                    ("my-llama-8b", "Llama-3.1-8B-Instruct"),
                )
            ]
        ),
    )
    download_started: Final = threading.Event()
    second_download: Final = threading.Event()
    hub_released: Final = threading.Event()
    downloads: Final[list[str]] = []

    def hanging_download(identifier: str) -> None:
        downloads.append(identifier)
        (second_download if download_started.is_set() else download_started).set()
        hub_released.wait(10)

    _stub_hub_downloads(monkeypatch, hanging_download)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_WAIT_SECONDS", 30)

    async def tokenizer_type(model: str) -> str:
        response: Final = await _token_counter_route(
            TokenCountRequest(model=model, messages=[{"role": "user", "content": "What is new?"}])
        )
        return response.tokenizer_type or ""

    first: Final = asyncio.ensure_future(tokenizer_type("my-llama-70b"))
    try:
        await asyncio.to_thread(download_started.wait, 10)
        second: Final = asyncio.ensure_future(tokenizer_type("my-llama-8b"))
        await asyncio.to_thread(second_download.wait, 0.5)
    finally:
        hub_released.set()

    assert await asyncio.wait_for(asyncio.gather(first, second), 8) == ["huggingface_tokenizer"] * 2
    assert downloads == [_LLAMA_3_HUB_TOKENIZER]


@pytest.mark.asyncio
async def test_deployment_whose_own_model_has_a_tokenizer_keeps_it_when_base_model_names_another(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials: Final = {"api_key": "sk-test", "api_base": "http://127.0.0.1:9/v1"}
    own_model: Final = "hosted_vllm/deepseek-ai/DeepSeek-V4-Flash"
    base_model: Final = "anthropic/claude-opus-5-5"
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "based",
                    "litellm_params": {"model": own_model, **credentials},
                    "model_info": {"base_model": base_model},
                },
                {"model_name": "own", "litellm_params": {"model": own_model, **credentials}},
                {"model_name": "base", "litellm_params": {"model": base_model, **credentials}},
            ]
        ),
    )
    messages: Final = [{"role": "user", "content": _MULTILINGUAL_TEXT}]

    based, own, base = [
        (await _token_counter_route(TokenCountRequest(model=name, messages=messages))).total_tokens
        for name in ("based", "own", "base")
    ]

    assert based == own != base


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_huggingface_tokenizer_loaded")
async def test_framed_count_still_estimates_while_base_model_tokenizer_downloads_hang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter: Final = _ScriptedCounter({"us.moonshotai.kimi-k3": _provider_reply("kimi", status_code=400)})
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "my-llama",
                    "litellm_params": {"model": "hosted_vllm/my-llama", "api_base": "http://127.0.0.1:9/v1"},
                    "model_info": {"base_model": "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct"},
                },
                {
                    "model_name": "kimi",
                    "litellm_params": {"model": "bedrock/us.moonshotai.kimi-k3"},
                    "model_info": {
                        "custom_tokenizer": {"identifier": "lit-tests/kimi", "revision": "main", "auth_token": None}
                    },
                },
            ]
        ),
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server, "_get_provider_token_counter", lambda deployment, model: (counter, model, "bedrock")
    )
    download_started: Final = threading.Event()
    hub_released: Final = threading.Event()
    _hanging_llama_download(monkeypatch, download_started, hub_released)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_MAX_RUNNING", 1)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_WAIT_SECONDS", 30)
    llama: Final = asyncio.ensure_future(
        _token_counter_route(
            TokenCountRequest(model="my-llama", messages=[{"role": "user", "content": "What is new?"}])
        )
    )
    try:
        await asyncio.to_thread(download_started.wait, 10)
        framed: Final = await asyncio.wait_for(_count_status("kimi"), 5)
    finally:
        hub_released.set()
        await llama

    assert framed == "local_estimate"


@pytest.mark.usefixtures("no_huggingface_tokenizer_loaded")
def test_framed_custom_named_deployment_counts_like_before_while_its_base_model_tokenizer_download_hangs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter: Final = _ScriptedCounter({"us.moonshotai.kimi-k3": _provider_reply("kimi", status_code=400)})
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "framed-llama",
                    "litellm_params": {"model": "bedrock/us.moonshotai.kimi-k3"},
                    "model_info": {
                        "base_model": "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
                        "token_count_framing": 7,
                    },
                }
            ]
        ),
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server, "_get_provider_token_counter", lambda deployment, model: (counter, model, "bedrock")
    )
    download_started: Final = threading.Event()
    hub_released: Final = threading.Event()
    _hanging_llama_download(monkeypatch, download_started, hub_released)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_WAIT_SECONDS", 30)

    async def count_past_the_wait(loop: _SkippableClockLoop) -> tuple[str, ...]:
        try:
            count: Final = asyncio.ensure_future(_count_status("framed-llama"))
            await asyncio.to_thread(download_started.wait, 10)
            loop.skip(31)
            return (await asyncio.wait_for(count, 8),)
        finally:
            hub_released.set()

    assert _run_on_skippable_clock(count_past_the_wait) == ("local_estimate",)


_COUNTED_HELLO: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"model": "count-log-gpt", "messages": [{"role": "user", "content": "Hello, how many tokens is this?"}]}
)


def _post_count(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    deployment: Mapping[str, JsonValue],
    body: Mapping[str, JsonValue],
    headers: Mapping[str, str],
) -> httpx.Response:
    from fastapi.testclient import TestClient

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.proxy_server import app

    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", Router(model_list=[dict(deployment)]))
    monkeypatch.setitem(app.dependency_overrides, user_api_key_auth, lambda: UserAPIKeyAuth(api_key="sk-test"))
    return TestClient(app, raise_server_exceptions=False).post(path, json=dict(body), headers=dict(headers))



def _utils_token_counter(
    monkeypatch: pytest.MonkeyPatch, body: Mapping[str, JsonValue], headers: Mapping[str, str]
) -> httpx.Response:
    deployment: Final[Mapping[str, JsonValue]] = {
        "model_name": "count-log-gpt",
        "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "test-key"},
        "model_info": {"id": "count-log-utils-deployment"},
    }
    return _post_count(monkeypatch, "/utils/token_counter", deployment, body, headers)


def test_utils_token_counter_logs_one_line_with_the_callers_call_id_model_and_count(
    monkeypatch: pytest.MonkeyPatch, token_count_log_lines: Callable[[], tuple[dict[str, object], ...]]
) -> None:
    response: Final = _utils_token_counter(monkeypatch, _COUNTED_HELLO, {"x-litellm-call-id": "count-log-utils"})

    assert response.status_code == 200, response.text
    assert token_count_log_lines() == (
        {
            "litellm_call_id": "count-log-utils",
            "request_model": "count-log-gpt",
            "model_id": "count-log-utils-deployment",
            "model_used": "gpt-4o-mini",
            "input_tokens": response.json()["total_tokens"],
            "count_source": "local_estimate",
            "status_code": 200,
        },
    )


def test_utils_token_counter_logs_a_generated_call_id_when_the_client_sends_none(
    monkeypatch: pytest.MonkeyPatch, token_count_log_lines: Callable[[], tuple[dict[str, object], ...]]
) -> None:
    response: Final = _utils_token_counter(monkeypatch, _COUNTED_HELLO, {})

    assert response.status_code == 200, response.text
    (line,) = token_count_log_lines()
    assert str(uuid.UUID(str(line["litellm_call_id"]))) == line["litellm_call_id"]


def test_utils_token_counter_without_messages_logs_the_400_it_answers(
    monkeypatch: pytest.MonkeyPatch, token_count_log_lines: Callable[[], tuple[dict[str, object], ...]]
) -> None:
    response: Final = _utils_token_counter(
        monkeypatch, {"model": "count-log-gpt"}, {"x-litellm-call-id": "count-log-utils-empty"}
    )

    assert response.status_code == 400, response.text
    assert token_count_log_lines() == (
        {
            "litellm_call_id": "count-log-utils-empty",
            "request_model": "count-log-gpt",
            "model_id": None,
            "model_used": None,
            "input_tokens": None,
            "count_source": None,
            "status_code": 400,
        },
    )


@pytest.mark.parametrize("nested", [False, True], ids=["user-content", "tool-result"])
def test_utils_token_counter_answers_400_naming_a_content_block_the_local_counter_cannot_read(
    monkeypatch: pytest.MonkeyPatch, token_count_log_lines: Callable[[], tuple[dict[str, object], ...]], nested: bool
) -> None:
    block: Final[JsonValue] = {"type": "future_block", "data": "opaque"}
    content: Final[JsonValue] = (
        [{"type": "tool_result", "tool_use_id": "toolu_1", "content": [block]}] if nested else [block]
    )

    response: Final = _utils_token_counter(
        monkeypatch,
        {"model": "count-log-gpt", "messages": [{"role": "user", "content": content}]},
        {"x-litellm-call-id": "count-log-utils-unreadable"},
    )

    assert response.status_code == 400, response.text
    error: Final = response.json()["error"]
    assert (error["type"], error["param"], error["code"]) == ("invalid_request_error", "messages", "400"), error
    assert error["message"] == "The token counter can't count content blocks of type future_block", error
    assert [line["status_code"] for line in token_count_log_lines()] == [400]


def test_utils_token_counter_answers_400_naming_a_malformed_server_tool_block(monkeypatch: pytest.MonkeyPatch) -> None:
    block: Final[JsonValue] = {
        "type": "web_search_tool_result",
        "tool_use_id": "srv_1",
        "content": [{"url": "https://example.invalid"}],
    }

    response: Final = _utils_token_counter(
        monkeypatch, {"model": "count-log-gpt", "messages": [{"role": "user", "content": [block]}]}, {}
    )

    assert response.status_code == 400, response.text
    error: Final = response.json()["error"]
    assert (error["type"], error["param"]) == ("invalid_request_error", "messages"), error
    assert error["message"].startswith(
        "The token counter can't read the content field of a web_search_tool_result block: "
    ), error


_FUTURE_BLOCK: Final = MappingProxyType({"type": "future_block", "data": "opaque"})


@pytest.mark.parametrize(
    ("fields", "param"),
    [
        ({"system": [dict(_FUTURE_BLOCK)], "messages": [{"role": "user", "content": "hi"}]}, "system"),
        ({"system": "Be brief.", "messages": [{"role": "user", "content": [dict(_FUTURE_BLOCK)]}]}, "messages"),
        (
            {
                "system": [{"type": "text", "text": "Be brief."}],
                "messages": [{"role": "user", "content": [dict(_FUTURE_BLOCK)]}],
            },
            None,
        ),
        (
            {
                "messages": [{"role": "user", "content": [dict(_FUTURE_BLOCK)]}],
                "contents": [{"role": "user", "parts": [{"text": "Be brief."}]}],
            },
            "messages",
        ),
    ],
    ids=["block-in-system", "block-in-messages", "blocks-in-both-fields", "contents-sent-beside-messages"],
)
def test_utils_token_counter_points_param_only_at_the_one_field_that_can_hold_the_block(
    monkeypatch: pytest.MonkeyPatch, fields: Mapping[str, JsonValue], param: str | None
) -> None:
    response: Final = _utils_token_counter(monkeypatch, {"model": "count-log-gpt", **fields}, {})

    assert response.status_code == 400, response.text
    error: Final = response.json()["error"]
    assert (error["type"], error["param"]) == ("invalid_request_error", param), error


def test_utils_token_counter_keeps_500_for_a_known_block_it_fails_to_count(monkeypatch: pytest.MonkeyPatch) -> None:
    response: Final = _utils_token_counter(
        monkeypatch, {"model": "count-log-gpt", "messages": [{"role": "user", "content": [{"type": "image"}]}]}, {}
    )

    assert response.status_code == 500, response.text


_INPUT_AUDIO: Final = MappingProxyType({"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}})


@pytest.mark.parametrize(
    ("fields", "param"),
    [
        ({"input": [{"role": "user", "content": [dict(_INPUT_AUDIO)]}]}, "input"),
        ({"input": "hi", "instructions": [dict(_INPUT_AUDIO)]}, "instructions"),
    ],
    ids=["block-in-input", "block-in-instructions"],
)
def test_responses_input_tokens_points_param_at_the_field_holding_a_block_the_local_counter_cannot_read(
    monkeypatch: pytest.MonkeyPatch, fields: Mapping[str, JsonValue], param: str
) -> None:
    deployment: Final[Mapping[str, JsonValue]] = {
        "model_name": "count-local-vllm",
        "litellm_params": {"model": "hosted_vllm/zai-org/GLM-5.3", "api_base": "http://127.0.0.1:9/v1"},
    }

    response: Final = _post_count(
        monkeypatch, "/v1/responses/input_tokens", deployment, {"model": "count-local-vllm", **fields}, {}
    )

    assert response.status_code == 400, response.text
    error: Final = response.json()["error"]
    assert (error["type"], error["param"]) == ("invalid_request_error", param), error
    assert error["message"] == "The token counter can't count content blocks of type input_audio", error


def test_utils_token_counter_failed_count_logs_its_status_in_place_of_a_count(
    monkeypatch: pytest.MonkeyPatch, token_count_log_lines: Callable[[], tuple[dict[str, object], ...]]
) -> None:
    response: Final = _utils_token_counter(
        monkeypatch, {**_COUNTED_HELLO, "prompt": "and a prompt"}, {"x-litellm-call-id": "count-log-utils-error"}
    )

    assert response.status_code == 400, response.text
    assert response.json() == {
        "error": {
            "message": "prompt cannot be sent with messages or contents. Set drop_params to count them without the prompt",
            "type": "invalid_request_error",
            "param": "prompt",
            "code": "400",
        }
    }
    assert token_count_log_lines() == (
        {
            "litellm_call_id": "count-log-utils-error",
            "request_model": "count-log-gpt",
            "model_id": "count-log-utils-deployment",
            "model_used": None,
            "input_tokens": None,
            "count_source": None,
            "status_code": 400,
        },
    )


@pytest.mark.parametrize(
    ("body", "headers", "status_code"),
    [
        (_COUNTED_HELLO, {}, 200),
        (_COUNTED_HELLO, {"x-litellm-call-id": "count-echo-utils"}, 200),
        ({"model": "count-log-gpt"}, {}, 400),
        ({**_COUNTED_HELLO, "prompt": "and a prompt"}, {"x-litellm-call-id": "count-echo-utils"}, 400),
        (
            {"model": "count-log-gpt", "messages": [{"role": "user", "content": "hi", "tool_calls": "x"}]},
            {"x-litellm-call-id": "count-echo-utils"},
            500,
        ),
    ],
    ids=["counted", "counted-with-client-id", "no-messages", "failed-count-with-client-id", "crash-with-client-id"],
)
def test_utils_token_counter_answers_the_call_id_its_count_line_names(
    monkeypatch: pytest.MonkeyPatch,
    token_count_log_lines: Callable[[], tuple[dict[str, object], ...]],
    body: Mapping[str, JsonValue],
    headers: Mapping[str, str],
    status_code: int,
) -> None:
    response: Final = _utils_token_counter(monkeypatch, body, headers)

    (line,) = token_count_log_lines()
    assert (response.status_code, response.headers.get("x-litellm-call-id")) == (
        status_code,
        line["litellm_call_id"],
    ), response.text
    assert headers.get("x-litellm-call-id") in (None, line["litellm_call_id"])


def test_utils_token_counter_json_line_keeps_its_own_fields_when_the_call_id_is_a_dict(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    token_count_log_lines: Callable[[], tuple[dict[str, object], ...]],
) -> None:
    forged_call_id: Final = "{'component':'forged-comp','logger':'forged.py:1','injected_key':'injected'}"

    _utils_token_counter(monkeypatch, _COUNTED_HELLO, {"x-litellm-call-id": forged_call_id})

    (line,) = token_count_log_lines()
    (record,) = (record for record in caplog.records if "count_source" in record.__dict__)
    rendered: Final = json.loads(JsonFormatter().format(record))
    assert line["litellm_call_id"] == forged_call_id
    assert {key: rendered.get(key) for key in ("litellm_call_id", "component", "logger", "injected_key")} == {
        "litellm_call_id": forged_call_id,
        "component": verbose_proxy_logger.name,
        "logger": f"{record.filename}:{record.lineno}",
        "injected_key": None,
    }


def test_utils_token_counter_line_stays_one_line_when_the_model_carries_a_newline(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    token_count_log_lines: Callable[[], tuple[dict[str, object], ...]],
) -> None:
    forged_model: Final = "count-log-gpt\ntoken count - litellm_call_id=forged input_tokens=1"

    _utils_token_counter(monkeypatch, {**_COUNTED_HELLO, "model": forged_model}, {"x-litellm-call-id": "count log"})

    (line,) = token_count_log_lines()
    (message,) = (record.getMessage() for record in caplog.records if "count_source" in record.__dict__)
    assert line["request_model"] == forged_model
    assert "\n" not in message
    assert message.startswith(f'token count - litellm_call_id="count log" request_model={json.dumps(forged_model)} ')


@pytest.mark.asyncio
async def test_mistral_proxy_token_counter_chat_and_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "mistral-proxy-model",
                "litellm_params": {
                    "model": "mistral/mistral-large-4",
                    "api_key": "test-mistral-key",
                },
            }
        ]
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)

    def count_peer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 33}, request=request)

    with patch.object(
        mistral_count_tokens_http,
        "get_async_httpx_client",
        return_value=AsyncHTTPHandler(transport=httpx.MockTransport(count_peer)),
    ):
        utils_result: Final = await count_request_tokens(
            request=TokenCountRequest(
                model="mistral-proxy-model", messages=[{"role": "user", "content": "hello mistral"}]
            ),
            call_endpoint=True,
        )
        responses_result: Final = await count_request_tokens(
            request=TokenCountRequest(
                model="mistral-proxy-model", messages=[{"role": "user", "content": "hello mistral"}]
            ),
            call_endpoint=True,
            responses_request={"model": "mistral-proxy-model", "input": "hello mistral"},
        )

    assert utils_result.total_tokens == 33
    assert utils_result.tokenizer_type == "mistral_api"
    assert responses_result.total_tokens == 33
    assert responses_result.tokenizer_type == "mistral_api"


def _input_texts(item: JsonValue) -> list[str]:
    assert isinstance(item, dict)
    content: Final = item.get("content")
    if isinstance(content, str):
        return [content]
    assert isinstance(content, list)
    return [str(part["text"]) for part in content if isinstance(part, dict) and "text" in part]


class _SpendLogsTable:
    def __init__(self, rows: tuple[Mapping[str, object], ...]) -> None:
        self.rows: Final = rows

    async def query_raw(self, query: str, request_ids: list[str]) -> list[Mapping[str, object]]:
        sessions: Final = {row["session_id"] for row in self.rows if row["request_id"] in request_ids}
        return [row for row in self.rows if row["session_id"] in sessions]


@pytest.mark.asyncio
async def test_responses_count_on_mistral_counts_the_turns_previous_response_id_replays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "mistral-counted",
                "litellm_params": {"model": "mistral/mistral-large-4", "api_key": "test-mistral-key"},
            }
        ]
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
    stored_turn: Final = {
        "request_id": "chatcmpl-stored",
        "call_type": "aresponses",
        "session_id": "session-1",
        "proxy_server_request": {"input": "Who won in 1998?", "model": "mistral-counted"},
        "response": {
            "id": "chatcmpl-stored",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "The Bulls."}, "finish_reason": "stop"}
            ],
        },
        "status": "success",
    }
    bodies: Final[list[dict[str, JsonValue]]] = []  # mutable-ok: collects what the mock peer receives

    def count_peer(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 41}, request=request)

    monkeypatch.setattr(litellm.proxy.proxy_server, "prisma_client", SimpleNamespace(db=_SpendLogsTable((stored_turn,))))
    with patch.object(
        mistral_count_tokens_http,
        "get_async_httpx_client",
        return_value=AsyncHTTPHandler(transport=httpx.MockTransport(count_peer)),
    ):
        result: Final = await count_request_tokens(
            request=TokenCountRequest(model="mistral-counted", messages=[{"role": "user", "content": "And in 1999?"}]),
            call_endpoint=True,
            responses_request={
                "model": "mistral-counted",
                "input": "And in 1999?",
                "previous_response_id": "chatcmpl-stored",
            },
        )

    assert (result.error, result.total_tokens, result.tokenizer_type) == (False, 41, "mistral_api")
    (body,) = bodies
    assert "previous_response_id" not in body
    sent_input: Final = body["input"]
    assert isinstance(sent_input, list)
    assert list(itertools.chain.from_iterable(map(_input_texts, sent_input))) == [
        "Who won in 1998?",
        "The Bulls.",
        "And in 1999?",
    ]


_COUNTED_OPTION_FIELDS: Final = ("thinking", "tool_choice", "output_config")
_DEEPSEEK_COUNT_ROUTE: Final = "https://api.deepseek.com/anthropic/v1/messages/count_tokens"
_CUSTOM_READ_FILE_TOOL: Final[dict[str, JsonValue]] = {**_READ_FILE_TOOL, "type": "custom"}


def _messages_inference_fields(
    config: object, model: str, body: Mapping[str, JsonValue], fields: tuple[str, ...]
) -> dict[str, object]:
    from litellm.types.router import GenericLiteLLMParams

    inference: Final = config.transform_anthropic_messages_request(
        model=model,
        messages=json.loads(json.dumps(body["messages"])),
        anthropic_messages_optional_request_params={
            **{key: json.loads(json.dumps(value)) for key, value in body.items() if key not in ("model", "messages")},
            "max_tokens": 4096,
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    return {field: inference.get(field) for field in fields}


async def _count_messages_body(
    monkeypatch: pytest.MonkeyPatch,
    deployment_params: Mapping[str, JsonValue],
    body: Mapping[str, JsonValue],
    count_route: str,
    provider_reply: httpx.Response,
) -> tuple[object, respx.Route]:
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.delenv("DEEPSEEK_API_BASE", raising=False)
    monkeypatch.delenv("DEEPSEEK_ANTHROPIC_API_BASE", raising=False)
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        Router(model_list=[{"model_name": "counted", "litellm_params": dict(deployment_params)}]),
    )
    monkeypatch.setattr(proxy_server, "proxy_config", proxy_server.ProxyConfig())
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    with respx.mock:
        route: Final = respx.post(count_route).mock(return_value=provider_reply)
        response: Final = await anthropic_count_tokens(
            _json_request({"model": "counted", **body}, "python-httpx/0.28.1"), Response(), UserAPIKeyAuth()
        )
    return response, route


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "options"),
    (
        ("claude-sonnet-4-5", {"tool_choice": {"type": "any"}}),
        ("claude-sonnet-4-5", {"thinking": {"type": "enabled", "budget_tokens": 2048}}),
        ("claude-haiku-5-5", {"thinking": {"type": "enabled", "budget_tokens": 2048}, "tool_choice": {"type": "auto"}}),
        ("claude-opus-5-5", {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}),
    ),
)
async def test_messages_count_tokens_counts_the_thinking_tool_choice_and_effort_messages_inference_sends(
    monkeypatch: pytest.MonkeyPatch, model: str, options: Mapping[str, JsonValue]
) -> None:
    from litellm.llms.anthropic.messages.native.transformation import AnthropicMessagesConfig

    body: Final[dict[str, JsonValue]] = {
        "messages": [{"role": "user", "content": "Read the README"}],
        "tools": [_READ_FILE_TOOL],
        **options,
    }

    response, route = await _count_messages_body(
        monkeypatch,
        {"model": f"anthropic/{model}", "api_key": "sk-ant-test"},
        body,
        "https://api.anthropic.com/v1/messages/count_tokens",
        httpx.Response(200, json={"input_tokens": 120}),
    )

    sent: Final = json.loads(route.calls.last.request.content)
    assert response == {"input_tokens": 120, "litellm_count_source": "provider"}
    assert {field: sent.get(field) for field in _COUNTED_OPTION_FIELDS} == _messages_inference_fields(
        AnthropicMessagesConfig(), model, body, _COUNTED_OPTION_FIELDS
    )


@pytest.mark.asyncio
async def test_messages_count_tokens_leaves_out_an_option_the_request_drops(monkeypatch: pytest.MonkeyPatch) -> None:
    response, route = await _count_messages_body(
        monkeypatch,
        {"model": "anthropic/claude-sonnet-4-5", "api_key": "sk-ant-test"},
        {
            "messages": [{"role": "user", "content": "Read the README"}],
            "tools": [_READ_FILE_TOOL],
            "tool_choice": {"type": "any"},
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "additional_drop_params": ["thinking"],
        },
        "https://api.anthropic.com/v1/messages/count_tokens",
        httpx.Response(200, json={"input_tokens": 120}),
    )

    sent: Final = json.loads(route.calls.last.request.content)
    assert response == {"input_tokens": 120, "litellm_count_source": "provider"}
    assert (sent["tool_choice"], "thinking" in sent) == ({"type": "any"}, False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body_options", "sent_thinking_type"),
    (
        ({}, "disabled"),
        ({"thinking": {"type": "enabled"}}, "enabled"),
        ({"additional_drop_params": ["thinking"]}, None),
    ),
)
async def test_messages_count_tokens_counts_the_thinking_the_deployment_sets_unless_the_request_overrides_it(
    monkeypatch: pytest.MonkeyPatch, body_options: Mapping[str, JsonValue], sent_thinking_type: str | None
) -> None:
    _, route = await _count_messages_body(
        monkeypatch,
        {"model": "deepseek/deepseek-v4-pro", "api_key": "sk-deepseek-test", "thinking": {"type": "disabled"}},
        {"messages": [{"role": "user", "content": "Read the README"}], **body_options},
        _DEEPSEEK_COUNT_ROUTE,
        httpx.Response(200, json={"input_tokens": 91}),
    )

    sent: Final = json.loads(route.calls.last.request.content)
    assert (sent.get("thinking") or {}).get("type") == sent_thinking_type


@pytest.mark.asyncio
async def test_messages_count_tokens_answers_400_for_a_thinking_budget_messages_inference_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.exceptions import BadRequestError
    from litellm.llms.anthropic.messages.native.transformation import AnthropicMessagesConfig

    body: Final[dict[str, JsonValue]] = {
        "messages": [{"role": "user", "content": "Read the README"}],
        "thinking": {"type": "enabled", "budget_tokens": "lots"},
    }
    with pytest.raises(BadRequestError) as rejected:
        _messages_inference_fields(AnthropicMessagesConfig(), "claude-opus-5-5", body, _COUNTED_OPTION_FIELDS)

    response, route = await _count_messages_body(
        monkeypatch,
        {"model": "anthropic/claude-opus-5-5", "api_key": "sk-ant-test"},
        body,
        "https://api.anthropic.com/v1/messages/count_tokens",
        httpx.Response(200, json={"input_tokens": 120}),
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert rejected.value.message in json.loads(response.body)["error"]["message"]
    assert not route.called


@pytest.mark.asyncio
async def test_messages_count_tokens_counts_deepseek_on_its_anthropic_count_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.llms.deepseek.messages.transformation import DeepSeekAnthropicMessagesConfig

    body: Final[dict[str, JsonValue]] = {
        "messages": [{"role": "user", "content": "Read the README"}],
        "tools": [_CUSTOM_READ_FILE_TOOL],
        "tool_choice": {"type": "any"},
        "thinking": {"type": "enabled"},
        "output_config": {"effort": "max"},
    }

    response, route = await _count_messages_body(
        monkeypatch,
        {"model": "deepseek/deepseek-v4-pro", "api_key": "sk-deepseek-test"},
        body,
        _DEEPSEEK_COUNT_ROUTE,
        httpx.Response(200, json={"input_tokens": 91}),
    )

    sent_request: Final = route.calls.last.request
    sent: Final = json.loads(sent_request.content)
    inference: Final = _messages_inference_fields(
        DeepSeekAnthropicMessagesConfig(), "deepseek-v4-pro", body, ("tool_choice", "output_config", "tools", "thinking")
    )
    assert response == {"input_tokens": 91, "litellm_count_source": "provider"}
    assert sent_request.headers["x-api-key"] == "sk-deepseek-test"
    assert (sent["tool_choice"], sent.get("output_config"), sent["tools"], sent["thinking"]["type"]) == (
        inference["tool_choice"],
        inference["output_config"],
        inference["tools"],
        inference["thinking"]["type"],
    )


@pytest.mark.asyncio
async def test_messages_count_tokens_on_deepseek_on_another_host_answers_the_estimate_without_a_count_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response, route = await _count_messages_body(
        monkeypatch,
        {"model": "deepseek/deepseek-v4-pro", "api_key": "sk-other-host", "api_base": "https://gateway.example/v1"},
        {"messages": [{"role": "user", "content": "Read the README"}]},
        "https://gateway.example/anthropic/v1/messages/count_tokens",
        httpx.Response(401, json={"error": "Authentication required."}),
    )

    assert not route.called
    assert isinstance(response, dict)
    assert response["litellm_count_source"] == "local_estimate"
    assert response["input_tokens"] > 0


async def _count_deepseek_responses(
    monkeypatch: pytest.MonkeyPatch,
    deployment_options: Mapping[str, JsonValue],
    responses_options: Mapping[str, JsonValue] | None,
    request_options: Mapping[str, JsonValue] = MappingProxyType({}),
) -> tuple[TokenCountResponse, dict[str, JsonValue]]:
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.delenv("DEEPSEEK_API_BASE", raising=False)
    monkeypatch.delenv("DEEPSEEK_ANTHROPIC_API_BASE", raising=False)
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "deepseek-counted",
                    "litellm_params": {
                        "model": "deepseek/deepseek-v4-pro",
                        "api_key": "sk-deepseek-test",
                        **deployment_options,
                    },
                }
            ]
        ),
    )
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    with respx.mock:
        route: Final = respx.post(_DEEPSEEK_COUNT_ROUTE).mock(return_value=httpx.Response(200, json={"input_tokens": 91}))
        result: Final = await count_request_tokens(
            request=TokenCountRequest.model_validate(
                {"model": "deepseek-counted", "messages": [{"role": "user", "content": "Say hi."}], **request_options}
            ),
            call_endpoint=True,
            responses_request=None
            if responses_options is None
            else {
                "model": "deepseek-counted",
                "input": "Say hi.",
                "tools": [_WEATHER_TOOL],
                **responses_options,
            },
        )
    return result, json.loads(route.calls.last.request.content)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reasoning", "expected_options"),
    (
        ({}, {}),
        ({"reasoning": {"effort": "none"}}, {"thinking": {"type": "disabled"}}),
        ({"reasoning": {"effort": "high"}}, {"thinking": {"type": "enabled"}, "output_config": {"effort": "high"}}),
        ({"reasoning": {"effort": "max"}}, {"thinking": {"type": "enabled"}, "output_config": {"effort": "max"}}),
    ),
)
async def test_responses_count_on_deepseek_maps_reasoning_effort_the_way_deepseek_chat_inference_does(
    monkeypatch: pytest.MonkeyPatch, reasoning: Mapping[str, JsonValue], expected_options: Mapping[str, JsonValue]
) -> None:
    result, sent = await _count_deepseek_responses(monkeypatch, {}, reasoning)

    assert (result.total_tokens, result.count_source) == (91, "provider")
    assert {field: sent[field] for field in ("thinking", "output_config") if field in sent} == expected_options
    assert all("type" not in tool for tool in sent["tools"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("deployment_options", "responses_options", "expected_options"),
    (
        ({"thinking": {"type": "disabled"}}, {}, {"thinking": {"type": "disabled"}}),
        ({"reasoning_effort": "none"}, {}, {"thinking": {"type": "disabled"}}),
        ({"thinking": {"type": "disabled"}}, {"additional_drop_params": ["thinking"]}, {}),
        ({"thinking": {"type": "disabled"}}, {"thinking": {"type": "enabled"}}, {"thinking": {"type": "enabled"}}),
    ),
)
async def test_responses_count_on_deepseek_counts_the_thinking_inference_sends_from_the_request_or_deployment(
    monkeypatch: pytest.MonkeyPatch,
    deployment_options: Mapping[str, JsonValue],
    responses_options: Mapping[str, JsonValue],
    expected_options: Mapping[str, JsonValue],
) -> None:
    result, sent = await _count_deepseek_responses(monkeypatch, deployment_options, responses_options)

    assert (result.total_tokens, result.count_source) == (91, "provider")
    assert {field: sent[field] for field in ("thinking", "output_config") if field in sent} == expected_options


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("deployment_options", "expected_options"),
    (
        ({"thinking": {"type": "disabled"}}, {"thinking": {"type": "disabled"}}),
        ({"reasoning_effort": "high"}, {"thinking": {"type": "enabled"}, "output_config": {"effort": "high"}}),
        ({"thinking": {"type": "disabled"}, "additional_drop_params": ["thinking"]}, {}),
    ),
)
async def test_chat_count_on_deepseek_counts_the_thinking_the_deployment_sets_unless_it_drops_it(
    monkeypatch: pytest.MonkeyPatch,
    deployment_options: Mapping[str, JsonValue],
    expected_options: Mapping[str, JsonValue],
) -> None:
    result, sent = await _count_deepseek_responses(monkeypatch, deployment_options, None)

    assert (result.total_tokens, result.count_source) == (91, "provider")
    assert {field: sent[field] for field in ("thinking", "output_config") if field in sent} == expected_options


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("deployment_options", "request_options", "expected_options"),
    (
        ({}, {"reasoning_effort": "none"}, {"thinking": {"type": "disabled"}}),
        ({}, {"reasoning_effort": "max"}, {"thinking": {"type": "enabled"}, "output_config": {"effort": "max"}}),
        ({"thinking": {"type": "disabled"}}, {"thinking": {"type": "enabled"}}, {"thinking": {"type": "enabled"}}),
    ),
)
async def test_chat_count_on_deepseek_counts_the_thinking_the_request_sends_over_the_deployment_default(
    monkeypatch: pytest.MonkeyPatch,
    deployment_options: Mapping[str, JsonValue],
    request_options: Mapping[str, JsonValue],
    expected_options: Mapping[str, JsonValue],
) -> None:
    result, sent = await _count_deepseek_responses(monkeypatch, deployment_options, None, request_options)

    assert (result.total_tokens, result.count_source) == (91, "provider")
    assert {field: sent[field] for field in ("thinking", "output_config") if field in sent} == expected_options


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_drops", "sent_options"),
    (
        ({}, (True, None)),
        ({"additional_drop_params": ["tool_choice"]}, (False, "enabled")),
    ),
)
async def test_messages_count_tokens_drops_what_the_deployment_drops_unless_the_request_sends_its_own_list(
    monkeypatch: pytest.MonkeyPatch, request_drops: Mapping[str, JsonValue], sent_options: tuple[bool, str | None]
) -> None:
    _, route = await _count_messages_body(
        monkeypatch,
        {"model": "anthropic/claude-sonnet-4-5", "api_key": "sk-ant-test", "additional_drop_params": ["thinking"]},
        {
            "messages": [{"role": "user", "content": "Read the README"}],
            "tools": [_READ_FILE_TOOL],
            "tool_choice": {"type": "auto"},
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            **request_drops,
        },
        "https://api.anthropic.com/v1/messages/count_tokens",
        httpx.Response(200, json={"input_tokens": 120}),
    )

    sent: Final = json.loads(route.calls.last.request.content)
    assert ("tool_choice" in sent, (sent.get("thinking") or {}).get("type")) == sent_options


_TOOL_CHOICE_MODEL: Final = "qwen/qwen3.8-27b"
_TOOL_CHOICE_QUESTION: Final = "What is the weather in Paris?"
_CITY_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
_CHAT_WEATHER_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {"name": "get_weather", "description": "Get the weather for a city", "parameters": _CITY_SCHEMA},
}


def _tool_choice_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "no-count-api",
                "litellm_params": {"model": f"groq/{_TOOL_CHOICE_MODEL}", "api_key": "sk-test"},
            }
        ]
    )


async def _utils_route_count(tool_choice: Mapping[str, JsonValue], model: str = "no-count-api") -> int:
    response: Final = await token_counter(
        request=TokenCountRequest(
            model=model,
            messages=[{"role": "user", "content": _TOOL_CHOICE_QUESTION}],
            tools=[_CHAT_WEATHER_TOOL],
            **tool_choice,
        ),
        http_request=_json_request({}, "python-httpx/0.28.1"),
        fastapi_response=Response(),
        user_api_key_dict=UserAPIKeyAuth(),
    )
    return response.total_tokens


async def _messages_route_count(tool_choice: Mapping[str, JsonValue]) -> int:
    body: Final = {
        "model": "no-count-api",
        "messages": [{"role": "user", "content": _TOOL_CHOICE_QUESTION}],
        "tools": [{"name": "get_weather", "description": "Get the weather for a city", "input_schema": _CITY_SCHEMA}],
        **tool_choice,
    }
    response: Final = await anthropic_count_tokens(
        _json_request(body, "python-httpx/0.28.1"), Response(), UserAPIKeyAuth()
    )
    assert isinstance(response, dict)
    return response["input_tokens"]


async def _responses_route_count(tool_choice: Mapping[str, JsonValue]) -> int:
    from litellm.proxy.response_api_endpoints.endpoints import responses_input_tokens

    body: Final = {
        "model": "no-count-api",
        "input": _TOOL_CHOICE_QUESTION,
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": _CITY_SCHEMA,
            }
        ],
        **tool_choice,
    }
    response: Final = await responses_input_tokens(
        request=_json_request(body, "python-httpx/0.28.1"),
        fastapi_response=Response(),
        general_settings=MappingProxyType({}),
        token_counter=count_request_tokens,
        llm_router=litellm.proxy.proxy_server.llm_router,
        user_api_key_dict=UserAPIKeyAuth(),
        response_id_security=None,
        managed_files=None,
    )
    assert isinstance(response, dict)
    return response["input_tokens"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route_count", "tool_choice"),
    [
        (_utils_route_count, {"type": "function", "function": {"name": "get_weather"}}),
        (_utils_route_count, "none"),
        (_messages_route_count, {"type": "tool", "name": "get_weather"}),
        (_messages_route_count, {"type": "none"}),
        (_responses_route_count, {"type": "function", "name": "get_weather"}),
        (_responses_route_count, "none"),
    ],
    ids=["utils-named", "utils-none", "messages-named", "messages-none", "responses-named", "responses-none"],
)
async def test_count_routes_without_a_count_api_count_the_tool_choice_like_token_counter(
    monkeypatch: pytest.MonkeyPatch,
    route_count: Callable[[Mapping[str, JsonValue]], Coroutine[None, None, int]],
    tool_choice: JsonValue,
) -> None:
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", _tool_choice_router())
    monkeypatch.setattr(litellm.proxy.proxy_server, "proxy_config", litellm.proxy.proxy_server.ProxyConfig())
    question: Final = [{"role": "user", "content": _TOOL_CHOICE_QUESTION}]
    sdk_added: Final = litellm.token_counter(
        model=_TOOL_CHOICE_MODEL, messages=question, tools=[_CHAT_WEATHER_TOOL], tool_choice=tool_choice
    ) - litellm.token_counter(model=_TOOL_CHOICE_MODEL, messages=question, tools=[_CHAT_WEATHER_TOOL])

    route_added: Final = await route_count({"tool_choice": tool_choice}) - await route_count({})

    assert route_added == sdk_added > 0


_AZURE_GROK_PARAMS: Final = {"api_base": "https://example.services.ai.azure.com", "api_key": "sk-test"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("deployment", "served_by_xai"),
    [
        ({"litellm_params": {"model": "xai/grok-4-1-fast-reasoning", "api_key": "sk-test"}}, True),
        ({"litellm_params": {"model": "openrouter/x-ai/grok-4.3", "api_key": "sk-test"}}, True),
        (
            {"litellm_params": {"model": "openai/grok-4.3", "api_base": "https://api.x.ai/v1", "api_key": "sk-test"}},
            True,
        ),
        ({"litellm_params": {"model": "openai/grok-4.3", "api_base": "http://[bad", "api_key": "sk-test"}}, False),
        (
            {
                "litellm_params": {"model": "azure_ai/grok-4-1", **_AZURE_GROK_PARAMS},
                "model_info": {"base_model": "azure_ai/grok-4-1-fast-reasoning"},
            },
            False,
        ),
        ({"litellm_params": {"model": "azure_ai/grok-4-1-fast-reasoning", **_AZURE_GROK_PARAMS}}, False),
        ({"litellm_params": {"model": "grok-4.3", "custom_llm_provider": "azure_ai", **_AZURE_GROK_PARAMS}}, False),
        (
            {
                "litellm_params": {
                    "model": "vertex_ai/xai/grok-4.3",
                    "vertex_project": "test-project",
                    "vertex_location": "global",
                }
            },
            False,
        ),
    ],
)
async def test_utils_token_counter_counts_xai_tool_choice_costs_only_on_grok_that_xai_serves(
    monkeypatch: pytest.MonkeyPatch, deployment: dict[str, dict[str, str]], served_by_xai: bool
) -> None:
    monkeypatch.setattr(
        litellm.proxy.proxy_server, "llm_router", Router(model_list=[{"model_name": "grok", **deployment}])
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "proxy_config", litellm.proxy.proxy_server.ProxyConfig())
    named: Final = {"tool_choice": {"type": "function", "function": {"name": "get_weather"}}}
    question: Final = [{"role": "user", "content": _TOOL_CHOICE_QUESTION}]
    sdk_added: Final = litellm.token_counter(
        model="xai/grok-4.3", messages=question, tools=[_CHAT_WEATHER_TOOL], **named
    ) - litellm.token_counter(model="xai/grok-4.3", messages=question, tools=[_CHAT_WEATHER_TOOL])

    route_added: Final = await _utils_route_count(named, "grok") - await _utils_route_count({}, "grok")

    assert route_added == (sdk_added if served_by_xai else 0)
    assert sdk_added > 0


def _litellm_warnings(caplog: pytest.LogCaptureFixture) -> tuple[str, ...]:
    return tuple(record.getMessage() for record in caplog.records if record.name == verbose_logger.name)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route_count", "tool_choice"),
    [(_utils_route_count, "none"), (_messages_route_count, {"type": "none"}), (_responses_route_count, "none")],
    ids=["utils", "messages", "responses"],
)
@pytest.mark.parametrize("model", ["chatgpt/gpt-6.1-sol", "github_copilot/kimi-k3"])
async def test_count_routes_leave_the_provider_login_alone_when_counting_a_tool_choice(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    route_count: Callable[[Mapping[str, JsonValue]], Coroutine[None, None, int]],
    tool_choice: JsonValue,
    model: str,
) -> None:
    (tmp_path / "auth.json").write_text("not a saved login")
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", str(tmp_path))
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(model_list=[{"model_name": "no-count-api", "litellm_params": {"model": model}}]),
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "proxy_config", litellm.proxy.proxy_server.ProxyConfig())

    with caplog.at_level(logging.WARNING, logger=verbose_logger.name):
        without_choice: Final = await route_count({})
        warnings_without_choice: Final = _litellm_warnings(caplog)
        caplog.clear()
        with_choice: Final = await route_count({"tool_choice": tool_choice})

    assert with_choice > without_choice
    assert _litellm_warnings(caplog) == warnings_without_choice


def _nested_json(depth: int) -> JsonValue:
    return "none" if depth == 0 else {"nested": _nested_json(depth - 1)}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route_count", "field"),
    [
        (_utils_route_count, "tool_choice"),
        (_utils_route_count, "additional_drop_params"),
        (_messages_route_count, "tool_choice"),
        (_messages_route_count, "thinking"),
        (_responses_route_count, "tool_choice"),
    ],
    ids=["utils-tool-choice", "utils-drop-params", "messages-tool-choice", "messages-thinking", "responses-tool-choice"],
)
async def test_count_routes_skip_a_value_nested_too_deep_to_validate(
    monkeypatch: pytest.MonkeyPatch,
    route_count: Callable[[Mapping[str, JsonValue]], Coroutine[None, None, int]],
    field: str,
) -> None:
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", _tool_choice_router())
    monkeypatch.setattr(litellm.proxy.proxy_server, "proxy_config", litellm.proxy.proxy_server.ProxyConfig())

    assert await route_count({field: _nested_json(300)}) == await route_count({})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route_count", "tool_choice"),
    [(_utils_route_count, "none"), (_messages_route_count, {"type": "none"}), (_responses_route_count, "none")],
    ids=["utils", "messages", "responses"],
)
@pytest.mark.parametrize(
    ("drop_params", "counts_choice"),
    [({"additional_drop_params": ["tool_choice"]}, False), ({"drop_params": True}, True)],
    ids=["tool-choice-dropped", "only-unsupported-dropped"],
)
async def test_count_routes_count_no_tool_choice_the_deployment_drops(
    monkeypatch: pytest.MonkeyPatch,
    route_count: Callable[[Mapping[str, JsonValue]], Coroutine[None, None, int]],
    tool_choice: JsonValue,
    drop_params: dict[str, JsonValue],
    counts_choice: bool,
) -> None:
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "no-count-api",
                    "litellm_params": {"model": f"groq/{_TOOL_CHOICE_MODEL}", "api_key": "sk-test", **drop_params},
                }
            ]
        ),
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "proxy_config", litellm.proxy.proxy_server.ProxyConfig())

    route_added: Final = await route_count({"tool_choice": tool_choice}) - await route_count({})

    assert (route_added > 0) == counts_choice


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route_count", "tool_choice"),
    [(_utils_route_count, "none"), (_messages_route_count, {"type": "none"}), (_responses_route_count, "none")],
    ids=["utils", "messages", "responses"],
)
@pytest.mark.parametrize(
    ("dropped", "counts_choice"),
    [(["tool_choice"], False), (["temperature"], True)],
    ids=["tool-choice-dropped", "other-param-dropped"],
)
async def test_count_routes_count_no_tool_choice_the_request_drops(
    monkeypatch: pytest.MonkeyPatch,
    route_count: Callable[[Mapping[str, JsonValue]], Coroutine[None, None, int]],
    tool_choice: JsonValue,
    dropped: list[JsonValue],
    counts_choice: bool,
) -> None:
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "no-count-api",
                    "litellm_params": {"model": f"groq/{_TOOL_CHOICE_MODEL}", "api_key": "sk-test"},
                }
            ]
        ),
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "proxy_config", litellm.proxy.proxy_server.ProxyConfig())
    drop_list: Final = {"additional_drop_params": dropped}

    route_added: Final = await route_count({"tool_choice": tool_choice, **drop_list}) - await route_count(drop_list)

    assert (route_added > 0) == counts_choice


@pytest.mark.asyncio
async def test_utils_token_counter_counts_a_bare_prompt_with_a_tool_choice_like_the_prompt_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", _tool_choice_router())

    with_choice: Final = await token_counter(
        request=TokenCountRequest(model="no-count-api", prompt=_TOOL_CHOICE_QUESTION, tool_choice="required"),
        http_request=_json_request({}, "python-httpx/0.28.1"),
        fastapi_response=Response(),
        user_api_key_dict=UserAPIKeyAuth(),
    )
    prompt_alone: Final = await token_counter(
        request=TokenCountRequest(model="no-count-api", prompt=_TOOL_CHOICE_QUESTION),
        http_request=_json_request({}, "python-httpx/0.28.1"),
        fastapi_response=Response(),
        user_api_key_dict=UserAPIKeyAuth(),
    )

    assert (with_choice.error, with_choice.total_tokens) == (False, prompt_alone.total_tokens)
