from collections.abc import Mapping
from types import MappingProxyType
from typing import Final
from unittest.mock import patch

import httpx
import pytest
from fastapi import Request

import litellm.proxy.proxy_server
from litellm.proxy._types import OsoAuthorizationConfig, ProxyException, UserAPIKeyAuth, hash_token
from litellm.proxy.auth.oso_authorization import (
    OsoAuthorizeRequest,
    OsoCloudAuthorizer,
    OsoContextFact,
    OsoValue,
    enforce_oso_model_authorization,
)
from litellm.proxy.auth.user_api_key_auth import (
    _enforce_configured_oso_authorization,
    _run_centralized_common_checks,
)


class RecordingAuthorizer:
    def __init__(self, decisions: tuple[bool, ...] = (True,), error: Exception | None = None) -> None:
        self._decisions: Final = decisions
        self._error: Final = error
        self.requests: list[OsoAuthorizeRequest] = []

    async def authorize(self, request: OsoAuthorizeRequest) -> bool:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return self._decisions[len(self.requests) - 1]


class RecordingHTTPClient:
    def __init__(self, response: httpx.Response) -> None:
        self._response: Final = response
        self.calls: list[tuple[str, dict[str, object], dict[str, str], float]] = []

    async def post(
        self,
        url: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
        timeout: float,
    ) -> httpx.Response:
        self.calls.append((url, json, headers, timeout))
        return self._response


def _enabled_settings() -> Mapping[str, object]:
    return {"oso_authorization": {"enabled": True, "api_key": "oso-test-secret", "timeout": 2.5}}


@pytest.mark.asyncio
async def test_oso_allow_sends_model_and_non_secret_authenticated_identity() -> None:
    authorizer: Final = RecordingAuthorizer()
    raw_litellm_key: Final = "sk-litellm-secret"
    token: Final = UserAPIKeyAuth(
        token=raw_litellm_key,
        user_id="user-alice",
        team_id="team-platform",
        org_id="org-example",
        project_id="project-gateway",
    )

    await enforce_oso_model_authorization(
        general_settings=_enabled_settings(),
        valid_token=token,
        model="gpt-5.4",
        route="/v1/chat/completions",
        request_method="POST",
        authorizer=authorizer,
    )

    assert len(authorizer.requests) == 1
    decision: Final = authorizer.requests[0]
    assert (decision.actor_type, decision.actor_id, decision.action) == ("User", "user-alice", "invoke")
    assert (decision.resource_type, decision.resource_id) == ("Model", "gpt-5.4")
    actor: Final = OsoValue(type="User", id="user-alice")
    assert decision.context_facts == (
        OsoContextFact(predicate="has_relation", args=(actor, "team", OsoValue(type="Team", id="team-platform"))),
        OsoContextFact(
            predicate="has_relation", args=(actor, "organization", OsoValue(type="Organization", id="org-example"))
        ),
        OsoContextFact(
            predicate="has_relation", args=(actor, "project", OsoValue(type="Project", id="project-gateway"))
        ),
    )
    serialized: Final = decision.model_dump_json()
    assert raw_litellm_key not in serialized
    assert "ApiKey" not in serialized
    assert all(secret not in serialized for secret in ("oso-test-secret", "Bearer"))


@pytest.mark.asyncio
async def test_oso_deny_returns_403() -> None:
    with pytest.raises(ProxyException) as exc_info:
        await enforce_oso_model_authorization(
            general_settings=_enabled_settings(),
            valid_token=UserAPIKeyAuth(user_id="user-alice"),
            model="restricted-model",
            route="/v1/responses",
            request_method="POST",
            authorizer=RecordingAuthorizer(decisions=(False,)),
        )

    assert exc_info.value.code == "403"
    assert exc_info.value.param == "model"


@pytest.mark.asyncio
async def test_oso_provider_error_fails_closed_with_503() -> None:
    with pytest.raises(ProxyException) as exc_info:
        await enforce_oso_model_authorization(
            general_settings=_enabled_settings(),
            valid_token=UserAPIKeyAuth(user_id="user-alice"),
            model="gpt-5.4",
            route="/v1/responses",
            request_method="POST",
            authorizer=RecordingAuthorizer(error=httpx.ReadTimeout("Oso timed out")),
        )

    assert exc_info.value.code == "503"


@pytest.mark.asyncio
async def test_oso_missing_identity_fails_closed_without_calling_provider() -> None:
    authorizer: Final = RecordingAuthorizer()
    with pytest.raises(ProxyException) as exc_info:
        await enforce_oso_model_authorization(
            general_settings=_enabled_settings(),
            valid_token=UserAPIKeyAuth(),
            model="gpt-5.4",
            route="/v1/embeddings",
            request_method="POST",
            authorizer=authorizer,
        )

    assert exc_info.value.code == "403"
    assert authorizer.requests == []


@pytest.mark.asyncio
async def test_oso_uses_existing_hashed_key_as_api_key_actor() -> None:
    authorizer: Final = RecordingAuthorizer()
    hashed_key: Final = hash_token("sk-litellm-secret")

    await enforce_oso_model_authorization(
        general_settings=_enabled_settings(),
        valid_token=UserAPIKeyAuth(token=hashed_key),
        model="gpt-5.4",
        route="/v1/embeddings",
        request_method="POST",
        authorizer=authorizer,
    )

    assert tuple((request.actor_type, request.actor_id) for request in authorizer.requests) == (("ApiKey", hashed_key),)


@pytest.mark.asyncio
async def test_oso_disabled_preserves_existing_behavior() -> None:
    authorizer: Final = RecordingAuthorizer(error=AssertionError("disabled integration called Oso"))

    await enforce_oso_model_authorization(
        general_settings={},
        valid_token=UserAPIKeyAuth(),
        model="gpt-5.4",
        route="/v1/chat/completions",
        request_method="POST",
        authorizer=authorizer,
    )

    assert authorizer.requests == []


@pytest.mark.asyncio
async def test_custom_auth_cannot_bypass_enabled_oso_authorization() -> None:
    authorizer: Final = RecordingAuthorizer()
    request: Final = Request(
        scope={
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": (),
            "query_string": b"",
        }
    )
    with patch.multiple(
        litellm.proxy.proxy_server,
        general_settings=_enabled_settings(),
        llm_router=None,
        master_key="sk-test-master",
        user_custom_auth=object(),
    ):
        await _run_centralized_common_checks(
            user_api_key_auth_obj=UserAPIKeyAuth(user_id="user-alice"),
            request=request,
            request_data={"model": "gpt-5.4"},
            route="/v1/chat/completions",
            oso_authorizer=authorizer,
        )

    assert tuple(decision.resource_id for decision in authorizer.requests) == ("gpt-5.4",)


@pytest.mark.asyncio
async def test_oso_requires_every_requested_model_to_be_allowed() -> None:
    authorizer: Final = RecordingAuthorizer(decisions=(True, False, True))
    with pytest.raises(ProxyException) as exc_info:
        await enforce_oso_model_authorization(
            general_settings=_enabled_settings(),
            valid_token=UserAPIKeyAuth(team_id="team-platform"),
            model=["model-a", "model-b", "model-c"],
            route="/cost/predict-cache",
            request_method="POST",
            authorizer=authorizer,
        )

    assert exc_info.value.code == "403"
    assert tuple(request.resource_id for request in authorizer.requests) == ("model-a", "model-b")


@pytest.mark.asyncio
async def test_oso_rejects_list_model_on_single_model_route_without_provider_calls() -> None:
    authorizer: Final = RecordingAuthorizer()
    with pytest.raises(ProxyException) as exc_info:
        await enforce_oso_model_authorization(
            general_settings=_enabled_settings(),
            valid_token=UserAPIKeyAuth(user_id="user-alice"),
            model=["model-a", "model-b"],
            route="/v1/chat/completions",
            request_method="POST",
            authorizer=authorizer,
        )

    assert exc_info.value.code == "400"
    assert authorizer.requests == []


@pytest.mark.asyncio
async def test_oso_rejects_excessive_multi_model_authorization_without_provider_calls() -> None:
    authorizer: Final = RecordingAuthorizer()
    with pytest.raises(ProxyException) as exc_info:
        await enforce_oso_model_authorization(
            general_settings=_enabled_settings(),
            valid_token=UserAPIKeyAuth(user_id="user-alice"),
            model=[f"model-{index}" for index in range(11)],
            route="/cost/predict-cache",
            request_method="POST",
            authorizer=authorizer,
        )

    assert exc_info.value.code == "400"
    assert authorizer.requests == []


@pytest.mark.asyncio
async def test_oso_allows_model_optional_route_without_model() -> None:
    authorizer: Final = RecordingAuthorizer()

    await enforce_oso_model_authorization(
        general_settings=_enabled_settings(),
        valid_token=UserAPIKeyAuth(user_id="user-alice"),
        model=None,
        route="/v1/vector_stores",
        request_method="POST",
        authorizer=authorizer,
    )

    assert authorizer.requests == []


@pytest.mark.asyncio
async def test_oso_preserves_proxy_exception_from_authorizer() -> None:
    expected: Final = ProxyException(message="specific denial", type="auth_error", param="model", code=401)

    with pytest.raises(ProxyException) as exc_info:
        await enforce_oso_model_authorization(
            general_settings=_enabled_settings(),
            valid_token=UserAPIKeyAuth(user_id="user-alice"),
            model="gpt-5.4",
            route="/v1/responses",
            request_method="POST",
            authorizer=RecordingAuthorizer(error=expected),
        )

    assert exc_info.value is expected


@pytest.mark.asyncio
async def test_oso_invalid_enabled_configuration_fails_closed() -> None:
    with pytest.raises(ProxyException) as exc_info:
        await enforce_oso_model_authorization(
            general_settings={"oso_authorization": {"enabled": True}},
            valid_token=UserAPIKeyAuth(user_id="user-alice"),
            model="gpt-5.4",
            route="/v1/responses",
            request_method="POST",
        )

    assert exc_info.value.code == "503"


@pytest.mark.asyncio
async def test_oso_missing_model_fails_closed_on_model_endpoint() -> None:
    authorizer: Final = RecordingAuthorizer()
    with pytest.raises(ProxyException) as exc_info:
        await enforce_oso_model_authorization(
            general_settings=_enabled_settings(),
            valid_token=UserAPIKeyAuth(user_id="user-alice"),
            model=None,
            route="/v1/responses",
            request_method="POST",
            authorizer=authorizer,
        )

    assert exc_info.value.code == "403"
    assert authorizer.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route",
    (
        "/v1/chat/completions",
        "/v1/responses",
        "/v1/messages",
        "/v1/embeddings",
        "/v1/images/generations",
        "/v1/audio/speech",
        "/v1/audio/transcriptions",
        "/v1/moderations",
    ),
)
async def test_major_model_endpoints_share_oso_authorization(route: str) -> None:
    authorizer: Final = RecordingAuthorizer()
    request: Final = Request(
        scope={"type": "http", "method": "POST", "path": route, "headers": (), "query_string": b""}
    )

    await _enforce_configured_oso_authorization(
        user_api_key_auth_obj=UserAPIKeyAuth(user_id="user-alice"),
        request=request,
        request_data={"model": "model-for-route"},
        route=route,
        general_settings=_enabled_settings(),
        llm_router=None,
        oso_authorizer=authorizer,
    )

    assert tuple(decision.resource_id for decision in authorizer.requests) == ("model-for-route",)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "settings_key", "configured_model"),
    (
        ("/v1/chat/completions", "completion_model", "server-chat-model"),
        ("/v1/images/generations", "image_generation_model", "server-image-model"),
        ("/v1/images/edits", "image_generation_model", "server-image-edit-model"),
        ("/v1/moderations", "moderation_model", "server-moderation-model"),
        ("/v1/audio/transcriptions", "moderation_model", "server-audio-model"),
    ),
)
async def test_oso_authorizes_server_selected_model(
    route: str,
    settings_key: str,
    configured_model: str,
) -> None:
    authorizer: Final = RecordingAuthorizer()
    request: Final = Request(
        scope={"type": "http", "method": "POST", "path": route, "headers": (), "query_string": b""}
    )

    await _enforce_configured_oso_authorization(
        user_api_key_auth_obj=UserAPIKeyAuth(user_id="user-alice"),
        request=request,
        request_data={"model": "client-decoy-model"},
        route=route,
        general_settings={**_enabled_settings(), settings_key: configured_model},
        llm_router=None,
        oso_authorizer=authorizer,
    )

    assert tuple(decision.resource_id for decision in authorizer.requests) == (configured_model,)


@pytest.mark.asyncio
async def test_oso_image_query_model_takes_precedence_over_image_default() -> None:
    authorizer: Final = RecordingAuthorizer()
    request: Final = Request(
        scope={
            "type": "http",
            "method": "POST",
            "path": "/v1/images/generations",
            "headers": (),
            "query_string": b"model=query-model",
        }
    )

    await _enforce_configured_oso_authorization(
        user_api_key_auth_obj=UserAPIKeyAuth(user_id="user-alice"),
        request=request,
        request_data={"model": "body-model"},
        route="/v1/images/generations",
        general_settings={**_enabled_settings(), "image_generation_model": "server-image-model"},
        llm_router=None,
        oso_authorizer=authorizer,
    )

    assert tuple(decision.resource_id for decision in authorizer.requests) == ("query-model",)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "settings", "expected_model"),
    (
        (
            "/v1/images/generations",
            MappingProxyType({"image_generation_model": "server-image-model"}),
            "cli-model",
        ),
        (
            "/v1/images/edits",
            MappingProxyType(
                {"completion_model": "server-completion-model", "image_generation_model": "server-image-model"}
            ),
            "server-completion-model",
        ),
    ),
)
async def test_oso_image_model_precedence_matches_final_dispatch(
    route: str,
    settings: Mapping[str, str],
    expected_model: str,
) -> None:
    authorizer: Final = RecordingAuthorizer()
    request: Final = Request(
        scope={
            "type": "http",
            "method": "POST",
            "path": route,
            "headers": (),
            "query_string": b"model=query-model",
        }
    )
    with patch.object(litellm.proxy.proxy_server, "user_model", "cli-model"):
        await _enforce_configured_oso_authorization(
            user_api_key_auth_obj=UserAPIKeyAuth(user_id="user-alice"),
            request=request,
            request_data={"model": "body-model"},
            route=route,
            general_settings={**_enabled_settings(), **settings},
            llm_router=None,
            oso_authorizer=authorizer,
        )

    assert tuple(decision.resource_id for decision in authorizer.requests) == (expected_model,)


@pytest.mark.asyncio
async def test_oso_speech_authorizes_cli_model_without_undispatched_key_alias() -> None:
    authorizer: Final = RecordingAuthorizer()
    request: Final = Request(
        scope={"type": "http", "method": "POST", "path": "/v1/audio/speech", "headers": (), "query_string": b""}
    )
    with patch.object(litellm.proxy.proxy_server, "user_model", "cli-model"):
        await _enforce_configured_oso_authorization(
            user_api_key_auth_obj=UserAPIKeyAuth(user_id="user-alice", aliases={"cli-model": "aliased-model"}),
            request=request,
            request_data={"model": "client-decoy-model"},
            route="/v1/audio/speech",
            general_settings=_enabled_settings(),
            llm_router=None,
            oso_authorizer=authorizer,
        )

    assert tuple(decision.resource_id for decision in authorizer.requests) == ("cli-model",)


@pytest.mark.asyncio
async def test_oso_chat_authorizes_key_alias_used_for_dispatch() -> None:
    authorizer: Final = RecordingAuthorizer()
    request: Final = Request(
        scope={"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": (), "query_string": b""}
    )

    await _enforce_configured_oso_authorization(
        user_api_key_auth_obj=UserAPIKeyAuth(user_id="user-alice", aliases={"body-model": "aliased-model"}),
        request=request,
        request_data={"model": "body-model"},
        route="/v1/chat/completions",
        general_settings=_enabled_settings(),
        llm_router=None,
        oso_authorizer=authorizer,
    )

    assert tuple(decision.resource_id for decision in authorizer.requests) == ("aliased-model",)


@pytest.mark.asyncio
async def test_oso_image_generation_applies_global_but_not_key_alias() -> None:
    authorizer: Final = RecordingAuthorizer()
    request: Final = Request(
        scope={"type": "http", "method": "POST", "path": "/v1/images/generations", "headers": (), "query_string": b""}
    )
    with patch.object(litellm, "model_alias_map", {"body-model": "global-alias"}):
        await _enforce_configured_oso_authorization(
            user_api_key_auth_obj=UserAPIKeyAuth(user_id="user-alice", aliases={"global-alias": "key-alias"}),
            request=request,
            request_data={"model": "body-model"},
            route="/v1/images/generations",
            general_settings=_enabled_settings(),
            llm_router=None,
            oso_authorizer=authorizer,
        )

    assert tuple(decision.resource_id for decision in authorizer.requests) == ("global-alias",)


@pytest.mark.asyncio
async def test_oso_audio_without_any_effective_model_fails_closed() -> None:
    request: Final = Request(
        scope={
            "type": "http",
            "method": "POST",
            "path": "/v1/audio/transcriptions",
            "headers": (),
            "query_string": b"",
        }
    )

    with pytest.raises(ProxyException) as exc_info:
        await _enforce_configured_oso_authorization(
            user_api_key_auth_obj=UserAPIKeyAuth(user_id="user-alice"),
            request=request,
            request_data={},
            route="/v1/audio/transcriptions",
            general_settings=_enabled_settings(),
            llm_router=None,
            oso_authorizer=RecordingAuthorizer(),
        )

    assert exc_info.value.code == "403"


@pytest.mark.asyncio
async def test_oso_cloud_authorizer_uses_documented_check_api_shape() -> None:
    response: Final = httpx.Response(200, json={"allowed": True})
    http_client: Final = RecordingHTTPClient(response)
    config: Final = OsoAuthorizationConfig(enabled=True, api_key="oso-test-secret", timeout=2.5)
    authorizer: Final = OsoCloudAuthorizer(config=config, http_client=http_client)
    request: Final = OsoAuthorizeRequest(
        actor_type="User",
        actor_id="user-alice",
        action="invoke",
        resource_type="Model",
        resource_id="gpt-5.4",
        context_facts=(),
    )

    allowed: Final = await authorizer.authorize(request)

    assert allowed is True
    assert http_client.calls == [
        (
            "https://api.osohq.com/api/authorize",
            {
                "actor_type": "User",
                "actor_id": "user-alice",
                "action": "invoke",
                "resource_type": "Model",
                "resource_id": "gpt-5.4",
                "context_facts": [],
            },
            {"Authorization": "Bearer oso-test-secret", "Content-Type": "application/json"},
            2.5,
        )
    ]


@pytest.mark.asyncio
async def test_oso_cloud_authorizer_rejects_missing_api_key_before_network_call() -> None:
    http_client: Final = RecordingHTTPClient(httpx.Response(200, json={"allowed": True}))
    authorizer: Final = OsoCloudAuthorizer(config=OsoAuthorizationConfig(), http_client=http_client)
    request: Final = OsoAuthorizeRequest(
        actor_type="User",
        actor_id="user-alice",
        action="invoke",
        resource_type="Model",
        resource_id="gpt-5.4",
        context_facts=(),
    )

    with pytest.raises(ValueError, match="API key"):
        await authorizer.authorize(request)

    assert http_client.calls == []
