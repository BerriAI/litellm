import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Response

from litellm.llms.vertex_ai.interactions_passthrough.id_codec import decode, encode
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    _base_vertex_proxy_route,
    _encode_interaction_response,
    _resolve_interactions_input_routing,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    LITELLM_PASS_THROUGH_CUSTOM_BODY_STATE_KEY,
)


def _mock_request(method: str, body: dict | None):
    request = MagicMock()
    request.method = method
    request.headers = {}
    state = MagicMock()
    request.state = state
    request.json = AsyncMock(return_value=(body or {}))
    return request, state


def _mock_router(vertex_project, vertex_location):
    router = MagicMock()
    router.get_available_deployment_for_pass_through.return_value = {
        "litellm_params": {
            "vertex_project": vertex_project,
            "vertex_location": vertex_location,
            "model": "vertex_ai/gemini-omni-flash-preview",
        }
    }
    return router


def _make_response(payload: dict) -> Response:
    return Response(content=json.dumps(payload), status_code=200, media_type="application/json")


@pytest.mark.asyncio
async def test_create_auto_routes_and_encodes_id():
    request, state = _mock_request("POST", {"model": "gemini-omni-flash-preview", "background": True})
    fastapi_response = MagicMock()
    handler = MagicMock()
    handler.get_default_base_target_url.return_value = "https://aiplatform.googleapis.com"
    router = _mock_router("real-proj", "global")

    captured = {}

    def fake_create_pass_through_route(endpoint, target, custom_headers, is_streaming_request):
        captured["target"] = target

        async def endpoint_func(request, fastapi_response, user_api_key_dict):
            return _make_response({"id": "video-abc", "status": "in_progress"})

        return endpoint_func

    async def _echo_prep_headers(**kwargs):
        return (
            {},
            "https://aiplatform.googleapis.com",
            False,
            kwargs["vertex_project"],
            kwargs["vertex_location"],
        )

    endpoint = "/vertex_ai/v1beta1/projects/PLACEHOLDER/locations/global/interactions"

    with (
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router"
        ) as mock_pt_router,
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
            side_effect=fake_create_pass_through_route,
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_vertex_auth_headers",
            new=AsyncMock(side_effect=_echo_prep_headers),
        ),
        patch("litellm.proxy.proxy_server.llm_router", new=router),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            new={"vertex_interactions_passthrough_auto_routing": True},
        ),
    ):
        mock_pt_router.get_vertex_credentials.return_value = MagicMock()
        result = await _base_vertex_proxy_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            get_vertex_pass_through_handler=handler,
        )

    assert "projects/real-proj/" in captured["target"]
    payload = json.loads(bytes(result.body))
    decoded = decode(payload["id"])
    assert decoded is not None
    assert (decoded.project, decoded.location, decoded.raw_id) == ("real-proj", "global", "video-abc")


@pytest.mark.asyncio
async def test_create_short_form_url_without_project_auto_routes():
    # Short form like generateContent: no projects/locations in the URL at all.
    request, state = _mock_request("POST", {"model": "gemini-omni-flash-preview", "background": True})
    fastapi_response = MagicMock()
    handler = MagicMock()
    handler.get_default_base_target_url.return_value = "https://aiplatform.googleapis.com"
    router = _mock_router("real-proj", "global")

    captured = {}

    def fake_create_pass_through_route(endpoint, target, custom_headers, is_streaming_request):
        captured["target"] = target

        async def endpoint_func(request, fastapi_response, user_api_key_dict):
            return _make_response({"id": "video-abc", "status": "in_progress"})

        return endpoint_func

    async def _echo_prep_headers(**kwargs):
        return (
            {},
            "https://aiplatform.googleapis.com",
            False,
            kwargs["vertex_project"],
            kwargs["vertex_location"],
        )

    endpoint = "v1beta1/interactions"

    with (
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router"
        ) as mock_pt_router,
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
            side_effect=fake_create_pass_through_route,
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_vertex_auth_headers",
            new=AsyncMock(side_effect=_echo_prep_headers),
        ),
        patch("litellm.proxy.proxy_server.llm_router", new=router),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            new={"vertex_interactions_passthrough_auto_routing": True},
        ),
    ):
        mock_pt_router.get_vertex_credentials.return_value = MagicMock()
        result = await _base_vertex_proxy_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            get_vertex_pass_through_handler=handler,
        )

    # litellm filled in the resolved project/location for the projectless URL.
    assert "projects/real-proj/locations/global/interactions" in captured["target"]
    payload = json.loads(bytes(result.body))
    decoded = decode(payload["id"])
    assert decoded is not None
    assert (decoded.project, decoded.location, decoded.raw_id) == ("real-proj", "global", "video-abc")


@pytest.mark.asyncio
async def test_get_short_form_url_routes_back_via_opaque_id():
    # Short form get: no projects/locations; the opaque id supplies them.
    opaque = encode("real-proj", "global", "video-abc")
    request, state = _mock_request("GET", None)
    fastapi_response = MagicMock()
    handler = MagicMock()
    handler.get_default_base_target_url.return_value = "https://aiplatform.googleapis.com"

    captured = {}

    def fake_create_pass_through_route(endpoint, target, custom_headers, is_streaming_request):
        captured["target"] = target

        async def endpoint_func(request, fastapi_response, user_api_key_dict):
            return _make_response({"id": "video-abc", "status": "completed"})

        return endpoint_func

    async def _echo_prep_headers(**kwargs):
        return (
            {},
            "https://aiplatform.googleapis.com",
            False,
            kwargs["vertex_project"],
            kwargs["vertex_location"],
        )

    endpoint = f"v1beta1/interactions/{opaque}"

    with (
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router"
        ) as mock_pt_router,
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
            side_effect=fake_create_pass_through_route,
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_vertex_auth_headers",
            new=AsyncMock(side_effect=_echo_prep_headers),
        ),
        patch("litellm.proxy.proxy_server.llm_router", new=MagicMock()),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            new={"vertex_interactions_passthrough_auto_routing": True},
        ),
    ):
        mock_pt_router.get_vertex_credentials.return_value = MagicMock()
        result = await _base_vertex_proxy_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            get_vertex_pass_through_handler=handler,
        )

    # Routed to the decoded project with the raw id, project filled in by litellm.
    assert "projects/real-proj/locations/global/interactions/video-abc" in captured["target"]
    payload = json.loads(bytes(result.body))
    assert payload["id"] == opaque


@pytest.mark.asyncio
async def test_get_routes_back_via_opaque_id_ignoring_url_project():
    opaque = encode("real-proj", "global", "video-abc")
    request, state = _mock_request("GET", None)
    fastapi_response = MagicMock()
    handler = MagicMock()
    handler.get_default_base_target_url.return_value = "https://aiplatform.googleapis.com"

    captured = {}

    def fake_create_pass_through_route(endpoint, target, custom_headers, is_streaming_request):
        captured["target"] = target

        async def endpoint_func(request, fastapi_response, user_api_key_dict):
            return _make_response({"id": "video-abc", "status": "completed"})

        return endpoint_func

    async def _echo_prep_headers(**kwargs):
        return (
            {},
            "https://aiplatform.googleapis.com",
            False,
            kwargs["vertex_project"],
            kwargs["vertex_location"],
        )

    endpoint = f"/vertex_ai/v1beta1/projects/PLACEHOLDER/locations/global/interactions/{opaque}"

    with (
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router"
        ) as mock_pt_router,
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
            side_effect=fake_create_pass_through_route,
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_vertex_auth_headers",
            new=AsyncMock(side_effect=_echo_prep_headers),
        ),
        patch("litellm.proxy.proxy_server.llm_router", new=MagicMock()),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            new={"vertex_interactions_passthrough_auto_routing": True},
        ),
    ):
        mock_pt_router.get_vertex_credentials.return_value = MagicMock()
        result = await _base_vertex_proxy_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            get_vertex_pass_through_handler=handler,
        )

    assert "projects/real-proj/" in captured["target"]
    assert "interactions/video-abc" in captured["target"]
    payload = json.loads(bytes(result.body))
    assert payload["id"] == opaque


@pytest.mark.asyncio
async def test_flag_off_leaves_url_untouched():
    request, state = _mock_request("POST", {"model": "gemini-omni-flash-preview"})
    fastapi_response = MagicMock()
    handler = MagicMock()
    handler.get_default_base_target_url.return_value = "https://aiplatform.googleapis.com"
    router = _mock_router("real-proj", "global")

    captured = {}

    def fake_create_pass_through_route(endpoint, target, custom_headers, is_streaming_request):
        captured["target"] = target

        async def endpoint_func(request, fastapi_response, user_api_key_dict):
            return _make_response({"id": "video-abc", "status": "in_progress"})

        return endpoint_func

    async def _echo_prep_headers(**kwargs):
        return (
            {},
            "https://aiplatform.googleapis.com",
            False,
            kwargs["vertex_project"],
            kwargs["vertex_location"],
        )

    endpoint = "/vertex_ai/v1beta1/projects/PLACEHOLDER/locations/global/interactions"

    with (
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router"
        ) as mock_pt_router,
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
            side_effect=fake_create_pass_through_route,
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_vertex_auth_headers",
            new=AsyncMock(side_effect=_echo_prep_headers),
        ),
        patch("litellm.proxy.proxy_server.llm_router", new=router),
        patch("litellm.proxy.proxy_server.general_settings", new={}),
    ):
        mock_pt_router.get_vertex_credentials.return_value = MagicMock()
        result = await _base_vertex_proxy_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            get_vertex_pass_through_handler=handler,
        )

    assert "projects/PLACEHOLDER/" in captured["target"]
    payload = json.loads(bytes(result.body))
    assert payload["id"] == "video-abc"
    assert decode(payload["id"]) is None


@pytest.mark.asyncio
async def test_create_response_preserves_upstream_headers():
    request, state = _mock_request("POST", {"model": "gemini-omni-flash-preview"})
    fastapi_response = MagicMock()
    handler = MagicMock()
    handler.get_default_base_target_url.return_value = "https://aiplatform.googleapis.com"
    router = _mock_router("real-proj", "global")

    def fake_create_pass_through_route(endpoint, target, custom_headers, is_streaming_request):
        async def endpoint_func(request, fastapi_response, user_api_key_dict):
            return Response(
                content=json.dumps({"id": "video-abc", "status": "in_progress"}),
                status_code=200,
                media_type="application/json",
                headers={"x-goog-request-id": "trace-123"},
            )

        return endpoint_func

    async def _echo_prep_headers(**kwargs):
        return (
            {},
            "https://aiplatform.googleapis.com",
            False,
            kwargs["vertex_project"],
            kwargs["vertex_location"],
        )

    endpoint = "/vertex_ai/v1beta1/projects/PLACEHOLDER/locations/global/interactions"

    with (
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router"
        ) as mock_pt_router,
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
            side_effect=fake_create_pass_through_route,
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_vertex_auth_headers",
            new=AsyncMock(side_effect=_echo_prep_headers),
        ),
        patch("litellm.proxy.proxy_server.llm_router", new=router),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            new={"vertex_interactions_passthrough_auto_routing": True},
        ),
    ):
        mock_pt_router.get_vertex_credentials.return_value = MagicMock()
        result = await _base_vertex_proxy_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            get_vertex_pass_through_handler=handler,
        )

    assert result.headers["x-goog-request-id"] == "trace-123"
    assert int(result.headers["content-length"]) == len(bytes(result.body))
    payload = json.loads(bytes(result.body))
    assert decode(payload["id"]) is not None


@pytest.mark.asyncio
async def test_get_round_trip_through_real_prepare_auth_headers():
    from litellm.types.passthrough_endpoints.vertex_ai import (
        VertexPassThroughCredentials,
    )

    opaque = encode("real-proj", "global", "video-abc")
    request, state = _mock_request("GET", None)
    fastapi_response = MagicMock()
    handler = MagicMock()
    handler.get_default_base_target_url.return_value = "https://aiplatform.googleapis.com"
    handler.update_base_target_url_with_credential_location.side_effect = lambda base_url, location: base_url

    captured = {}

    def fake_create_pass_through_route(endpoint, target, custom_headers, is_streaming_request):
        captured["target"] = target

        async def endpoint_func(request, fastapi_response, user_api_key_dict):
            return _make_response({"id": "video-abc", "status": "completed"})

        return endpoint_func

    endpoint = f"/vertex_ai/v1beta1/projects/PLACEHOLDER/locations/global/interactions/{opaque}"

    with (
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router"
        ) as mock_pt_router,
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
            side_effect=fake_create_pass_through_route,
        ),
        patch(
            "litellm.llms.vertex_ai.vertex_llm_base.VertexBase._ensure_access_token_async",
            new=AsyncMock(side_effect=lambda credentials, project_id, custom_llm_provider: ("tok", project_id)),
        ),
        patch(
            "litellm.llms.vertex_ai.vertex_llm_base.VertexBase._get_token_and_url",
            new=MagicMock(return_value=("tok", "https://aiplatform.googleapis.com")),
        ),
        patch("litellm.proxy.proxy_server.llm_router", new=MagicMock()),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            new={"vertex_interactions_passthrough_auto_routing": True},
        ),
    ):
        # A credential registered for the decoded project: its vertex_project
        # equals the lookup key, so _prepare_vertex_auth_headers is idempotent.
        mock_pt_router.get_vertex_credentials.return_value = VertexPassThroughCredentials(
            vertex_project="real-proj",
            vertex_location="global",
            vertex_credentials="/fake/creds.json",
        )
        result = await _base_vertex_proxy_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            get_vertex_pass_through_handler=handler,
        )

    # Credentials were looked up by the DECODED project, not the placeholder URL.
    assert mock_pt_router.get_vertex_credentials.call_args.kwargs["project_id"] == "real-proj"
    # Routed to the decoded project with the raw id, through the real auth-header path.
    assert "projects/real-proj/" in captured["target"]
    assert "interactions/video-abc" in captured["target"]
    # Response re-encodes to the SAME opaque id the caller sent (stable polling).
    payload = json.loads(bytes(result.body))
    assert payload["id"] == opaque


@pytest.mark.asyncio
async def test_streaming_create_response_passes_through_untouched():
    from fastapi.responses import StreamingResponse

    request, state = _mock_request("POST", {"model": "gemini-omni-flash-preview", "stream": True})
    fastapi_response = MagicMock()
    handler = MagicMock()
    handler.get_default_base_target_url.return_value = "https://aiplatform.googleapis.com"
    router = _mock_router("real-proj", "global")

    captured = {}
    sentinel = StreamingResponse(iter([b"data: {}\n\n"]), media_type="text/event-stream")

    def fake_create_pass_through_route(endpoint, target, custom_headers, is_streaming_request):
        captured["target"] = target

        async def endpoint_func(request, fastapi_response, user_api_key_dict):
            return sentinel

        return endpoint_func

    async def _echo_prep_headers(**kwargs):
        return (
            {},
            "https://aiplatform.googleapis.com",
            False,
            kwargs["vertex_project"],
            kwargs["vertex_location"],
        )

    endpoint = "/vertex_ai/v1beta1/projects/PLACEHOLDER/locations/global/interactions"

    with (
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router"
        ) as mock_pt_router,
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
            side_effect=fake_create_pass_through_route,
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_vertex_auth_headers",
            new=AsyncMock(side_effect=_echo_prep_headers),
        ),
        patch("litellm.proxy.proxy_server.llm_router", new=router),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            new={"vertex_interactions_passthrough_auto_routing": True},
        ),
    ):
        mock_pt_router.get_vertex_credentials.return_value = MagicMock()
        result = await _base_vertex_proxy_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            get_vertex_pass_through_handler=handler,
        )

    # Input-side model resolution still applied for a streaming create.
    assert "projects/real-proj/" in captured["target"]
    # A StreamingResponse has no `.body`, so the output encode degrades to a
    # no-op and the streaming response is returned untouched (never buffered).
    assert result is sentinel


def test_encode_response_noop_for_non_response_return():
    obj = object()
    assert _encode_interaction_response(obj, "proj", "global") is obj


def test_encode_response_noop_for_non_json_body():
    resp = Response(content=b"not json", status_code=200, media_type="text/plain")
    assert _encode_interaction_response(resp, "proj", "global") is resp


def test_encode_response_noop_for_non_dict_json_body():
    resp = Response(content=json.dumps([1, 2, 3]), status_code=200, media_type="application/json")
    assert _encode_interaction_response(resp, "proj", "global") is resp


def test_encode_response_noop_when_no_id_field():
    resp = Response(content=json.dumps({"status": "in_progress"}), status_code=200, media_type="application/json")
    assert _encode_interaction_response(resp, "proj", "global") is resp


@pytest.mark.asyncio
async def test_resolve_input_routing_body_read_error_falls_back_to_url():
    request = MagicMock()
    request.method = "POST"
    request.state = MagicMock()
    request.json = AsyncMock(side_effect=ValueError("bad body"))
    endpoint = "/vertex_ai/v1beta1/projects/url-proj/locations/global/interactions"

    new_endpoint, project, location = await _resolve_interactions_input_routing(
        endpoint=endpoint,
        request=request,
        vertex_project="url-proj",
        vertex_location="global",
        llm_router=MagicMock(),
    )
    assert new_endpoint == endpoint
    assert (project, location) == ("url-proj", "global")


@pytest.mark.asyncio
async def test_resolve_input_routing_non_post_non_id_url_is_untouched():
    request = MagicMock()
    request.method = "DELETE"
    request.state = MagicMock()
    # A DELETE on the collection-level URL (no id): neither branch applies, values pass through.
    endpoint = "/vertex_ai/v1beta1/projects/url-proj/locations/global/interactions"

    new_endpoint, project, location = await _resolve_interactions_input_routing(
        endpoint=endpoint,
        request=request,
        vertex_project="url-proj",
        vertex_location="global",
        llm_router=MagicMock(),
    )
    assert new_endpoint == endpoint
    assert (project, location) == ("url-proj", "global")


@pytest.mark.asyncio
async def test_create_forwards_decoded_previous_interaction_id_upstream():
    # A caller passes back an opaque previous_interaction_id it received earlier.
    # The body forwarded upstream must carry the DECODED raw id, not the opaque
    # string, or Vertex cannot parse it. The rewritten body is stashed on
    # request.state under LITELLM_PASS_THROUGH_CUSTOM_BODY_STATE_KEY, which is what
    # pass_through_request forwards.
    prev_opaque = encode("real-proj", "global", "video-prev")
    request, state = _mock_request(
        "POST",
        {"model": "gemini-omni-flash-preview", "previous_interaction_id": prev_opaque},
    )

    new_endpoint, project, location = await _resolve_interactions_input_routing(
        endpoint="v1beta1/interactions",
        request=request,
        vertex_project=None,
        vertex_location=None,
        llm_router=_mock_router("real-proj", "global"),
    )

    # The body handed to pass_through_request carries the raw id, not the opaque one.
    forwarded_body = getattr(state, LITELLM_PASS_THROUGH_CUSTOM_BODY_STATE_KEY)
    assert forwarded_body["previous_interaction_id"] == "video-prev"
    assert forwarded_body["previous_interaction_id"] != prev_opaque
    # Model resolution still drives project/location on create.
    assert (project, location) == ("real-proj", "global")
