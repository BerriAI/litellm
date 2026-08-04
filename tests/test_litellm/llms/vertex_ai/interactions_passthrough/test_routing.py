from unittest.mock import MagicMock

from litellm.llms.vertex_ai.interactions_passthrough.id_codec import decode, encode
from litellm.llms.vertex_ai.interactions_passthrough.routing import (
    encode_interaction_response_id,
    resolve_create_project_location,
    rewrite_interaction_input,
)


def _router_with_deployment(vertex_project, vertex_location, model="vertex_ai/gemini-omni-flash-preview"):
    router = MagicMock()
    router.get_available_deployment_for_pass_through.return_value = {
        "litellm_params": {
            "vertex_project": vertex_project,
            "vertex_location": vertex_location,
            "model": model,
        }
    }
    return router


def test_resolves_project_location_from_model():
    router = _router_with_deployment("real-proj", "global")
    result = resolve_create_project_location(
        body={"model": "gemini-omni-flash-preview", "input": []},
        url_project="PLACEHOLDER",
        url_location="global",
        llm_router=router,
    )
    assert result.project == "real-proj"
    assert result.location == "global"
    router.get_available_deployment_for_pass_through.assert_called_once_with(model="gemini-omni-flash-preview")


def test_no_model_falls_back_to_url():
    router = MagicMock()
    result = resolve_create_project_location(
        body={"input": []},
        url_project="url-proj",
        url_location="global",
        llm_router=router,
    )
    assert result.project == "url-proj"
    assert result.location == "global"
    router.get_available_deployment_for_pass_through.assert_not_called()


def test_unknown_model_falls_back_to_url():
    router = MagicMock()
    router.get_available_deployment_for_pass_through.return_value = None
    result = resolve_create_project_location(
        body={"model": "not-configured"},
        url_project="url-proj",
        url_location="us-central1",
        llm_router=router,
    )
    assert result.project == "url-proj"
    assert result.location == "us-central1"


def test_previous_interaction_id_is_decoded_and_used_as_fallback():
    router = MagicMock()
    router.get_available_deployment_for_pass_through.return_value = None
    prev = encode("prev-proj", "global", "video-old")
    result = resolve_create_project_location(
        body={"model": "unknown", "previous_interaction_id": prev},
        url_project="PLACEHOLDER",
        url_location="global",
        llm_router=router,
    )
    assert result.project == "prev-proj"
    assert result.location == "global"
    assert result.body["previous_interaction_id"] == "video-old"


def test_model_resolution_wins_over_previous_interaction_id():
    router = _router_with_deployment("model-proj", "global")
    prev = encode("prev-proj", "us-central1", "video-old")
    result = resolve_create_project_location(
        body={"model": "gemini-omni-flash-preview", "previous_interaction_id": prev},
        url_project="PLACEHOLDER",
        url_location="global",
        llm_router=router,
    )
    assert result.project == "model-proj"
    assert result.body["previous_interaction_id"] == "video-old"


def test_body_is_not_mutated_in_place():
    router = MagicMock()
    router.get_available_deployment_for_pass_through.return_value = None
    prev = encode("prev-proj", "global", "video-old")
    original = {"model": "x", "previous_interaction_id": prev}
    resolve_create_project_location(body=original, url_project="p", url_location="global", llm_router=router)
    assert original["previous_interaction_id"] == prev


def test_input_rewrite_decodes_and_overrides():
    opaque = encode("real-proj", "global", "video-abc")
    endpoint = f"/vertex_ai/v1beta1/projects/PLACEHOLDER/locations/global/interactions/{opaque}"
    result = rewrite_interaction_input(endpoint, url_project="PLACEHOLDER", url_location="global")
    assert result.project == "real-proj"
    assert result.location == "global"
    assert result.endpoint.endswith("/interactions/video-abc")
    assert opaque not in result.endpoint


def test_input_rewrite_preserves_cancel_suffix():
    opaque = encode("real-proj", "us-central1", "video-abc")
    endpoint = f"/vertex_ai/v1beta1/projects/X/locations/global/interactions/{opaque}:cancel"
    result = rewrite_interaction_input(endpoint, url_project="X", url_location="global")
    assert result.project == "real-proj"
    assert result.location == "us-central1"
    assert result.endpoint.endswith("/interactions/video-abc:cancel")


def test_input_rewrite_noop_for_raw_id():
    endpoint = "/vertex_ai/v1beta1/projects/real/locations/global/interactions/video-raw"
    result = rewrite_interaction_input(endpoint, url_project="real", url_location="global")
    assert result.project == "real"
    assert result.location == "global"
    assert result.endpoint == endpoint


def test_input_rewrite_noop_for_create_url_without_id():
    # A collection-level create URL has no interaction id, so the endpoint and the
    # URL project/location are returned untouched.
    endpoint = "/vertex_ai/v1beta1/projects/real/locations/global/interactions"
    result = rewrite_interaction_input(endpoint, url_project="real", url_location="global")
    assert result.project == "real"
    assert result.location == "global"
    assert result.endpoint == endpoint


def test_resolve_falls_back_when_router_lacks_passthrough_method():
    # A router object without get_available_deployment_for_pass_through resolves to URL values.
    router = object()
    result = resolve_create_project_location(
        body={"model": "gemini-omni-flash-preview"},
        url_project="url-proj",
        url_location="global",
        llm_router=router,
    )
    assert (result.project, result.location) == ("url-proj", "global")


def test_resolve_falls_back_when_router_raises():
    router = MagicMock()
    router.get_available_deployment_for_pass_through.side_effect = RuntimeError("boom")
    result = resolve_create_project_location(
        body={"model": "gemini-omni-flash-preview"},
        url_project="url-proj",
        url_location="us-central1",
        llm_router=router,
    )
    assert (result.project, result.location) == ("url-proj", "us-central1")


def test_resolve_falls_back_when_deployment_has_no_litellm_params():
    router = MagicMock()
    router.get_available_deployment_for_pass_through.return_value = {"model_info": {}}
    result = resolve_create_project_location(
        body={"model": "gemini-omni-flash-preview"},
        url_project="url-proj",
        url_location="global",
        llm_router=router,
    )
    assert (result.project, result.location) == ("url-proj", "global")


def test_output_encode_rewrites_top_level_id():
    body = {"id": "video-abc", "status": "in_progress", "object": "interaction"}
    out = encode_interaction_response_id(body, project="real-proj", location="global")
    assert out["status"] == "in_progress"
    decoded = decode(out["id"])
    assert decoded is not None
    assert (decoded.project, decoded.location, decoded.raw_id) == ("real-proj", "global", "video-abc")


def test_output_encode_is_stable_round_trip_with_input():
    opaque = encode("real-proj", "global", "video-abc")
    endpoint = f"/vertex_ai/v1beta1/projects/PLACEHOLDER/locations/global/interactions/{opaque}"
    rewritten = rewrite_interaction_input(endpoint, "PLACEHOLDER", "global")
    body = {"id": "video-abc", "status": "completed"}
    out = encode_interaction_response_id(body, rewritten.project, rewritten.location)
    assert out["id"] == opaque


def test_output_encode_noop_when_no_id():
    body = {"status": "in_progress"}
    out = encode_interaction_response_id(body, "p", "global")
    assert out == {"status": "in_progress"}


def test_output_encode_noop_when_project_none():
    body = {"id": "video-abc"}
    out = encode_interaction_response_id(body, None, "global")
    assert out["id"] == "video-abc"


def test_output_encode_does_not_mutate_input():
    body = {"id": "video-abc", "status": "in_progress"}
    encode_interaction_response_id(body, "p", "global")
    assert body["id"] == "video-abc"
