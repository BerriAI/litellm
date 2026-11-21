"""Vertex AI Model Garden, dedicated endpoints and Private Service Connect (PSC) endpoint URL shapes.

Three routes address a model by an endpoint rather than a publisher model:

- Model Garden OpenAI compatible models (`vertex_ai/openai/<endpoint id>` for a self deployed endpoint,
  `vertex_ai/xai/<model>` for a catalog publisher model) go to `VertexAIModelGardenModels.completion`
  (`litellm/llms/vertex_ai/vertex_model_garden/main.py`). With an `api_base` it skips `_check_custom_proxy`
  and lets `OpenAILikeChatHandler` append `/chat/completions` to the base. A single segment endpoint id
  is carried by the URL and the body `model` is the empty string; a `publisher/model` id stays in the body.
- A numeric model with an `api_base` (`vertex_ai/<digits>`) is routed to the Gemini HTTP path
  (`get_vertex_ai_model_route`, `litellm/llms/vertex_ai/common_utils.py:336`) and becomes
  `.../endpoints/<id>:generateContent` (`get_vertex_url`, `common_utils.py:613`).
- `use_psc_endpoint_format: true` on such a deployment builds
  `{api_base}/v1/projects/{project}/locations/{location}/endpoints/{id}:{method}` even when the `api_base`
  has its own path (`_check_custom_proxy`, `litellm/llms/vertex_ai/vertex_llm_base.py:822`); without it a
  path-bearing `api_base` is taken as the full resource and only `:{method}` is appended.

These shapes are what customers with private networking depend on, and every one of them is built by
string rules a Rust port has to reproduce exactly. The `google_genai` adapter routes for Model Garden are
already covered by `test_google_genai_vertex_routes_wire.py`.
"""

from __future__ import annotations

import json
from typing import Final

from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply
from integration.providers.dec1_vertex3_support import (
    GARDEN_ENDPOINT_ID,
    GARDEN_PUBLISHER,
    PSC_ENDPOINT_ID,
    is_token_from,
    location_path,
    openai_echo,
    openai_reply,
)
from integration.providers.dec1_vertex_support import (
    LOCATION,
    body_of,
    gemini_echo,
    marker,
    message_of,
    path_of,
    query_of,
    vertex,
)
from pydantic import JsonValue

_GARDEN_MODEL: Final = f"openai/{GARDEN_ENDPOINT_ID}"
_DEDICATED_PATH: Final = f"{location_path(LOCATION, version='v1beta1')}/endpoints/{GARDEN_ENDPOINT_ID}"
_PSC_ENDPOINT: Final = f"{location_path(LOCATION)}/endpoints/{PSC_ENDPOINT_ID}"


# ---------------------------------------------------------------- Model Garden (OpenAI compatible)


def test_a_model_garden_endpoint_id_posts_to_chat_completions_under_the_api_base(gateway: Gateway) -> None:
    """`vertex_ai/openai/5464397967697903616` with a bare host `api_base` posts once to `/chat/completions`
    on that host: the Model Garden handler uses the `api_base` as the OpenAI base and
    `OpenAILike._validate_environment` appends `chat/completions`. No `:predict` or `:rawPredict` is used."""
    with vertex(gateway, openai_echo, backend=_GARDEN_MODEL) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        call: Final = rig.peer.one_call()
        assert (path_of(call), query_of(call)) == ("/chat/completions", "")


def test_a_dedicated_endpoint_api_base_keeps_its_path_and_gets_chat_completions(gateway: Gateway) -> None:
    """A dedicated endpoint `api_base` of `<peer>/v1beta1/projects/.../endpoints/5464397967697903616` is used
    unchanged with `/chat/completions` appended, so the call lands on
    `/v1beta1/projects/dec1-vertex-project/locations/us-central1/endpoints/5464397967697903616/chat/completions`.
    The comment in `vertex_model_garden/main.py` says why: the `:verb` rewrite of `_check_custom_proxy`
    corrupted these URLs."""
    with vertex(gateway, openai_echo, backend=_GARDEN_MODEL, base_suffix=_DEDICATED_PATH) as rig:
        assert rig.chat().status_code == 200
        assert path_of(rig.peer.one_call()) == f"{_DEDICATED_PATH}/chat/completions"


def test_a_model_garden_endpoint_id_sends_an_empty_body_model(gateway: Gateway) -> None:
    """For a single segment endpoint id the model lives in the URL, so the body carries `model: ""`
    (`vertex_model_garden_model_id_in_json_body` is false and the handler blanks `model`). Sending the id,
    or `openai/<id>`, in the body is rejected by self deployed vLLM servers."""
    with vertex(gateway, openai_echo, backend=_GARDEN_MODEL) as rig:
        assert rig.chat().status_code == 200
        assert rig.peer.one_body()["model"] == ""


def test_a_model_garden_publisher_model_keeps_its_id_in_the_body(gateway: Gateway) -> None:
    """`vertex_ai/xai/grok-4.1-fast-non-reasoning` routes to Model Garden (the `xai/` prefix) and, because
    the id has a `/`, the body `model` is the full `xai/grok-4.1-fast-non-reasoning`: catalog models share
    one OpenAPI endpoint and are told apart by the body. `get_vertex_base_model_name` only strips routing
    prefixes such as `openai/`, never `xai/`."""
    with vertex(gateway, openai_echo, backend=GARDEN_PUBLISHER) as rig:
        assert rig.chat().status_code == 200
        call: Final = rig.peer.one_call()
        assert path_of(call) == "/chat/completions"
        assert body_of(call)["model"] == GARDEN_PUBLISHER


def test_a_model_garden_call_carries_the_minted_google_token(gateway: Gateway) -> None:
    """The Model Garden call is authenticated with `Authorization: Bearer <minted token>` from the
    deployment's service account; the empty `api_key` is never sent."""
    with vertex(gateway, openai_echo, backend=_GARDEN_MODEL) as rig:
        assert rig.chat().status_code == 200
        assert is_token_from(rig.peer.plan, rig.peer.one_call())


def test_a_model_garden_answer_is_returned_as_an_openai_message(gateway: Gateway) -> None:
    """The vLLM style OpenAI answer comes back unchanged in shape: the assistant text and the upstream
    `usage` (11 prompt, 7 completion tokens)."""
    with vertex(gateway, lambda _r: openai_reply(text="garden reply"), backend=_GARDEN_MODEL) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert message_of(response)["content"] == "garden reply"
        usage: Final = object_value(response.json()["usage"])
        assert (usage["prompt_tokens"], usage["completion_tokens"]) == (11, 7)


def test_a_self_deployed_endpoint_drops_params_its_server_cannot_take(gateway: Gateway) -> None:
    """With `drop_params: true`, `store`, `service_tier` and `prompt_cache_key` are removed before a self
    deployed Model Garden endpoint is called (`VERTEX_SELF_DEPLOYED_ENDPOINT_UNSUPPORTED_PARAMS`,
    `litellm/llms/vertex_ai/common_utils.py:94`), while `temperature` is still forwarded. A vLLM server
    answers 400 to the unknown OpenAI fields."""
    with vertex(gateway, openai_echo, backend=_GARDEN_MODEL) as rig:
        response: Final = rig.chat(
            store=True, service_tier="auto", prompt_cache_key=f"k-{marker()}", temperature=0.2, drop_params=True
        )
        assert response.status_code == 200, response.text
        body: Final = rig.peer.one_body()
        assert {"store", "service_tier", "prompt_cache_key"}.isdisjoint(body), body
        assert body["temperature"] == 0.2


def test_a_model_garden_streaming_call_relays_openai_chunks(gateway: Gateway) -> None:
    """`stream: true` on a Model Garden endpoint posts to the same `/chat/completions` with `stream: true`
    in the body and no `alt=sse` query, and the OpenAI SSE frames are relayed ending with `[DONE]`."""
    with vertex(gateway, openai_echo, backend=_GARDEN_MODEL) as rig:
        response, text = rig.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": marker()}]})
        assert response.status_code == 200
        call: Final = rig.peer.one_call()
        assert (path_of(call), query_of(call)) == ("/chat/completions", "")
        assert body_of(call)["stream"] is True
        assert "data: [DONE]" in text


# ---------------------------------------------------------------- numeric endpoints on the Gemini path (PSC)


def test_a_numeric_endpoint_with_an_api_base_posts_gemini_generate_content_to_endpoints(gateway: Gateway) -> None:
    """`vertex_ai/4965075652664360960` with a bare host `api_base` takes the Gemini HTTP path and posts
    to `/v1/projects/dec1-vertex-project/locations/us-central1/endpoints/4965075652664360960:generateContent`
    (not `publishers/google/models/...`), with a Gemini `contents` body."""
    with vertex(gateway, gemini_echo, backend=PSC_ENDPOINT_ID) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        call: Final = rig.peer.one_call()
        assert (path_of(call), query_of(call)) == (f"{_PSC_ENDPOINT}:generateContent", "")
        assert "contents" in body_of(call)


def test_a_numeric_endpoint_streams_to_stream_generate_content_with_alt_sse(gateway: Gateway) -> None:
    """Streaming on a numeric endpoint posts to `.../endpoints/4965075652664360960:streamGenerateContent`
    with the query `alt=sse`."""
    with vertex(gateway, gemini_echo, backend=PSC_ENDPOINT_ID) as rig:
        response, _text = rig.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": marker()}]})
        assert response.status_code == 200
        call: Final = rig.peer.one_call()
        assert (path_of(call), query_of(call)) == (f"{_PSC_ENDPOINT}:streamGenerateContent", "alt=sse")


def test_a_gemini_prefixed_numeric_endpoint_is_also_sent_to_endpoints(gateway: Gateway) -> None:
    """`vertex_ai/gemini/4965075652664360960` (the fine tuned Gemini spelling) has the `gemini/` stripped
    by `get_model_for_vertex_ai_url` and is posted to the same `.../endpoints/4965075652664360960:
    generateContent`. UNVERIFIED: that the router accepts the `gemini/<digits>` spelling with an `api_base`
    and that `_is_model_gemini_spec_model` matches it."""
    with vertex(gateway, gemini_echo, backend=f"gemini/{PSC_ENDPOINT_ID}") as rig:
        assert rig.chat().status_code == 200
        assert path_of(rig.peer.one_call()) == f"{_PSC_ENDPOINT}:generateContent"


def test_a_path_bearing_api_base_without_psc_format_is_the_full_resource(gateway: Gateway) -> None:
    """Without `use_psc_endpoint_format`, an `api_base` of `<peer>/psc/dedicated-endpoint` is neither empty
    nor a bare version, so it is taken as the whole resource and the call goes to
    `/psc/dedicated-endpoint:generateContent`."""
    with vertex(gateway, gemini_echo, backend=PSC_ENDPOINT_ID, base_suffix="/psc/dedicated-endpoint") as rig:
        assert rig.chat().status_code == 200
        assert path_of(rig.peer.one_call()) == "/psc/dedicated-endpoint:generateContent"


def test_psc_endpoint_format_builds_the_full_endpoint_path_under_the_api_base(gateway: Gateway) -> None:
    """With `use_psc_endpoint_format: true` the same `<peer>/psc` base gets the full endpoint resource
    appended: `/psc/v1/projects/dec1-vertex-project/locations/us-central1/endpoints/4965075652664360960:
    generateContent`. This is how a PSC forwarding rule IP, which serves the whole Vertex API surface, is
    addressed."""
    with vertex(gateway, gemini_echo, backend=PSC_ENDPOINT_ID, base_suffix="/psc", use_psc_endpoint_format=True) as rig:
        assert rig.chat().status_code == 200
        assert path_of(rig.peer.one_call()) == f"/psc{_PSC_ENDPOINT}:generateContent"


def test_psc_endpoint_format_streams_with_alt_sse(gateway: Gateway) -> None:
    """Streaming with `use_psc_endpoint_format: true` posts to `/psc/v1/projects/.../endpoints/<id>:
    streamGenerateContent?alt=sse`."""
    with vertex(gateway, gemini_echo, backend=PSC_ENDPOINT_ID, base_suffix="/psc", use_psc_endpoint_format=True) as rig:
        response, _text = rig.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": marker()}]})
        assert response.status_code == 200
        call: Final = rig.peer.one_call()
        assert (path_of(call), query_of(call)) == (f"/psc{_PSC_ENDPOINT}:streamGenerateContent", "alt=sse")


def test_psc_endpoint_format_is_not_sent_to_google(gateway: Gateway) -> None:
    """`use_psc_endpoint_format` is a routing switch: it is absent from the Gemini body's top level and
    from `generationConfig` (only `GenerationConfig` keys survive the filter in
    `litellm/llms/vertex_ai/gemini/transformation.py:1208`)."""
    with vertex(gateway, gemini_echo, backend=PSC_ENDPOINT_ID, base_suffix="/psc", use_psc_endpoint_format=True) as rig:
        assert rig.chat(temperature=0.3).status_code == 200
        body: Final = rig.peer.one_body()
        assert "use_psc_endpoint_format" not in body
        config: Final = object_value(body["generationConfig"])
        assert "use_psc_endpoint_format" not in config
        assert config["temperature"] == 0.3


# ---------------------------------------------------------------- numeric embedding endpoints


def _fine_tuned_embedding_peer(_request: object) -> Reply:
    return Reply(body=json.dumps({"predictions": [[[0.25, 0.75]], [[0.5, 0.5]]]}).encode())


def test_a_numeric_embedding_endpoint_posts_inputs_instances_to_predict(gateway: Gateway) -> None:
    """`/v1/embeddings` on `vertex_ai/4965075652664360960` with a bare host `api_base` posts to
    `.../endpoints/4965075652664360960:predict` with `instances: [{"inputs": a}, {"inputs": b}]` (the fine
    tuned shape, `_transform_openai_request_to_fine_tuned_embedding_request`), not `{"content": ...}`."""
    first: Final = f"first {marker()}"
    with vertex(gateway, _fine_tuned_embedding_peer, backend=PSC_ENDPOINT_ID) as rig:
        response: Final = rig.post("/v1/embeddings", {"input": [first, "second"]})
        assert response.status_code == 200, response.text
        call: Final = rig.peer.one_call()
        assert path_of(call) == f"{_PSC_ENDPOINT}:predict"
        assert body_of(call)["instances"] == [{"inputs": first}, {"inputs": "second"}]


def test_a_numeric_embedding_endpoint_unwraps_the_nested_vectors(gateway: Gateway) -> None:
    """The fine tuned embedding answer `predictions: [[[0.25, 0.75]], [[0.5, 0.5]]]` is one level deeper
    than OpenAI's, so the proxy returns `data[0].embedding == [0.25, 0.75]` and `data[1].embedding ==
    [0.5, 0.5]` with zero usage tokens (the endpoint reports no counts)."""
    with vertex(gateway, _fine_tuned_embedding_peer, backend=PSC_ENDPOINT_ID) as rig:
        response: Final = rig.post("/v1/embeddings", {"input": [f"a {marker()}", "b"]})
        assert response.status_code == 200, response.text
        data: Final[list[JsonValue]] = list_value(response.json()["data"])
        assert [object_value(item)["embedding"] for item in data] == [[0.25, 0.75], [0.5, 0.5]]
        assert object_value(response.json()["usage"])["prompt_tokens"] == 0


def test_psc_endpoint_format_applies_to_numeric_embedding_endpoints(gateway: Gateway) -> None:
    """`use_psc_endpoint_format: true` on an embedding deployment with `api_base <peer>/psc` posts to
    `/psc/v1/projects/dec1-vertex-project/locations/us-central1/endpoints/4965075652664360960:predict`
    (`vertex_embeddings/embedding_handler.py:77` reads the flag the same way chat does)."""
    with vertex(
        gateway, _fine_tuned_embedding_peer, backend=PSC_ENDPOINT_ID, base_suffix="/psc", use_psc_endpoint_format=True
    ) as rig:
        assert rig.post("/v1/embeddings", {"input": [f"a {marker()}", "b"]}).status_code == 200
        assert path_of(rig.peer.one_call()) == f"/psc{_PSC_ENDPOINT}:predict"
