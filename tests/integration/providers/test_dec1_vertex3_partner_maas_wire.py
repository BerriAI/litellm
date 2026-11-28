"""Vertex AI Model-as-a-Service partner models on `/v1/chat/completions` and `/v1/completions`.

`get_vertex_ai_model_route` (`litellm/llms/vertex_ai/common_utils.py:295`) sends any `vertex_ai/<model>` whose
name starts with a partner prefix (`meta/`, `deepseek-ai`, `qwen`, `openai/gpt-oss-`, `minimaxai/`,
`moonshotai/`, `zai-org/`, `mistral`, `codestral`, `jamba`, `claude`, `google/gemma-`) to
`VertexAIPartnerModels.completion` (`litellm/llms/vertex_ai/vertex_ai_partner_models/main.py`). That handler
mints a Google access token and picks one of three wire shapes:

- the Llama family (Llama, DeepSeek, Qwen, gpt-oss, MiniMax, Kimi, GLM) all post OpenAI chat bodies to ONE
  shared route, `{api_base}/v1/projects/{project}/locations/{location}/endpoints/openapi/chat/completions`, with
  the full model name in the body (`VertexBase.create_vertex_url`, `litellm/llms/vertex_ai/vertex_llm_base.py:352`).
  This route skips `_check_custom_proxy`, so streaming adds no `alt=sse`;
- Mistral and Codestral post OpenAI bodies to `publishers/mistralai/models/{model}:rawPredict` (or
  `:streamRawPredict?alt=sse`), with the `@version` kept in the URL but stripped from the body `model`;
- AI21 Jamba posts to `v1beta1/.../publishers/ai21/models/{model}:rawPredict`.

The location is first passed through `get_vertex_region`, which reroutes a location the cost map's
`supported_regions` does not list to the first listed region, so a `global`-only MaaS model never reaches a
regional host that would 404. A Rust port of the Vertex provider must keep all of these URL shapes, the bearer,
and the body `model` rules; the Claude rawPredict shape is already pinned in `test_dec1_vertex_claude_chat.py`.
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, list_value, object_value, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply
from integration.providers.dec1_vertex3_support import (
    CODESTRAL,
    DEEPSEEK,
    DEEPSEEK_GLOBAL_ONLY,
    GLM_GLOBAL_ONLY,
    GPT_OSS,
    JAMBA,
    KIMI,
    LLAMA,
    MINIMAX,
    MISTRAL,
    MISTRAL_SMALL,
    QWEN_GLOBAL_ONLY,
    is_token_from,
    openai_echo,
    openai_reply,
    openapi_chat_path,
    raw_predict_path,
)
from integration.providers.dec1_vertex_support import (
    body_of,
    marker,
    message_of,
    path_of,
    query_of,
    streamed_text,
    vertex,
)
from pydantic import JsonValue

_OPENAPI: Final = openapi_chat_path()
_WEATHER_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def _openai_error(status: int, message: str) -> Reply:
    return Reply(
        status=status,
        body=json.dumps({"error": {"message": message, "type": "invalid_request_error", "code": status}}).encode(),
    )


# ---------------------------------------------------------------- the shared openapi route


@pytest.mark.parametrize("backend", [LLAMA, DEEPSEEK, GPT_OSS, MINIMAX, KIMI])
def test_llama_family_partners_post_to_the_shared_openapi_chat_route(gateway: Gateway, backend: str) -> None:
    """A Llama, DeepSeek, gpt-oss, MiniMax or Kimi deployment on a bare host `api_base` posts once to
    `/v1/projects/dec1-vertex-project/locations/us-central1/endpoints/openapi/chat/completions` with no query
    (`create_vertex_url` llama branch). All of them share this one route: a rewrite that builds
    `publishers/<vendor>/models/<model>:rawPredict` for them would 404 on Google."""
    with vertex(gateway, openai_echo, backend=backend) as rig:
        response: Final = rig.chat(max_tokens=16)
        assert response.status_code == 200, response.text
        call: Final = rig.peer.one_call()
        assert (path_of(call), query_of(call)) == (_OPENAPI, "")


@pytest.mark.parametrize("backend", [LLAMA, DEEPSEEK, GPT_OSS, MINIMAX, KIMI])
def test_the_openapi_route_names_the_full_partner_model_in_the_body(gateway: Gateway, backend: str) -> None:
    """Because the openapi route is shared, the model is chosen by the body: the request body's `model` is
    the deployment's model without the `vertex_ai/` prefix, publisher segment included (for example
    `meta/llama-3.3-70b-instruct-maas`), exactly as `OpenAIGPTConfig.transform_request` builds it."""
    with vertex(gateway, openai_echo, backend=backend) as rig:
        assert rig.chat(max_tokens=16).status_code == 200
        assert rig.peer.one_body()["model"] == backend


def test_the_openapi_route_carries_the_minted_google_token(gateway: Gateway) -> None:
    """The Llama call is authenticated with `Authorization: Bearer <token>` where the token is the one the
    peer's OAuth endpoint just minted for the deployment's service account; no `x-api-key` and no
    `x-goog-api-key` is sent."""
    with vertex(gateway, openai_echo, backend=LLAMA) as rig:
        assert rig.chat(max_tokens=16).status_code == 200
        call: Final = rig.peer.one_call()
        assert is_token_from(rig.peer.plan, call), call.headers
        assert "x-api-key" not in call.headers and "x-goog-api-key" not in call.headers


def test_the_openapi_route_sends_the_openai_messages_unchanged(gateway: Gateway) -> None:
    """A system and a user message reach the openapi route as OpenAI `messages` in the same order and roles
    (no Gemini `contents`, no Anthropic top level `system`), since the MaaS endpoint speaks the OpenAI wire."""
    prompt: Final = f"what is the weather {marker()}"
    with vertex(gateway, openai_echo, backend=LLAMA) as rig:
        response: Final = rig.post(
            "/v1/chat/completions",
            {"messages": [{"role": "system", "content": "be terse"}, {"role": "user", "content": prompt}]},
        )
        assert response.status_code == 200, response.text
        sent: Final = list_value(rig.peer.one_body()["messages"])
        assert [object_value(item)["role"] for item in sent] == ["system", "user"]
        assert object_value(sent[1])["content"] == prompt


def test_max_completion_tokens_is_sent_as_max_tokens_on_llama(gateway: Gateway) -> None:
    """`VertexAILlama3Config.map_openai_params` renames `max_completion_tokens` to `max_tokens`, so the
    openapi body carries `max_tokens: 64` and no `max_completion_tokens` (the MaaS endpoint rejects the newer
    name)."""
    with vertex(gateway, openai_echo, backend=LLAMA) as rig:
        assert rig.chat(max_completion_tokens=64).status_code == 200
        body: Final = rig.peer.one_body()
        assert body["max_tokens"] == 64
        assert "max_completion_tokens" not in body


def test_openai_tools_reach_the_openapi_route_in_openai_shape(gateway: Gateway) -> None:
    """An OpenAI function tool is forwarded as-is in `tools` (type `function`, the same name and
    parameters), not converted to Gemini `function_declarations`, and a `tool_calls` answer comes back as an
    OpenAI tool call with `finish_reason: "tool_calls"`."""
    tool_call: Final[dict[str, JsonValue]] = {
        "id": "call_maas_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
    }
    reply: Final = openai_reply(text=None, finish="tool_calls", tool_calls=[tool_call])
    with vertex(gateway, lambda _r: reply, backend=LLAMA) as rig:
        response: Final = rig.chat(tools=[_WEATHER_TOOL])
        assert response.status_code == 200, response.text
        sent_tool: Final = object_value(list_value(rig.peer.one_body()["tools"])[0])
        assert sent_tool["type"] == "function"
        assert object_value(sent_tool["function"])["name"] == "get_weather"
        calls: Final = list_value(message_of(response)["tool_calls"])
        assert object_value(object_value(calls[0])["function"])["name"] == "get_weather"


def test_a_maas_answer_comes_back_with_its_text_and_usage(gateway: Gateway) -> None:
    """The partner's OpenAI answer is returned as an assistant message with the same text and `usage`
    `prompt_tokens: 11`, `completion_tokens: 7`, `total_tokens: 18` copied from the upstream body
    (`VertexAILlama3Config.transform_response`)."""
    with vertex(gateway, lambda _r: openai_reply(text="maas says hi"), backend=LLAMA) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert message_of(response)["content"] == "maas says hi"
        usage: Final = object_value(response.json()["usage"])
        assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (11, 7, 18)


def test_a_streaming_maas_call_uses_the_same_route_without_alt_sse(gateway: Gateway) -> None:
    """`stream: true` on a Llama family model posts to the same openapi path with an empty query (the llama
    branch of `get_complete_vertex_url` returns before `_check_custom_proxy` would add `alt=sse`) and a body
    with `stream: true`; the OpenAI SSE frames are relayed and join to "hello maas"."""
    with vertex(gateway, openai_echo, backend=LLAMA) as rig:
        response, text = rig.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": marker()}]})
        assert response.status_code == 200
        call: Final = rig.peer.one_call()
        assert (path_of(call), query_of(call)) == (_OPENAPI, "")
        assert body_of(call)["stream"] is True
        assert streamed_text(text) == "hello maas"
        assert "data: [DONE]" in text


@pytest.mark.parametrize(
    ("status", "expected"), [(429, 429), (400, 400), (404, 404)], ids=["rate-limited", "bad-request", "not-found"]
)
def test_a_maas_error_keeps_its_status(gateway: Gateway, status: int, expected: int) -> None:
    """An OpenAI error envelope from the openapi route leaves `/v1/chat/completions` with the same status
    and the upstream message in the body; a 429 must stay a 429 so router cooldowns and client retries behave.
    UNVERIFIED: the status mapping goes through `VertexAILlama3Config.get_error_class` (OpenAI's), which keeps
    the status; look at `litellm/litellm_core_utils/exception_mapping_utils.py` if a 500 comes back.

    The call gets a 60 second read timeout instead of the client's 15: on a 400 or 404 the proxy loads the
    `Xenova/llama-3-tokenizer` Hugging Face tokenizer for this Llama model before it answers
    (`_load_huggingface_tokenizer`, litellm/utils.py:3314-3329 on main), and with no route to huggingface.co the
    hub client retries for about 24 seconds before falling back to tiktoken. Round 2 timed out on exactly that."""
    message: Final = f"synthetic maas failure {status}"
    with vertex(gateway, lambda _r: _openai_error(status, message), backend=LLAMA) as rig:
        response: Final = rig.gateway.client.post(
            "/v1/chat/completions",
            json={"model": rig.model, "messages": [{"role": "user", "content": f"ping {marker()}"}]},
            headers={"Authorization": f"Bearer {rig.gateway.key}"},
            timeout=60,
        )
        assert response.status_code == expected, response.text
        assert message in response.text


# ---------------------------------------------------------------- location rerouting


@pytest.mark.parametrize("backend", [DEEPSEEK_GLOBAL_ONLY, QWEN_GLOBAL_ONLY, GLM_GLOBAL_ONLY])
def test_a_global_only_maas_model_is_rerouted_to_the_global_location(gateway: Gateway, backend: str) -> None:
    """A model whose cost map entry lists `supported_regions: ["global"]` and a deployment that names
    `vertex_location: us-central1` is sent to `locations/global` in the openapi path (`get_vertex_region`
    overrides an unsupported region with the first supported one), so the call does not hit a regional
    endpoint that does not serve the model."""
    with vertex(gateway, openai_echo, backend=backend, vertex_location="us-central1") as rig:
        assert rig.chat().status_code == 200
        assert path_of(rig.peer.one_call()) == openapi_chat_path("global")


def test_a_regional_maas_model_named_on_another_region_is_sent_to_its_listed_region(gateway: Gateway) -> None:
    """`deepseek-ai/deepseek-r1-0528-maas` lists `supported_regions: ["us-central1"]`; a deployment on
    `europe-west4` is rerouted so the path names `locations/us-central1`."""
    with vertex(gateway, openai_echo, backend=DEEPSEEK, vertex_location="europe-west4") as rig:
        assert rig.chat().status_code == 200
        assert path_of(rig.peer.one_call()) == openapi_chat_path("us-central1")


def test_a_maas_model_without_supported_regions_keeps_the_deployment_location(gateway: Gateway) -> None:
    """`meta/llama-3.3-70b-instruct-maas` has no `supported_regions`, so `vertex_location: us-east5` is
    used verbatim in the openapi path."""
    with vertex(gateway, openai_echo, backend=LLAMA, vertex_location="us-east5") as rig:
        assert rig.chat().status_code == 200
        assert path_of(rig.peer.one_call()) == openapi_chat_path("us-east5")


# ---------------------------------------------------------------- Mistral and Codestral rawPredict


def test_mistral_posts_to_the_mistralai_raw_predict_path_with_its_version(gateway: Gateway) -> None:
    """`vertex_ai/mistral-large@2407` reaches `POST /v1/projects/dec1-vertex-project/locations/us-central1/
    publishers/mistralai/models/mistral-large@2407:rawPredict` with no query: the URL is built before the
    handler strips the `@version`, so the version stays in the path."""
    with vertex(gateway, openai_echo, backend=MISTRAL) as rig:
        assert rig.chat().status_code == 200
        call: Final = rig.peer.one_call()
        assert (path_of(call), query_of(call)) == (raw_predict_path("mistralai", MISTRAL) + ":rawPredict", "")


@pytest.mark.parametrize(("backend", "body_model"), [(MISTRAL, "mistral-large"), (MISTRAL_SMALL, "mistral-small-2503")])
def test_mistral_body_model_drops_the_version_suffix(gateway: Gateway, backend: str, body_model: str) -> None:
    """The Mistral body `model` is the deployment model cut at `@` (`model.split("@")[0]` in
    `VertexAIPartnerModels.completion`), so `mistral-large@2407` is sent as `mistral-large` while the URL
    keeps the version."""
    with vertex(gateway, openai_echo, backend=backend) as rig:
        assert rig.chat().status_code == 200
        assert rig.peer.one_body()["model"] == body_model


def test_mistral_non_streaming_body_says_stream_false(gateway: Gateway) -> None:
    """The partner handler always sets `optional_params["stream"]` and `OpenAILikeChatHandler` writes it
    back, so a non-streaming Mistral body carries an explicit `stream: false`."""
    with vertex(gateway, openai_echo, backend=MISTRAL) as rig:
        assert rig.chat().status_code == 200
        assert rig.peer.one_body()["stream"] is False


def test_mistral_streaming_uses_stream_raw_predict_with_alt_sse(gateway: Gateway) -> None:
    """`stream: true` on Mistral posts to `...mistral-large@2407:streamRawPredict` with the query
    `alt=sse` (added by `_check_custom_proxy` because the deployment has an `api_base`), and the OpenAI
    SSE frames are relayed as chat chunks joining to "hello maas"."""
    with vertex(gateway, openai_echo, backend=MISTRAL) as rig:
        response, text = rig.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": marker()}]})
        assert response.status_code == 200
        call: Final = rig.peer.one_call()
        assert (path_of(call), query_of(call)) == (
            raw_predict_path("mistralai", MISTRAL) + ":streamRawPredict",
            "alt=sse",
        )
        assert streamed_text(text) == "hello maas"


def test_mistral_carries_the_minted_token_and_a_json_content_type(gateway: Gateway) -> None:
    """The Mistral rawPredict call carries the minted bearer and `Content-Type: application/json`
    (`OpenAILike._validate_environment` default headers)."""
    with vertex(gateway, openai_echo, backend=MISTRAL) as rig:
        assert rig.chat().status_code == 200
        call: Final = rig.peer.one_call()
        assert is_token_from(rig.peer.plan, call)
        assert call.headers["content-type"].startswith("application/json")


def test_codestral_fill_in_the_middle_posts_prompt_and_suffix_to_raw_predict(gateway: Gateway) -> None:
    """`POST /v1/completions` with `prompt` and `suffix` on `vertex_ai/codestral@2405` takes the Codestral
    FIM branch (`litellm_params.text_completion is True`): one `:rawPredict` call on
    `publishers/mistralai/models/codestral@2405` whose body has `model: "codestral"`, a `prompt` holding the
    caller's text and the caller's `suffix`, and the answer's `choices[0].message.content` comes back as
    `choices[0].text`. UNVERIFIED: that the proxy's `/v1/completions` sets `text_completion=True` for this
    route (`litellm/main.py:8376`) and that `prompt_factory` keeps the prompt text verbatim."""
    prompt: Final = f"def add(a, b):\n    {marker()}"
    with vertex(gateway, lambda _r: openai_reply(text="return a + b"), backend=CODESTRAL) as rig:
        response: Final = rig.post(
            "/v1/completions", {"prompt": prompt, "suffix": "\n\nprint(add(1, 2))", "max_tokens": 16}
        )
        assert response.status_code == 200, response.text
        call: Final = rig.peer.one_call()
        assert path_of(call) == raw_predict_path("mistralai", CODESTRAL) + ":rawPredict"
        body: Final = body_of(call)
        assert body["model"] == "codestral"
        assert prompt in string_value(body["prompt"])
        assert body["suffix"] == "\n\nprint(add(1, 2))"
        assert object_value(list_value(response.json()["choices"])[0])["text"] == "return a + b"


# ---------------------------------------------------------------- AI21 Jamba


def test_jamba_posts_to_the_v1beta1_ai21_raw_predict_path(gateway: Gateway) -> None:
    """`vertex_ai/jamba-1.5-large@001` reaches `POST /v1beta1/projects/dec1-vertex-project/locations/
    us-central1/publishers/ai21/models/jamba-1.5-large@001:rawPredict`: AI21 is the only partner on
    `v1beta1`, and a bare host `api_base` keeps that version because the default path is joined as-is."""
    with vertex(gateway, openai_echo, backend=JAMBA) as rig:
        assert rig.chat().status_code == 200
        call: Final = rig.peer.one_call()
        assert (path_of(call), query_of(call)) == (
            raw_predict_path("ai21", JAMBA, version="v1beta1") + ":rawPredict",
            "",
        )


def test_jamba_keeps_the_version_in_the_body_model(gateway: Gateway) -> None:
    """Unlike Mistral, the `@` suffix is not stripped for AI21, so the Jamba body `model` is
    `jamba-1.5-large@001`, and `max_completion_tokens: 40` is sent as `max_tokens: 40`
    (`VertexAIAi21Config.map_openai_params`)."""
    with vertex(gateway, openai_echo, backend=JAMBA) as rig:
        assert rig.chat(max_completion_tokens=40).status_code == 200
        body: Final = rig.peer.one_body()
        assert body["model"] == JAMBA
        assert body["max_tokens"] == 40


# ---------------------------------------------------------------- spend


def test_a_maas_call_is_billed_at_deployment_prices_under_vertex_ai(gateway: Gateway) -> None:
    """One Llama call writes one SpendLogs row with `custom_llm_provider = "vertex_ai"`, 11 prompt and 7
    completion tokens, priced at the deployment's custom prices (`11 * 0.001 + 7 * 0.002 = 0.025`)."""
    with vertex(gateway, openai_echo, backend=LLAMA, input_cost_per_token=0.001, output_cost_per_token=0.002) as rig:
        assert rig.chat().status_code == 200
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT custom_llm_provider, prompt_tokens, completion_tokens, spend FROM "LiteLLM_SpendLogs" WHERE model_group=%s',
                (rig.model,),
            ),
            lambda found: len(found) >= 1,
            seconds=90,
        )
        assert len(rows) == 1, rows
        assert (rows[0]["custom_llm_provider"], rows[0]["prompt_tokens"], rows[0]["completion_tokens"]) == (
            "vertex_ai",
            11,
            7,
        )
        assert float(str(rows[0]["spend"])) == pytest.approx(0.025)


def test_a_maas_call_without_custom_prices_is_billed_from_the_cost_map(gateway: Gateway) -> None:
    """Without deployment prices the Llama 3.3 row is priced from `vertex_ai/meta/llama-3.3-70b-instruct-maas`
    in the cost map (`7.2e-07` per input and output token), so 11 plus 7 tokens cost `18 * 7.2e-07 = 1.296e-05`.
    UNVERIFIED: that the cost lookup keys on the publisher qualified name, not on `llama-3.3-70b-instruct-maas`."""
    with vertex(gateway, openai_echo, backend=LLAMA) as rig:
        assert rig.chat().status_code == 200
        rows: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (rig.model,)),
            lambda found: len(found) >= 1,
            seconds=90,
        )
        assert float(str(rows[0]["spend"])) == pytest.approx(18 * 7.2e-07)
