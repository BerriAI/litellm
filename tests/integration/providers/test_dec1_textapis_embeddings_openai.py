"""Contract: `POST /v1/embeddings` served by an OpenAI deployment, from the caller's body to the provider's wire.

The embeddings endpoint is used by every RAG client and has far fewer tests than chat. This file pins what a
caller of an `openai/text-embedding-3-small` deployment can observe: the exact request the provider receives
(path `/v1/embeddings` under the deployment's `api_base`, `Authorization: Bearer <deployment key>`, a JSON body
with the upstream model name and the caller's `input` untouched), which optional parameters are forwarded
(`user`, `dimensions`, unknown provider fields) and which are refused before the provider is called
(`dimensions` on a model older than `text-embedding-3`), the response shape (`object: "list"`, one
`{"object": "embedding", "index", "embedding"}` per input, `usage.prompt_tokens` and `usage.total_tokens`), the
`x-litellm-*` response headers, request validation errors, and how provider errors 400, 401, 429 and 500 come
back in the OpenAI error envelope `{"error": {"message", "type", "param", "code"}}`.

Token array inputs, `encoding_format` and the `/engines/...` and `/openai/deployments/...` path aliases already
have their own files (`test_embedding_token_array_wire.py`, `test_openai_embedding_wire.py`,
`compatibility/test_embedding_path_aliases.py`) and are not repeated here. All tests use the shared gateway, a
fresh deployment per test and unique inputs, so the shared response cache never answers for the provider.
"""

from __future__ import annotations

import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration.providers.dec1_textapis_support import (
    EMBEDDING_COST,
    EMBEDDING_PROMPT_TOKENS,
    PROVIDER_KEY,
    VECTOR,
    Call,
    default_respond,
    embed,
    error_message,
    error_of,
    error_reply,
    failing,
    marker,
    openai_embedding_model,
    peer,
    served,
)
from pydantic import JsonValue

from tests.integration._support.wire import Reply

_UPSTREAM: Final = "text-embedding-3-small"
_OPENAI_PREFIX: Final = {
    400: "litellm.BadRequestError: OpenAIException - Error code: 400 - ",
    401: "litellm.AuthenticationError: OpenAIException - Error code: 401 - ",
    429: "litellm.RateLimitError: RateLimitError: OpenAIException - Error code: 429 - ",
    500: "litellm.InternalServerError: InternalServerError: OpenAIException - Error code: 500 - ",
}
_OPENAI_ERROR_TYPE: Final = {
    400: "invalid_request_error",
    401: "authentication_error",
    429: "throttling_error",
    500: "internal_server_error",
}


def test_a_string_input_reaches_the_provider_as_the_same_string_on_the_embeddings_path(gateway: Gateway) -> None:
    """A caller embeds one string on an `openai/text-embedding-3-small` deployment whose `api_base` ends in `/v1`.

    The provider must receive exactly one `POST /v1/embeddings` whose JSON body is
    `{"model": "text-embedding-3-small", "input": "<the string>"}`: the public model name is replaced by the
    upstream name, the input stays a string (it is not wrapped in a list) and nothing else is added.
    """
    text: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = openai_embedding_model(scenario, provider)
        served(embed(gateway, model, text))
        call: Final = provider.only()
        assert (call.method, call.path, dict(call.query)) == ("POST", "/v1/embeddings", {})
        assert call.body == {"model": _UPSTREAM, "input": text}


def test_the_provider_sees_the_deployment_key_and_never_the_callers_virtual_key(gateway: Gateway) -> None:
    """A virtual key calls embeddings on a deployment configured with its own provider `api_key`.

    The provider request must carry `Authorization: Bearer <deployment api_key>` and
    `Content-Type: application/json`, and the virtual key must not appear in any header value sent upstream.
    A front door rewrite that forwards the inbound Authorization header would leak proxy credentials.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = openai_embedding_model(scenario, provider)
        key: Final = scenario.key(models=[model])
        served(embed(gateway, model, key=key))
        call: Final = provider.only()
        assert call.headers["authorization"] == f"Bearer {PROVIDER_KEY}"
        assert call.headers["content-type"] == "application/json"
        assert all(key not in value for value in call.headers.values()), dict(call.headers)


def test_a_list_of_strings_is_forwarded_in_order_and_returns_one_indexed_vector_per_input(gateway: Gateway) -> None:
    """A caller embeds a list of three strings.

    The provider must receive the same three strings in the same order in one request, and the caller gets
    three `data` entries with `index` 0, 1 and 2, each `{"object": "embedding", "index": n, "embedding":
    [0.25, 0.5, 0.75]}`.
    """
    inputs: Final[list[JsonValue]] = [marker(), marker(), marker()]
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = openai_embedding_model(scenario, provider)
        body: Final = served(embed(gateway, model, inputs))
        assert provider.only().body == {"model": _UPSTREAM, "input": inputs}
        assert body["data"] == [
            {"object": "embedding", "index": index, "embedding": list(VECTOR)} for index in range(3)
        ], body


def test_the_response_is_an_openai_list_with_prompt_and_total_token_usage(gateway: Gateway) -> None:
    """The provider answers one vector and `usage: {"prompt_tokens": 7, "total_tokens": 7}`.

    The caller's body must have `object` `"list"`, a string `model`, exactly one `data` entry and a `usage`
    object whose `prompt_tokens` and `total_tokens` are both 7. OpenAI SDKs parse these fields strictly.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = openai_embedding_model(scenario, provider)
        body: Final = served(embed(gateway, model))
        usage: Final = object_value(body["usage"])
        assert body["object"] == "list", body
        assert isinstance(body["model"], str), body
        assert len(list_value(body["data"])) == 1, body
        assert (usage["prompt_tokens"], usage["total_tokens"]) == (EMBEDDING_PROMPT_TOKENS, EMBEDDING_PROMPT_TOKENS)


def test_the_user_parameter_is_forwarded_to_the_provider(gateway: Gateway) -> None:
    """A caller sends `user: "dec1-end-user"` with an embeddings request.

    `user` is one of the three OpenAI embedding parameters (`dimensions`, `encoding_format`, `user`), so the
    provider body must be `{"model", "input", "user": "dec1-end-user"}`.
    """
    text: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = openai_embedding_model(scenario, provider)
        served(embed(gateway, model, text, user="dec1-end-user"))
        assert provider.only().body == {"model": _UPSTREAM, "input": text, "user": "dec1-end-user"}


def test_dimensions_is_forwarded_for_a_text_embedding_3_model(gateway: Gateway) -> None:
    """A caller sends `dimensions: 256` to an `openai/text-embedding-3-small` deployment.

    The provider body must carry `"dimensions": 256` unchanged next to the model and input.
    """
    text: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = openai_embedding_model(scenario, provider)
        served(embed(gateway, model, text, dimensions=256))
        assert provider.only().body == {"model": _UPSTREAM, "input": text, "dimensions": 256}


def test_dimensions_on_an_older_openai_model_is_refused_with_400_before_the_provider(gateway: Gateway) -> None:
    """A caller sends `dimensions: 256` to an `openai/text-embedding-ada-002` deployment.

    Only `text-embedding-3` and later accept `dimensions`, so the proxy must answer 400 without calling the
    provider, and the message must contain `litellm.UnsupportedParamsError: Setting dimensions is not supported
    for OpenAI `text-embedding-3` and later models. To drop it from the call, set `litellm.drop_params = True`.`
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = openai_embedding_model(scenario, provider, upstream="text-embedding-ada-002")
        response: Final = embed(gateway, model, dimensions=256)
        assert response.status_code == 400, response.text
        assert (
            "litellm.UnsupportedParamsError: Setting dimensions is not supported for OpenAI `text-embedding-3` and "
            "later models. To drop it from the call, set `litellm.drop_params = True`." in error_message(response)
        ), response.text
        assert provider.posts() == ()


def test_dimensions_on_an_older_openai_model_is_dropped_when_the_deployment_sets_drop_params(
    gateway: Gateway,
) -> None:
    """The same request, on an ada-002 deployment created with `litellm_params.drop_params: true`.

    The request must be served and the provider body must be `{"model", "input"}` without `dimensions`.
    UNVERIFIED: `embedding()` filters litellm-owned kwargs before `get_optional_params_embeddings`, so the
    deployment flag may not reach it (only global `litellm_settings.drop_params` is certain to); see
    `litellm/main.py` around `filter_out_litellm_params` and `litellm/utils.py:get_optional_params_embeddings`.
    """
    text: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = openai_embedding_model(scenario, provider, upstream="text-embedding-ada-002", drop_params=True)
        served(embed(gateway, model, text, dimensions=256))
        assert provider.only().body == {"model": "text-embedding-ada-002", "input": text}


def test_an_unknown_body_field_is_passed_through_to_an_openai_provider(gateway: Gateway) -> None:
    """A caller adds `input_type: "query"`, which is not an OpenAI embeddings parameter.

    For OpenAI-compatible providers unknown fields travel as `extra_body`, so the provider body must contain
    `"input_type": "query"` at the top level. Self-hosted OpenAI-compatible embedding servers rely on this.
    """
    text: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = openai_embedding_model(scenario, provider)
        served(embed(gateway, model, text, input_type="query"))
        assert provider.only().body == {"model": _UPSTREAM, "input": text, "input_type": "query"}


def test_proxy_only_fields_in_the_body_are_not_forwarded_to_the_provider(gateway: Gateway) -> None:
    """A caller sends `metadata: {"tags": ["dec1"]}` and `litellm_session_id`, which only the proxy understands.

    The provider body must be exactly `{"model", "input"}`. A leaked proxy field makes strict providers answer
    400 and is invisible from the caller's side, which is why it is asserted on the wire.
    """
    text: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = openai_embedding_model(scenario, provider)
        served(embed(gateway, model, text, metadata={"tags": ["dec1"]}, litellm_session_id=str(uuid.uuid4())))
        assert provider.only().body == {"model": _UPSTREAM, "input": text}


def test_three_hundred_inputs_are_sent_in_one_provider_request(gateway: Gateway) -> None:
    """A caller embeds a list of 300 strings.

    The proxy has no batch size limit of its own and must not split the batch: the provider receives one
    request with all 300 inputs and the caller gets 300 vectors indexed 0 to 299.
    """
    inputs: Final[list[JsonValue]] = [f"{marker()}-{index}" for index in range(300)]
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = openai_embedding_model(scenario, provider)
        body: Final = served(embed(gateway, model, inputs))
        assert provider.only().body["input"] == inputs
        assert [object_value(entry)["index"] for entry in list_value(body["data"])] == list(range(300))


def test_a_missing_input_is_refused_with_the_exact_missing_parameter_error(gateway: Gateway) -> None:
    """A caller posts `{"model": <existing model>}` with no `input`.

    The proxy must answer 400 with exactly `{"error": {"message": "/embeddings: Missing required parameter:
    'input'.", "type": "invalid_request_error", "param": "input", "code": "400"}}` and not call the provider.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = openai_embedding_model(scenario, provider)
        response: Final = gateway.request("POST", "/v1/embeddings", {"model": model})
        assert response.status_code == 400, response.text
        assert response.json() == {
            "error": {
                "message": "/embeddings: Missing required parameter: 'input'.",
                "type": "invalid_request_error",
                "param": "input",
                "code": "400",
            }
        }
        assert provider.posts() == ()


def test_a_null_input_is_refused_like_a_missing_input(gateway: Gateway) -> None:
    """A caller posts `{"model": ..., "input": null}`.

    For embeddings a null `input` counts as missing: 400, `param` `"input"` and the message
    `/embeddings: Missing required parameter: 'input'.`, with no provider call.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = openai_embedding_model(scenario, provider)
        response: Final = gateway.request("POST", "/v1/embeddings", {"model": model, "input": None})
        assert response.status_code == 400, response.text
        error: Final = error_of(response)
        assert (error["message"], error["param"]) == ("/embeddings: Missing required parameter: 'input'.", "input")
        assert provider.posts() == ()


def test_the_missing_input_error_carries_a_call_id_header(gateway: Gateway) -> None:
    """The 400 for a missing `input` must still carry an `x-litellm-call-id` response header that parses as a
    UUID, so a caller can quote it to an operator. UNVERIFIED: the embeddings error path builds its headers in
    `ProxyBaseLLMRequestProcessing.handle_llm_api_exception`; the header value was not traced for this early error."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = openai_embedding_model(scenario, provider)
        response: Final = gateway.request("POST", "/v1/embeddings", {"model": model})
        assert response.status_code == 400, response.text
        assert uuid.UUID(response.headers["x-litellm-call-id"]), response.headers


def test_an_unknown_model_is_refused_with_400_naming_the_model(gateway: Gateway) -> None:
    """A caller embeds with a model name that no deployment, alias or wildcard matches.

    The proxy must answer 400 and the message must contain `Invalid model name passed in model=<name>`.
    """
    ghost: Final = f"dec1-textapis-ghost-{uuid.uuid4().hex}"
    response: Final = embed(gateway, ghost)
    assert response.status_code == 400, response.text
    assert f"Invalid model name passed in model={ghost}" in response.text


def test_an_empty_input_list_ends_in_a_400(gateway: Gateway) -> None:
    """A caller posts `input: []` and the provider, like OpenAI, rejects an empty input with 400.

    The caller must get 400 whether the proxy refuses the empty list itself or relays the provider's refusal.
    UNVERIFIED: the proxy only checks that `input` is present, so the expected path is the relayed provider
    error; see `REQUIRED_BODY_PARAMS_BY_ROUTE` in `litellm/proxy/route_llm_request.py`.
    """

    def respond(call: Call) -> Reply:
        if call.method == "POST" and call.body.get("input") == []:
            return error_reply(400, "'$.input' is invalid. Please check the API reference.")
        return default_respond(call)

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = openai_embedding_model(scenario, provider)
        response: Final = embed(gateway, model, [])
        assert response.status_code == 400, response.text
        assert error_of(response)["code"] == "400"


def test_a_numeric_input_ends_in_a_400(gateway: Gateway) -> None:
    """A caller posts `input: 5`, which is neither a string nor a list, and the provider answers 400 for it.

    The caller must get 400 with `code` `"400"`, never a 500. UNVERIFIED: the proxy does not type-check
    `input`, so this assumes the value is forwarded and the provider's 400 is relayed.
    """

    def respond(call: Call) -> Reply:
        if call.method == "POST" and not isinstance(call.body.get("input"), (str, list)):
            return error_reply(400, "Invalid type for 'input': expected a string or array.")
        return default_respond(call)

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = openai_embedding_model(scenario, provider)
        response: Final = embed(gateway, model, 5)
        assert response.status_code == 400, response.text
        assert error_of(response)["code"] == "400"


def test_a_body_that_is_not_json_is_refused_with_a_4xx(gateway: Gateway) -> None:
    """A caller posts the bytes `{"model": ` with `Content-Type: application/json`.

    The proxy must answer a client error (400 or 422) with a JSON body and must not answer 500.
    UNVERIFIED: the exact status and message come from `read_request_body`
    (`litellm/proxy/common_utils/http_parsing_utils.py`) and were not traced.
    """
    response: Final = gateway.client.post(
        "/v1/embeddings",
        content=b'{"model": ',
        headers={"Authorization": f"Bearer {gateway.key}", "Content-Type": "application/json"},
    )
    assert response.status_code in (400, 422), response.text
    assert response.headers["content-type"].startswith("application/json"), response.headers


@pytest.mark.parametrize("status", [400, 401, 429, 500])
def test_a_provider_error_is_returned_with_its_status_in_the_openai_error_envelope(
    gateway: Gateway, status: int
) -> None:
    """The provider answers 400, 401, 429 or 500 with an OpenAI error body whose message is
    `dec1 embeddings upstream said no`.

    The caller gets the same HTTP status, `error.code` equal to that status as a string, `error.param` null,
    `error.type` `invalid_request_error`, `authentication_error`, `throttling_error` or `internal_server_error`
    respectively, and a message that starts with litellm's exception prefix for an OpenAI provider and carries
    the provider's text.
    Pinned current behaviour: on embeddings the message after `OpenAIException - ` is the OpenAI client's own
    rendering, `Error code: <status> - ` followed by the provider's error body as a Python dict, so the
    provider's message appears inside it rather than right after the prefix, and the 401 prefix names
    `AuthenticationError` once while the 429 and 500 prefixes repeat the class name.
    """
    with peer(failing(status, "dec1 embeddings upstream said no")) as provider, gateway.scenario() as scenario:
        model: Final = openai_embedding_model(scenario, provider)
        response: Final = embed(gateway, model)
        assert response.status_code == status, response.text
        error: Final = error_of(response)
        assert {name: error[name] for name in ("type", "param", "code")} == {
            "type": _OPENAI_ERROR_TYPE[status],
            "param": None,
            "code": str(status),
        }, response.text
        message: Final = error_message(response)
        assert message.startswith(
            _OPENAI_PREFIX[status] + "{'error': {'message': 'dec1 embeddings upstream said no'"
        ), message
        assert len(provider.posts()) >= 1


def test_a_provider_error_response_carries_the_call_id_and_model_id_headers(gateway: Gateway) -> None:
    """The provider answers 400. The error response must carry `x-litellm-call-id` (a UUID) and
    `x-litellm-model-id` (non-empty), the same headers a success carries, so failed calls can be traced to a
    deployment. UNVERIFIED: `x-litellm-model-id` on errors depends on `maybe_get_model_id` finding the
    deployment in the logging object."""
    with peer(failing(400)) as provider, gateway.scenario() as scenario:
        model: Final = openai_embedding_model(scenario, provider)
        response: Final = embed(gateway, model)
        assert response.status_code == 400, response.text
        assert uuid.UUID(response.headers["x-litellm-call-id"]), response.headers
        assert response.headers["x-litellm-model-id"], response.headers


def test_a_success_carries_the_litellm_tracing_and_cost_headers(gateway: Gateway) -> None:
    """One served embeddings call priced at 0.001 per input token with 7 prompt tokens.

    The response must carry `x-litellm-call-id` (a version 4 UUID), a non-empty `x-litellm-model-id`,
    `x-litellm-model-api-base` equal to the deployment's `api_base`, a non-empty `x-litellm-version`,
    `x-litellm-response-cost` equal to 0.007 and `x-litellm-model-group` equal to the public model name.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = openai_embedding_model(scenario, provider)
        response: Final = embed(gateway, model)
        assert response.status_code == 200, response.text
        headers: Final = response.headers
        assert uuid.UUID(headers["x-litellm-call-id"]).version == 4, headers
        assert headers["x-litellm-model-id"], headers
        assert headers["x-litellm-model-api-base"] == f"{provider.url}/v1", headers
        assert headers["x-litellm-version"], headers
        assert float(headers["x-litellm-response-cost"]) == pytest.approx(EMBEDDING_COST), headers
        assert headers["x-litellm-model-group"] == model, headers


def test_a_call_id_sent_by_the_caller_is_echoed_in_the_response_header(gateway: Gateway) -> None:
    """A caller sends `x-litellm-call-id: <uuid>` with an embeddings request.

    The response header `x-litellm-call-id` must be that same value, so callers can correlate their own ids
    with proxy logs.
    """
    call_id: Final = str(uuid.uuid4())
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = openai_embedding_model(scenario, provider)
        response: Final = embed(gateway, model, headers={"x-litellm-call-id": call_id})
        assert response.status_code == 200, response.text
        assert response.headers["x-litellm-call-id"] == call_id


def test_the_response_content_type_is_json(gateway: Gateway) -> None:
    """A served embeddings response has `Content-Type: application/json` (the handler uses an ORJSON response
    class). SDKs refuse to parse a body announced as anything else."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = openai_embedding_model(scenario, provider)
        response: Final = embed(gateway, model)
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].split(";")[0] == "application/json"


def test_provider_rate_limit_headers_are_relayed_with_the_llm_provider_prefix(gateway: Gateway) -> None:
    """The provider answers 200 with `x-ratelimit-remaining-requests: 41`.

    The caller must see `llm_provider-x-ratelimit-remaining-requests: 41` on the embeddings response, the same
    relay chat completions does. UNVERIFIED: the embeddings path stores provider headers in
    `_hidden_params["additional_headers"]`; the prefixed name was read from the chat path
    (`litellm/litellm_core_utils/core_helpers.py:process_response_headers`), not confirmed for embeddings.
    """

    def respond(call: Call) -> Reply:
        reply: Final = default_respond(call)
        return Reply(body=reply.body, headers={"x-ratelimit-remaining-requests": "41"})

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = openai_embedding_model(scenario, provider)
        response: Final = embed(gateway, model)
        assert response.status_code == 200, response.text
        assert response.headers["llm_provider-x-ratelimit-remaining-requests"] == "41", response.headers
