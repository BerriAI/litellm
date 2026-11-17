"""The proxy's Azure compatible routes reaching Azure deployments end to end.

The proxy serves Azure OpenAI SDK clients on `/openai/deployments/{model}/chat/completions`,
`/openai/deployments/{model}/embeddings` and the older `/engines/{model}/chat/completions`
(`litellm/proxy/proxy_server.py:13577`). The path segment names the proxy model group and wins over a body
`model` (`resolve_inference_model`, `litellm/proxy/common_utils/http_parsing_utils.py:100`). Azure SDK clients
authenticate with an `api-key` header (`SpecialHeaders.azure_authorization = "API-Key"`) or through API
Management with `Ocp-Apim-Subscription-Key`, and the proxy accepts both as the virtual key
(`litellm/proxy/auth/user_api_key_auth.py:1550`).

The router (Rust router core) then picks one of the group's Azure deployments and the call goes out with that
deployment's own `api_version` and key, never the caller's. `compatibility/test_dec1_http_route_aliases.py`
already pins that the path model wins over the body model, so these tests focus on the Azure deployment that
is reached, the Azure style credentials and routing across deployments.
"""

from __future__ import annotations

import uuid
from typing import Final

import httpx
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from integration.providers.dec1_azure_support import (
    API_KEY,
    DATED_API_VERSION,
    DEPLOYMENT,
    EMBEDDING_DEPLOYMENT,
    always,
    azure,
    azure_error,
    chat_reply,
    chat_stream,
    embedding_json,
    json_reply,
    marker,
    path_of,
    query_of,
    spend_row,
    sse_events,
    streamed_text,
)


def _ok(_: Request) -> Reply:
    return chat_reply("via deployments route")


def _deployments_chat(
    gateway: Gateway, group: str, *, headers: dict[str, str] | None = None, **fields: object
) -> httpx.Response:
    """POST the Azure SDK shaped call: model in the path, `api-version` in the query, no `model` in the body."""
    body: Final = {"messages": [{"role": "user", "content": marker()}], **fields}
    if headers is None:
        return gateway.request(
            "POST",
            f"/openai/deployments/{group}/chat/completions",
            body,
            params={"api-version": "2024-02-01"},  # pyright: ignore[reportArgumentType]
        )
    return gateway.client.post(
        f"/openai/deployments/{group}/chat/completions",
        json=body,
        params={"api-version": "2024-02-01"},
        headers=headers,
    )


def test_the_deployments_route_reaches_the_azure_deployment_of_the_path_group(gateway: Gateway) -> None:
    """The caller posts `/openai/deployments/<group>/chat/completions?api-version=2024-02-01` with no body `model`.

    The proxy must route to the group's `azure/gpt-4o-dec1` deployment: Azure sees one
    `POST /openai/deployments/gpt-4o-dec1/chat/completions`, and the caller gets HTTP 200 with
    `via deployments route`.
    """
    with azure(gateway, _ok) as rig:
        response: Final = _deployments_chat(gateway, rig.model)
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "via deployments route"
        assert path_of(rig.one_post()) == f"/openai/deployments/{DEPLOYMENT}/chat/completions"


def test_the_callers_api_version_query_is_not_forwarded_to_azure(gateway: Gateway) -> None:
    """The caller sends `api-version=2024-02-01` in its query; the deployment is configured with `2024-10-21`.

    Azure must receive the deployment's own `api-version: 2024-10-21`, because the version is a deployment
    setting and the caller's query is only the Azure SDK's habit. UNVERIFIED: no code was found that copies the
    inbound query into the call, so the deployment value is expected.
    """
    with azure(gateway, _ok) as rig:
        assert _deployments_chat(gateway, rig.model).status_code == 200
        assert query_of(rig.one_post()) == {"api-version": DATED_API_VERSION}


def test_an_api_key_header_authenticates_an_azure_sdk_caller(gateway: Gateway) -> None:
    """The caller sends its virtual key as `api-key: <key>` and no `Authorization` header, as AzureOpenAI does.

    The proxy must accept it (`azure_api_key_header`, `user_api_key_auth.py:1550`): HTTP 200, and Azure receives
    the deployment key `synthetic-azure-key`, not the caller's virtual key.
    """
    with azure(gateway, _ok) as rig:
        key: Final = rig.scenario.key()
        response: Final = _deployments_chat(gateway, rig.model, headers={"api-key": key})
        assert response.status_code == 200, response.text
        assert rig.one_post().headers.get("api-key") == API_KEY


def test_an_ocp_apim_subscription_key_header_authenticates_a_caller(gateway: Gateway) -> None:
    """The caller sends its virtual key as `Ocp-Apim-Subscription-Key`, as an Azure API Management client does.

    The proxy must accept it as the virtual key and answer HTTP 200 (`azure_apim_header`, `user_api_key_auth.py:1557`).
    """
    with azure(gateway, _ok) as rig:
        key: Final = rig.scenario.key()
        response: Final = _deployments_chat(gateway, rig.model, headers={"Ocp-Apim-Subscription-Key": key})
        assert response.status_code == 200, response.text


def test_an_unknown_api_key_header_is_refused_with_401(gateway: Gateway) -> None:
    """The caller sends `api-key: sk-not-a-real-key` on the deployments route.

    The proxy must answer HTTP 401 and Azure must receive nothing.
    """
    with azure(gateway, _ok) as rig:
        response: Final = _deployments_chat(gateway, rig.model, headers={"api-key": "sk-not-a-real-key"})
        assert response.status_code == 401, response.text
        assert rig.posts() == ()


def test_the_engines_route_reaches_the_same_azure_deployment(gateway: Gateway) -> None:
    """The caller posts `/engines/<group>/chat/completions` (the pre 2023 Azure SDK path).

    Azure must see one `POST /openai/deployments/gpt-4o-dec1/chat/completions` and the caller gets HTTP 200.
    """
    with azure(gateway, _ok) as rig:
        response: Final = gateway.request(
            "POST", f"/engines/{rig.model}/chat/completions", {"messages": [{"role": "user", "content": marker()}]}
        )
        assert response.status_code == 200, response.text
        assert path_of(rig.one_post()) == f"/openai/deployments/{DEPLOYMENT}/chat/completions"


def test_the_deployments_embeddings_route_reaches_the_azure_embedding_deployment(gateway: Gateway) -> None:
    """The caller posts `/openai/deployments/<group>/embeddings` with only `input` in the body.

    Azure must see `POST /openai/deployments/text-embedding-3-small/embeddings` and the caller gets the vector back.
    """
    reply: Final = json_reply(embedding_json([[0.5, 0.25]]))
    with azure(gateway, always(reply), model=f"azure/{EMBEDDING_DEPLOYMENT}") as rig:
        response: Final = gateway.request(
            "POST",
            f"/openai/deployments/{rig.model}/embeddings",
            {"input": [marker()]},
            params={"api-version": "2024-02-01"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["data"][0]["embedding"] == [0.5, 0.25]
        assert path_of(rig.one_post()) == f"/openai/deployments/{EMBEDDING_DEPLOYMENT}/embeddings"


def test_the_deployments_route_streams(gateway: Gateway) -> None:
    """The caller streams on `/openai/deployments/<group>/chat/completions` with `stream: true`.

    The caller must receive an event stream whose deltas join to `hello stream`.
    """
    with azure(gateway, always(chat_stream(("hello ", "stream")))) as rig:
        response: Final = _deployments_chat(gateway, rig.model, stream=True)
        assert response.status_code == 200, response.text
        assert streamed_text(sse_events(response)) == "hello stream"


def test_a_deployments_route_call_is_logged_under_the_path_group(gateway: Gateway) -> None:
    """A priced call through the deployments route.

    The SpendLogs row must name the path group as `model_group` and `custom_llm_provider: azure`.
    """
    with azure(gateway, _ok, input_cost_per_token=0.001, output_cost_per_token=0.002) as rig:
        response: Final = _deployments_chat(gateway, rig.model)
        assert response.status_code == 200, response.text
        row: Final = spend_row(response.json()["id"])
        assert (row["model_group"], row["custom_llm_provider"]) == (rig.model, "azure")


def test_the_router_spreads_deployments_route_calls_over_two_azure_deployments(gateway: Gateway) -> None:
    """One group holds two Azure deployments on two different peers; the caller makes 12 calls on the deployments route.

    Both peers must receive at least one POST and together exactly 12, so the route goes through the router
    rather than pinning the first deployment. With simple shuffle the chance of a false failure is 2 / 4096.
    """
    group: Final = f"dec1-azure-pair-{uuid.uuid4().hex}"
    with wire_server(_ok) as first, wire_server(_ok) as second, gateway.scenario() as scenario:
        for wire in (first, second):
            created = gateway.post(
                "/model/new",
                {
                    "model_name": group,
                    "litellm_params": {
                        "model": f"azure/{DEPLOYMENT}",
                        "api_base": wire.url,
                        "api_key": API_KEY,
                        "api_version": DATED_API_VERSION,
                    },
                },
            )
            scenario.cleanups.callback(scenario.delete_model, str(created["model_info"]["id"]))  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]
        for _ in range(12):
            assert _deployments_chat(gateway, group).status_code == 200
        counts: Final = [sum(1 for request in wire.drain() if request.method == "POST") for wire in (first, second)]
        assert sum(counts) == 12, counts
        assert min(counts) >= 1, counts


def test_a_429_on_the_deployments_route_retries_on_the_other_azure_deployment(gateway: Gateway) -> None:
    """One group holds a deployment that always answers 429 and a healthy one; SDK `max_retries: 0`; 6 calls with `num_retries: 2`.

    Every call must end in HTTP 200 with the healthy deployment's answer, because the router retries on the
    other deployment of the group. UNVERIFIED: assumes a retry avoids the deployment that just failed; if the
    router can pick the 429 deployment three times in a row, this fails about one call in eight.
    """
    group: Final = f"dec1-azure-pair-{uuid.uuid4().hex}"
    with (
        wire_server(always(azure_error(429, "429", "busy"))) as limited,
        wire_server(always(chat_reply("healthy"))) as healthy,
        gateway.scenario() as scenario,
    ):
        for wire in (limited, healthy):
            created = gateway.post(
                "/model/new",
                {
                    "model_name": group,
                    "litellm_params": {
                        "model": f"azure/{DEPLOYMENT}",
                        "api_base": wire.url,
                        "api_key": API_KEY,
                        "api_version": DATED_API_VERSION,
                        "max_retries": 0,
                    },
                },
            )
            scenario.cleanups.callback(scenario.delete_model, str(created["model_info"]["id"]))  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]
        for _ in range(6):
            response = _deployments_chat(gateway, group, num_retries=2)
            assert response.status_code == 200, response.text
            assert response.json()["choices"][0]["message"]["content"] == "healthy"
