"""`/v1/messages` on Claude through Vertex AI: custom `api_base` shapes, credential and project sources, betas, cost.

On main `/v1/messages` for `vertex_ai/claude-*` still runs on Python (`litellm/rust_bridge/catalog.py:209-216`
has no Vertex rule) through `VertexAIPartnerModelsAnthropicMessagesConfig`
(`litellm/llms/vertex_ai/vertex_ai_partner_models/anthropic/messages/transformation.py`), and the Rust port
(draft #51294) already exists as `litellm/rust/crates/llms/src/vertex_ai/messages/transformation.rs`. These tests
pin what both must agree on where they currently differ or where nothing pins it yet.

URL: Python builds `create_vertex_url` on the custom base and then `_check_custom_proxy` grafts or appends
(`litellm/llms/vertex_ai/vertex_llm_base.py:352-424,771-852`); the Rust port `complete_vertex_anthropic_url`
(`transformation.rs:196-234`) strips one leading `/v1` instead. For a base ending in `/v1` or `/v1beta1` Python
doubles the version (`/v1/v1/projects/...`), which is a product bug pinned as a skip. Credentials: Python reads
`vertex_project` or `vertex_ai_project`, `vertex_location` or `vertex_ai_location`, `vertex_credentials` or
`vertex_ai_credentials` (`vertex_llm_base.py:1354-1426`) and always overwrites `Authorization` with the minted
token (`transformation.py:181`); the Rust port forwards an existing `Authorization` header and mints nothing
(`transformation.rs:140-146`, credential-gaps row 26). Betas derived from the body go in the `anthropic-beta`
header, sorted and comma joined (`transformation.py:194-238`).

Access tokens come from the shared scripted upstream's `/_oauth/token` (`scripted-token`); the Vertex host is an
owned loopback peer.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.wire import Reply, Request
from integration.messages_endpoint.dec1_messages_support import JSON_OBJECT, Rig
from integration.messages_endpoint.dec1_msgprov_support import (
    VERTEX_BACKEND,
    VERTEX_LOCATION,
    VERTEX_PROJECT,
    VERTEX_TOKEN,
    anthropic_json_reply,
    message_id,
    only_spend_row,
    path_of,
    provider_deployment,
    query_of,
    spend_of,
    vertex_credentials,
    vertex_echo,
    vertex_path,
)
from pydantic import JsonValue

_OTHER_PROJECT: Final = "synthetic-vertex-other-project"
_TOOL: Final[dict[str, JsonValue]] = {
    "name": "get_weather",
    "description": "Weather for a city",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
}


def _vertex(
    gateway: Gateway,
    respond: Callable[[Request], Reply] = vertex_echo,
    *,
    base_suffix: str = "",
    **params: JsonValue,
) -> AbstractContextManager[Rig]:
    settings: Final[dict[str, JsonValue]] = {
        "vertex_project": VERTEX_PROJECT,
        "vertex_location": VERTEX_LOCATION,
        "vertex_credentials": vertex_credentials(gateway),
        **params,
    }
    return provider_deployment(
        gateway, respond, model=f"vertex_ai/{VERTEX_BACKEND}", base_suffix=base_suffix, api_key=None, **settings
    )


def _betas(request: Request) -> list[str]:
    return [value.strip() for value in request.headers.get("anthropic-beta", "").split(",") if value.strip()]


# ---------------------------------------------------------------- custom api_base shapes


def test_a_trailing_slash_api_base_reaches_the_default_raw_predict_path(gateway: Gateway) -> None:
    """`api_base: http://<peer>/` has an empty path, so the default
    `/v1/projects/<p>/locations/us-east5/publishers/anthropic/models/<model>:rawPredict` is joined under it with a
    single slash (`_check_custom_proxy` `path in ("", "/")` branch, `vertex_llm_base.py:838-839`; Rust
    `transformation.rs:216-222`)."""
    with _vertex(gateway, base_suffix="/") as rig:
        assert rig.post(rig.body()).status_code == 200
        assert path_of(rig.one_call()) == vertex_path() + ":rawPredict"


def test_an_api_base_ending_in_v1_is_not_doubled(gateway: Gateway) -> None:
    """`api_base: http://<peer>/v1` should post to `/v1/projects/.../models/<model>:rawPredict`, as the Rust port
    does (`"/v1" | "/v1beta1"` arm, `transformation.rs:224`). Python joins `v1/projects/...` onto the custom base
    in `create_vertex_url` (`vertex_llm_base.py:377`) and `graft_default_vertex_path` (`:45-50`) then drops only
    one leading `v1` of the already doubled path, so the peer receives `/v1/v1/projects/...:rawPredict`. The same
    bug on chat is skipped in `providers/test_dec1_vertex_api_base_shapes.py:107`."""
    pytest.skip("BUG: Vertex Claude /v1/messages with api_base ending in /v1 posts to /v1/v1/projects/...:rawPredict")
    with _vertex(gateway, base_suffix="/v1") as rig:
        assert rig.post(rig.body()).status_code == 200
        assert path_of(rig.one_call()) == vertex_path() + ":rawPredict"


def test_an_api_base_ending_in_v1beta1_keeps_its_own_version_once(gateway: Gateway) -> None:
    """`api_base: http://<peer>/v1beta1` should post to `/v1beta1/projects/.../models/<model>:rawPredict` (Rust:
    `format!("{custom}{}", path.trim_start_matches("/v1"))`, `transformation.rs:224`). Python grafts onto
    `/v1beta1/v1/projects/...` and keeps the inner `v1`, so the peer receives `/v1beta1/v1/projects/...`."""
    pytest.skip("BUG: Vertex Claude /v1/messages with api_base ending in /v1beta1 posts to /v1beta1/v1/projects/...")
    with _vertex(gateway, base_suffix="/v1beta1") as rig:
        assert rig.post(rig.body()).status_code == 200
        assert path_of(rig.one_call()) == "/v1beta1" + vertex_path().removeprefix("/v1") + ":rawPredict"


def test_a_custom_path_api_base_is_the_resource_with_raw_predict_appended(gateway: Gateway) -> None:
    """`api_base: http://<peer>/gateway/claude` is neither empty nor a version, so it is taken as the whole model
    resource and the call goes to `/gateway/claude:rawPredict` (`vertex_llm_base.py:842-848`; Rust `_` arm,
    `transformation.rs:225`). Gateways that front Vertex rely on this shape."""
    with _vertex(gateway, base_suffix="/gateway/claude") as rig:
        assert rig.post(rig.body()).status_code == 200
        assert path_of(rig.one_call()) == "/gateway/claude:rawPredict"


def test_a_custom_path_api_base_streams_to_stream_raw_predict_with_alt_sse(gateway: Gateway) -> None:
    """Streaming through the same custom base posts to `/gateway/claude:streamRawPredict` with query `alt=sse`
    (`vertex_llm_base.py:849-852`; Rust `transformation.rs:227-233`)."""

    def respond(request: Request) -> Reply:
        return vertex_echo(request)

    with _vertex(gateway, respond, base_suffix="/gateway/claude") as rig:
        response, _text = rig.stream(rig.body(stream=True))
        assert response.status_code == 200
        call: Final = rig.one_call()
        assert (path_of(call), query_of(call)) == ("/gateway/claude:streamRawPredict", "alt=sse")


def test_a_unary_call_through_a_bare_host_carries_no_alt_query(gateway: Gateway) -> None:
    """A unary call through a bare host `api_base` has an empty query string: `alt=sse` is only added when
    `stream is True` (`vertex_llm_base.py:849`)."""
    with _vertex(gateway) as rig:
        assert rig.post(rig.body()).status_code == 200
        assert query_of(rig.one_call()) == ""


# ---------------------------------------------------------------- project, location and credential sources


def test_vertex_ai_project_and_vertex_ai_location_aliases_build_the_url(gateway: Gateway) -> None:
    """A deployment configured only with the alias names `vertex_ai_project: synthetic-vertex-other-project` and
    `vertex_ai_location: europe-west1` posts to `/v1/projects/synthetic-vertex-other-project/locations/europe-west1/
    publishers/anthropic/models/<model>:rawPredict` (`safe_get_vertex_ai_project` and `explicit_vertex_ai_location`,
    `vertex_llm_base.py:1367-1402`). UNVERIFIED: that the Rust `VertexParams` reads the same alias names."""
    with _vertex(
        gateway,
        vertex_project=None,
        vertex_location=None,
        vertex_ai_project=_OTHER_PROJECT,
        vertex_ai_location="europe-west1",
    ) as rig:
        assert rig.post(rig.body()).status_code == 200
        assert path_of(rig.one_call()) == vertex_path(project=_OTHER_PROJECT, location="europe-west1") + ":rawPredict"


def test_vertex_ai_credentials_alias_is_used_to_mint_the_token(gateway: Gateway) -> None:
    """With the service account given as `vertex_ai_credentials` (and no `vertex_credentials`) the call still carries
    `Authorization: Bearer scripted-token` minted from it (`safe_get_vertex_ai_credentials`,
    `vertex_llm_base.py:1388-1392`)."""
    with _vertex(gateway, vertex_credentials=None, vertex_ai_credentials=vertex_credentials(gateway)) as rig:
        assert rig.post(rig.body()).status_code == 200
        assert rig.one_call().headers["authorization"] == f"Bearer {VERTEX_TOKEN}"


def test_service_account_credentials_given_as_an_object_are_accepted(gateway: Gateway) -> None:
    """`vertex_credentials` given as a JSON object rather than JSON text mints the same `scripted-token` and posts to
    the usual path (credential-gaps row 1: Python `_credentials_from_json(dict)`, Rust serializes the object back to
    text). YAML configs usually give credentials this way."""
    as_object: Final = JSON_OBJECT.validate_json(vertex_credentials(gateway))
    with _vertex(gateway, vertex_credentials=as_object) as rig:
        assert rig.post(rig.body()).status_code == 200
        call: Final = rig.one_call()
        assert call.headers["authorization"] == f"Bearer {VERTEX_TOKEN}"
        assert path_of(call) == vertex_path() + ":rawPredict"


def test_vertex_project_beats_the_service_account_project(gateway: Gateway) -> None:
    """With credentials whose `project_id` is `synthetic-vertex-project` and `vertex_project:
    synthetic-vertex-other-project`, the URL names the configured project: `create_vertex_url` is called with
    `vertex_project or project_id` (`vertex_llm_base.py:396`), and Rust resolves `get_vertex_ai_project` before
    `get_vertex_ai_project_from_credentials` (`transformation.rs:84-87`)."""
    with _vertex(gateway, vertex_project=_OTHER_PROJECT) as rig:
        assert rig.post(rig.body()).status_code == 200
        assert path_of(rig.one_call()) == vertex_path(project=_OTHER_PROJECT) + ":rawPredict"


def test_a_deployment_authorization_header_is_replaced_by_the_minted_token(gateway: Gateway) -> None:
    """A deployment with `extra_headers: {"Authorization": "Bearer stale-operator-token"}` still sends
    `Authorization: Bearer scripted-token`, because Python always sets the minted token over whatever headers came in
    (`_finalize_vertex_messages_environment`, `transformation.py:180-181`). The Rust port forwards an existing
    `Authorization` and mints nothing (`transformation.rs:140-146`, credential-gaps row 26), so this pins the
    behaviour the port must keep. UNVERIFIED: that deployment `extra_headers` reach the messages headers here."""
    with _vertex(gateway, extra_headers={"Authorization": "Bearer stale-operator-token"}) as rig:
        assert rig.post(rig.body()).status_code == 200
        assert rig.one_call().headers["authorization"] == f"Bearer {VERTEX_TOKEN}"


def test_the_callers_x_api_key_never_reaches_vertex(gateway: Gateway) -> None:
    """A caller authenticating with `x-api-key: <proxy key>` (the Anthropic SDK default) gets a 200 and the Vertex
    request carries no `x-api-key` header and only the minted bearer, so the proxy key never leaves the proxy."""
    with _vertex(gateway) as rig:
        response: Final = rig.post_raw(
            {"x-api-key": gateway.key, "content-type": "application/json", "anthropic-version": "2023-06-01"},
            content=json.dumps(rig.body()).encode(),
        )
        assert response.status_code == 200, response.text
        call: Final = rig.one_call()
        assert "x-api-key" not in call.headers
        assert call.headers["authorization"] == f"Bearer {VERTEX_TOKEN}"


# ---------------------------------------------------------------- derived betas


def test_context_management_adds_its_beta_header(gateway: Gateway) -> None:
    """A body with `context_management: {"edits": [{"type": "clear_tool_uses_20250919"}]}` makes the Vertex request
    carry `context-management-2025-06-27` in `anthropic-beta` (`context_management_betas`, `transformation.py:202`)."""
    edits: Final[dict[str, JsonValue]] = {"edits": [{"type": "clear_tool_uses_20250919"}]}
    with _vertex(gateway) as rig:
        assert rig.post(rig.body(context_management=edits)).status_code == 200
        assert "context-management-2025-06-27" in _betas(rig.one_call())


def test_derived_betas_are_sorted_and_comma_joined_with_the_callers(gateway: Gateway) -> None:
    """A caller beta `fine-grained-tool-streaming-2025-05-14` plus a `web_search_20250305` tool gives one
    `anthropic-beta` header whose values are sorted, `fine-grained-tool-streaming-2025-05-14,web-search-2025-03-05`
    (`",".join(sorted(all_beta_values))`, `transformation.py:237-238`). UNVERIFIED: that the Vertex beta row keeps
    `fine-grained-tool-streaming-2025-05-14` rather than dropping it."""
    tool: Final[dict[str, JsonValue]] = {"type": "web_search_20250305", "name": "web_search", "max_uses": 1}
    with _vertex(gateway) as rig:
        response: Final = rig.post(
            rig.body(tools=[tool]), headers={"anthropic-beta": "fine-grained-tool-streaming-2025-05-14"}
        )
        assert response.status_code == 200, response.text
        assert (
            rig.one_call().headers["anthropic-beta"] == "fine-grained-tool-streaming-2025-05-14,web-search-2025-03-05"
        )


def test_a_plain_call_sends_no_anthropic_beta_header(gateway: Gateway) -> None:
    """A text-only call with no caller beta, no tools and no thinking sends no `anthropic-beta` header at all, since
    the header is only set when the beta set is non-empty (`transformation.py:237`)."""
    with _vertex(gateway) as rig:
        assert rig.post(rig.body(tools=[_TOOL])).status_code == 200
        assert "anthropic-beta" not in rig.one_call().headers


# ---------------------------------------------------------------- cost from the cost map


def test_a_vertex_claude_call_without_deployment_prices_is_priced_from_the_vertex_row(gateway: Gateway) -> None:
    """With no prices on the deployment, 1000 input, 100 output, 2000 cache read and 400 cache creation tokens are
    priced on `vertex_ai/claude-sonnet-4-5@20250929` (3e-06, 1.5e-05, 3e-07, 3.75e-06 in
    `litellm/model_prices_and_context_window_backup.json`): `0.003 + 0.0015 + 0.0006 + 0.0015 = 0.0066`, on the row
    and in `x-litellm-response-cost`."""
    usage: Final[dict[str, JsonValue]] = {
        "input_tokens": 1000,
        "output_tokens": 100,
        "cache_read_input_tokens": 2000,
        "cache_creation_input_tokens": 400,
    }
    reply: Final = anthropic_json_reply(message_id(), model="claude-sonnet-4-5-20250929", usage=usage)
    with _vertex(gateway, lambda _request: reply) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 200, response.text
        assert object_value(JSON_OBJECT.validate_json(response.content)["usage"])["cache_read_input_tokens"] == 2000
        expected: Final = 1000 * 3e-06 + 100 * 1.5e-05 + 2000 * 3e-07 + 400 * 3.75e-06
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(expected)
        assert spend_of(only_spend_row(rig)) == pytest.approx(expected)
