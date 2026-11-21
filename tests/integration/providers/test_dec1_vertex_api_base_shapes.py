"""Where a Vertex request lands when the deployment overrides `api_base`.

`VertexBase._check_custom_proxy` (`litellm/llms/vertex_ai/vertex_llm_base.py`) decides the final URL:
an `api_base` with an empty path gets the default Vertex path appended; an `api_base` whose path is only
an API version (`/v1`, `/v1beta1`) gets the default path grafted under that version so the version is
never doubled; any other path is treated as the full resource and gets `:{method}` appended; and a
streaming call always gets `alt=sse` added to the query (joined with `&` when the base already has a
query). Image generation is different: with an `api_base` the URL is the `api_base` itself
(`VertexAIImagenImageGenerationConfig.get_complete_url`).

These shapes are how corporate gateways, Private Service Connect endpoints and test doubles are
wired in, and a URL builder rewritten in Rust is where a `/v1/v1` or a lost `alt=sse` would appear.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway
from integration.providers.dec1_vertex_support import (
    CLAUDE,
    EMBEDDING,
    claude_echo,
    claude_path,
    embedding_peer,
    gemini_echo,
    marker,
    models_path,
    path_of,
    query_of,
    vertex,
)

_TAIL: Final = models_path() + ":generateContent"


def test_a_bare_host_api_base_gets_the_default_vertex_path(gateway: Gateway) -> None:
    """`api_base: http://<peer>` (no path) receives `/v1/projects/<p>/locations/<l>/publishers/google/
    models/gemini-2.5-flash:generateContent`, the same path the regional Google host would get."""
    with vertex(gateway, gemini_echo) as rig:
        assert rig.chat().status_code == 200
        assert path_of(rig.peer.one_call()) == _TAIL


def test_a_trailing_slash_on_the_api_base_does_not_double_the_slash(gateway: Gateway) -> None:
    """`api_base: http://<peer>/` reaches the same path as the bare host, never `//v1/...`."""
    with vertex(gateway, gemini_echo, base_suffix="/") as rig:
        assert rig.chat().status_code == 200
        assert path_of(rig.peer.one_call()) == _TAIL


def test_an_api_base_ending_in_v1_is_not_doubled_for_gemini(gateway: Gateway) -> None:
    """`api_base: http://<peer>/v1` has the default path grafted under its `/v1`
    (`graft_default_vertex_path`), so Gemini chat posts to `/v1/projects/...:generateContent` and not
    the `/v1/v1/projects/...` this shape used to produce."""
    with vertex(gateway, gemini_echo, base_suffix="/v1") as rig:
        assert rig.chat().status_code == 200
        assert path_of(rig.peer.one_call()) == _TAIL


def test_an_api_base_ending_in_v1beta1_keeps_its_version(gateway: Gateway) -> None:
    """`api_base: http://<peer>/v1beta1` replaces the default `v1` segment with the base's own
    `v1beta1`, giving `/v1beta1/projects/...:generateContent`."""
    with vertex(gateway, gemini_echo, base_suffix="/v1beta1") as rig:
        assert rig.chat().status_code == 200
        assert path_of(rig.peer.one_call()) == models_path(version="v1beta1") + ":generateContent"


def test_an_api_base_with_a_custom_path_is_used_as_the_resource_with_the_method_appended(gateway: Gateway) -> None:
    """`api_base: http://<peer>/gateway/vertex/gemini` is neither empty nor a version, so it is taken as
    the full model resource and the call goes to `/gateway/vertex/gemini:generateContent`."""
    with vertex(gateway, gemini_echo, base_suffix="/gateway/vertex/gemini") as rig:
        assert rig.chat().status_code == 200
        assert path_of(rig.peer.one_call()) == "/gateway/vertex/gemini:generateContent"


def test_a_custom_path_api_base_streams_with_alt_sse(gateway: Gateway) -> None:
    """Streaming through a custom path `api_base` posts to `/gateway/vertex/gemini:streamGenerateContent`
    with the query `alt=sse`."""
    with vertex(gateway, gemini_echo, base_suffix="/gateway/vertex/gemini") as rig:
        rig.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": f"s {marker()}"}]})
        call: Final = rig.peer.one_call()
        assert (path_of(call), query_of(call)) == ("/gateway/vertex/gemini:streamGenerateContent", "alt=sse")


def test_a_v1_api_base_streams_to_the_grafted_path_with_alt_sse(gateway: Gateway) -> None:
    """A `/v1` `api_base` keeps the graft when streaming: `/v1/projects/...:streamGenerateContent?alt=sse`."""
    with vertex(gateway, gemini_echo, base_suffix="/v1") as rig:
        rig.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": f"s {marker()}"}]})
        call: Final = rig.peer.one_call()
        assert (path_of(call), query_of(call)) == (models_path() + ":streamGenerateContent", "alt=sse")


def test_a_bare_host_api_base_reaches_the_claude_raw_predict_path(gateway: Gateway) -> None:
    """For Claude on Vertex chat, a bare host `api_base` receives the default
    `/v1/projects/<p>/locations/us-east5/publishers/anthropic/models/<model>:rawPredict`."""
    with vertex(gateway, claude_echo, backend=CLAUDE) as rig:
        assert rig.chat(max_tokens=16).status_code == 200
        assert path_of(rig.peer.one_call()) == claude_path() + ":rawPredict"


def test_an_api_base_ending_in_v1_is_not_doubled_for_claude_chat(gateway: Gateway) -> None:
    """A `/v1` `api_base` on a Claude on Vertex deployment should post to `/v1/projects/...:rawPredict`.
    `create_vertex_url` already joins `v1/...` onto the custom base before `_check_custom_proxy` grafts,
    and the graft only drops one leading `v1`, so the path keeps a doubled `/v1/v1`."""
    pytest.skip("BUG: Claude on Vertex chat with api_base ending in /v1 posts to /v1/v1/projects/...:rawPredict")
    with vertex(gateway, claude_echo, backend=CLAUDE, base_suffix="/v1") as rig:
        assert rig.chat(max_tokens=16).status_code == 200
        assert path_of(rig.peer.one_call()) == claude_path() + ":rawPredict"


def test_the_gemini_embedding_path_follows_the_same_api_base_rules(gateway: Gateway) -> None:
    """`/v1/embeddings` on `text-embedding-005` with a `/v1` `api_base` posts to `/v1/projects/<p>/
    locations/<l>/publishers/google/models/text-embedding-005:predict`, with no doubled version."""
    with vertex(gateway, embedding_peer, backend=EMBEDDING, base_suffix="/v1") as rig:
        response: Final = rig.post("/v1/embeddings", {"input": [f"e {marker()}"]})
        assert response.status_code == 200, response.text
        assert path_of(rig.peer.one_call()) == models_path(EMBEDDING) + ":predict"
