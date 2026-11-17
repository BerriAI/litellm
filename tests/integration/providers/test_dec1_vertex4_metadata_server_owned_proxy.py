"""Vertex credentials from the GCE metadata server (Application Default Credentials), chat and messages.

On GCE, GKE or Cloud Run a Vertex deployment usually names no credentials at all: `VertexBase.load_auth`
(`litellm/llms/vertex_ai/vertex_llm_base.py:220-225`) calls `google.auth.default()`, which finds no key file
and no gcloud file, pings the metadata server, reads the project from `project/project-id` and returns
Compute Engine credentials whose refresh reads `instance/service-accounts/default/` and then
`instance/service-accounts/default/token?scopes=...`. Every metadata request carries `Metadata-Flavor: Google`.

This module boots its own proxy with `GCE_METADATA_HOST` and `GCE_METADATA_IP` on a loopback double
(`MetadataPeer` in `dec1_vertex4_support.py`), an empty `CLOUDSDK_CONFIG` and no
`GOOGLE_APPLICATION_CREDENTIALS`, so the metadata server is the only ADC source and no request reaches
`metadata.google.internal`. The double is also the Vertex model host.

For the Rust port of `/v1/messages` (#51294) `credential-gaps.md` lists two gaps this module pins:
row 9 (gcp_auth ignores `GCE_METADATA_HOST` and always calls `metadata.google.internal`) and the project
table (Rust cannot read the project from the metadata server and refuses with "Missing Vertex AI project").
Expected values about google-auth's request shapes were read from google-auth 2.53 sources
(`compute_engine/_metadata.py`, `compute_engine/credentials.py`); the branch pins 2.52.0 (`uv.lock:2771`), so
they are marked UNVERIFIED where a minor version could differ.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

import pytest
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration.providers.dec1_vertex4_support import (
    CLEARED_GOOGLE_ENVIRONMENT,
    MDS_PROJECT,
    METADATA_ROOT,
    MetadataPeer,
    bearer,
    metadata_environment,
    metadata_peer,
    owned_config,
    query_params,
)
from integration.providers.dec1_vertex_support import (
    CLAUDE,
    PROJECT,
    SCOPE,
    claude_path,
    marker,
    models_path,
    path_of,
)

pytestmark: Final = [pytest.mark.timeout(2 * graceful_stop_seconds() + 240)]


@pytest.fixture(scope="module")
def mds() -> Iterator[MetadataPeer]:
    with metadata_peer() as owned:
        yield owned


@pytest.fixture(scope="module")
def mds_proxy(tmp_path_factory: pytest.TempPathFactory, mds: MetadataPeer) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("dec1-vertex4-mds")
    gcloud: Final = directory / "empty-gcloud-config"
    gcloud.mkdir()
    overrides: Final = metadata_environment(mds, gcloud)
    removed: Final = tuple(name for name in CLEARED_GOOGLE_ENVIRONMENT if name not in overrides)
    with (
        gateway_from_environment() as gateway,
        owned_proxy(
            gateway, directory, overrides, config=owned_config(directory), remove_environment=removed, workers=1
        ) as proxy,
    ):
        yield proxy


@contextmanager
def _deployment(proxy: Gateway, mds: MetadataPeer, backend: str, *, vertex_project: str) -> Iterator[str]:
    """A Vertex deployment on the metadata double with no `vertex_credentials` but a `vertex_project`."""
    with proxy.scenario() as scenario:
        yield scenario.model(
            model=f"vertex_ai/{backend}",
            api_key=None,
            api_base=mds.url,
            vertex_project=vertex_project,
            vertex_location="us-east5" if backend.startswith("claude") else "us-central1",
        )


def _chat(proxy: Gateway, model: str) -> int:
    body: Final = {"model": model, "messages": [{"role": "user", "content": f"mds {marker()}"}]}
    return proxy.request("POST", "/v1/chat/completions", body).status_code


def _messages(proxy: Gateway, model: str) -> int:
    body: Final = {"model": model, "max_tokens": 32, "messages": [{"role": "user", "content": f"mds {marker()}"}]}
    return proxy.request("POST", "/v1/messages", body).status_code


@pytest.fixture(scope="module")
def gemini_model(mds_proxy: Gateway, mds: MetadataPeer) -> Iterator[str]:
    """One Gemini deployment without credentials or project, kept for the whole module."""
    with mds_proxy.scenario() as scenario:
        yield scenario.model(
            model="vertex_ai/gemini-2.5-flash", api_key=None, api_base=mds.url, vertex_location="us-central1"
        )


@pytest.fixture(scope="module")
def claude_model(mds_proxy: Gateway, mds: MetadataPeer) -> Iterator[str]:
    """One Vertex Claude deployment without credentials or project, kept for the whole module."""
    with mds_proxy.scenario() as scenario:
        yield scenario.model(model=f"vertex_ai/{CLAUDE}", api_key=None, api_base=mds.url, vertex_location="us-east5")


# ---------------------------------------------------------------- Gemini chat


def test_gemini_chat_without_credentials_uses_a_metadata_server_token(
    mds_proxy: Gateway, mds: MetadataPeer, gemini_model: str
) -> None:
    """A Gemini deployment with no credentials on a proxy whose ADC is the metadata server answers 200 and
    its `generateContent` call carries a token the metadata double issued (`mds-token-N`)."""
    assert _chat(mds_proxy, gemini_model) == 200
    assert bearer(mds.model_calls()[-1]) in mds.issued


def test_gemini_chat_takes_the_project_from_the_metadata_server(
    mds_proxy: Gateway, mds: MetadataPeer, gemini_model: str
) -> None:
    """With no `vertex_project`, the path names the project the metadata server reports at
    `/computeMetadata/v1/project/project-id` (`google.auth.default()` returns it, `vertex_llm_base.py:224`)."""
    assert _chat(mds_proxy, gemini_model) == 200
    assert path_of(mds.model_calls()[-1]) == models_path(project=MDS_PROJECT) + ":generateContent"


def test_every_metadata_request_carries_the_metadata_flavor_header(
    mds_proxy: Gateway, mds: MetadataPeer, gemini_model: str
) -> None:
    """Every GET the proxy sends to the metadata double (ping, project id, account info, token) carries
    `Metadata-Flavor: Google`, which the real metadata server requires. UNVERIFIED against google-auth 2.52
    (read from 2.53 `_metadata.py:93-95`)."""
    assert _chat(mds_proxy, gemini_model) == 200
    calls: Final = mds.metadata_calls()
    assert calls
    assert all(call.headers.get("metadata-flavor") == "Google" for call in calls)


def test_the_metadata_token_is_requested_for_the_cloud_platform_scope(
    mds_proxy: Gateway, mds: MetadataPeer, gemini_model: str
) -> None:
    """The token request goes to `/computeMetadata/v1/instance/service-accounts/default/token` with
    `scopes=https://www.googleapis.com/auth/cloud-platform`, the scope `load_auth` passes to
    `google.auth.default()` (`vertex_llm_base.py:222`). UNVERIFIED: query shape read from google-auth 2.53."""
    assert _chat(mds_proxy, gemini_model) == 200
    tokens: Final = mds.token_calls()
    assert tokens
    assert path_of(tokens[0]) == f"{METADATA_ROOT}instance/service-accounts/default/token"
    assert query_params(tokens[0]).get("scopes") == SCOPE


def test_the_metadata_token_is_reused_across_requests(mds_proxy: Gateway, mds: MetadataPeer, gemini_model: str) -> None:
    """With `expires_in: 3600` three chat calls fetch no new metadata token after the first one; a rewrite
    that asks the metadata server per request would add a round trip to every call."""
    assert _chat(mds_proxy, gemini_model) == 200
    fetched: Final = len(mds.token_calls())
    for _ in range(3):
        assert _chat(mds_proxy, gemini_model) == 200
    assert len(mds.token_calls()) == fetched


def test_gemini_chat_vertex_project_wins_over_the_metadata_project(mds_proxy: Gateway, mds: MetadataPeer) -> None:
    """A deployment that sets `vertex_project` posts to that project even though the metadata server
    reports another one; the metadata server still supplies the token."""
    with _deployment(mds_proxy, mds, "gemini-2.5-flash", vertex_project=PROJECT) as model:
        assert _chat(mds_proxy, model) == 200
    call: Final = mds.model_calls()[-1]
    assert path_of(call) == models_path(project=PROJECT) + ":generateContent"
    assert bearer(call) in mds.issued


# ---------------------------------------------------------------- /v1/messages (Vertex Claude)


def test_messages_without_credentials_uses_a_metadata_server_token(
    mds_proxy: Gateway, mds: MetadataPeer, claude_model: str
) -> None:
    """`/v1/messages` to a Vertex Claude deployment without credentials answers 200 and the rawPredict
    call carries a metadata token. `credential-gaps.md` row 9: the Rust port ignores `GCE_METADATA_HOST`,
    so on this proxy it would call `metadata.google.internal` instead and fail."""
    assert _messages(mds_proxy, claude_model) == 200
    assert bearer(mds.model_calls()[-1]) in mds.issued


def test_messages_takes_the_project_from_the_metadata_server(
    mds_proxy: Gateway, mds: MetadataPeer, claude_model: str
) -> None:
    """With no `vertex_project`, the rawPredict path names the metadata server's project
    (`.../projects/dec1-mds-project/locations/us-east5/publishers/anthropic/models/<model>:rawPredict`).
    `credential-gaps.md` lists this as a Rust gap ("Missing Vertex AI project"), so this test fails on the
    port until the project is resolved asynchronously."""
    assert _messages(mds_proxy, claude_model) == 200
    assert path_of(mds.model_calls()[-1]) == claude_path(project=MDS_PROJECT) + ":rawPredict"


def test_chat_and_messages_each_reuse_their_metadata_token(
    mds_proxy: Gateway, mds: MetadataPeer, gemini_model: str, claude_model: str
) -> None:
    """Gemini chat and Claude messages keep their own `VertexBase` token cache (one handler instance per
    route), each keyed by "no credentials, no project": after one call on each route, a second round of
    both fetches no further metadata token."""
    assert _chat(mds_proxy, gemini_model) == 200
    assert _messages(mds_proxy, claude_model) == 200
    fetched: Final = len(mds.token_calls())
    assert _chat(mds_proxy, gemini_model) == 200
    assert _messages(mds_proxy, claude_model) == 200
    assert len(mds.token_calls()) == fetched


def test_messages_vertex_project_wins_over_the_metadata_project(mds_proxy: Gateway, mds: MetadataPeer) -> None:
    """A Vertex Claude deployment with `vertex_project` posts `/v1/messages` to that project; only the
    token comes from the metadata server."""
    with _deployment(mds_proxy, mds, CLAUDE, vertex_project=PROJECT) as model:
        assert _messages(mds_proxy, model) == 200
    call: Final = mds.model_calls()[-1]
    assert path_of(call) == claude_path(project=PROJECT) + ":rawPredict"
    assert bearer(call) in mds.issued
