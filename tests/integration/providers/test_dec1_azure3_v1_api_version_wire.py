"""The `v1` / `latest` / `preview` api_version family: which Azure routes send an `api-version` query.

`BaseAzureLLM.is_azure_v1_api_version` treats `v1`, `latest` and `preview` alike: they select Azure's
`/openai/v1/...` surface. What reaches the query differs by route, read from code:

- Chat, embeddings and files use the SDK path (`common_utils.py`, the `is_azure_v1_api_version` branch of
  `get_azure_openai_client`): a plain OpenAI client on `{root}/openai/v1/` whose `default_query` is the api_base
  query minus `api-version`. So no `api-version` is ever sent, whichever of the three was configured.
- Responses, vector stores and images use `_get_base_azure_url`, which adds `api-version=<v>` as given, so
  `v1`, `latest` and `preview` each reach Azure as the literal query value.

Azure documents the v1 surface as needing no api-version for GA features and `api-version=preview` for preview
features. Chat on `preview` therefore silently loses access to preview-only features: that is pinned below as a
skipped BUG. The other cells pin today's behaviour so a Rust port changes it only on purpose.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Request
from integration.providers.dec1_azure_support import (
    EMBEDDING_DEPLOYMENT,
    GPT5_DEPLOYMENT,
    always,
    azure,
    chat_reply,
    embedding_json,
    json_reply,
    path_of,
    query_of,
    responses_json,
)


def _chat(_: Request):  # noqa: ANN202
    return chat_reply("pong")


def test_chat_on_preview_sends_api_version_preview(gateway: Gateway) -> None:
    """The deployment sets `api_version: preview`.

    Intended: Azure must see `POST /openai/v1/chat/completions?api-version=preview`, as Azure documents for preview
    features on the v1 surface and as the Responses route already does. Today the v1 client drops every
    `api-version` (`common_utils.py`, `v1_query = v1_base.params.remove("api-version")`), so the query is empty.
    """
    pytest.skip(
        "BUG: api_version preview on Azure chat sends no api-version query, so preview features are unreachable"
    )
    with azure(gateway, _chat, api_version="preview") as rig:
        rig.chat()
        request: Final = rig.one_post()
        assert (path_of(request), query_of(request)) == ("/openai/v1/chat/completions", {"api-version": "preview"})


def test_embeddings_on_preview_send_api_version_preview(gateway: Gateway) -> None:
    """Same intended behaviour for `/v1/embeddings` on a `preview` deployment: `POST /openai/v1/embeddings` with
    `{"api-version": "preview"}`. Today the query is empty for the same reason as chat.
    """
    pytest.skip("BUG: api_version preview on Azure embeddings sends no api-version query")
    with azure(
        gateway,
        always(json_reply(embedding_json([[0.1]]))),
        model=f"azure/{EMBEDDING_DEPLOYMENT}",
        api_version="preview",
    ) as rig:
        rig.embed("x")
        request: Final = rig.one_post()
        assert (path_of(request), query_of(request)) == ("/openai/v1/embeddings", {"api-version": "preview"})


@pytest.mark.parametrize("version", ["latest", "preview"])
def test_chat_on_latest_or_preview_sends_no_api_version_today(gateway: Gateway, version: str) -> None:
    """The deployment sets `api_version` to `latest` or `preview`.

    Current behaviour: Azure sees `POST /openai/v1/chat/completions` with an empty query, exactly like `v1`.
    Round 1 pinned only `v1`; this pins the other two members of the family. When the preview BUG above is fixed,
    the `preview` cell flips and must be updated with it.
    """
    with azure(gateway, _chat, api_version=version) as rig:
        assert rig.chat().status_code == 200
        request: Final = rig.one_post()
        assert (path_of(request), query_of(request)) == ("/openai/v1/chat/completions", {})


def test_chat_on_v1_keeps_other_api_base_query_parameters(gateway: Gateway) -> None:
    """`api_base` is `<peer>?tenant=blue&api-version=2024-02-01` and `api_version` is `v1`.

    The v1 client's `default_query` is the api_base query minus `api-version`, so Azure must see exactly
    `{"tenant": "blue"}` on `/openai/v1/chat/completions`.
    """
    with azure(gateway, _chat, api_version="v1", base_path="?tenant=blue&api-version=2024-02-01") as rig:
        rig.chat()
        request: Final = rig.one_post()
        assert (path_of(request), query_of(request)) == ("/openai/v1/chat/completions", {"tenant": "blue"})


@pytest.mark.parametrize("version", ["v1", "latest", "preview"])
def test_responses_on_the_v1_family_send_the_literal_version_as_the_query(gateway: Gateway, version: str) -> None:
    """A Responses deployment sets `api_version` to `v1`, `latest` or `preview`.

    `_resolve_responses_api_version` keeps a non dated version and `_get_base_azure_url` adds it, so Azure must
    see `POST /openai/v1/responses` with query exactly `{"api-version": <version>}`. UNVERIFIED that Azure accepts
    `api-version=v1` and `api-version=latest`; this pins the bytes litellm sends.
    """
    with azure(
        gateway, always(json_reply(responses_json())), model=f"azure/{GPT5_DEPLOYMENT}", api_version=version
    ) as rig:
        assert rig.responses().status_code == 200
        request: Final = rig.one_post()
        assert (path_of(request), query_of(request)) == ("/openai/v1/responses", {"api-version": version})


@pytest.mark.parametrize("version", ["latest", "preview"])
def test_images_on_latest_or_preview_send_the_literal_version(gateway: Gateway, version: str) -> None:
    """An image deployment sets `api_version` to `latest` or `preview`.

    `get_azure_v1_image_url` keeps the version, so Azure must see `POST /openai/v1/images/generations` with
    query exactly `{"api-version": <version>}`.
    """
    image: Final = {"created": 1, "data": [{"b64_json": "AAAA"}]}
    with azure(gateway, always(json_reply(image)), model="azure/gpt-image-dec1", api_version=version) as rig:
        rig.gateway.request("POST", "/v1/images/generations", {"model": rig.model, "prompt": "p"})
        request: Final = rig.one_post()
        assert (path_of(request), query_of(request)) == ("/openai/v1/images/generations", {"api-version": version})


def test_audio_speech_on_v1_posts_to_openai_v1_audio_speech_with_no_api_version(gateway: Gateway) -> None:
    """An `azure/tts-1` deployment with `api_version: v1`.

    Speech uses the same SDK client factory as chat, so Azure must see `POST /openai/v1/audio/speech` with an
    empty query and body `model: tts-1`. UNVERIFIED: `audio_speech` is assumed to go through
    `get_azure_openai_client` like chat.
    """
    from integration.providers.dec1_azure3_support import SPEECH_BYTES, binary_reply
    from integration.providers.dec1_azure_support import body_of

    with azure(gateway, always(binary_reply(SPEECH_BYTES, "audio/mpeg")), model="azure/tts-1", api_version="v1") as rig:
        rig.gateway.request("POST", "/v1/audio/speech", {"model": rig.model, "input": "hi", "voice": "alloy"})
        request: Final = rig.one_post()
        assert (path_of(request), query_of(request)) == ("/openai/v1/audio/speech", {})
        assert body_of(request)["model"] == "tts-1"
