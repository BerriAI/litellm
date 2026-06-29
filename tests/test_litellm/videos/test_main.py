"""
Dispatch-contract tests for litellm/videos/main.py

Each public video operation is a pair: a sync `video_*` worker (decorated with
@client) that resolves the provider, fetches the provider config, logs, and then
forwards to exactly one `base_llm_http_handler.video_*_handler`; and an async
`avideo_*` wrapper that delegates to the sync worker in an executor.

This file locks the contract of that layer so a regression fails loudly:

  1. DISPATCH   - the one correct handler fired and every sibling video handler
                  asserted NOT called. A copy-paste that calls the wrong handler
                  (e.g. remix -> edit) flips this.
  2. RESULT     - the handler's return value is propagated by identity.
  3. PROVIDER   - custom_llm_provider is decoded from an encoded video id when not
                  passed (status/content/remix/edit/extension), or defaults to
                  "openai" (list/create_character/get_character). This is the exact
                  surface of the historical "content defaulted to openai" bug.
  4. PAYLOAD    - the provider config object and the operation's identifying args
                  (video_id/prompt/name/...) reach the handler; _is_async is False
                  on the sync path.
  5. SHORT-CIRCUIT - mock_response returns a typed object without any handler call.
  6. UNSUPPORTED   - a None provider config raises before any handler fires.
  7. DELEGATION    - avideo_* returns the sync worker's result untouched, sets
                     async_call=True, and pre-resolves the provider where it must.

Seams mocked: the http handler (network), the provider-config registry lookup,
get_llm_provider, and the video-generation optional-param builders. The id decode
helper runs for real against genuinely-encoded ids, so the provider assertions
reflect production.
"""

import os
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.videos.main import CharacterObject, VideoObject
from litellm.types.videos.utils import encode_video_id_with_provider
from litellm.videos import main as videos_main

# A real model-encoded video id: decodes (for real) to provider "azure". Used to
# prove the sync workers derive custom_llm_provider from the id, not a hardcode.
AZURE_VIDEO_ID = encode_video_id_with_provider("video_raw", "azure", "deployment-1")

# The nine sync handlers on base_llm_http_handler. Dispatch tests assert exactly
# one fired and the other eight did not.
SYNC_HANDLERS = (
    "video_generation_handler",
    "video_content_handler",
    "video_remix_handler",
    "video_create_character_handler",
    "video_get_character_handler",
    "video_edit_handler",
    "video_extension_handler",
    "video_list_handler",
    "video_status_handler",
)

GEN_OPTIONAL_PARAMS = {"seconds": "8", "size": "720x1280"}


@dataclass
class Seams:
    handler: MagicMock
    get_config: MagicMock
    config: MagicMock

    def kwargs_of(self, handler_name: str) -> Dict[str, Any]:
        method = getattr(self.handler, handler_name)
        assert method.call_count == 1
        return dict(method.call_args.kwargs)

    def assert_only(self, handler_name: str) -> None:
        for name in SYNC_HANDLERS:
            method = getattr(self.handler, name)
            if name == handler_name:
                method.assert_called_once()
            else:
                method.assert_not_called()


@pytest.fixture
def seams():
    handler = MagicMock(spec=BaseLLMHTTPHandler)
    config = MagicMock(name="provider_video_config")
    get_config = MagicMock(return_value=config)

    with ExitStack() as stack:
        stack.enter_context(patch.object(videos_main, "base_llm_http_handler", handler))
        stack.enter_context(
            patch.object(
                videos_main.ProviderConfigManager,
                "get_provider_video_config",
                get_config,
            )
        )
        # video_generation resolves model+provider through get_llm_provider and
        # builds optional params; mock those so the dispatch payload is deterministic.
        stack.enter_context(
            patch.object(
                videos_main,
                "get_llm_provider",
                MagicMock(return_value=("sora-2", "openai", None, None)),
            )
        )
        stack.enter_context(
            patch.object(
                videos_main.VideoGenerationRequestUtils,
                "get_requested_video_generation_optional_param",
                MagicMock(return_value={"seconds": "8"}),
            )
        )
        stack.enter_context(
            patch.object(
                videos_main.VideoGenerationRequestUtils,
                "get_optional_params_video_generation",
                MagicMock(return_value=dict(GEN_OPTIONAL_PARAMS)),
            )
        )
        yield Seams(handler=handler, get_config=get_config, config=config)


# =========================================================================== #
# Dispatch contract - one rich test per sync worker.
# =========================================================================== #


def test_video_generation__dispatch(seams):
    result = videos_main.video_generation(prompt="a sunset", model="sora-2")

    seams.assert_only("video_generation_handler")
    assert result is seams.handler.video_generation_handler.return_value
    kw = seams.kwargs_of("video_generation_handler")
    assert kw["model"] == "sora-2"
    assert kw["prompt"] == "a sunset"
    assert kw["custom_llm_provider"] == "openai"
    assert kw["video_generation_provider_config"] is seams.config
    assert kw["video_generation_optional_request_params"] == GEN_OPTIONAL_PARAMS
    assert kw["_is_async"] is False


def test_video_status__dispatch_and_provider_from_id(seams):
    result = videos_main.video_status(video_id=AZURE_VIDEO_ID)

    seams.assert_only("video_status_handler")
    assert result is seams.handler.video_status_handler.return_value
    kw = seams.kwargs_of("video_status_handler")
    assert kw["video_id"] == AZURE_VIDEO_ID
    assert kw["custom_llm_provider"] == "azure"  # decoded from the id, not openai
    assert kw["video_status_provider_config"] is seams.config
    assert kw["_is_async"] is False
    # provider config requested for the decoded provider, not a hardcode.
    assert seams.get_config.call_args.kwargs["provider"] == litellm.LlmProviders.AZURE


def test_video_content__dispatch_and_provider_from_id(seams):
    result = videos_main.video_content(video_id=AZURE_VIDEO_ID, variant="thumbnail")

    seams.assert_only("video_content_handler")
    assert result is seams.handler.video_content_handler.return_value
    kw = seams.kwargs_of("video_content_handler")
    assert kw["video_id"] == AZURE_VIDEO_ID
    assert kw["custom_llm_provider"] == "azure"
    assert kw["variant"] == "thumbnail"
    assert kw["video_content_provider_config"] is seams.config
    assert kw["_is_async"] is False


def test_video_content__plain_id_defaults_to_openai(seams):
    videos_main.video_content(video_id="video_plain")

    assert seams.kwargs_of("video_content_handler")["custom_llm_provider"] == "openai"


def test_video_remix__dispatch_and_provider_from_id(seams):
    result = videos_main.video_remix(video_id=AZURE_VIDEO_ID, prompt="new colors")

    seams.assert_only("video_remix_handler")
    assert result is seams.handler.video_remix_handler.return_value
    kw = seams.kwargs_of("video_remix_handler")
    assert kw["video_id"] == AZURE_VIDEO_ID
    assert kw["prompt"] == "new colors"
    assert kw["custom_llm_provider"] == "azure"
    assert kw["video_remix_provider_config"] is seams.config
    assert kw["_is_async"] is False


def test_video_edit__dispatch_and_provider_from_id(seams):
    result = videos_main.video_edit(video_id=AZURE_VIDEO_ID, prompt="brighter")

    seams.assert_only("video_edit_handler")
    assert result is seams.handler.video_edit_handler.return_value
    kw = seams.kwargs_of("video_edit_handler")
    assert kw["video_id"] == AZURE_VIDEO_ID
    assert kw["prompt"] == "brighter"
    assert kw["custom_llm_provider"] == "azure"
    assert kw["video_provider_config"] is seams.config
    assert kw["_is_async"] is False


def test_video_extension__dispatch_and_provider_from_id(seams):
    result = videos_main.video_extension(
        video_id=AZURE_VIDEO_ID, prompt="continue", seconds="5"
    )

    seams.assert_only("video_extension_handler")
    assert result is seams.handler.video_extension_handler.return_value
    kw = seams.kwargs_of("video_extension_handler")
    assert kw["video_id"] == AZURE_VIDEO_ID
    assert kw["prompt"] == "continue"
    assert kw["seconds"] == "5"
    assert kw["custom_llm_provider"] == "azure"
    assert kw["video_provider_config"] is seams.config
    assert kw["_is_async"] is False


def test_video_list__dispatch_defaults_to_openai(seams):
    result = videos_main.video_list(after="cur", limit=5, order="desc")

    seams.assert_only("video_list_handler")
    assert result is seams.handler.video_list_handler.return_value
    kw = seams.kwargs_of("video_list_handler")
    assert kw["after"] == "cur"
    assert kw["limit"] == 5
    assert kw["order"] == "desc"
    assert kw["custom_llm_provider"] == "openai"
    assert kw["video_list_provider_config"] is seams.config
    assert kw["_is_async"] is False


def test_video_create_character__dispatch_defaults_to_openai(seams):
    video = MagicMock(name="video_upload")
    result = videos_main.video_create_character(name="hero", video=video)

    seams.assert_only("video_create_character_handler")
    assert result is seams.handler.video_create_character_handler.return_value
    kw = seams.kwargs_of("video_create_character_handler")
    assert kw["name"] == "hero"
    assert kw["video"] is video
    assert kw["custom_llm_provider"] == "openai"
    assert kw["video_provider_config"] is seams.config
    assert kw["_is_async"] is False


def test_video_get_character__dispatch_defaults_to_openai(seams):
    result = videos_main.video_get_character(character_id="char_1")

    seams.assert_only("video_get_character_handler")
    assert result is seams.handler.video_get_character_handler.return_value
    kw = seams.kwargs_of("video_get_character_handler")
    assert kw["character_id"] == "char_1"
    assert kw["custom_llm_provider"] == "openai"
    assert kw["video_provider_config"] is seams.config
    assert kw["_is_async"] is False


def test_explicit_provider_beats_decoded_id(seams):
    """An explicit custom_llm_provider wins over the one encoded in the id."""
    videos_main.video_status(video_id=AZURE_VIDEO_ID, custom_llm_provider="vertex_ai")

    assert seams.kwargs_of("video_status_handler")["custom_llm_provider"] == "vertex_ai"


# =========================================================================== #
# mock_response short-circuit - returns a typed object, no handler call.
# =========================================================================== #


def test_generation__mock_response_short_circuits(seams):
    resp = videos_main.video_generation(
        prompt="x",
        model="sora-2",
        mock_response={"id": "v1", "object": "video", "status": "queued"},
    )

    assert isinstance(resp, VideoObject)
    assert resp.id == "v1"
    seams.handler.video_generation_handler.assert_not_called()


def test_list__mock_response_short_circuits(seams):
    resp = videos_main.video_list(
        mock_response=[{"id": "v1", "object": "video", "status": "completed"}]
    )

    assert isinstance(resp, list)
    assert resp[0].id == "v1"
    seams.handler.video_list_handler.assert_not_called()


def test_get_character__mock_response_short_circuits(seams):
    resp = videos_main.video_get_character(
        character_id="char_1",
        mock_response={
            "id": "char_1",
            "object": "character",
            "created_at": 1,
            "name": "hero",
        },
    )

    assert isinstance(resp, CharacterObject)
    assert resp.id == "char_1"
    seams.handler.video_get_character_handler.assert_not_called()


# =========================================================================== #
# Unsupported provider - a None provider config raises before any dispatch.
# =========================================================================== #


def test_unsupported_provider_raises_without_dispatch(seams):
    seams.get_config.return_value = None

    with pytest.raises(Exception):
        videos_main.video_status(video_id=AZURE_VIDEO_ID)

    seams.handler.video_status_handler.assert_not_called()


# =========================================================================== #
# Async-wrapper delegation - representative coverage.
# =========================================================================== #


@pytest.mark.asyncio
async def test_avideo_generation__delegates_with_async_flag():
    sentinel = VideoObject(id="v-async", object="video", status="queued")
    with (
        patch.object(
            videos_main, "video_generation", MagicMock(return_value=sentinel)
        ) as sync,
        patch.object(
            litellm,
            "get_llm_provider",
            MagicMock(return_value=("sora-2", "openai", None, None)),
        ),
    ):
        result = await videos_main.avideo_generation(prompt="x", model="sora-2")

    assert result is sentinel
    assert sync.call_args.kwargs["async_call"] is True
    assert sync.call_args.kwargs["custom_llm_provider"] == "openai"


@pytest.mark.asyncio
async def test_avideo_status__delegates_untouched():
    sentinel = VideoObject(id="v-async", object="video", status="queued")
    with patch.object(
        videos_main, "video_status", MagicMock(return_value=sentinel)
    ) as sync:
        result = await videos_main.avideo_status(video_id="video_plain")

    assert result is sentinel
    assert sync.call_args.kwargs["async_call"] is True
    assert sync.call_args.kwargs["video_id"] == "video_plain"


@pytest.mark.asyncio
async def test_avideo_content__pre_decodes_provider_before_delegating():
    """avideo_content resolves the provider from the encoded id itself before
    handing off, so the sync worker receives the decoded provider, not None."""
    sentinel = b"mp4-bytes"
    with patch.object(
        videos_main, "video_content", MagicMock(return_value=sentinel)
    ) as sync:
        result = await videos_main.avideo_content(video_id=AZURE_VIDEO_ID)

    assert result is sentinel
    assert sync.call_args.kwargs["async_call"] is True
    assert sync.call_args.kwargs["custom_llm_provider"] == "azure"


# =========================================================================== #
# Credential passthrough - DB/YAML model-config credentials the router injects
# via kwargs must reach the provider call for EVERY video handler, carried in
# litellm_params. Distinct per-field values catch a cross-wired field.
# =========================================================================== #

DB_YAML_CREDS = {
    "api_key": "sk-db-credential",
    "api_base": "https://db-resource.test",
    "api_version": "2024-12-31",
    "vertex_project": "db-project-xyz",
}

CREDENTIAL_OPERATIONS = [
    (
        "video_generation_handler",
        lambda: videos_main.video_generation(
            prompt="p", model="sora-2", **DB_YAML_CREDS
        ),
    ),
    (
        "video_status_handler",
        lambda: videos_main.video_status(video_id=AZURE_VIDEO_ID, **DB_YAML_CREDS),
    ),
    (
        "video_content_handler",
        lambda: videos_main.video_content(video_id=AZURE_VIDEO_ID, **DB_YAML_CREDS),
    ),
    (
        "video_remix_handler",
        lambda: videos_main.video_remix(
            video_id=AZURE_VIDEO_ID, prompt="p", **DB_YAML_CREDS
        ),
    ),
    (
        "video_edit_handler",
        lambda: videos_main.video_edit(
            video_id=AZURE_VIDEO_ID, prompt="p", **DB_YAML_CREDS
        ),
    ),
    (
        "video_extension_handler",
        lambda: videos_main.video_extension(
            video_id=AZURE_VIDEO_ID, prompt="p", seconds="5", **DB_YAML_CREDS
        ),
    ),
    (
        "video_list_handler",
        lambda: videos_main.video_list(**DB_YAML_CREDS),
    ),
    (
        "video_create_character_handler",
        lambda: videos_main.video_create_character(
            name="hero", video=MagicMock(name="vid"), **DB_YAML_CREDS
        ),
    ),
    (
        "video_get_character_handler",
        lambda: videos_main.video_get_character(character_id="char_1", **DB_YAML_CREDS),
    ),
]


@pytest.mark.parametrize(
    "handler_name,invoke",
    CREDENTIAL_OPERATIONS,
    ids=[op[0] for op in CREDENTIAL_OPERATIONS],
)
def test_db_yaml_credentials_reach_every_handler(seams, handler_name, invoke):
    invoke()

    litellm_params = seams.kwargs_of(handler_name)["litellm_params"]
    assert litellm_params.get("api_key") == DB_YAML_CREDS["api_key"]
    assert litellm_params.get("api_base") == DB_YAML_CREDS["api_base"]
    assert litellm_params.get("api_version") == DB_YAML_CREDS["api_version"]
    assert litellm_params.get("vertex_project") == DB_YAML_CREDS["vertex_project"]
