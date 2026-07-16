"""
Routing-contract tests for litellm/proxy/video_endpoints/endpoints.py

Unlike the batches layer, every video endpoint funnels into a single downstream
seam - ProxyBaseLLMRequestProcessing.base_process_llm_request - so there is no
provider-dispatch to assert. All of the video-specific, regression-worthy logic
runs *before* that call, while the endpoint assembles the `data` dict. Each test
therefore locks four things:

  1. ROUTE_TYPE   - the exact route_type each endpoint forwards
                    (avideo_generation/status/content/edit). Swapping two would
                    silently route requests to the wrong handler.
  2. DATA SHAPE   - the entire `data` dict the processor is constructed with:
                    provider-precedence resolution, video_id passthrough/extraction,
                    model resolution from the decoded model_id, and file attachment.
  3. RESULT       - base_process_llm_request's return value is propagated untouched
                    (except where the endpoint transforms it).
  4. OUTPUT SHAPE - video_content wraps raw bytes in a Response (video/mp4 +
                    Content-Disposition).

Only true I/O boundaries are mocked (the downstream processor call, request body
parsing, file->bytes conversion, the provider-from-request readers, and the
router's model-id resolver). The id decode helpers and get_custom_provider_from_data
run for real, so the data assertions reflect production exactly. base_process is
patched with autospec so the real __init__ still stores self.data (captured via the
mock's call args), and a brand-new kwarg added to this layer surfaces as a failure.
"""

import os
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm.proxy.proxy_server as proxy_server
import litellm.proxy.video_endpoints.endpoints as endpoints
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.utils import ProxyLogging
from litellm.router import Router
from litellm.types.videos.utils import (
    encode_character_id_with_provider,
    encode_video_id_with_provider,
)

from fastapi import Response

# --------------------------------------------------------------------------- #
# A real model-encoded video id: decodes (for real) to provider "azure",
# model_id VIDEO_MODEL_ID, original video id "video_orig123". The router's
# resolver maps that model_id to a model name; an unknown id resolves to None,
# so a wrong/hardcoded model_id cannot produce a plausible-looking result.
# --------------------------------------------------------------------------- #

VIDEO_MODEL_ID = "deployment-123"
AZURE_VIDEO_ID = encode_video_id_with_provider("video_orig123", "azure", VIDEO_MODEL_ID)
# A real model-encoded character id: decodes to provider "azure", VIDEO_MODEL_ID,
# original character id "char_orig". Distinct from the video id so a test cannot
# pass by reusing the wrong constant.
AZURE_CHARACTER_ID = encode_character_id_with_provider(
    "char_orig", "azure", VIDEO_MODEL_ID
)
RESOLVED_MODELS: Dict[str, str] = {VIDEO_MODEL_ID: "azure-sora"}

# Sentinel propagated by base_process for the passthrough endpoints.
SENTINEL = object()


class FakeRequest:
    """Minimal stand-in. headers/query_params are read by the provider readers
    (mocked) and on the edit path the raw body is parsed for real via orjson."""

    def __init__(
        self,
        headers: Optional[Dict[str, str]] = None,
        query: Optional[Dict[str, str]] = None,
        raw_body: bytes = b"{}",
    ):
        self.headers = headers or {}
        self.query_params = query or {}
        self._raw_body = raw_body

    async def body(self) -> bytes:
        return self._raw_body


@dataclass
class Harness:
    read_body: AsyncMock
    batch_to_bytesio: AsyncMock
    base_process: MagicMock
    handle_exc: AsyncMock
    provider_from_headers: MagicMock
    provider_from_query: MagicMock
    provider_from_body: AsyncMock
    router: MagicMock
    resolve_model: MagicMock

    def processor_data(self) -> Dict[str, Any]:
        """The exact `data` dict the processor was constructed with."""
        assert self.base_process.call_count == 1
        return dict(self.base_process.call_args.args[0].data)

    def route_type(self) -> str:
        return self.base_process.call_args.kwargs["route_type"]


@pytest.fixture
def harness():
    logging = MagicMock(spec=ProxyLogging)

    router = MagicMock(spec=Router)
    resolve_model = MagicMock(
        side_effect=lambda model_id, custom_llm_provider=None: RESOLVED_MODELS.get(
            model_id
        )
    )
    router.resolve_model_name_from_model_id = resolve_model

    read_body = AsyncMock(return_value={})
    batch_to_bytesio = AsyncMock(return_value=[b"filebytes"])
    handle_exc = AsyncMock(return_value=RuntimeError("handled"))
    provider_from_headers = MagicMock(return_value=None)
    provider_from_query = MagicMock(return_value=None)
    provider_from_body = AsyncMock(return_value=None)

    with ExitStack() as stack:
        base_process = stack.enter_context(
            patch.object(
                ProxyBaseLLMRequestProcessing,
                "base_process_llm_request",
                autospec=True,
            )
        )
        base_process.return_value = SENTINEL
        stack.enter_context(
            patch.object(
                ProxyBaseLLMRequestProcessing,
                "_handle_llm_api_exception",
                handle_exc,
            )
        )
        stack.enter_context(patch.object(endpoints, "_read_request_body", read_body))
        stack.enter_context(
            patch.object(endpoints, "batch_to_bytesio", batch_to_bytesio)
        )
        stack.enter_context(
            patch.object(
                endpoints,
                "get_custom_llm_provider_from_request_headers",
                provider_from_headers,
            )
        )
        stack.enter_context(
            patch.object(
                endpoints,
                "get_custom_llm_provider_from_request_query",
                provider_from_query,
            )
        )
        stack.enter_context(
            patch.object(
                endpoints,
                "get_custom_llm_provider_from_request_body",
                provider_from_body,
            )
        )
        stack.enter_context(patch.object(proxy_server, "llm_router", router))
        stack.enter_context(patch.object(proxy_server, "proxy_logging_obj", logging))
        stack.enter_context(patch.object(proxy_server, "general_settings", {}))
        stack.enter_context(patch.object(proxy_server, "proxy_config", MagicMock()))
        stack.enter_context(
            patch.object(proxy_server, "select_data_generator", MagicMock())
        )
        stack.enter_context(patch.object(proxy_server, "user_model", None))
        stack.enter_context(patch.object(proxy_server, "user_temperature", None))
        stack.enter_context(patch.object(proxy_server, "user_request_timeout", None))
        stack.enter_context(patch.object(proxy_server, "user_max_tokens", None))
        stack.enter_context(patch.object(proxy_server, "user_api_base", None))
        stack.enter_context(patch.object(proxy_server, "version", "test-version"))

        yield Harness(
            read_body=read_body,
            batch_to_bytesio=batch_to_bytesio,
            base_process=base_process,
            handle_exc=handle_exc,
            provider_from_headers=provider_from_headers,
            provider_from_query=provider_from_query,
            provider_from_body=provider_from_body,
            router=router,
            resolve_model=resolve_model,
        )


def _user() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-test")


# =========================================================================== #
#   POST /v1/videos  -  video_generation                                       #
# =========================================================================== #


async def call_generation(
    harness: Harness, *, body: Dict[str, Any], input_reference=None
):
    harness.read_body.return_value = body
    return await endpoints.video_generation(
        request=FakeRequest(),
        fastapi_response=Response(),
        input_reference=input_reference,
        user_api_key_dict=_user(),
    )


@pytest.mark.asyncio
async def test_generation__route_type_data_and_no_provider_default(harness):
    body = {"model": "sora-2", "prompt": "a sunset"}

    resp = await call_generation(harness, body=body)

    assert resp is SENTINEL
    assert harness.route_type() == "avideo_generation"
    # generation does NOT resolve a provider; data is the body, untouched. A
    # future default custom_llm_provider injection would break this row.
    assert harness.processor_data() == {"model": "sora-2", "prompt": "a sunset"}
    harness.batch_to_bytesio.assert_not_called()


@pytest.mark.asyncio
async def test_generation__input_reference_attached(harness):
    body = {"model": "sora-2", "prompt": "a sunset"}
    upload = MagicMock(name="upload_file")

    await call_generation(harness, body=body, input_reference=upload)

    harness.batch_to_bytesio.assert_called_once_with([upload])
    assert harness.processor_data() == {
        "model": "sora-2",
        "prompt": "a sunset",
        "input_reference": b"filebytes",
    }


@pytest.mark.asyncio
async def test_generation__exception_routed_through_handler(harness):
    harness.base_process.side_effect = ValueError("provider boom")

    with pytest.raises(RuntimeError, match="handled"):
        await call_generation(harness, body={"model": "sora-2"})

    harness.handle_exc.assert_called_once()
    assert harness.handle_exc.call_args.kwargs["e"].args[0] == "provider boom"


# =========================================================================== #
#   GET /v1/videos/{video_id}  -  video_status                                 #
# =========================================================================== #


async def call_status(harness: Harness, video_id: str, *, headers=None, query=None):
    return await endpoints.video_status(
        video_id=video_id,
        request=FakeRequest(headers=headers, query=query),
        fastapi_response=Response(),
        user_api_key_dict=_user(),
    )


@pytest.mark.asyncio
async def test_status__model_encoded_id_full_contract(harness):
    resp = await call_status(harness, AZURE_VIDEO_ID)

    assert resp is SENTINEL
    assert harness.route_type() == "avideo_status"
    # provider comes from the decoded id; model_id resolved to a model name.
    harness.resolve_model.assert_called_once_with(VIDEO_MODEL_ID)
    assert harness.processor_data() == {
        "video_id": AZURE_VIDEO_ID,
        "custom_llm_provider": "azure",
        "model": "azure-sora",
    }


@pytest.mark.asyncio
async def test_status__plain_id_defaults_to_openai(harness):
    await call_status(harness, "video_plain")

    # plain id -> nothing decoded, no header/query/body provider -> "openai".
    harness.resolve_model.assert_not_called()
    assert harness.processor_data() == {
        "video_id": "video_plain",
        "custom_llm_provider": "openai",
    }


@pytest.mark.asyncio
async def test_status__header_provider_beats_decoded_id(harness):
    harness.provider_from_headers.return_value = "bedrock"

    await call_status(harness, AZURE_VIDEO_ID)

    data = harness.processor_data()
    # header wins over the provider decoded from the id ...
    assert data["custom_llm_provider"] == "bedrock"
    # ... but the model is still resolved from the decoded model_id.
    assert data["model"] == "azure-sora"


# =========================================================================== #
#   GET /v1/videos/{video_id}/content  -  video_content                        #
# =========================================================================== #


async def call_content(harness: Harness, video_id: str, *, headers=None, query=None):
    return await endpoints.video_content(
        video_id=video_id,
        request=FakeRequest(headers=headers, query=query),
        fastapi_response=Response(),
        user_api_key_dict=_user(),
    )


@pytest.mark.asyncio
async def test_content__wraps_raw_bytes_in_response(harness):
    harness.base_process.return_value = b"VIDEOBYTES"

    resp = await call_content(harness, "video_plain")

    assert harness.route_type() == "avideo_content"
    assert isinstance(resp, Response)
    assert resp.body == b"VIDEOBYTES"
    assert resp.media_type == "video/mp4"
    assert (
        resp.headers["content-disposition"]
        == "attachment; filename=video_video_plain.mp4"
    )


@pytest.mark.asyncio
async def test_content__plain_id_has_no_openai_default(harness):
    """The high-value asymmetry vs video_status: content stops at the decoded
    provider and never injects an 'openai' default, so a plain id leaves
    custom_llm_provider unset. A copy-paste of status' fallback breaks this."""
    harness.base_process.return_value = b"x"

    await call_content(harness, "video_plain")

    assert harness.processor_data() == {"video_id": "video_plain"}


@pytest.mark.asyncio
async def test_content__model_encoded_id(harness):
    harness.base_process.return_value = b"x"

    await call_content(harness, AZURE_VIDEO_ID)

    harness.resolve_model.assert_called_once_with(VIDEO_MODEL_ID)
    assert harness.processor_data() == {
        "video_id": AZURE_VIDEO_ID,
        "custom_llm_provider": "azure",
        "model": "azure-sora",
    }


# =========================================================================== #
#   POST /v1/videos/edits  -  video_edit                                       #
# =========================================================================== #


async def call_edit(
    harness: Harness, *, body: Dict[str, Any], headers=None, query=None
):
    return await endpoints.video_edit(
        request=FakeRequest(headers=headers, query=query, raw_body=orjson.dumps(body)),
        fastapi_response=Response(),
        user_api_key_dict=_user(),
    )


@pytest.mark.asyncio
async def test_edit__extracts_nested_video_id_full_contract(harness):
    resp = await call_edit(
        harness, body={"prompt": "brighter", "video": {"id": AZURE_VIDEO_ID}}
    )

    assert resp is SENTINEL
    assert harness.route_type() == "avideo_edit"
    harness.resolve_model.assert_called_once_with(VIDEO_MODEL_ID)
    # nested video object is popped; its id becomes video_id; provider/model
    # derived from the encoded id.
    assert harness.processor_data() == {
        "prompt": "brighter",
        "video_id": AZURE_VIDEO_ID,
        "custom_llm_provider": "azure",
        "model": "azure-sora",
    }


@pytest.mark.asyncio
async def test_edit__provider_from_body_data_for_plain_id(harness):
    """For a plain id, get_custom_provider_from_data (run for real) pulls the
    provider out of the request body before the 'openai' default."""
    await call_edit(
        harness,
        body={
            "prompt": "x",
            "video": {"id": "video_plain"},
            "custom_llm_provider": "vertex_ai",
        },
    )

    data = harness.processor_data()
    assert data["video_id"] == "video_plain"
    assert data["custom_llm_provider"] == "vertex_ai"
    harness.resolve_model.assert_not_called()


@pytest.mark.asyncio
async def test_edit__missing_video_object_defaults_to_openai(harness):
    await call_edit(harness, body={"prompt": "x"})

    data = harness.processor_data()
    # no video object -> empty video_id; plain -> default provider.
    assert data["video_id"] == ""
    assert data["custom_llm_provider"] == "openai"
    assert "video" not in data


# =========================================================================== #
#   GET /v1/videos  -  video_list                                              #
# =========================================================================== #


async def call_list(harness: Harness, *, headers=None, query=None):
    return await endpoints.video_list(
        request=FakeRequest(headers=headers, query=query),
        fastapi_response=Response(),
        user_api_key_dict=_user(),
    )


@pytest.mark.asyncio
async def test_list__query_params_and_no_provider(harness):
    resp = await call_list(harness, query={"limit": "5"})

    assert resp is SENTINEL
    assert harness.route_type() == "avideo_list"
    # no provider anywhere -> custom_llm_provider stays absent (only set if truthy).
    assert harness.processor_data() == {"query_params": {"limit": "5"}}


@pytest.mark.asyncio
async def test_list__provider_from_header(harness):
    harness.provider_from_headers.return_value = "bedrock"

    await call_list(harness)

    assert harness.processor_data() == {
        "query_params": {},
        "custom_llm_provider": "bedrock",
    }


# =========================================================================== #
#   POST /v1/videos/{video_id}/remix  -  video_remix                           #
# =========================================================================== #


async def call_remix(
    harness: Harness, video_id: str, *, body, headers=None, query=None
):
    return await endpoints.video_remix(
        video_id=video_id,
        request=FakeRequest(headers=headers, query=query, raw_body=orjson.dumps(body)),
        fastapi_response=Response(),
        user_api_key_dict=_user(),
    )


@pytest.mark.asyncio
async def test_remix__model_encoded_id_full_contract(harness):
    resp = await call_remix(harness, AZURE_VIDEO_ID, body={"prompt": "new colors"})

    assert resp is SENTINEL
    assert harness.route_type() == "avideo_remix"
    harness.resolve_model.assert_called_once_with(VIDEO_MODEL_ID)
    assert harness.processor_data() == {
        "prompt": "new colors",
        "video_id": AZURE_VIDEO_ID,
        "custom_llm_provider": "azure",
        "model": "azure-sora",
    }


@pytest.mark.asyncio
async def test_remix__provider_from_body_data_not_request_body_reader(harness):
    """remix resolves the provider from data.get('custom_llm_provider'), never
    from the async request-body reader (unlike status/get_character). Setting
    that reader to a sentinel and asserting it is untouched locks the difference."""
    harness.provider_from_body.return_value = "must-not-win"

    await call_remix(
        harness,
        "video_plain",
        body={"prompt": "x", "custom_llm_provider": "vertex_ai"},
    )

    harness.provider_from_body.assert_not_called()
    data = harness.processor_data()
    assert data["video_id"] == "video_plain"
    assert data["custom_llm_provider"] == "vertex_ai"


@pytest.mark.asyncio
async def test_remix__plain_id_has_no_openai_default(harness):
    await call_remix(harness, "video_plain", body={"prompt": "x"})

    # like video_content, remix stops at provider_from_id with no 'openai' default.
    assert harness.processor_data() == {"prompt": "x", "video_id": "video_plain"}


# =========================================================================== #
#   POST /v1/videos/characters  -  video_create_character                      #
# =========================================================================== #


async def call_create_character(harness: Harness, *, body, video=None, name="my_char"):
    harness.read_body.return_value = body
    return await endpoints.video_create_character(
        request=FakeRequest(),
        fastapi_response=Response(),
        video=video if video is not None else MagicMock(name="video_upload"),
        name=name,
        user_api_key_dict=_user(),
    )


@pytest.mark.asyncio
async def test_create_character__video_attached_default_provider_no_encode(harness):
    upload = MagicMock(name="video_upload")

    resp = await call_create_character(harness, body={"prompt": "x"}, video=upload)

    assert resp is SENTINEL
    assert harness.route_type() == "avideo_create_character"
    harness.batch_to_bytesio.assert_called_once_with([upload])
    # no target_model_names -> no model injected, no id re-encoding.
    assert harness.processor_data() == {
        "prompt": "x",
        "video": b"filebytes",
        "custom_llm_provider": "openai",
    }


@pytest.mark.asyncio
async def test_create_character__target_model_sets_model_and_encodes_id(harness):
    harness.base_process.return_value = {"id": "char_raw"}

    resp = await call_create_character(
        harness,
        body={"target_model_names": "azure-sora-model", "custom_llm_provider": "azure"},
    )

    data = harness.processor_data()
    assert data["model"] == "azure-sora-model"
    assert data["custom_llm_provider"] == "azure"
    # response id re-encoded with the resolved provider + model for the round-trip.
    assert resp["id"] == encode_character_id_with_provider(
        "char_raw", "azure", "azure-sora-model"
    )


# =========================================================================== #
#   GET /v1/videos/characters/{character_id}  -  video_get_character           #
# =========================================================================== #


async def call_get_character(
    harness: Harness, character_id: str, *, headers=None, query=None
):
    return await endpoints.video_get_character(
        character_id=character_id,
        request=FakeRequest(headers=headers, query=query),
        fastapi_response=Response(),
        user_api_key_dict=_user(),
    )


@pytest.mark.asyncio
async def test_get_character__encoded_id_full_contract(harness):
    harness.base_process.return_value = {"id": "char_raw2"}

    resp = await call_get_character(harness, AZURE_CHARACTER_ID)

    assert harness.route_type() == "avideo_get_character"
    harness.resolve_model.assert_called_once_with(VIDEO_MODEL_ID)
    # character_id decoded to its inner value; provider/model from the encoded id.
    assert harness.processor_data() == {
        "character_id": "char_orig",
        "custom_llm_provider": "azure",
        "model": "azure-sora",
    }
    # response id re-encoded for the client round-trip.
    assert resp["id"] == encode_character_id_with_provider(
        "char_raw2", "azure", VIDEO_MODEL_ID
    )


@pytest.mark.asyncio
async def test_get_character__plain_id_defaults_openai_no_encode(harness):
    harness.base_process.return_value = {"id": "char_raw3"}

    resp = await call_get_character(harness, "char_plain")

    harness.resolve_model.assert_not_called()
    assert harness.processor_data() == {
        "character_id": "char_plain",
        "custom_llm_provider": "openai",
    }
    # id does not start with 'character_' -> returned untouched.
    assert resp["id"] == "char_raw3"


# =========================================================================== #
#   POST /v1/videos/extensions  -  video_extension                            #
# =========================================================================== #


async def call_extension(harness: Harness, *, body, headers=None, query=None):
    return await endpoints.video_extension(
        request=FakeRequest(headers=headers, query=query, raw_body=orjson.dumps(body)),
        fastapi_response=Response(),
        user_api_key_dict=_user(),
    )


@pytest.mark.asyncio
async def test_extension__extracts_nested_video_id_full_contract(harness):
    resp = await call_extension(
        harness, body={"prompt": "continue", "video": {"id": AZURE_VIDEO_ID}}
    )

    assert resp is SENTINEL
    assert harness.route_type() == "avideo_extension"
    harness.resolve_model.assert_called_once_with(VIDEO_MODEL_ID)
    assert harness.processor_data() == {
        "prompt": "continue",
        "video_id": AZURE_VIDEO_ID,
        "custom_llm_provider": "azure",
        "model": "azure-sora",
    }
