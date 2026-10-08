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

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
import respx


import litellm
import litellm.proxy.proxy_server as proxy_server
import litellm.proxy.video_endpoints.endpoints as endpoints
from litellm.llms.fal_ai.videos.transformation import FalAIVideoConfig
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.utils import ProxyLogging, _check_and_merge_model_level_guardrails
from litellm.router import Router
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_character_id_with_provider,
    encode_video_id_with_provider,
)

from fastapi import Response
from starlette.datastructures import UploadFile as StarletteUploadFile

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
SENTINEL = VideoObject(id="video_raw", object="video", status="processing")


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

    def pinned_deployment_id(self) -> str | None:
        return self.base_process.call_args.args[0].pinned_deployment_id

    def route_type(self) -> str:
        return self.base_process.call_args.kwargs["route_type"]


@pytest.fixture
def harness():
    SENTINEL.id = "video_raw"
    logging = MagicMock(spec=ProxyLogging)

    router = MagicMock(spec=Router)
    resolve_model = MagicMock(
        side_effect=lambda model_id: RESOLVED_MODELS.get(model_id)
    )
    router.resolve_model_name_from_model_id = resolve_model
    router.has_model_id.return_value = False
    router.get_deployment.return_value = None

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
                "handle_llm_api_exception",
                handle_exc,
            )
        )
        stack.enter_context(patch.object(endpoints, "read_request_body", read_body))
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
async def test_generation__reencodes_id_with_model_id(harness):
    response = VideoObject(
        id=encode_video_id_with_provider("video_raw", "openai", None),
        object="video",
        status="processing",
    )
    response._hidden_params = {
        "custom_llm_provider": "openai",
        "model_id": VIDEO_MODEL_ID,
    }
    harness.base_process.return_value = response

    resp = await call_generation(harness, body={"model": "sora-2"})

    decoded_video_id = decode_video_id_with_provider(resp.id)
    assert decoded_video_id["model_id"] == VIDEO_MODEL_ID
    assert decoded_video_id["video_id"] == "video_raw"


@pytest.mark.asyncio
async def test_generation__reencodes_id_with_routed_deployment_id(harness):
    # data["model"] is the public group here; only litellm_metadata carries the
    # deployment the router picked.
    response = VideoObject(
        id=encode_video_id_with_provider("video_raw", "openai", None),
        object="video",
        status="processing",
    )
    response._hidden_params = {"custom_llm_provider": "openai"}
    harness.base_process.return_value = response

    resp = await call_generation(
        harness,
        body={
            "model": "sora-2",
            "litellm_metadata": {"model_info": {"id": VIDEO_MODEL_ID}},
        },
    )

    decoded_video_id = decode_video_id_with_provider(resp.id)
    assert decoded_video_id["model_id"] == VIDEO_MODEL_ID
    assert decoded_video_id["video_id"] == "video_raw"


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
    harness.read_body.return_value = dict(body)
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
async def test_edit__reencodes_id_with_model_id(harness):
    response = VideoObject(
        id=encode_video_id_with_provider("video_raw", "openai", None),
        object="video",
        status="processing",
    )
    response._hidden_params = {
        "custom_llm_provider": "openai",
        "model_id": VIDEO_MODEL_ID,
    }
    harness.base_process.return_value = response

    resp = await call_edit(
        harness, body={"prompt": "brighter", "video": {"id": "video_plain"}}
    )

    decoded_video_id = decode_video_id_with_provider(resp.id)
    assert decoded_video_id["model_id"] == VIDEO_MODEL_ID
    assert decoded_video_id["video_id"] == "video_raw"


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


@pytest.mark.asyncio
async def test_edit__bare_string_video_id_from_form_field(harness):
    await call_edit(harness, body={"prompt": "brighter", "video": "video_plain"})

    assert harness.processor_data() == {
        "prompt": "brighter",
        "video_id": "video_plain",
        "custom_llm_provider": "openai",
    }


@pytest.mark.asyncio
async def test_edit__json_string_video_reference_from_form_field(harness):
    await call_edit(
        harness,
        body={"prompt": "brighter", "video": orjson.dumps({"id": "video_plain"}).decode()},
    )

    assert harness.processor_data()["video_id"] == "video_plain"


@pytest.mark.asyncio
async def test_edit__uploaded_video_file_is_forwarded_not_dropped(harness):
    """A multipart-uploaded source video must be converted to bytes and attached
    under ``video`` so the provider receives the file. Before the fix the upload
    was popped, coerced to an empty ``video_id``, and silently dropped."""
    import io

    upload = StarletteUploadFile(file=io.BytesIO(b"rawmp4"), filename="clip.mp4")
    harness.read_body.return_value = {"prompt": "make it nighttime", "video": upload}

    await endpoints.video_edit(
        request=FakeRequest(raw_body=b"multipart"),
        fastapi_response=Response(),
        user_api_key_dict=_user(),
    )

    harness.batch_to_bytesio.assert_called_once_with((upload,))
    assert harness.processor_data() == {
        "prompt": "make it nighttime",
        "video": b"filebytes",
        "video_id": "",
        "custom_llm_provider": "openai",
    }


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
    harness.read_body.return_value = dict(body)
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
async def test_remix__reencodes_id_with_model_id(harness):
    response = VideoObject(
        id=encode_video_id_with_provider("video_raw", "openai", None),
        object="video",
        status="processing",
    )
    response._hidden_params = {
        "custom_llm_provider": "openai",
        "model_id": VIDEO_MODEL_ID,
    }
    harness.base_process.return_value = response

    resp = await call_remix(harness, "video_plain", body={"prompt": "new colors"})

    decoded_video_id = decode_video_id_with_provider(resp.id)
    assert decoded_video_id["model_id"] == VIDEO_MODEL_ID
    assert decoded_video_id["video_id"] == "video_raw"


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
    harness.read_body.return_value = dict(body)
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


@pytest.mark.asyncio
async def test_extension__reencodes_id_with_model_id(harness):
    response = VideoObject(
        id=encode_video_id_with_provider("video_raw", "openai", None),
        object="video",
        status="processing",
    )
    response._hidden_params = {
        "custom_llm_provider": "openai",
        "model_id": VIDEO_MODEL_ID,
    }
    harness.base_process.return_value = response

    resp = await call_extension(harness, body={"prompt": "continue"})

    decoded_video_id = decode_video_id_with_provider(resp.id)
    assert decoded_video_id["model_id"] == VIDEO_MODEL_ID
    assert decoded_video_id["video_id"] == "video_raw"


# =========================================================================== #
#   Follow-up calls against a real Router                                     #
# =========================================================================== #

FAL_MODEL = "fal-ai/kling-video/v2/master/text-to-video"


def _video_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "sora-2",
                "litellm_params": {
                    "model": "openai/sora-2",
                    "api_key": "sk-mock-a",
                    "api_base": "http://a.localhost/v1",
                    "weight": 1,
                },
                "model_info": {"id": "deployment-a"},
            },
            {
                "model_name": "sora-2",
                "litellm_params": {
                    "model": "openai/sora-2",
                    "api_key": "sk-mock-b",
                    "api_base": "http://b.localhost/v1",
                    "weight": 0,
                },
                "model_info": {"id": "deployment-b"},
            },
            {
                "model_name": "kling",
                "litellm_params": {"model": f"fal_ai/{FAL_MODEL}", "api_key": "sk-mock-fal"},
                "model_info": {"id": "deployment-fal"},
            },
        ]
    )


def _video_id(model_id: str) -> str:
    return encode_video_id_with_provider("video_orig", "openai", model_id)


FOLLOW_UP_CALLS = {
    "status": lambda h, model_id: call_status(h, _video_id(model_id)),
    "content": lambda h, model_id: call_content(h, _video_id(model_id)),
    "remix": lambda h, model_id: call_remix(h, _video_id(model_id), body={"prompt": "x"}),
    "edit": lambda h, model_id: call_edit(h, body={"prompt": "x", "video": {"id": _video_id(model_id)}}),
    "extension": lambda h, model_id: call_extension(h, body={"prompt": "x", "video": {"id": _video_id(model_id)}}),
    "get_character": lambda h, model_id: call_get_character(
        h, encode_character_id_with_provider("char_orig", "openai", model_id)
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", sorted(FOLLOW_UP_CALLS))
@pytest.mark.parametrize(
    ("model_id", "expected_model"),
    [
        pytest.param("deployment-b", "sora-2", id="deployment_id_pins"),
        pytest.param("sora-2", "sora-2", id="group_id_load_balances"),
        pytest.param("", None, id="no_model_in_id"),
        pytest.param("deployment-gone", None, id="deleted_deployment_calls_provider_directly"),
    ],
)
async def test_follow_up__model_comes_from_the_id(harness, endpoint, model_id, expected_model):
    harness.base_process.return_value = b"video-bytes"

    with patch.object(proxy_server, "llm_router", _video_router()):
        await FOLLOW_UP_CALLS[endpoint](harness, model_id)

    assert harness.processor_data().get("model") == expected_model
    assert harness.pinned_deployment_id() == ("deployment-b" if model_id == "deployment-b" else None)


@pytest.mark.asyncio
async def test_status__real_router_reaches_the_creating_deployment(harness, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    llm_router = _video_router()

    with patch.object(proxy_server, "llm_router", llm_router):
        await call_status(harness, _video_id("deployment-b"))
    with respx.mock(assert_all_called=True) as respx_mock:
        deployment_b = respx_mock.get("http://b.localhost/v1/videos/video_orig").respond(
            json={"id": "video_orig", "object": "video", "status": "completed", "created_at": 0}
        )
        data = harness.processor_data()
        status = await llm_router.avideo_status(
            **{**data, "metadata": {**data.get("metadata", {}), "pinned_deployment_id": harness.pinned_deployment_id()}}
        )

    assert deployment_b.call_count == 1
    assert status.status == "completed"


@pytest.mark.asyncio
async def test_status__fal_id_keeps_the_model_path_fal_reads(harness):
    with patch.object(proxy_server, "llm_router", _video_router()):
        await call_status(harness, encode_video_id_with_provider("req-123", "fal_ai", "deployment-fal"))
    data = harness.processor_data()
    url, _ = FalAIVideoConfig().transform_video_status_retrieve_request(
        video_id=data["video_id"],
        api_base="https://queue.fal.run",
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert data["model"] == "kling"
    assert url == "https://queue.fal.run/fal-ai/kling-video/requests/req-123/status"


@pytest.mark.asyncio
async def test_status__response_id_keeps_the_deployment_for_content(harness):
    harness.base_process.return_value = VideoObject(
        id=encode_video_id_with_provider("video_orig", "openai", None), object="video", status="completed"
    )

    with patch.object(proxy_server, "llm_router", _video_router()):
        status = await call_status(harness, _video_id("deployment-b"))
        harness.base_process.reset_mock()
        harness.base_process.return_value = b"video-bytes"
        await call_content(harness, status.id)

    assert decode_video_id_with_provider(status.id)["model_id"] == "deployment-b"
    assert harness.processor_data().get("model") == "sora-2"
    assert harness.pinned_deployment_id() == "deployment-b"


@pytest.mark.asyncio
async def test_status__without_router_keeps_the_provider_model_in_the_id(harness):
    fal_status_id = encode_video_id_with_provider("req-123", "fal_ai", "fal-ai/kling-video")
    harness.base_process.return_value = VideoObject(id=fal_status_id, object="video", status="completed")

    with patch.object(proxy_server, "llm_router", None):
        status = await call_status(harness, fal_status_id)

    assert status.id == fal_status_id


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", sorted(FOLLOW_UP_CALLS))
async def test_follow_up__pinned_deployment_keeps_its_model_level_guardrails(harness, endpoint):
    harness.base_process.return_value = b"video-bytes"
    llm_router = Router(
        model_list=[
            {
                "model_name": "sora-2",
                "litellm_params": {"model": "openai/sora-2", "api_key": "sk-mock-b", "guardrails": ["video-guard"]},
                "model_info": {"id": "deployment-b"},
            }
        ]
    )

    with patch.object(proxy_server, "llm_router", llm_router):
        await FOLLOW_UP_CALLS[endpoint](harness, "deployment-b")
    merged = _check_and_merge_model_level_guardrails(
        data=harness.processor_data(), llm_router=llm_router, trust_client_model_info=False
    )

    assert merged["metadata"]["guardrails"] == ["video-guard"]


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [False, [], "", '{"tag": "blue"}'])
async def test_remix__pinned_id_leaves_the_client_metadata_for_the_proxy_to_validate(harness, metadata):
    harness.base_process.return_value = b"video-bytes"

    with patch.object(proxy_server, "llm_router", _video_router()):
        await call_remix(harness, _video_id("deployment-b"), body={"prompt": "x", "metadata": metadata})

    assert harness.processor_data()["metadata"] == metadata
    assert harness.pinned_deployment_id() == "deployment-b"


async def _call_router_as_proxy(llm_router: Router, harness, key: UserAPIKeyAuth):
    data = harness.processor_data()
    metadata = {
        **data.get("metadata", {}),
        "pinned_deployment_id": harness.pinned_deployment_id(),
        "user_api_key_auth": key,
        "user_api_key_team_id": key.team_id,
    }
    return await getattr(llm_router, harness.route_type())(**{**data, "metadata": metadata})


def _access_group_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "sora-2",
                "litellm_params": {"model": "openai/sora-2", "api_key": "sk-mock-a"},
                "model_info": {"id": "deployment-a", "access_groups": ["group-a"]},
            },
            {
                "model_name": "sora-2",
                "litellm_params": {"model": "openai/sora-2", "api_key": "sk-mock-b"},
                "model_info": {"id": "deployment-b", "access_groups": ["group-b"]},
            },
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", sorted(FOLLOW_UP_CALLS))
async def test_follow_up__id_outside_the_key_access_group_is_rejected(harness, monkeypatch, endpoint):
    key = UserAPIKeyAuth(api_key="sk-test", models=["group-a"])
    monkeypatch.setitem(globals(), "_user", lambda: key)
    harness.base_process.return_value = b"video-bytes"
    llm_router = _access_group_router()

    with patch.object(proxy_server, "llm_router", llm_router):
        await FOLLOW_UP_CALLS[endpoint](harness, "deployment-b")
    with pytest.raises(litellm.BadRequestError) as rejected:
        await _call_router_as_proxy(llm_router, harness, key)

    assert rejected.value.status_code == 400
    assert "You passed in model=sora-2. There are no healthy deployments for this model" in rejected.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "model_id"),
    [
        pytest.param(UserAPIKeyAuth(api_key="sk-test", models=["group-a"]), "deployment-a", id="own_access_group"),
        pytest.param(UserAPIKeyAuth(api_key="sk-test", models=["group-a", "group-b"]), "deployment-b", id="both_groups"),
        pytest.param(UserAPIKeyAuth(api_key="sk-test", models=["sora-2"]), "deployment-b", id="group_name"),
        pytest.param(UserAPIKeyAuth(api_key="sk-test", models=["*"]), "deployment-b", id="wildcard"),
        pytest.param(UserAPIKeyAuth(api_key="sk-test", models=["all-proxy-models"]), "deployment-b", id="all_proxy"),
        pytest.param(
            UserAPIKeyAuth(api_key="sk-test", user_role=LitellmUserRoles.PROXY_ADMIN), "deployment-b", id="admin"
        ),
    ],
)
async def test_status__id_inside_the_key_access_pins_the_deployment(harness, monkeypatch, key, model_id):
    monkeypatch.setitem(globals(), "_user", lambda: key)
    harness.base_process.return_value = b"video-bytes"

    with patch.object(proxy_server, "llm_router", _access_group_router()):
        await call_status(harness, _video_id(model_id))

    assert harness.processor_data().get("model") == "sora-2"
    assert harness.pinned_deployment_id() == model_id


def _team_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "model_name_team-b_sora",
                "litellm_params": {"model": "openai/sora-2", "api_key": "sk-mock-team-b"},
                "model_info": {"id": "team-b-deployment", "team_id": "team-b", "team_public_model_name": "sora-2"},
            }
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", sorted(FOLLOW_UP_CALLS))
async def test_follow_up__id_of_another_team_deployment_is_rejected(harness, monkeypatch, endpoint):
    key = UserAPIKeyAuth(api_key="sk-test", team_id="team-a")
    monkeypatch.setitem(globals(), "_user", lambda: key)
    harness.base_process.return_value = b"video-bytes"
    llm_router = _team_router()

    with patch.object(proxy_server, "llm_router", llm_router):
        await FOLLOW_UP_CALLS[endpoint](harness, "team-b-deployment")
    with pytest.raises(litellm.BadRequestError) as rejected:
        await _call_router_as_proxy(llm_router, harness, key)

    assert rejected.value.status_code == 400
    assert "You passed in model=model_name_team-b_sora. There are no healthy deployments" in rejected.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key",
    [
        pytest.param(UserAPIKeyAuth(api_key="sk-test", team_id="team-b"), id="same_team"),
        pytest.param(UserAPIKeyAuth(api_key="sk-test", user_role=LitellmUserRoles.PROXY_ADMIN), id="admin"),
    ],
)
async def test_status__id_of_a_team_deployment_pins_it_for_that_team(harness, monkeypatch, key):
    monkeypatch.setitem(globals(), "_user", lambda: key)
    harness.base_process.return_value = b"video-bytes"

    with patch.object(proxy_server, "llm_router", _team_router()):
        await call_status(harness, _video_id("team-b-deployment"))

    assert harness.processor_data().get("model") == "model_name_team-b_sora"
    assert harness.pinned_deployment_id() == "team-b-deployment"
