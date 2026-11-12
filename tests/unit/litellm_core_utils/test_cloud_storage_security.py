import sys
from base64 import urlsafe_b64encode
from functools import reduce
from typing import Final, cast
from urllib.parse import quote

import pytest

from litellm.litellm_core_utils.cloud_storage_security import (
    MAX_FILE_ID_DECODE_PASSES,
    is_managed_cloud_storage_uri,
    is_managed_s3_uri,
    media_file_ids,
    names_an_unlisted_s3_bucket,
    s3_media_uris,
)


def test_is_managed_cloud_storage_uri_detects_raw_object_uris():
    assert is_managed_cloud_storage_uri("s3://bucket/litellm-batch-outputs/x.jsonl.out")
    assert is_managed_cloud_storage_uri("gs://bucket/litellm-vertex-files/x")


def test_is_managed_cloud_storage_uri_ignores_provider_and_unified_ids():
    # Plain provider ids and base64 unified ids carry no storage scheme.
    assert not is_managed_cloud_storage_uri("file-abc123")
    assert not is_managed_cloud_storage_uri("bGl0ZWxsbV9wcm94eQ==")
    assert not is_managed_cloud_storage_uri("")


def _quote_once(current: str, _: int) -> str:
    return quote(current, safe="")


def _quoted_times(value: str, times: int) -> str:
    return reduce(_quote_once, range(times), value)


_MANAGED_UPLOAD: Final = "s3://media-bucket/litellm-bedrock-files/other-team.pdf"


def _unpadded_base64url(text: str) -> str:
    return urlsafe_b64encode(text.encode()).decode().rstrip("=")


_MODEL_EMBEDDED_UPLOAD_ID: Final = "file-" + _unpadded_base64url(f"litellm:{_MANAGED_UPLOAD};model,nova")
_UNIFIED_BATCH_OUTPUT_ID: Final = _unpadded_base64url(
    f"litellm_proxy:application/json;unified_id,u1;target_model_names,nova;llm_output_file_id,{_MANAGED_UPLOAD};"
)


def _user_turn(*parts: object) -> dict[str, object]:
    return {"messages": [{"role": "user", "content": list(parts)}]}


def _responses_turn(*parts: object) -> dict[str, object]:
    return {"input": [{"role": "user", "content": list(parts)}]}


def _gemini_turn(*parts: object) -> dict[str, object]:
    return {"contents": [{"role": "user", "parts": list(parts)}]}


@pytest.mark.parametrize(
    "request_body",
    (
        pytest.param(_user_turn({"type": "image_url", "image_url": {"url": _MANAGED_UPLOAD}}), id="chat-image-url"),
        pytest.param(_user_turn({"type": "image_url", "image_url": _MANAGED_UPLOAD}), id="chat-image-url-string"),
        pytest.param(_user_turn({"type": "file", "file": {"file_id": _MANAGED_UPLOAD}}), id="chat-file-id"),
        pytest.param(_user_turn({"type": "file", "file": {"file_data": _MANAGED_UPLOAD}}), id="chat-file-data"),
        pytest.param(
            {
                "messages": [
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": [{"type": "image_url", "image_url": {"url": _MANAGED_UPLOAD}}],
                    }
                ]
            },
            id="chat-tool-result",
        ),
        pytest.param(
            _user_turn({"type": "file", "file": {"file_id": _MODEL_EMBEDDED_UPLOAD_ID}}),
            id="chat-model-embedded-file-id",
        ),
        pytest.param(
            _responses_turn({"type": "input_file", "file_id": _UNIFIED_BATCH_OUTPUT_ID}),
            id="responses-unified-file-id",
        ),
        pytest.param(
            _responses_turn({"type": "input_file", "file_url": _UNIFIED_BATCH_OUTPUT_ID}),
            id="responses-unified-file-id-in-file-url",
        ),
        pytest.param(
            _gemini_turn({"file_data": {"mime_type": "application/pdf", "file_uri": _UNIFIED_BATCH_OUTPUT_ID}}),
            id="gemini-unified-file-id-in-file-uri",
        ),
        pytest.param(
            _gemini_turn({"fileData": {"mimeType": "application/pdf", "fileUri": _MODEL_EMBEDDED_UPLOAD_ID}}),
            id="gemini-model-embedded-file-id-in-camel-case-file-uri",
        ),
        pytest.param(
            {
                **_user_turn({"type": "file", "file": {"file_id": "alias"}}),
                "model_file_id_mapping": {"alias": {"deployment-1": _MANAGED_UPLOAD}},
            },
            id="model-file-id-mapping",
        ),
        pytest.param(
            {
                **_responses_turn({"type": "input_file", "file_id": "alias"}),
                "model_file_id_mapping": {"alias": {"deployment-1": _MODEL_EMBEDDED_UPLOAD_ID}},
            },
            id="model-file-id-mapping-to-a-model-embedded-file-id",
        ),
        pytest.param(_responses_turn({"type": "input_image", "image_url": _MANAGED_UPLOAD}), id="responses-image"),
        pytest.param(_responses_turn({"type": "input_file", "file_url": _MANAGED_UPLOAD}), id="responses-file-url"),
        pytest.param(_responses_turn({"type": "input_file", "file_id": _MANAGED_UPLOAD}), id="responses-file-id"),
        pytest.param(
            {
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": [{"type": "input_image", "image_url": _MANAGED_UPLOAD}],
                    }
                ]
            },
            id="responses-function-call-output",
        ),
        pytest.param(
            _user_turn({"type": "image", "source": {"type": "url", "url": _MANAGED_UPLOAD}}), id="anthropic-image"
        ),
        pytest.param(
            _user_turn({"type": "document", "source": {"type": "url", "url": _MANAGED_UPLOAD}}),
            id="anthropic-document",
        ),
        pytest.param(
            {"system": [{"type": "image", "source": {"type": "url", "url": _MANAGED_UPLOAD}}], "messages": []},
            id="anthropic-system",
        ),
        pytest.param(
            _gemini_turn({"file_data": {"mime_type": "image/png", "file_uri": _MANAGED_UPLOAD}}),
            id="gemini-file-data-snake-case",
        ),
        pytest.param(
            _gemini_turn({"fileData": {"mimeType": "image/png", "fileUri": _MANAGED_UPLOAD}}),
            id="gemini-file-data-camel-case",
        ),
        pytest.param(
            {"system_instruction": {"parts": [{"file_data": {"file_uri": _MANAGED_UPLOAD}}]}, "contents": []},
            id="gemini-system-instruction",
        ),
        pytest.param(
            {
                "input": [
                    {
                        "type": "computer_call_output",
                        "call_id": "call_1",
                        "output": {"type": "computer_screenshot", "image_url": _MANAGED_UPLOAD},
                    }
                ]
            },
            id="responses-computer-screenshot",
        ),
        pytest.param(
            {
                "messages": (
                    {"role": "user", "content": ({"type": "image_url", "image_url": {"url": _MANAGED_UPLOAD}},)},
                )
            },
            id="tuples",
        ),
        pytest.param(
            _user_turn(1, 2.5, True, None, {"type": "image_url", "image_url": {"url": _MANAGED_UPLOAD}}),
            id="numbers-before-the-media-part",
        ),
    ),
)
def test_s3_media_uris_finds_an_s3_object_on_every_media_surface(request_body: dict[str, object]):
    assert s3_media_uris(request_body) == (_MANAGED_UPLOAD,)


@pytest.mark.parametrize(
    "uri",
    (
        "s3://media-bucket/litellm-bedrock-files/x.pdf",
        "s3://media-bucket/litellm-bedrock-files-batch-1/input.jsonl",
        "s3://media-bucket/litellm-batch-outputs/x.jsonl.out",
        "s3://media-bucket/tenant-a/litellm-bedrock-files/x.pdf",
        "s3://media-bucket//litellm-bedrock-files/x.pdf",
        "s3://media-bucket/./litellm-bedrock-files/x.pdf",
        "s3://media-bucket/litellm%2Dbedrock-files/x.pdf",
        "s3://media-bucket/litellm%252Dbedrock-files/x.pdf",
        "s3://media-bucket%2Flitellm-bedrock-files/x.pdf",
        "s3:///litellm-bedrock-files/x.pdf",
    ),
)
def test_is_managed_s3_uri_sees_a_managed_prefix_anywhere_in_the_decoded_key(uri: str):
    assert s3_media_uris(_user_turn({"type": "file", "file": {"file_id": uri}})) == (uri,)
    assert is_managed_s3_uri(uri)


def test_is_managed_s3_uri_fails_closed_on_a_key_encoded_past_the_pass_cap():
    uri: Final = "s3://media-bucket/" + _quoted_times("litellm-bedrock-files/x.pdf", MAX_FILE_ID_DECODE_PASSES + 2)

    assert s3_media_uris(_user_turn({"type": "image_url", "image_url": {"url": uri}})) == (uri,)
    assert is_managed_s3_uri(uri)


@pytest.mark.parametrize(
    "uri",
    (
        pytest.param("s3://media-bucket/dir/cat.png", id="unmanaged-key"),
        pytest.param("s3://litellm-bedrock-files-bucket/cat.png", id="managed-prefix-in-bucket-name-only"),
        pytest.param("s3://media-bucket", id="no-key"),
    ),
)
def test_is_managed_s3_uri_leaves_other_objects_alone(uri: str):
    assert not is_managed_s3_uri(uri)


@pytest.mark.parametrize(
    "request_body",
    (
        pytest.param(
            _user_turn({"type": "image_url", "image_url": {"url": "https://example.test/litellm-bedrock-files/x.png"}}),
            id="https-url",
        ),
        pytest.param(
            _user_turn({"type": "image_url", "image_url": {"url": "data:image/png;base64,czM6Ly9idWNrZXQva2V5"}}),
            id="data-url",
        ),
        pytest.param(
            _user_turn(
                {
                    "type": "file",
                    "file": {
                        "file_id": _unpadded_base64url(
                            "litellm_proxy:application/pdf;unified_id,u2;target_model_names,nova;"
                            "llm_output_file_id,file-abc123;"
                        )
                    },
                }
            ),
            id="unified-file-id-of-a-provider-file",
        ),
        pytest.param(
            _user_turn(
                {
                    "type": "file",
                    "file": {"file_data": "data:application/pdf;base64," + _unpadded_base64url(_MANAGED_UPLOAD)},
                }
            ),
            id="file-data-payload-naming-an-s3-object",
        ),
        pytest.param(
            _user_turn({"type": "text", "text": _MODEL_EMBEDDED_UPLOAD_ID}), id="encoded-file-id-outside-a-media-part"
        ),
        pytest.param(_user_turn({"type": "text", "text": _MANAGED_UPLOAD}), id="text-part"),
        pytest.param({"messages": [{"role": "user", "content": _MANAGED_UPLOAD}]}, id="string-content"),
        pytest.param({"input": _MANAGED_UPLOAD}, id="responses-string-input"),
        pytest.param({"system": _MANAGED_UPLOAD, "messages": []}, id="string-system"),
        pytest.param({"metadata": {"note": _MANAGED_UPLOAD}}, id="non-media-key"),
        pytest.param({"metadata": {"url": _MANAGED_UPLOAD}}, id="metadata-url"),
        pytest.param({"extra_body": {"url": _MANAGED_UPLOAD}}, id="extra-body-url"),
        pytest.param({"name": "fetch", "arguments": {"url": _MANAGED_UPLOAD}}, id="mcp-tool-arguments"),
        pytest.param(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "See the source.",
                        "annotations": [{"type": "url_citation", "url_citation": {"url": _MANAGED_UPLOAD}}],
                    }
                ]
            },
            id="chat-url-citation",
        ),
        pytest.param(
            {
                "input": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "See the file.",
                                "annotations": [{"type": "file_citation", "file_id": _MANAGED_UPLOAD, "index": 0}],
                            }
                        ],
                    }
                ]
            },
            id="responses-file-citation",
        ),
        pytest.param(
            _gemini_turn({"functionCall": {"name": "fetch", "args": {"url": _MANAGED_UPLOAD}}}),
            id="gemini-function-call-camel-case",
        ),
        pytest.param(
            _gemini_turn({"function_call": {"name": "fetch", "args": {"url": _MANAGED_UPLOAD}}}),
            id="gemini-function-call-snake-case",
        ),
        pytest.param(
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "fetch",
                            "parameters": {
                                "type": "object",
                                "properties": {"url": {"type": "string", "enum": [_MANAGED_UPLOAD]}},
                                "default": {"url": _MANAGED_UPLOAD},
                            },
                        },
                    }
                ]
            },
            id="tool-schema",
        ),
        pytest.param(
            _user_turn(
                {
                    "type": "file",
                    "file": {"file_data": "data:application/pdf;base64,JVBERi0=", "filename": _MANAGED_UPLOAD},
                }
            ),
            id="media-part-filename",
        ),
        pytest.param(
            _user_turn({"type": "image_url", "image_url": {"url": "S3://media-bucket/litellm-bedrock-files/x.png"}}),
            id="uppercase-scheme",
        ),
        pytest.param(
            _user_turn(
                {"type": "file", "file": {"file_id": "s3%3A%2F%2Fmedia-bucket%2Flitellm-bedrock-files%2Fx.pdf"}}
            ),
            id="encoded-scheme",
        ),
        pytest.param(
            _user_turn(
                {"type": "file", "file": {"file_id": "%73%33%3A%2F%2Fmedia-bucket%2Flitellm-bedrock-files%2Fx.pdf"}}
            ),
            id="fully-encoded-scheme",
        ),
    ),
)
def test_s3_media_uris_ignores_what_never_becomes_an_s3_media_source(request_body: dict[str, object]):
    assert s3_media_uris(request_body) == ()


def test_s3_media_uris_finds_an_s3_object_nested_deeper_than_the_recursion_limit():
    nested: Final = reduce(
        lambda inner, _: [inner],
        range(sys.getrecursionlimit() + 100),
        cast(object, [{"type": "image_url", "image_url": {"url": _MANAGED_UPLOAD}}]),
    )

    assert s3_media_uris({"messages": [{"role": "user", "content": nested}]}) == (_MANAGED_UPLOAD,)


def test_s3_media_uris_skips_only_the_objects_a_trusted_file_id_embeds():
    listed_image: Final = "s3://media-bucket/dir/cat.png"
    request_body: Final = _responses_turn(
        {"type": "input_file", "file_id": _UNIFIED_BATCH_OUTPUT_ID},
        {"type": "input_file", "file_id": _MODEL_EMBEDDED_UPLOAD_ID},
        {"type": "input_image", "image_url": listed_image},
    )

    assert s3_media_uris(request_body, frozenset({_UNIFIED_BATCH_OUTPUT_ID})) == (_MANAGED_UPLOAD, listed_image)
    assert s3_media_uris(request_body, frozenset({_UNIFIED_BATCH_OUTPUT_ID, _MODEL_EMBEDDED_UPLOAD_ID})) == (
        listed_image,
    )


def test_media_file_ids_lists_the_file_ids_of_media_parts_only():
    request_body: Final = {
        **_user_turn(
            {"type": "file", "file": {"file_id": "file-a"}},
            {"type": "text", "text": "file-b", "file_id": "file-b"},
            {"type": "file", "file": {"file_id": "file-a"}},
        ),
        "metadata": {"file_id": "file-c"},
    }

    assert media_file_ids(request_body) == ("file-a",)


def test_media_file_ids_lists_every_field_a_bridge_reads_as_a_file_id():
    request_body: Final = {
        **_responses_turn({"type": "input_file", "file_url": "file-a"}),
        **_gemini_turn({"file_data": {"file_uri": "file-b"}}, {"fileData": {"fileUri": "file-c"}}),
    }

    assert set(media_file_ids(request_body)) == {"file-a", "file-b", "file-c"}


def test_s3_media_uris_names_each_s3_object_once_in_the_order_it_first_appears():
    other_upload: Final = "s3://media-bucket/dir/cat.png"
    request_body: Final = _user_turn(
        {"type": "image_url", "image_url": {"url": other_upload}},
        {"type": "file", "file": {"file_id": _MANAGED_UPLOAD}},
        {"type": "image_url", "image_url": {"url": other_upload}},
    )

    assert s3_media_uris(request_body) == (other_upload, _MANAGED_UPLOAD)


@pytest.mark.parametrize(
    ("uri", "names_an_unlisted_bucket"),
    (
        pytest.param("s3://listed-bucket/dir/cat.png", False, id="listed"),
        pytest.param("s3://listed-bucket/dir/a%20cat.png", False, id="listed-with-an-encoded-key"),
        pytest.param("s3://other-bucket/dir/cat.png", True, id="unlisted"),
        pytest.param("s3://listed-bucket.example/cat.png", True, id="listed-name-as-a-prefix"),
        pytest.param("s3://listed%2Dbucket/cat.png", True, id="encoded-bucket"),
        pytest.param("s3://other-bucket%2Flisted-bucket/cat.png", True, id="encoded-slash-in-the-bucket"),
        pytest.param("s3:///listed-bucket/cat.png", True, id="empty-bucket"),
        pytest.param("s3%3A%2F%2Fother-bucket%2Fcat.png", True, id="encoded-scheme"),
        pytest.param(
            "s3://listed-bucket/" + _quoted_times("a cat.png", MAX_FILE_ID_DECODE_PASSES + 2),
            True,
            id="encoded-past-the-pass-cap",
        ),
    ),
)
def test_names_an_unlisted_s3_bucket_checks_the_bucket_of_every_decoding(uri: str, names_an_unlisted_bucket: bool):
    assert names_an_unlisted_s3_bucket(uri, frozenset({"listed-bucket"}).__contains__) is names_an_unlisted_bucket
