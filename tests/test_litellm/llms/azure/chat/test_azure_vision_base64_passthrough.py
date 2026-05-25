"""
Regression tests pinning that LiteLLM's Azure chat path preserves base64 vision
``image_url`` payloads byte-for-byte.

Context: LIT-2870. A customer reported `ContentPolicyViolationError` when
sending a base64-encoded image to Azure OpenAI / Azure AI Foundry vision through
LiteLLM. Investigation showed that LiteLLM does not mutate base64 ``image_url``
content beyond wrapping the legacy string form into ``{"url": ...}`` for Azure
SDK compatibility (see ``_azure_image_url_helper`` in
``litellm/litellm_core_utils/prompt_templates/factory.py``). The reported
content-policy error originates in Azure's content filter, not in LiteLLM.

These tests pin that no future change silently re-encodes, truncates, or strips
the base64 payload on the Azure path. If a future regression starts mangling
the base64, these tests will fail loudly with a clear "bytes differ" message
rather than the customer-visible symptom (an opaque Azure 400 / content-filter
error).
"""

import base64
import copy
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

from litellm.litellm_core_utils.prompt_templates.factory import (  # noqa: E402
    _azure_image_url_helper,
    convert_to_azure_openai_messages,
)
from litellm.llms.azure.chat.gpt_transformation import AzureOpenAIConfig  # noqa: E402


# A tiny 2x2 red PNG. Real PNG bytes (not made up) so any byte-level corruption
# (e.g. accidental decode/re-encode through a charset) is visible.
_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x02"
    b"\x08\x02\x00\x00\x00\xfd\xd4\x9as\x00\x00\x00\x16IDATx\x9cc\xfc\xcf\xc0"
    b"\xc0\xc0\xc4\xc0\xc0\xc0\xc4\xc0\xc0\xc0\xc4\xc0\x00\x00\x008\x00\x01"
    b"F\xa9\\\xa1\x00\x00\x00\x00IEND\xaeB`\x82"
)
_PNG_B64 = base64.b64encode(_PNG_BYTES).decode("ascii")
_DATA_URL = f"data:image/png;base64,{_PNG_B64}"


def _extract_image_url(message):
    """Extract the image_url value (string or dict) from a chat message content list."""
    for part in message["content"]:
        if isinstance(part, dict) and part.get("type") == "image_url":
            return part["image_url"]
    raise AssertionError("no image_url part found in message")


def test_azure_image_url_helper_preserves_dict_base64_data_url_verbatim():
    """Dict-form base64 ``image_url`` must pass through ``_azure_image_url_helper``
    without any byte mutation (the typical OpenAI-spec input shape)."""
    content = {
        "type": "image_url",
        "image_url": {"url": _DATA_URL, "detail": "auto"},
    }
    before = copy.deepcopy(content)

    _azure_image_url_helper(content)

    assert content == before, (
        "dict-form base64 image_url was mutated by _azure_image_url_helper"
    )
    assert content["image_url"]["url"] == _DATA_URL


def test_azure_image_url_helper_wraps_string_form_without_altering_bytes():
    """Legacy string-form base64 ``image_url`` must be wrapped into ``{"url": ...}``
    but the URL contents (including the base64 payload) must be byte-identical."""
    content = {"type": "image_url", "image_url": _DATA_URL}

    _azure_image_url_helper(content)

    assert isinstance(content["image_url"], dict), (
        "_azure_image_url_helper must wrap a string image_url into a dict"
    )
    assert content["image_url"] == {"url": _DATA_URL}
    # Decode round-trip — same bytes that went in must come out.
    fwd_b64 = content["image_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(fwd_b64) == _PNG_BYTES


def test_convert_to_azure_openai_messages_round_trips_base64_image_bytes():
    """End-to-end on the chat-message conversion: a base64 image survives the
    full ``convert_to_azure_openai_messages`` pass with the underlying PNG
    bytes byte-identical on the way out."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this image?"},
                {"type": "image_url", "image_url": {"url": _DATA_URL}},
            ],
        }
    ]
    original_messages = copy.deepcopy(messages)

    out = convert_to_azure_openai_messages(messages)

    assert len(out) == 1
    out_msg = out[0]
    assert out_msg["role"] == "user"
    # Image part — image_url dict-form preserved.
    fwd_image_url = _extract_image_url(out_msg)
    assert isinstance(fwd_image_url, dict)
    assert fwd_image_url["url"] == _DATA_URL
    # Decode the forwarded base64 and verify it equals the original PNG bytes.
    fwd_b64 = fwd_image_url["url"].split(",", 1)[1]
    decoded = base64.b64decode(fwd_b64)
    assert decoded == _PNG_BYTES, (
        f"Azure conversion altered base64 image bytes: "
        f"got {len(decoded)} bytes, expected {len(_PNG_BYTES)}"
    )
    # Text part is also untouched.
    text_parts = [
        p for p in out_msg["content"] if isinstance(p, dict) and p.get("type") == "text"
    ]
    assert text_parts and text_parts[0]["text"] == original_messages[0]["content"][0]["text"]


def test_convert_to_azure_openai_messages_round_trips_string_form_base64():
    """Same as above but using the legacy string-form ``image_url`` — must be
    wrapped to dict form, and the base64 payload must still decode byte-equal."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this image?"},
                {"type": "image_url", "image_url": _DATA_URL},
            ],
        }
    ]

    out = convert_to_azure_openai_messages(messages)

    fwd_image_url = _extract_image_url(out[0])
    assert isinstance(fwd_image_url, dict), (
        "string-form image_url must be wrapped to dict form for Azure"
    )
    assert fwd_image_url == {"url": _DATA_URL}
    fwd_b64 = fwd_image_url["url"].split(",", 1)[1]
    assert base64.b64decode(fwd_b64) == _PNG_BYTES


def test_azure_transform_request_preserves_base64_image_url_in_messages():
    """``AzureOpenAIConfig.transform_request`` is the entrypoint used by the
    Azure chat handler. Pin that the request body it produces contains the
    base64 ``image_url`` byte-identical to the input."""
    config = AzureOpenAIConfig()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this image?"},
                {"type": "image_url", "image_url": {"url": _DATA_URL}},
            ],
        }
    ]

    body = config.transform_request(
        model="gpt-4o",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert body["model"] == "gpt-4o"
    fwd_image_url = _extract_image_url(body["messages"][0])
    assert isinstance(fwd_image_url, dict)
    assert fwd_image_url["url"] == _DATA_URL
    fwd_b64 = fwd_image_url["url"].split(",", 1)[1]
    assert base64.b64decode(fwd_b64) == _PNG_BYTES
