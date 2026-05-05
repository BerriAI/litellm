"""Stream / EventStream envelope tests for passthrough flush plumbing."""

import base64
import json

import pytest


def test_event_stream_message_to_response_dict_has_no_chunk_key():
    """
    ``EventStreamMessage`` is botocore's decoded *frame* object. ``to_response_dict()``
    only exposes ``status_code``, ``headers``, and ``body`` (``body`` == ``payload`` bytes).

    A ``"chunk"`` member (Bedrock ResponseStream union) appears only *after*
    ``EventStreamJSONParser.parse(response_dict, ...)`` on ``common_utils`` — not on this dict.
    """
    from botocore.eventstream import EventStreamMessage, MessagePrelude

    prelude = MessagePrelude(total_length=256, headers_length=0, crc=0)
    headers = {":message-type": "event"}
    outer_json = {
        "chunk": {
            "bytes": base64.b64encode(b'{"delta":{"text":"x"}}').decode("ascii"),
        }
    }
    payload = json.dumps(outer_json).encode("utf-8")
    event = EventStreamMessage(prelude, headers, payload, crc=0)

    response_dict = event.to_response_dict()

    assert "chunk" not in response_dict
    assert set(response_dict.keys()) == {"status_code", "headers", "body"}
    assert response_dict["body"] is event.payload

    # Inner service JSON can still use a ``chunk`` key inside ``body`` bytes.
    loaded = json.loads(response_dict["body"].decode("utf-8"))
    assert "chunk" in loaded


@pytest.mark.parametrize("message_type", ["error", "exception"])
def test_event_stream_message_error_types_response_dict_has_no_chunk_key(
    message_type: str,
):
    from botocore.eventstream import EventStreamMessage, MessagePrelude

    prelude = MessagePrelude(total_length=128, headers_length=0, crc=0)
    headers = {":message-type": message_type}
    payload = b'{"message":"oops"}'
    event = EventStreamMessage(prelude, headers, payload, crc=0)

    response_dict = event.to_response_dict()

    assert "chunk" not in response_dict
    assert response_dict["status_code"] == 400
    assert set(response_dict.keys()) == {"status_code", "headers", "body"}
