import sys
import os

# Add litellm to sys.path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)

from litellm.llms.a2a.chat.transformation import A2AConfig
from litellm.llms.a2a.common_utils import extract_text_from_a2a_message


def test_regression_issue_28577_a2a_discriminator():
    """
    Test that A2A transformation adds the mandatory 'kind': 'message' discriminator.
    Fixes Bug 1 in #28577.
    """
    config = A2AConfig()
    messages = [{"role": "user", "content": "ping"}]

    # transform_request creates the A2A JSON-RPC payload
    request_data = config.transform_request(
        model="a2a/demo",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    # Check Bug 1: message.kind missing
    a2a_message = request_data["params"]["message"]
    assert a2a_message["kind"] == "message"
    assert a2a_message["role"] == "user"
    assert "parts" in a2a_message


def test_regression_issue_28577_a2a_data_serialization():
    """
    Test that A2A common_utils handle kind: 'data' parts by serializing them.
    Fixes Bug 2 in #28577.
    """
    message_with_data = {
        "kind": "message",
        "role": "assistant",
        "parts": [{"kind": "data", "data": {"result": {"msg": "pong"}}}],
        "messageId": "msg-123",
    }

    text = extract_text_from_a2a_message(message_with_data)
    assert '"result": {"msg": "pong"}' in text


def test_regression_issue_28577_a2a_blocking_param():
    """
    Test that A2A requests include configuration.blocking: True.
    Fixes Bug 3 in #28577 (async task unblocking).
    """
    config = A2AConfig()
    messages = [{"role": "user", "content": "ping"}]

    request_data = config.transform_request(
        model="a2a/demo",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    # Check Bug 3 fix: configuration.blocking = True
    assert "configuration" in request_data["params"]
    assert request_data["params"]["configuration"]["blocking"] is True
