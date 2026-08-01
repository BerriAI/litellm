"""
Performance benchmarks for the A2A (agent-to-agent) message-translation hot path.

Both directions are covered: the client direction (litellm.completion talking to
an upstream A2A agent) converts OpenAI messages into a prompt and extracts text
from the A2A response, and the proxy server-ingress direction converts an inbound
A2A message into OpenAI messages before bridging to a completion. All are pure-CPU
per-request transforms.
"""

import pytest

from litellm.a2a_protocol.litellm_completion_bridge.transformation import (
    A2ACompletionBridgeTransformation,
)
from litellm.llms.a2a.common_utils import (
    convert_messages_to_prompt,
    extract_text_from_a2a_response,
)

MESSAGES = [
    {"role": "system", "content": "You are a helpful research assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "And what is its population?"},
]

MESSAGE_RESPONSE = {
    "result": {
        "kind": "message",
        "parts": [
            {"kind": "text", "text": "The population of Paris is about 2.1 million."},
            {"kind": "text", "text": "The metro area has over 12 million people."},
        ],
    }
}

TASK_RESPONSE = {
    "result": {
        "kind": "task",
        "artifacts": [{"parts": [{"kind": "text", "text": "Paris has a population of about 2.1 million."}]}],
    }
}

A2A_INBOUND_MESSAGE = {
    "role": "user",
    "parts": [
        {"kind": "text", "text": "Summarize the latest quarterly report."},
        {"kind": "text", "text": "Focus on revenue and margins."},
    ],
    "messageId": "msg-1",
}


@pytest.mark.benchmark
def test_convert_messages_to_a2a_prompt():
    """Benchmark converting OpenAI messages into an A2A prompt string."""
    convert_messages_to_prompt(messages=MESSAGES)


@pytest.mark.benchmark
def test_extract_text_from_a2a_message_response():
    """Benchmark extracting text from a direct-message A2A response."""
    extract_text_from_a2a_response(response_dict=MESSAGE_RESPONSE)


@pytest.mark.benchmark
def test_extract_text_from_a2a_task_response():
    """Benchmark extracting text from a task-with-artifacts A2A response."""
    extract_text_from_a2a_response(response_dict=TASK_RESPONSE)


@pytest.mark.benchmark
def test_a2a_inbound_message_to_openai_messages():
    """Benchmark the proxy converting an inbound A2A message into OpenAI messages."""
    A2ACompletionBridgeTransformation.a2a_message_to_openai_messages(A2A_INBOUND_MESSAGE)
