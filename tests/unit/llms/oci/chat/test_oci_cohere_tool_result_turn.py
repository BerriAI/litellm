"""Regression tests for OCI Cohere tool-result turns (issue #46157).

Two independent defects break the call *after* a tool call for OCI Cohere
models:

1. ``OCIChatConfig.transform_request`` put the last user message in the
   top-level ``message`` while trailing ``tool``-role results stayed at the end
   of ``chatHistory``. OCI rejects this combination with HTTP 400 ("cannot
   specify message if the last entry in chat history contains tool results").
   The accepted shape for a tool-result turn is the trailing TOOL entries moved
   out of ``chatHistory`` into the top-level ``toolResults`` and ``message``
   sent as ``""``.

2. OCI writes citation document references in camelCase (``documentIds``), but
   ``CohereCitation`` only accepted ``document_ids``, so
   ``handle_cohere_response`` raised a validation error whenever the model cited
   tool output.
"""

from litellm.llms.oci.chat.cohere import (
    adapt_messages_to_cohere_standard,
    split_trailing_tool_results,
)
from litellm.llms.oci.chat.transformation import OCIChatConfig
from litellm.types.llms.oci import (
    CohereChatBotMessage,
    CohereChatResult,
    CohereCitation,
    CohereMessage,
    CohereToolMessage,
)

TEST_COMPARTMENT_ID = "ocid1.compartment.oc1..xxxxxx"
BASE_OCI_PARAMS = {
    "oci_region": "us-chicago-1",
    "oci_user": "ocid1.user.oc1..xxxxxxEXAMPLExxxxxx",
    "oci_fingerprint": "4f:29:77:cc:b1:3e:55:ab:61:2a:de:47:f1:38:4c:90",
    "oci_tenancy": "ocid1.tenancy.oc1..xxxxxxEXAMPLExxxxxx",
    "oci_compartment_id": TEST_COMPARTMENT_ID,
    "oci_key_file": "/path/to/private_key.pem",
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_policy",
            "description": "Look up a policy by topic.",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        },
    }
]


def _tool_result_turn_messages():
    """The exact agent-loop continuation from issue #46157."""
    return [
        {"role": "user", "content": "How is support ticket priority decided?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup_policy",
                        "arguments": '{"topic": "ticket priority"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "P1 for an outage, P2 for degraded service, P3 for questions.",
        },
    ]


def _transform(messages):
    return OCIChatConfig().transform_request(
        model="oci/cohere.command-a-03-2025",
        messages=messages,  # type: ignore[arg-type]
        optional_params={**BASE_OCI_PARAMS, "tools": TOOLS, "max_tokens": 300},
        litellm_params={},
        headers={},
    )


class TestTrailingToolResultHoisting:
    """Request-side fix: trailing TOOL entries → top-level ``toolResults``."""

    def test_tool_result_turn_hoists_to_tool_results_and_empties_message(self):
        request = _transform(_tool_result_turn_messages())
        chat_request = request["chatRequest"]

        # The top-level message must be empty for a tool-result turn.
        assert chat_request["message"] == ""

        # No TOOL entry may remain at the end of chatHistory.
        history = chat_request["chatHistory"]
        assert history
        assert history[-1]["role"] != "TOOL"
        # The assistant tool-call turn that produced the results stays in history.
        assert any(entry["role"] == "CHATBOT" for entry in history)

        # The trailing result is hoisted into top-level toolResults with the
        # originating call's name/parameters resolved from the assistant turn.
        assert chat_request["toolResults"] == [
            {
                "call": {"name": "lookup_policy", "parameters": {"topic": "ticket priority"}},
                "outputs": [{"output": "P1 for an outage, P2 for degraded service, P3 for questions."}],
            }
        ]

    def test_parallel_tool_results_hoisted_in_order(self):
        messages = [
            {"role": "user", "content": "Check both policies."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "lookup_policy", "arguments": '{"topic": "a"}'},
                    },
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {"name": "lookup_policy", "arguments": '{"topic": "b"}'},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": "result A"},
            {"role": "tool", "tool_call_id": "call_b", "content": "result B"},
        ]
        request = _transform(messages)
        chat_request = request["chatRequest"]

        assert chat_request["message"] == ""
        assert chat_request["toolResults"] == [
            {
                "call": {"name": "lookup_policy", "parameters": {"topic": "a"}},
                "outputs": [{"output": "result A"}],
            },
            {
                "call": {"name": "lookup_policy", "parameters": {"topic": "b"}},
                "outputs": [{"output": "result B"}],
            },
        ]
        assert not any(entry["role"] == "TOOL" for entry in chat_request["chatHistory"])

    def test_plain_turn_is_unchanged(self):
        """A normal user→assistant turn keeps message non-empty and no toolResults."""
        messages = [
            {"role": "user", "content": "How is support ticket priority decided?"},
            {"role": "assistant", "content": "P1 for an outage."},
        ]
        chat_request = _transform(messages)["chatRequest"]

        # The last user message becomes the top-level ``message``; the remaining
        # history (the assistant reply here) stays in ``chatHistory``.
        assert chat_request["message"] == "How is support ticket priority decided?"
        assert "toolResults" not in chat_request
        assert chat_request["chatHistory"] == [{"role": "CHATBOT", "message": "P1 for an outage."}]

    def test_tool_result_in_middle_is_not_hoisted_when_user_message_follows(self):
        """A trailing user message ends the turn; the TOOL entry stays in history."""
        messages = [
            {"role": "user", "content": "How is support ticket priority decided?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup_policy", "arguments": '{"topic": "ticket priority"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "P1 for an outage, P2 for degraded service, P3 for questions.",
            },
            {"role": "user", "content": "Thanks, that helps."},
        ]
        chat_request = _transform(messages)["chatRequest"]

        # The turn ends on a user message, so the request keeps the previous
        # shape: the new question is sent as ``message`` and the tool result
        # stays in ``chatHistory``.
        assert chat_request["message"] == "Thanks, that helps."
        assert "toolResults" not in chat_request
        assert chat_request["chatHistory"][-1]["role"] == "TOOL"


class TestSplitTrailingToolResults:
    """Unit coverage for the split helper itself (normal/None/boundary)."""

    def test_no_tool_message_returns_unchanged(self):
        history = [CohereChatBotMessage(role="CHATBOT", message="hi")]
        remaining, trailing = split_trailing_tool_results(history)
        assert trailing == []
        assert remaining == history

    def test_empty_history_returns_empty(self):
        remaining, trailing = split_trailing_tool_results([])
        assert remaining == []
        assert trailing == []

    def test_all_tool_messages_are_hoisted(self):
        history: list[CohereMessage] = [
            CohereChatBotMessage(role="CHATBOT", message=""),
            CohereToolMessage(
                role="TOOL",
                toolResults=[
                    {
                        "call": {"name": "lookup_policy", "parameters": {"topic": "a"}},
                        "outputs": [{"output": "result A"}],
                    }
                ],
            ),
            CohereToolMessage(
                role="TOOL",
                toolResults=[
                    {
                        "call": {"name": "lookup_policy", "parameters": {"topic": "b"}},
                        "outputs": [{"output": "result B"}],
                    }
                ],
            ),
        ]
        remaining, trailing = split_trailing_tool_results(history)
        assert [m.role for m in remaining] == ["CHATBOT"]
        assert [r.call.name for r in trailing] == ["lookup_policy", "lookup_policy"]
        assert [r.call.parameters for r in trailing] == [
            {"topic": "a"},
            {"topic": "b"},
        ]

    def test_adapter_output_is_accepted_by_split(self):
        """End-to-end: adapter output feeds straight into the split helper."""
        from litellm.types.llms.openai import AllMessageValues

        messages: list[AllMessageValues] = _tool_result_turn_messages()  # type: ignore[assignment]
        history = adapt_messages_to_cohere_standard(messages)
        remaining, trailing = split_trailing_tool_results(history)
        assert trailing
        assert not any(isinstance(m, CohereToolMessage) for m in remaining)


class TestCohereCitationAlias:
    """Response-side fix: accept OCI's camelCase ``documentIds``."""

    def test_citation_accepts_document_ids(self):
        citation = CohereCitation.model_validate(
            {
                "start": 0,
                "end": 10,
                "text": "cited",
                "documentIds": ["lookup_policy:0:1:0"],
            }
        )
        assert citation.document_ids == ["lookup_policy:0:1:0"]

    def test_citation_still_accepts_snake_case(self):
        citation = CohereCitation.model_validate(
            {
                "start": 0,
                "end": 10,
                "text": "cited",
                "document_ids": ["doc-1"],
            }
        )
        assert citation.document_ids == ["doc-1"]

    def test_citation_serializes_back_to_camel_case(self):
        citation = CohereCitation(
            start=44,
            end=70,
            text="the cited snippet",
            document_ids=["lookup_policy:0:1:0"],
        )
        assert citation.model_dump(exclude_none=True, by_alias=True)["documentIds"] == ["lookup_policy:0:1:0"]

    def test_full_chat_result_with_oci_citations_validates(self):
        """The exact failure from issue #46157: model cites tool output."""
        payload = {
            "modelId": "cohere.command-a-03-2025",
            "modelVersion": "1.0",
            "chatResponse": {
                "text": "According to the policy, outages map to P1.",
                "apiFormat": "COHERE",
                "finishReason": "COMPLETE",
                "citations": [
                    {
                        "start": 44,
                        "end": 70,
                        "text": "P1 for an outage, P2 for degraded service, P3 for questions.",
                        "documentIds": ["lookup_policy:0:1:0"],
                    }
                ],
            },
        }
        result = CohereChatResult.model_validate(payload)
        assert result.chatResponse.citations is not None
        assert result.chatResponse.citations[0].document_ids == ["lookup_policy:0:1:0"]
