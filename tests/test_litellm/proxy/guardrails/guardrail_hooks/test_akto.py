"""Tests for the Akto guardrail's ingest payload construction."""

from litellm.proxy.guardrails.guardrail_hooks.akto.akto import AktoGuardrail


class TestBuildRequestBody:
    def test_request_scan_uses_structured_messages(self):
        body = AktoGuardrail.build_request_body(
            inputs={
                "texts": ["hi"],
                "structured_messages": [{"role": "user", "content": "hi"}],
            },
            request_data={"messages": [{"role": "user", "content": "raw"}]},
        )
        assert body["messages"] == [{"role": "user", "content": "hi"}]

    def test_response_scan_ingests_request_messages_not_conversation(self):
        """On response scans structured_messages carries the response turns too;
        the ingested request body must stay the actual request."""
        body = AktoGuardrail.build_request_body(
            inputs={
                "texts": ["the reply"],
                "structured_messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "the reply"},
                ],
            },
            request_data={"messages": [{"role": "user", "content": "hi"}]},
            prefer_structured_messages=False,
        )
        assert body["messages"] == [{"role": "user", "content": "hi"}]
