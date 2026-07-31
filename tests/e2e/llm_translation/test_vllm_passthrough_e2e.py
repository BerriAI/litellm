"""Live e2e for the /vllm passthrough route.

/vllm/{endpoint} is a raw passthrough: the client sends an OpenAI-format request
and litellm forwards it verbatim to the configured vLLM backend (VLLM_API_BASE),
with no per-request model registration (unlike the managed hosted_vllm path in
tests/e2e/batches). This drives /vllm/v1/chat/completions and asserts the
forwarded completion comes back with real content.

On stage the backend is a CPU llama.cpp server standing in for vLLM (the cluster
is GPU-less and its CPU nodes lack the AVX512 vLLM's CPU build needs); from
litellm's side the passthrough code path is identical. Batch and file passthrough
(/vllm/v1/batches, /vllm/v1/files) is not covered: no self-hosted
vLLM-compatible server implements the OpenAI Batch API, so there is no backend to
forward those routes to.

A passthrough call returning non-2xx fails hard (never a skip); once it is 2xx, a
missing or empty completion fails too.
"""

import pytest

from e2e_config import unique_marker
from e2e_http import require_successful_call
from models import ChatResponse
from passthrough_client import PassthroughClient

pytestmark = pytest.mark.e2e

VLLM_PASSTHROUGH_MODEL = "qwen2.5-0.5b-instruct"


class TestVllmChatPassthrough:
    @pytest.mark.covers("llm.chat_completions.hosted_vllm.passthrough.nonstream.works")
    def test_vllm_chat_passthrough_returns_completion(
        self, client: PassthroughClient, scoped_key: str
    ) -> None:
        result = client.vllm_chat(
            scoped_key, VLLM_PASSTHROUGH_MODEL, f"Say hello in one word ({unique_marker()})"
        )
        require_successful_call(result)

        parsed = ChatResponse.model_validate_json(result.body)
        assert parsed.choices, f"/vllm chat passthrough returned no choices: {result.body[:300]}"
        message = parsed.choices[0].message
        content = (message.content if message else None) or ""
        assert content.strip(), f"/vllm chat passthrough returned empty content: {result.body[:300]}"
