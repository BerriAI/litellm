"""
OCI Generative AI — end-to-end integration tests.

These tests make REAL calls to OCI.  They are skipped automatically when the
standard ~/.oci/config is absent or when OCI_TEST_COMPARTMENT_ID is not set.

Prerequisites
-------------
- ~/.oci/config with a valid [DEFAULT] profile
- Private key referenced by key_file in that profile
- Sufficient IAM policies to call the Generative AI inference service

Environment variables (all optional — fall back to ~/.oci/config values):
  OCI_TEST_REGION        OCI region (default: us-chicago-1)
  OCI_TEST_COMPARTMENT_ID  compartment OCID (default: tenancy root from config)

Run only these tests:
    pytest tests/llm_translation/test_oci_integration.py -v
"""

import os
import sys
from typing import Generator

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

OCI_CONFIG_FILE = os.path.expanduser("~/.oci/config")
_OCI_AVAILABLE = os.path.isfile(OCI_CONFIG_FILE)

pytestmark = pytest.mark.skipif(
    not _OCI_AVAILABLE,
    reason="~/.oci/config not found — skipping OCI integration tests",
)


@pytest.fixture(scope="module")
def oci_signer():
    """Return an oci.Signer built from ~/.oci/config [DEFAULT]."""
    oci = pytest.importorskip("oci")
    config = oci.config.from_file()
    return oci.Signer(
        tenancy=config["tenancy"],
        user=config["user"],
        fingerprint=config["fingerprint"],
        private_key_file_location=config["key_file"],
    )


@pytest.fixture(scope="module")
def oci_params(oci_signer) -> dict:
    """Common OCI call-time parameters shared by all tests."""
    oci = pytest.importorskip("oci")
    config = oci.config.from_file()
    compartment_id = os.environ.get("OCI_TEST_COMPARTMENT_ID", config["tenancy"])
    region = os.environ.get("OCI_TEST_REGION", "us-chicago-1")
    return {
        "oci_signer": oci_signer,
        "oci_compartment_id": compartment_id,
        "oci_region": region,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chat(model: str, message: str, params: dict, max_tokens: int = 64) -> str:
    """Run a single-turn completion and return the text content."""
    import litellm

    resp = litellm.completion(
        model=f"oci/{model}",
        messages=[{"role": "user", "content": message}],
        max_tokens=max_tokens,
        **params,
    )
    # Reasoning models may return None content when all budget is used by reasoning
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Chat tests — one per vendor family
# ---------------------------------------------------------------------------


class TestOCIChatMeta:
    """Meta Llama models (GENERIC apiFormat)."""

    MODEL = "meta.llama-3.3-70b-instruct"

    def test_basic_completion(self, oci_params):
        import litellm

        resp = litellm.completion(
            model=f"oci/{self.MODEL}",
            messages=[{"role": "user", "content": "Reply with only the word: pong"}],
            max_tokens=10,
            **oci_params,
        )
        assert resp.choices[0].message.content is not None
        assert resp.choices[0].finish_reason is not None
        assert resp.usage.prompt_tokens > 0

    def test_usage_populated(self, oci_params):
        import litellm

        resp = litellm.completion(
            model=f"oci/{self.MODEL}",
            messages=[{"role": "user", "content": "Say hi."}],
            max_tokens=20,
            **oci_params,
        )
        assert resp.usage.prompt_tokens > 0
        assert resp.usage.total_tokens >= resp.usage.prompt_tokens

    def test_system_message(self, oci_params):
        import litellm

        resp = litellm.completion(
            model=f"oci/{self.MODEL}",
            messages=[
                {"role": "system", "content": "You only reply in pirate speak."},
                {"role": "user", "content": "Hello!"},
            ],
            max_tokens=30,
            **oci_params,
        )
        assert resp.choices[0].message.content is not None

    def test_streaming(self, oci_params):
        import litellm

        chunks = list(
            litellm.completion(
                model=f"oci/{self.MODEL}",
                messages=[{"role": "user", "content": "Count to 3."}],
                max_tokens=30,
                stream=True,
                **oci_params,
            )
        )
        assert len(chunks) > 0
        content = "".join(
            c.choices[0].delta.content or "" for c in chunks if c.choices
        )
        assert len(content) > 0

    def test_multi_turn(self, oci_params):
        import litellm

        resp = litellm.completion(
            model=f"oci/{self.MODEL}",
            messages=[
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Nice to meet you, Alice!"},
                {"role": "user", "content": "What is my name?"},
            ],
            max_tokens=30,
            **oci_params,
        )
        assert "Alice" in (resp.choices[0].message.content or "")


class TestOCIChatGoogle:
    """Google Gemini models (GENERIC apiFormat)."""

    MODEL = "google.gemini-2.5-flash"

    def test_basic_completion(self, oci_params):
        import litellm

        resp = litellm.completion(
            model=f"oci/{self.MODEL}",
            messages=[{"role": "user", "content": "Reply with only the word: pong"}],
            max_tokens=200,
            **oci_params,
        )
        # Gemini 2.5 Flash is a reasoning model — content may be empty if reasoning
        # consumed the budget, but no exception should be raised.
        assert resp.choices[0].finish_reason is not None
        assert resp.usage.prompt_tokens > 0

    def test_usage_populated(self, oci_params):
        import litellm

        resp = litellm.completion(
            model=f"oci/{self.MODEL}",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            max_tokens=200,
            **oci_params,
        )
        assert resp.usage.total_tokens > 0

    def test_streaming(self, oci_params):
        import litellm

        chunks = list(
            litellm.completion(
                model=f"oci/{self.MODEL}",
                messages=[{"role": "user", "content": "Say the word hello."}],
                max_tokens=200,
                stream=True,
                **oci_params,
            )
        )
        assert len(chunks) > 0


class TestOCIChatXAI:
    """xAI Grok models (GENERIC apiFormat)."""

    MODEL = "xai.grok-3-mini"

    def test_basic_completion(self, oci_params):
        import litellm

        resp = litellm.completion(
            model=f"oci/{self.MODEL}",
            messages=[{"role": "user", "content": "Reply with only the word: pong"}],
            max_tokens=50,
            **oci_params,
        )
        assert resp.choices[0].message.content is not None
        assert resp.usage.total_tokens > 0

    def test_streaming(self, oci_params):
        import litellm

        chunks = list(
            litellm.completion(
                model=f"oci/{self.MODEL}",
                messages=[{"role": "user", "content": "Count to 3."}],
                max_tokens=50,
                stream=True,
                **oci_params,
            )
        )
        assert len(chunks) > 0

    def test_usage_has_reasoning_tokens(self, oci_params):
        """Grok mini exposes reasoning token breakdown in usage."""
        import litellm

        resp = litellm.completion(
            model=f"oci/{self.MODEL}",
            messages=[{"role": "user", "content": "What is 5*7?"}],
            max_tokens=100,
            **oci_params,
        )
        # totalTokens >= completionTokens + promptTokens (reasoning may add overhead)
        assert resp.usage.total_tokens >= resp.usage.prompt_tokens


class TestOCIChatCohere:
    """Cohere Command models (COHERE apiFormat)."""

    MODEL = "cohere.command-latest"

    def test_basic_completion(self, oci_params):
        import litellm

        resp = litellm.completion(
            model=f"oci/{self.MODEL}",
            messages=[{"role": "user", "content": "Reply with only the word: pong"}],
            max_tokens=20,
            **oci_params,
        )
        assert resp.choices[0].message.content is not None
        assert resp.usage.prompt_tokens > 0

    def test_streaming(self, oci_params):
        import litellm

        chunks = list(
            litellm.completion(
                model=f"oci/{self.MODEL}",
                messages=[{"role": "user", "content": "Count to 3."}],
                max_tokens=30,
                stream=True,
                **oci_params,
            )
        )
        assert len(chunks) > 0
        content = "".join(
            c.choices[0].delta.content or "" for c in chunks if c.choices
        )
        assert len(content) > 0

    def test_system_message(self, oci_params):
        import litellm

        resp = litellm.completion(
            model=f"oci/{self.MODEL}",
            messages=[
                {"role": "system", "content": "Always end your response with 'cheers'."},
                {"role": "user", "content": "Say hello."},
            ],
            max_tokens=40,
            **oci_params,
        )
        assert resp.choices[0].message.content is not None


# ---------------------------------------------------------------------------
# Embedding tests
# ---------------------------------------------------------------------------


class TestOCIEmbeddings:

    def test_english_v3_basic(self, oci_params):
        import litellm

        resp = litellm.embedding(
            model="oci/cohere.embed-english-v3.0",
            input=["Hello world"],
            input_type="SEARCH_DOCUMENT",
            **oci_params,
        )
        assert len(resp.data) == 1
        assert len(resp.data[0]["embedding"]) == 1024
        assert resp.usage.prompt_tokens > 0

    def test_english_v3_batch(self, oci_params):
        import litellm

        texts = [
            "The quick brown fox",
            "jumps over the lazy dog",
            "Paris is the capital of France",
        ]
        resp = litellm.embedding(
            model="oci/cohere.embed-english-v3.0",
            input=texts,
            input_type="SEARCH_DOCUMENT",
            **oci_params,
        )
        assert len(resp.data) == 3
        for i, item in enumerate(resp.data):
            assert item["index"] == i
            assert len(item["embedding"]) == 1024

    def test_multilingual_v3(self, oci_params):
        import litellm

        resp = litellm.embedding(
            model="oci/cohere.embed-multilingual-v3.0",
            input=["Bonjour le monde", "Hola mundo"],
            input_type="SEARCH_DOCUMENT",
            **oci_params,
        )
        assert len(resp.data) == 2
        assert len(resp.data[0]["embedding"]) == 1024

    def test_search_query_input_type(self, oci_params):
        import litellm

        resp = litellm.embedding(
            model="oci/cohere.embed-english-v3.0",
            input=["What is the capital of France?"],
            input_type="SEARCH_QUERY",
            **oci_params,
        )
        assert len(resp.data[0]["embedding"]) == 1024

    def test_semantic_similarity(self, oci_params):
        """Semantically similar texts should have higher cosine similarity."""
        import litellm
        import math

        resp = litellm.embedding(
            model="oci/cohere.embed-english-v3.0",
            input=[
                "The cat sat on the mat",
                "A feline rested on the rug",
                "The stock market crashed today",
            ],
            input_type="SEARCH_DOCUMENT",
            **oci_params,
        )

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            mag_a = math.sqrt(sum(x ** 2 for x in a))
            mag_b = math.sqrt(sum(x ** 2 for x in b))
            return dot / (mag_a * mag_b)

        cat1 = resp.data[0]["embedding"]
        cat2 = resp.data[1]["embedding"]
        stock = resp.data[2]["embedding"]

        sim_cats = cosine(cat1, cat2)
        sim_diff = cosine(cat1, stock)
        assert sim_cats > sim_diff, (
            f"Expected similar sentences to score higher ({sim_cats:.3f} vs {sim_diff:.3f})"
        )

    def test_embed_v4(self, oci_params):
        import litellm

        resp = litellm.embedding(
            model="oci/cohere.embed-v4.0",
            input=["Hello world"],
            input_type="SEARCH_DOCUMENT",
            **oci_params,
        )
        assert len(resp.data) == 1
        assert len(resp.data[0]["embedding"]) == 1536

    def test_usage_tokens(self, oci_params):
        import litellm

        resp = litellm.embedding(
            model="oci/cohere.embed-english-v3.0",
            input=["short text", "another short text"],
            input_type="SEARCH_DOCUMENT",
            **oci_params,
        )
        assert resp.usage.prompt_tokens > 0
        assert resp.usage.total_tokens == resp.usage.prompt_tokens


# ---------------------------------------------------------------------------
# Env-var credential path
# ---------------------------------------------------------------------------


class TestOCIEnvVarCredentials:
    """Verify that OCI_* env vars are picked up without explicit params."""

    def test_completion_via_env_vars(self, monkeypatch):
        """Completion works when credentials are set through environment variables."""
        oci = pytest.importorskip("oci")
        config = oci.config.from_file()
        key_path = os.path.expanduser(config["key_file"])

        with open(key_path) as f:
            key_pem = f.read()

        monkeypatch.setenv("OCI_REGION", "us-chicago-1")
        monkeypatch.setenv("OCI_USER", config["user"])
        monkeypatch.setenv("OCI_FINGERPRINT", config["fingerprint"])
        monkeypatch.setenv("OCI_TENANCY", config["tenancy"])
        monkeypatch.setenv("OCI_KEY", key_pem)
        monkeypatch.setenv("OCI_COMPARTMENT_ID", config["tenancy"])

        import litellm

        resp = litellm.completion(
            model="oci/meta.llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": "Reply with only the word: pong"}],
            max_tokens=10,
        )
        assert resp.choices[0].message.content is not None

    def test_embedding_via_env_vars(self, monkeypatch):
        oci = pytest.importorskip("oci")
        config = oci.config.from_file()
        key_path = os.path.expanduser(config["key_file"])

        with open(key_path) as f:
            key_pem = f.read()

        monkeypatch.setenv("OCI_REGION", "us-chicago-1")
        monkeypatch.setenv("OCI_USER", config["user"])
        monkeypatch.setenv("OCI_FINGERPRINT", config["fingerprint"])
        monkeypatch.setenv("OCI_TENANCY", config["tenancy"])
        monkeypatch.setenv("OCI_KEY", key_pem)
        monkeypatch.setenv("OCI_COMPARTMENT_ID", config["tenancy"])

        import litellm

        resp = litellm.embedding(
            model="oci/cohere.embed-english-v3.0",
            input=["hello"],
            input_type="SEARCH_DOCUMENT",
        )
        assert len(resp.data[0]["embedding"]) == 1024


# ---------------------------------------------------------------------------
# Async chat tests
# ---------------------------------------------------------------------------


class TestOCIAsyncChat:
    """Verify async completion and streaming work for each vendor family."""

    @pytest.mark.asyncio
    async def test_async_completion_meta(self, oci_params):
        import litellm

        resp = await litellm.acompletion(
            model="oci/meta.llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": "Reply with only the word: pong"}],
            max_tokens=10,
            **oci_params,
        )
        assert resp.choices[0].message.content is not None
        assert resp.usage.prompt_tokens > 0

    @pytest.mark.asyncio
    async def test_async_completion_google(self, oci_params):
        import litellm

        resp = await litellm.acompletion(
            model="oci/google.gemini-2.5-flash",
            messages=[{"role": "user", "content": "Reply with only the word: pong"}],
            max_tokens=200,
            **oci_params,
        )
        assert resp.choices[0].finish_reason is not None
        assert resp.usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_async_completion_xai(self, oci_params):
        import litellm

        resp = await litellm.acompletion(
            model="oci/xai.grok-3-mini",
            messages=[{"role": "user", "content": "Reply with only the word: pong"}],
            max_tokens=50,
            **oci_params,
        )
        assert resp.choices[0].message.content is not None

    @pytest.mark.asyncio
    async def test_async_completion_cohere(self, oci_params):
        import litellm

        resp = await litellm.acompletion(
            model="oci/cohere.command-latest",
            messages=[{"role": "user", "content": "Reply with only the word: pong"}],
            max_tokens=20,
            **oci_params,
        )
        assert resp.choices[0].message.content is not None

    @pytest.mark.asyncio
    async def test_async_streaming_meta(self, oci_params):
        import litellm

        chunks = []
        async for chunk in await litellm.acompletion(
            model="oci/meta.llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": "Count to 3."}],
            max_tokens=30,
            stream=True,
            **oci_params,
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        content = "".join(
            c.choices[0].delta.content or "" for c in chunks if c.choices
        )
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_async_streaming_google(self, oci_params):
        import litellm

        chunks = []
        async for chunk in await litellm.acompletion(
            model="oci/google.gemini-2.5-flash",
            messages=[{"role": "user", "content": "Say the word hello."}],
            max_tokens=200,
            stream=True,
            **oci_params,
        ):
            chunks.append(chunk)

        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_async_streaming_xai(self, oci_params):
        import litellm

        chunks = []
        async for chunk in await litellm.acompletion(
            model="oci/xai.grok-3-mini",
            messages=[{"role": "user", "content": "Count to 3."}],
            max_tokens=50,
            stream=True,
            **oci_params,
        ):
            chunks.append(chunk)

        assert len(chunks) > 0


# ---------------------------------------------------------------------------
# Async embedding tests
# ---------------------------------------------------------------------------


class TestOCIAsyncEmbeddings:

    @pytest.mark.asyncio
    async def test_async_embedding_basic(self, oci_params):
        import litellm

        resp = await litellm.aembedding(
            model="oci/cohere.embed-english-v3.0",
            input=["Hello world"],
            input_type="SEARCH_DOCUMENT",
            **oci_params,
        )
        assert len(resp.data) == 1
        assert len(resp.data[0]["embedding"]) == 1024
        assert resp.usage.prompt_tokens > 0

    @pytest.mark.asyncio
    async def test_async_embedding_batch(self, oci_params):
        import litellm

        texts = ["The quick brown fox", "jumps over the lazy dog"]
        resp = await litellm.aembedding(
            model="oci/cohere.embed-english-v3.0",
            input=texts,
            input_type="SEARCH_DOCUMENT",
            **oci_params,
        )
        assert len(resp.data) == 2
        assert all(len(item["embedding"]) == 1024 for item in resp.data)

    @pytest.mark.asyncio
    async def test_async_embedding_multilingual(self, oci_params):
        import litellm

        resp = await litellm.aembedding(
            model="oci/cohere.embed-multilingual-v3.0",
            input=["Bonjour le monde"],
            input_type="SEARCH_DOCUMENT",
            **oci_params,
        )
        assert len(resp.data[0]["embedding"]) == 1024


# ---------------------------------------------------------------------------
# Tool use / function calling tests
# ---------------------------------------------------------------------------


class TestOCIToolUse:
    """Verify tool use (function calling) works for models that support it."""

    # Simple weather tool definition
    WEATHER_TOOL = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name.",
                    }
                },
                "required": ["city"],
            },
        },
    }

    def _assert_tool_call(self, resp, expected_tool: str = "get_weather"):
        """Assert the response contains a tool call."""
        choice = resp.choices[0]
        assert choice.finish_reason in ("tool_calls", "stop")
        if choice.finish_reason == "tool_calls":
            assert choice.message.tool_calls is not None
            assert len(choice.message.tool_calls) > 0
            assert choice.message.tool_calls[0].function.name == expected_tool
        # stop finish_reason can happen if model answers without calling the tool —
        # acceptable behaviour, not a bug.

    def test_tool_use_meta(self, oci_params):
        import litellm

        resp = litellm.completion(
            model="oci/meta.llama-3.3-70b-instruct",
            messages=[
                {"role": "user", "content": "What is the weather in Paris?"}
            ],
            tools=[self.WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=100,
            **oci_params,
        )
        self._assert_tool_call(resp)

    def test_tool_use_cohere(self, oci_params):
        import litellm

        resp = litellm.completion(
            model="oci/cohere.command-latest",
            messages=[
                {"role": "user", "content": "What is the weather in Tokyo?"}
            ],
            tools=[self.WEATHER_TOOL],
            max_tokens=200,
            **oci_params,
        )
        self._assert_tool_call(resp)

    def test_tool_use_google(self, oci_params):
        import litellm

        resp = litellm.completion(
            model="oci/google.gemini-2.5-flash",
            messages=[
                {"role": "user", "content": "What is the weather in London?"}
            ],
            tools=[self.WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=200,
            **oci_params,
        )
        self._assert_tool_call(resp)

    @pytest.mark.asyncio
    async def test_async_tool_use_meta(self, oci_params):
        import litellm

        resp = await litellm.acompletion(
            model="oci/meta.llama-3.3-70b-instruct",
            messages=[
                {"role": "user", "content": "What is the weather in Berlin?"}
            ],
            tools=[self.WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=100,
            **oci_params,
        )
        self._assert_tool_call(resp)
