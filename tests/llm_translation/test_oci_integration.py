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

import math
import os
import sys
from typing import NamedTuple, Optional

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

OCI_CONFIG_FILE = os.path.expanduser("~/.oci/config")
OCI_PROFILE = os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT")
_OCI_AVAILABLE = os.path.isfile(OCI_CONFIG_FILE)

pytestmark = pytest.mark.skipif(
    not _OCI_AVAILABLE,
    reason="~/.oci/config not found — skipping OCI integration tests",
)


@pytest.fixture(scope="module")
def oci_signer():
    """Return an OCI request signer built from ~/.oci/config [<profile>].

    Supports both API-key profiles and session-token (`oci session authenticate`)
    profiles — the latter are detected by the presence of `security_token_file`.
    """
    oci = pytest.importorskip("oci")
    config = oci.config.from_file(profile_name=OCI_PROFILE)
    if config.get("security_token_file"):
        with open(os.path.expanduser(config["security_token_file"]), "r") as f:
            token = f.read().strip()
        private_key = oci.signer.load_private_key_from_file(config["key_file"])
        return oci.auth.signers.SecurityTokenSigner(token, private_key)
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
    config = oci.config.from_file(profile_name=OCI_PROFILE)
    compartment_id = os.environ.get("OCI_TEST_COMPARTMENT_ID", config["tenancy"])
    region = os.environ.get("OCI_TEST_REGION", "us-chicago-1")
    return {
        "oci_signer": oci_signer,
        "oci_compartment_id": compartment_id,
        "oci_region": region,
    }


# ---------------------------------------------------------------------------
# Model registry
#
# Each entry drives the runtime pivot inside OCI's own transformation layer —
# the tests themselves are format-agnostic.  Per-model quirks are captured in
# the config fields below rather than in separate test classes.
# ---------------------------------------------------------------------------


class _M(NamedTuple):
    """Per-model test configuration."""

    model: str
    max_tokens: int
    # Reasoning models (Gemini 2.5, Grok mini) may return None content when the
    # reasoning budget is exhausted before the answer token budget starts.
    reasoning: bool = False
    # tool_choice value to send; None means omit the parameter entirely.
    tool_choice: Optional[str] = "auto"
    # Whether to include the model in tool-use parametrize list.
    supports_tool_use: bool = True


# All chat models under test.
CHAT_MODELS = [
    pytest.param(_M("meta.llama-3.3-70b-instruct", 64), id="meta"),
    pytest.param(_M("google.gemini-2.5-flash", 200, reasoning=True), id="google"),
    pytest.param(_M("xai.grok-3-mini", 100, reasoning=True), id="xai"),
    pytest.param(_M("cohere.command-latest", 64, tool_choice=None), id="cohere"),
]

# Subset of models that reliably support tool use in OCI.
# xAI Grok mini is omitted — OCI does not expose tool-use for it yet.
TOOL_USE_MODELS = [
    pytest.param(_M("meta.llama-3.3-70b-instruct", 100), id="meta"),
    pytest.param(_M("cohere.command-latest", 200, tool_choice=None), id="cohere"),
    pytest.param(_M("google.gemini-2.5-flash", 200, reasoning=True), id="google"),
]

# Simple weather tool used by all tool-use tests.
_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "The city name."}},
            "required": ["city"],
        },
    },
}


# ---------------------------------------------------------------------------
# Sync chat tests — model list drives the pivot, not separate test classes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", CHAT_MODELS)
def test_basic_completion(m: _M, oci_params):
    import litellm

    resp = litellm.completion(
        model=f"oci/{m.model}",
        messages=[{"role": "user", "content": "Reply with only the word: pong"}],
        max_tokens=m.max_tokens,
        **oci_params,
    )
    assert resp.choices[0].finish_reason is not None
    assert resp.usage.prompt_tokens > 0
    if not m.reasoning:
        assert resp.choices[0].message.content is not None


@pytest.mark.parametrize("m", CHAT_MODELS)
def test_usage_populated(m: _M, oci_params):
    import litellm

    resp = litellm.completion(
        model=f"oci/{m.model}",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        max_tokens=m.max_tokens,
        **oci_params,
    )
    assert resp.usage.prompt_tokens > 0
    assert resp.usage.total_tokens >= resp.usage.prompt_tokens


@pytest.mark.parametrize("m", CHAT_MODELS)
def test_system_message(m: _M, oci_params):
    import litellm

    resp = litellm.completion(
        model=f"oci/{m.model}",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        max_tokens=m.max_tokens,
        **oci_params,
    )
    assert resp.choices[0].finish_reason is not None


@pytest.mark.parametrize("m", CHAT_MODELS)
def test_streaming(m: _M, oci_params):
    import litellm

    chunks = list(
        litellm.completion(
            model=f"oci/{m.model}",
            messages=[{"role": "user", "content": "Count to 3."}],
            max_tokens=m.max_tokens,
            stream=True,
            **oci_params,
        )
    )
    assert len(chunks) > 0
    # Reasoning models may stream only reasoning tokens and return empty content.
    if not m.reasoning:
        content = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
        assert len(content) > 0


@pytest.mark.parametrize("m", CHAT_MODELS)
def test_multi_turn(m: _M, oci_params):
    import litellm

    resp = litellm.completion(
        model=f"oci/{m.model}",
        messages=[
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
            {"role": "user", "content": "What is my name?"},
        ],
        max_tokens=m.max_tokens,
        **oci_params,
    )
    # Reasoning models may have None content; skip text assertion for them.
    content = resp.choices[0].message.content or ""
    if not m.reasoning:
        assert "Alice" in content


# ---------------------------------------------------------------------------
# Async chat tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("m", CHAT_MODELS)
async def test_async_completion(m: _M, oci_params):
    import litellm

    resp = await litellm.acompletion(
        model=f"oci/{m.model}",
        messages=[{"role": "user", "content": "Reply with only the word: pong"}],
        max_tokens=m.max_tokens,
        **oci_params,
    )
    assert resp.choices[0].finish_reason is not None
    assert resp.usage.total_tokens > 0
    if not m.reasoning:
        assert resp.choices[0].message.content is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("m", CHAT_MODELS)
async def test_async_streaming(m: _M, oci_params):
    import litellm

    chunks = []
    async for chunk in await litellm.acompletion(
        model=f"oci/{m.model}",
        messages=[{"role": "user", "content": "Count to 3."}],
        max_tokens=m.max_tokens,
        stream=True,
        **oci_params,
    ):
        chunks.append(chunk)

    assert len(chunks) > 0
    if not m.reasoning:
        content = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
        assert len(content) > 0


# ---------------------------------------------------------------------------
# Tool-use tests
# ---------------------------------------------------------------------------


def _assert_tool_call(resp, expected_tool: str = "get_weather"):
    """Assert the response contains the expected tool call (or a plain stop)."""
    choice = resp.choices[0]
    assert choice.finish_reason in ("tool_calls", "stop")
    if choice.finish_reason == "tool_calls":
        assert choice.message.tool_calls is not None
        assert len(choice.message.tool_calls) > 0
        assert choice.message.tool_calls[0].function.name == expected_tool


@pytest.mark.parametrize("m", TOOL_USE_MODELS)
def test_tool_use(m: _M, oci_params):
    import litellm

    call_kwargs = dict(
        model=f"oci/{m.model}",
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=[_WEATHER_TOOL],
        max_tokens=m.max_tokens,
        **oci_params,
    )
    if m.tool_choice is not None:
        call_kwargs["tool_choice"] = m.tool_choice

    resp = litellm.completion(**call_kwargs)
    _assert_tool_call(resp)


@pytest.mark.asyncio
@pytest.mark.parametrize("m", TOOL_USE_MODELS)
async def test_async_tool_use(m: _M, oci_params):
    import litellm

    call_kwargs = dict(
        model=f"oci/{m.model}",
        messages=[{"role": "user", "content": "What is the weather in Berlin?"}],
        tools=[_WEATHER_TOOL],
        max_tokens=m.max_tokens,
        **oci_params,
    )
    if m.tool_choice is not None:
        call_kwargs["tool_choice"] = m.tool_choice

    resp = await litellm.acompletion(**call_kwargs)
    _assert_tool_call(resp)


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
            mag_a = math.sqrt(sum(x**2 for x in a))
            mag_b = math.sqrt(sum(x**2 for x in b))
            return dot / (mag_a * mag_b)

        cat1 = resp.data[0]["embedding"]
        cat2 = resp.data[1]["embedding"]
        stock = resp.data[2]["embedding"]
        sim_cats = cosine(cat1, cat2)
        sim_diff = cosine(cat1, stock)
        assert (
            sim_cats > sim_diff
        ), f"Expected similar sentences to score higher ({sim_cats:.3f} vs {sim_diff:.3f})"

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
# Env-var credential path
# ---------------------------------------------------------------------------


class TestOCIEnvVarCredentials:
    """Verify that OCI_* env vars are picked up without explicit params."""

    def test_completion_via_env_vars(self, monkeypatch):
        """Completion works when credentials are set through environment variables."""
        oci = pytest.importorskip("oci")
        config = oci.config.from_file(profile_name=OCI_PROFILE)
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
        config = oci.config.from_file(profile_name=OCI_PROFILE)
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
