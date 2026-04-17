"""
Unit tests for content-aware routing.

All tests are self-contained — no LLM calls, no external services.
"""
import math
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.router_strategy.content_aware_router.utils import (
    bm25_score,
    build_bm25_index,
    cosine_similarity,
    extract_prompt_text,
    stem,
    tokenize,
)


# ---------------------------------------------------------------------------
# Utils tests
# ---------------------------------------------------------------------------


class TestStem:
    def test_ing_suffix(self):
        assert stem("reasoning") == "reason"
        assert stem("debugging") == "debug"   # de-doubles trailing g
        assert stem("implementing") == "implement"

    def test_s_suffix(self):
        assert stem("functions") == "function"
        assert stem("algorithms") == "algorithm"

    def test_ed_suffix(self):
        # "sorted" → strip -ed → "sort"
        assert stem("sorted") == "sort"

    def test_ies_suffix(self):
        assert stem("stories") == "story"
        assert stem("entries") == "entry"

    def test_es_suffix(self):
        assert stem("processes") == "process"
        assert stem("classes") == "class"

    def test_no_change_on_short_word(self):
        assert stem("go") == "go"
        assert stem("code") == "code"

    def test_function_not_over_stemmed(self):
        # "function" must NOT be truncated by a -tion rule to "func"
        assert stem("function") == "function"
        # "computation" must stay intact (no destructive -tion stripping)
        assert stem("computation") == "computation"

    def test_morphological_variants_share_stem(self):
        # reasoning and reason must produce the same stem
        assert stem("reasoning") == stem("reason") == "reason"
        # functions and function must share stem
        assert stem("functions") == stem("function") == "function"


class TestTokenize:
    def test_lowercases(self):
        # "hello" is stem("hello") == "hello"
        assert "hello" in tokenize("Hello World")

    def test_removes_stop_words(self):
        tokens = tokenize("this is a test")
        assert "this" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens
        assert stem("test") in tokens

    def test_removes_punctuation(self):
        tokens = tokenize("code, debugging, and explaining!")
        assert stem("code") in tokens
        assert stem("debugging") in tokens
        assert stem("explaining") in tokens

    def test_empty_string(self):
        assert tokenize("") == []

    def test_single_char_removed(self):
        # Single-character tokens are filtered out
        tokens = tokenize("a b c hello")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "hello" in tokens

    def test_stemming_applied(self):
        # tokenize must apply stemming so inflected and base forms share the same token
        t_reasoning = tokenize("reasoning")
        t_reason = tokenize("reason")
        assert t_reasoning == t_reason, (
            f"'reasoning' tokenised to {t_reasoning} but 'reason' to {t_reason}; "
            "they should share the same stem"
        )


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_similarity(a, b)) < 1e-9

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_empty_returns_zero(self):
        assert cosine_similarity([], []) == 0.0

    def test_length_mismatch_returns_zero(self):
        assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0


class TestBuildBM25Index:
    def test_returns_correct_length(self):
        corpus = ["code debugging programming", "creative writing storytelling"]
        corpus_tokens, idf, avgdl = build_bm25_index(corpus)
        assert len(corpus_tokens) == 2
        assert len(idf) > 0
        assert avgdl > 0

    def test_unique_terms_get_positive_idf(self):
        corpus = ["code programming", "creative writing"]
        _, idf, _ = build_bm25_index(corpus)
        # stem("code") appears only in doc 0 → idf should be positive
        assert idf.get(stem("code"), 0) > 0

    def test_single_document(self):
        corpus = ["hello world testing"]
        corpus_tokens, idf, avgdl = build_bm25_index(corpus)
        assert len(corpus_tokens) == 1

    def test_bm25_score_nonzero_for_matching_term(self):
        corpus = ["code programming python", "storytelling creative fiction"]
        corpus_tokens, idf, avgdl = build_bm25_index(corpus)
        query = tokenize("write python code")
        score_code = bm25_score(query, corpus_tokens[0], idf, avgdl)
        score_story = bm25_score(query, corpus_tokens[1], idf, avgdl)
        assert score_code > score_story, (
            f"code description ({score_code:.4f}) should score higher than "
            f"story description ({score_story:.4f}) for a code prompt"
        )

    def test_bm25_score_zero_for_no_overlap(self):
        corpus = ["apple orange mango"]
        corpus_tokens, idf, avgdl = build_bm25_index(corpus)
        query = tokenize("quantum physics entanglement")
        assert bm25_score(query, corpus_tokens[0], idf, avgdl) == 0.0

    def test_stemmed_variants_match(self):
        # 'reasoning' in description should match 'reason' in query after stemming
        corpus = ["analyze explain reasoning logic mathematical math proof theorem equation"]
        corpus_tokens, idf, avgdl = build_bm25_index(corpus)
        query_base = tokenize("reason")
        query_inflected = tokenize("reasoning")
        # Both queries must produce identical scores (same stem)
        score_base = bm25_score(query_base, corpus_tokens[0], idf, avgdl)
        score_inflected = bm25_score(query_inflected, corpus_tokens[0], idf, avgdl)
        assert score_base == score_inflected, (
            f"'reason' ({score_base:.4f}) and 'reasoning' ({score_inflected:.4f}) "
            "should produce the same BM25 score after stemming"
        )
        assert score_base > 0.0


class TestExtractPromptText:
    def test_extracts_last_user_message(self):
        messages = [
            {"role": "user", "content": "first message"},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": "last user message"},
        ]
        user_text, system_text = extract_prompt_text(messages)
        assert user_text == "last user message"
        assert system_text is None

    def test_extracts_system_prompt(self):
        messages = [
            {"role": "system", "content": "You are a coding assistant"},
            {"role": "user", "content": "write a function"},
        ]
        user_text, system_text = extract_prompt_text(messages)
        assert user_text == "write a function"
        assert system_text == "You are a coding assistant"

    def test_handles_content_parts(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "write python code"}],
            }
        ]
        user_text, _ = extract_prompt_text(messages)
        assert user_text == "write python code"

    def test_empty_messages_returns_empty(self):
        user_text, system_text = extract_prompt_text([])
        assert user_text == ""
        assert system_text is None

    def test_none_returns_empty(self):
        user_text, system_text = extract_prompt_text(None)
        assert user_text == ""
        assert system_text is None


# ---------------------------------------------------------------------------
# ContentAwareRouter tests
# ---------------------------------------------------------------------------


def _make_router(preferences_by_model, classifier="rule_based", default_model="gpt-4o", threshold=0.01):
    """Create a ContentAwareRouter with minimal mocking."""
    from litellm.router_strategy.content_aware_router.content_aware_router import (
        ContentAwareRouter,
    )
    from litellm.types.router import ContentRoutingConfig, RoutingPreference

    prefs = {
        model: [RoutingPreference(**p) for p in pref_list]
        for model, pref_list in preferences_by_model.items()
    }
    config = ContentRoutingConfig(
        enabled=True,
        classifier=classifier,
        default_model=default_model,
        confidence_threshold=threshold,
    )
    mock_router = MagicMock()
    return ContentAwareRouter(
        preferences_by_model=prefs,
        config=config,
        litellm_router_instance=mock_router,
    )


class TestRuleBasedClassifier:
    def test_routes_code_prompt(self):
        """Code-heavy prompts should route to the code model."""
        router = _make_router(
            {
                "gpt-4o": [
                    {"name": "creative_writing", "description": "creative storytelling narrative fiction writing"},
                ],
                "claude-sonnet": [
                    {"name": "code_generation", "description": "code programming debugging python javascript function"},
                ],
            },
            threshold=0.0,
        )
        model, pref, score = router._classify_rule_based(
            "write a python function to sort a list", None
        )
        assert model == "claude-sonnet"
        assert pref == "code_generation"
        assert score > 0

    def test_routes_creative_prompt(self):
        """Creative prompts should route to the creative writing model."""
        router = _make_router(
            {
                "gpt-4o": [
                    {"name": "creative_writing", "description": "creative storytelling narrative fiction writing poetry"},
                ],
                "claude-sonnet": [
                    {"name": "code_generation", "description": "code programming debugging function"},
                ],
            },
            threshold=0.0,
        )
        model, pref, score = router._classify_rule_based(
            "write a short story about a lonely astronaut", None
        )
        assert model == "gpt-4o"
        assert pref == "creative_writing"

    def test_below_threshold_uses_default(self):
        """When confidence is below threshold, default_model is returned."""
        from litellm.router_strategy.content_aware_router.content_aware_router import (
            ContentAwareRouter,
        )
        from litellm.types.router import ContentRoutingConfig, RoutingPreference

        prefs = {
            "claude-sonnet": [
                RoutingPreference(name="code_generation", description="very specific coding words only"),
            ],
        }
        # Set threshold absurdly high so nothing matches
        config = ContentRoutingConfig(
            enabled=True,
            classifier="rule_based",
            default_model="gpt-4o",
            confidence_threshold=999.0,
        )
        mock_router = MagicMock()
        car = ContentAwareRouter(prefs, config, mock_router)

        import asyncio

        messages = [{"role": "user", "content": "hello how are you"}]
        result = asyncio.get_event_loop().run_until_complete(
            car.async_pre_routing_hook(
                model="any",
                request_kwargs={},
                messages=messages,
            )
        )
        assert result is not None
        assert result.model == "gpt-4o"

    def test_no_preferences_returns_none(self):
        """Router with no routing_preferences should return None."""
        from litellm.router_strategy.content_aware_router.content_aware_router import (
            ContentAwareRouter,
        )
        from litellm.types.router import ContentRoutingConfig

        config = ContentRoutingConfig(enabled=True, classifier="rule_based")
        mock_router = MagicMock()
        car = ContentAwareRouter({}, config, mock_router)

        import asyncio

        messages = [{"role": "user", "content": "write a python function"}]
        result = asyncio.get_event_loop().run_until_complete(
            car.async_pre_routing_hook(
                model="any",
                request_kwargs={},
                messages=messages,
            )
        )
        assert result is None

    def test_no_user_message_returns_none(self):
        """Missing user message should return None."""
        router = _make_router(
            {"gpt-4o": [{"name": "general", "description": "general conversation chat"}]},
            threshold=0.0,
        )
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            router.async_pre_routing_hook(
                model="any",
                request_kwargs={},
                messages=[{"role": "system", "content": "system only"}],
            )
        )
        assert result is None

    def test_routing_decision_stored_in_metadata(self):
        """Classification decision should be written into request_kwargs metadata."""
        router = _make_router(
            {
                "claude-sonnet": [
                    {"name": "code_generation", "description": "code programming debugging function python"},
                ],
            },
            threshold=0.0,
        )
        import asyncio

        request_kwargs: dict = {}
        messages = [{"role": "user", "content": "write a python function to sort a list"}]
        asyncio.get_event_loop().run_until_complete(
            router.async_pre_routing_hook(
                model="any",
                request_kwargs=request_kwargs,
                messages=messages,
            )
        )
        decision = request_kwargs.get("metadata", {}).get("content_routing_decision")
        assert decision is not None
        assert decision["model"] == "claude-sonnet"
        assert decision["matched_preference"] == "code_generation"
        assert decision["classifier"] == "rule_based"


class TestEmbeddingSimilarityClassifier:
    @pytest.mark.asyncio
    async def test_embedding_similarity_uses_cosine(self):
        """Embedding classifier should pick the model with highest cosine similarity."""
        from litellm.router_strategy.content_aware_router.content_aware_router import (
            ContentAwareRouter,
        )
        from litellm.types.router import ContentRoutingConfig, RoutingPreference

        prefs = {
            "gpt-4o": [
                RoutingPreference(name="creative_writing", description="storytelling"),
            ],
            "claude-sonnet": [
                RoutingPreference(name="code_generation", description="code programming"),
            ],
        }
        config = ContentRoutingConfig(
            enabled=True,
            classifier="embedding_similarity",
            default_model="gpt-4o",
            confidence_threshold=0.0,
        )
        mock_router = MagicMock()
        car = ContentAwareRouter(prefs, config, mock_router)

        # Pre-populate description embeddings (2 descriptions)
        # Code description embedding points towards [1, 0], creative towards [0, 1]
        car._description_embeddings = [
            [0.0, 1.0],  # gpt-4o / creative_writing
            [1.0, 0.0],  # claude-sonnet / code_generation
        ]

        # Prompt embedding pointing towards code
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [{"embedding": [0.9, 0.1]}]

        with patch("litellm.aembedding", new=AsyncMock(return_value=mock_embedding_response)):
            model, pref, score = await car._classify_embedding_similarity(
                "write python code", None
            )

        assert model == "claude-sonnet"
        assert pref == "code_generation"
        assert score > 0.5

    @pytest.mark.asyncio
    async def test_embedding_fallback_on_error(self):
        """Embedding failures should fall back to rule_based without raising."""
        from litellm.router_strategy.content_aware_router.content_aware_router import (
            ContentAwareRouter,
        )
        from litellm.types.router import ContentRoutingConfig, RoutingPreference

        prefs = {
            "claude-sonnet": [
                RoutingPreference(name="code_generation", description="code programming python function"),
            ],
        }
        config = ContentRoutingConfig(
            enabled=True,
            classifier="embedding_similarity",
            default_model="gpt-4o",
            confidence_threshold=0.0,
            embedding_model="text-embedding-3-small",
        )
        mock_router = MagicMock()
        car = ContentAwareRouter(prefs, config, mock_router)

        with patch("litellm.aembedding", new=AsyncMock(side_effect=Exception("API error"))):
            # Should not raise; falls back to rule_based
            model, pref, score = await car._classify_embedding_similarity(
                "write a python function", None
            )
        # Fell back to rule_based — claude-sonnet should win for code prompt
        assert model == "claude-sonnet"


class TestExternalModelClassifier:
    @pytest.mark.asyncio
    async def test_external_model_parses_response(self):
        """External classifier should parse JSON response correctly."""
        from litellm.router_strategy.content_aware_router.content_aware_router import (
            ContentAwareRouter,
        )
        from litellm.types.router import ContentRoutingConfig, RoutingPreference

        prefs = {
            "claude-sonnet": [
                RoutingPreference(name="code_generation", description="code programming"),
            ],
        }
        config = ContentRoutingConfig(
            enabled=True,
            classifier="external_model",
            default_model="gpt-4o",
            confidence_threshold=0.0,
            external_classifier_url="http://arch-router/classify",
        )
        mock_router = MagicMock()
        car = ContentAwareRouter(prefs, config, mock_router)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "matched_preference": "code_generation",
            "model": "claude-sonnet",
            "confidence": 0.95,
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            model, pref, confidence = await car._classify_external_model(
                "write python code", None
            )

        assert model == "claude-sonnet"
        assert pref == "code_generation"
        assert abs(confidence - 0.95) < 1e-9

    @pytest.mark.asyncio
    async def test_external_model_fallback_on_error(self):
        """HTTP errors should fall back to rule_based."""
        from litellm.router_strategy.content_aware_router.content_aware_router import (
            ContentAwareRouter,
        )
        from litellm.types.router import ContentRoutingConfig, RoutingPreference

        prefs = {
            "claude-sonnet": [
                RoutingPreference(name="code_generation", description="code programming python function"),
            ],
        }
        config = ContentRoutingConfig(
            enabled=True,
            classifier="external_model",
            default_model="gpt-4o",
            confidence_threshold=0.0,
            external_classifier_url="http://arch-router/classify",
        )
        mock_router = MagicMock()
        car = ContentAwareRouter(prefs, config, mock_router)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=Exception("connection refused"))

        with patch("httpx.AsyncClient", return_value=mock_client):
            model, pref, score = await car._classify_external_model(
                "write a python function", None
            )

        # Fell back to rule_based
        assert model == "claude-sonnet"

    @pytest.mark.asyncio
    async def test_external_model_no_url_falls_back(self):
        """Missing URL should fall back to rule_based."""
        from litellm.router_strategy.content_aware_router.content_aware_router import (
            ContentAwareRouter,
        )
        from litellm.types.router import ContentRoutingConfig, RoutingPreference

        prefs = {
            "claude-sonnet": [
                RoutingPreference(name="code_generation", description="code programming python function"),
            ],
        }
        config = ContentRoutingConfig(
            enabled=True,
            classifier="external_model",
            default_model="gpt-4o",
            confidence_threshold=0.0,
            external_classifier_url=None,  # no URL set
        )
        mock_router = MagicMock()
        car = ContentAwareRouter(prefs, config, mock_router)

        model, pref, score = await car._classify_external_model("write python code", None)
        assert model == "claude-sonnet"


class TestRouterIntegration:
    def test_router_init_with_content_routing(self):
        """Router should initialize ContentAwareRouter when content_routing is set."""
        import litellm

        router = litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-4o",
                    "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"},
                    "routing_preferences": [
                        {"name": "creative_writing", "description": "creative storytelling fiction"},
                    ],
                },
                {
                    "model_name": "claude-sonnet",
                    "litellm_params": {"model": "anthropic/claude-sonnet-4-20250514", "api_key": "fake"},
                    "routing_preferences": [
                        {"name": "code_generation", "description": "code programming debugging"},
                    ],
                },
            ],
            content_routing={
                "enabled": True,
                "classifier": "rule_based",
                "default_model": "gpt-4o",
                "confidence_threshold": 0.0,
            },
        )

        assert router.content_aware_router is not None
        assert len(router.content_aware_router._index) == 2
        assert router._content_routing_config is not None
        assert router._content_routing_config.enabled is True

    def test_router_disabled_content_routing(self):
        """Router should not set content_aware_router when enabled=False."""
        import litellm

        router = litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-4o",
                    "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"},
                },
            ],
            content_routing={"enabled": False},
        )
        assert router.content_aware_router is None

    def test_router_no_content_routing(self):
        """Router without content_routing param should have None content_aware_router."""
        import litellm

        router = litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-4o",
                    "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"},
                },
            ],
        )
        assert router.content_aware_router is None
