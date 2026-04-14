"""
Content-Aware Preference-Aligned Router

Routes incoming requests to the best-matching model by classifying the prompt
content against per-model routing_preference descriptions configured in the
LiteLLM YAML config.

Supports three classifiers:
  - rule_based: TF-IDF cosine similarity, zero latency, no external deps
  - embedding_similarity: uses litellm.aembedding() for semantic matching
  - external_model: delegates to an external HTTP classification endpoint
"""
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx

from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger

from .utils import (
    bm25_score,
    build_bm25_index,
    cosine_similarity,
    extract_prompt_text,
    tokenize,
)

if TYPE_CHECKING:
    from litellm.router import Router
    from litellm.types.router import (
        ContentRoutingConfig,
        PreRoutingHookResponse,
        RoutingPreference,
    )
else:
    Router = Any
    ContentRoutingConfig = Any
    PreRoutingHookResponse = Any
    RoutingPreference = Any


class ContentAwareRouter(CustomLogger):
    """
    Pre-routing hook that classifies prompt content and selects the best-matched
    model based on routing_preferences declared in the model_list config.

    Instantiated by the Router when router_settings.content_routing.enabled = true.
    """

    def __init__(
        self,
        preferences_by_model: Dict[str, List["RoutingPreference"]],
        config: "ContentRoutingConfig",
        litellm_router_instance: "Router",
    ):
        """
        Args:
            preferences_by_model: {model_name: [RoutingPreference, ...]} for all
                                   deployments that have routing_preferences set.
            config: ContentRoutingConfig parsed from router_settings.content_routing.
            litellm_router_instance: The Router instance (used for embedding calls).
        """
        self.preferences_by_model = preferences_by_model
        self.config = config
        self.litellm_router_instance = litellm_router_instance

        # Flat ordered list of (model_name, preference) for index alignment
        self._index: List[Tuple[str, "RoutingPreference"]] = [
            (model, pref)
            for model, prefs in preferences_by_model.items()
            for pref in prefs
        ]

        # Rule-based: BM25 index
        self._bm25_corpus: List[List[str]] = []   # stemmed tokens per preference
        self._bm25_idf: Dict[str, float] = {}
        self._bm25_avgdl: float = 0.0

        # Embedding-similarity: pre-computed description embeddings
        self._description_embeddings: List[List[float]] = []

        if self._index:
            # Always build the rule_based index — it is cheap and serves as a
            # fallback when embedding or external classifiers fail at runtime.
            self._build_rule_based_index()
            # Embedding vectors are built lazily on first request to allow async init

        verbose_router_logger.info(
            f"ContentAwareRouter initialized with {len(self._index)} preferences "
            f"across {len(preferences_by_model)} models, classifier={config.classifier}"
        )

    # ------------------------------------------------------------------
    # Index builders
    # ------------------------------------------------------------------

    def _build_rule_based_index(self) -> None:
        """Pre-compute BM25 index for all preference descriptions."""
        descriptions = [pref.description for _, pref in self._index]
        self._bm25_corpus, self._bm25_idf, self._bm25_avgdl = build_bm25_index(
            descriptions
        )

    async def _ensure_embedding_index(self) -> None:
        """Build embedding vectors for all preference descriptions (once)."""
        if self._description_embeddings:
            return  # already built

        import litellm

        embedding_model = self.config.embedding_model or "text-embedding-3-small"
        descriptions = [pref.description for _, pref in self._index]

        try:
            response = await litellm.aembedding(
                model=embedding_model,
                input=descriptions,
            )
            self._description_embeddings = [
                item["embedding"] for item in response.data
            ]
            verbose_router_logger.debug(
                f"ContentAwareRouter: built {len(self._description_embeddings)} "
                f"description embeddings with {embedding_model}"
            )
        except Exception as e:
            verbose_router_logger.warning(
                f"ContentAwareRouter: failed to build embedding index: {e}. "
                "Falling back to rule_based classifier."
            )
            self._build_rule_based_index()

    # ------------------------------------------------------------------
    # Classifiers — all return (model_name, preference_name, confidence)
    # ------------------------------------------------------------------

    def _classify_rule_based(
        self, user_text: str, system_text: Optional[str]
    ) -> Tuple[str, str, float]:
        """BM25 scoring against each preference description."""
        # Use full text for scoring (system prompt provides deployment context)
        combined = f"{system_text or ''} {user_text}".strip()
        prompt_tokens = tokenize(combined)

        best_score = -1.0
        best_model = self.config.default_model or ""
        best_pref = ""

        for i, (model_name, pref) in enumerate(self._index):
            score = bm25_score(
                prompt_tokens, self._bm25_corpus[i], self._bm25_idf, self._bm25_avgdl
            )
            verbose_router_logger.debug(
                f"ContentAwareRouter rule_based: model={model_name} "
                f"pref={pref.name} score={score:.4f}"
            )
            if score > best_score:
                best_score = score
                best_model = model_name
                best_pref = pref.name

        return best_model, best_pref, best_score

    async def _classify_embedding_similarity(
        self, user_text: str, system_text: Optional[str]
    ) -> Tuple[str, str, float]:
        """Embed the prompt and find the nearest preference description."""
        await self._ensure_embedding_index()

        # If embedding index failed, fall back to rule-based
        if not self._description_embeddings:
            return self._classify_rule_based(user_text, system_text)

        import litellm

        embedding_model = self.config.embedding_model or "text-embedding-3-small"
        try:
            response = await litellm.aembedding(
                model=embedding_model,
                input=[user_text],
            )
            prompt_embedding: List[float] = response.data[0]["embedding"]
        except Exception as e:
            verbose_router_logger.warning(
                f"ContentAwareRouter: embedding call failed ({e}), "
                "falling back to rule_based"
            )
            return self._classify_rule_based(user_text, system_text)

        best_score = -1.0
        best_model = self.config.default_model or ""
        best_pref = ""

        for i, (model_name, pref) in enumerate(self._index):
            score = cosine_similarity(prompt_embedding, self._description_embeddings[i])
            verbose_router_logger.debug(
                f"ContentAwareRouter embedding: model={model_name} "
                f"pref={pref.name} score={score:.4f}"
            )
            if score > best_score:
                best_score = score
                best_model = model_name
                best_pref = pref.name

        return best_model, best_pref, best_score

    async def _classify_external_model(
        self, user_text: str, system_text: Optional[str]
    ) -> Tuple[str, str, float]:
        """
        POST prompt to an external classifier endpoint.

        Expected response JSON:
            {"matched_preference": "code_generation", "model": "claude-sonnet", "confidence": 0.92}
        """
        url = self.config.external_classifier_url
        if not url:
            verbose_router_logger.warning(
                "ContentAwareRouter: external_classifier_url not set, "
                "falling back to rule_based"
            )
            return self._classify_rule_based(user_text, system_text)

        payload = {
            "prompt": user_text,
            "system_prompt": system_text,
            "preferences": [
                {"model": m, "name": p.name, "description": p.description}
                for m, p in self._index
            ],
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
            model_name = data.get("model", self.config.default_model or "")
            pref_name = data.get("matched_preference", "")
            confidence = float(data.get("confidence", 0.0))
            return model_name, pref_name, confidence
        except Exception as e:
            verbose_router_logger.warning(
                f"ContentAwareRouter: external classifier call failed ({e}), "
                "falling back to rule_based"
            )
            return self._classify_rule_based(user_text, system_text)

    # ------------------------------------------------------------------
    # Public pre-routing hook
    # ------------------------------------------------------------------

    async def async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: Dict,
        messages: Optional[List[Dict[str, Any]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
    ) -> Optional["PreRoutingHookResponse"]:
        """
        Called by Router.async_pre_routing_hook() before infrastructure routing.

        Classifies the prompt content and returns the best-matched model.
        Returns None when content routing should be skipped (no preferences, etc.).
        """
        from litellm.types.router import PreRoutingHookResponse

        if not self._index:
            return None

        user_text, system_text = extract_prompt_text(messages)
        if not user_text:
            verbose_router_logger.debug(
                "ContentAwareRouter: no user message found, skipping"
            )
            return None

        classifier = self.config.classifier

        if classifier == "rule_based":
            matched_model, matched_pref, confidence = self._classify_rule_based(
                user_text, system_text
            )
        elif classifier == "embedding_similarity":
            matched_model, matched_pref, confidence = (
                await self._classify_embedding_similarity(user_text, system_text)
            )
        else:  # external_model
            matched_model, matched_pref, confidence = (
                await self._classify_external_model(user_text, system_text)
            )

        threshold = self.config.confidence_threshold
        if confidence < threshold:
            fallback = self.config.default_model
            verbose_router_logger.info(
                f"ContentAwareRouter: confidence {confidence:.4f} below threshold "
                f"{threshold}, using default_model={fallback}"
            )
            if not fallback:
                return None
            matched_model = fallback
            matched_pref = "default"

        verbose_router_logger.info(
            f"ContentAwareRouter: classifier={classifier} "
            f"matched_preference={matched_pref} model={matched_model} "
            f"confidence={confidence:.4f}"
        )

        # Store decision in metadata for response header propagation
        metadata = request_kwargs.setdefault("metadata", {})
        metadata["content_routing_decision"] = {
            "matched_preference": matched_pref,
            "model": matched_model,
            "confidence": confidence,
            "classifier": classifier,
        }

        return PreRoutingHookResponse(
            model=matched_model,
            messages=messages,
        )
