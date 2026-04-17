"""
CONTENT-AWARE ROUTING ENDPOINTS

POST /utils/content_route_test  - Dry-run: classify a prompt without making an LLM call
GET  /router/content_routing/preferences - List all models with their routing_preferences
"""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


class ContentRouteTestRequest(BaseModel):
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None


class ContentRouteTestResponse(BaseModel):
    matched_preference: str
    matched_model: str
    confidence: float
    classifier: str
    all_scores: Optional[Dict[str, float]] = None


class ModelRoutingPreferences(BaseModel):
    model_name: str
    routing_preferences: List[Dict[str, str]]


class ContentRoutingPreferencesResponse(BaseModel):
    models: List[ModelRoutingPreferences]
    content_routing_enabled: bool
    classifier: Optional[str] = None
    default_model: Optional[str] = None


@router.post(
    "/utils/content_route_test",
    tags=["router"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ContentRouteTestResponse,
    summary="Test content-aware routing for a prompt without making an LLM call",
)
async def content_route_test(
    request: ContentRouteTestRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> ContentRouteTestResponse:
    """
    Classify a prompt against the configured routing_preferences and return the
    routing decision without actually calling any LLM.

    Useful for validating routing_preferences config and debugging routing decisions.
    """
    from litellm.proxy.proxy_server import llm_router
    from litellm.router_strategy.content_aware_router.utils import (
        build_tfidf_vectors,
        extract_prompt_text,
        tfidf_score,
        tokenize,
    )

    if llm_router is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Router not initialized")

    content_aware_router = llm_router.content_aware_router
    if content_aware_router is None:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail="Content-aware routing is not enabled. Set router_settings.content_routing.enabled=true in your config.",
        )

    # Build messages from prompt if needed
    messages: Optional[List[Dict[str, Any]]] = request.messages
    if not messages and request.prompt:
        messages = [{"role": "user", "content": request.prompt}]

    if not messages:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400, detail="Either 'prompt' or 'messages' must be provided"
        )

    user_text, system_text = extract_prompt_text(messages)
    if not user_text:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail="No user message content found")

    classifier = content_aware_router.config.classifier

    # Run classification and collect all scores for transparency
    all_scores: Dict[str, float] = {}

    if classifier == "rule_based":
        prompt_tokens = tokenize(f"{system_text or ''} {user_text}".strip())
        for i, (model_name, pref) in enumerate(content_aware_router._index):
            score = tfidf_score(
                prompt_tokens,
                content_aware_router._idf_weights,
                content_aware_router._tfidf_vectors[i],
            )
            all_scores[f"{model_name}/{pref.name}"] = round(score, 4)
        matched_model, matched_pref, confidence = (
            content_aware_router._classify_rule_based(user_text, system_text)
        )
    elif classifier == "embedding_similarity":
        matched_model, matched_pref, confidence = (
            await content_aware_router._classify_embedding_similarity(
                user_text, system_text
            )
        )
    else:  # external_model
        matched_model, matched_pref, confidence = (
            await content_aware_router._classify_external_model(user_text, system_text)
        )

    verbose_proxy_logger.info(
        f"content_route_test: classifier={classifier} matched={matched_pref} "
        f"model={matched_model} confidence={confidence:.4f}"
    )

    return ContentRouteTestResponse(
        matched_preference=matched_pref,
        matched_model=matched_model,
        confidence=round(confidence, 4),
        classifier=classifier,
        all_scores=all_scores if all_scores else None,
    )


@router.get(
    "/router/content_routing/preferences",
    tags=["router"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ContentRoutingPreferencesResponse,
    summary="List all models with their configured routing_preferences",
)
async def get_content_routing_preferences(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> ContentRoutingPreferencesResponse:
    """
    Returns all models in the router that have routing_preferences configured,
    along with the current content routing settings.
    """
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Router not initialized")

    models: List[ModelRoutingPreferences] = []
    for deployment in llm_router.model_list or []:
        if isinstance(deployment, dict):
            prefs = deployment.get("routing_preferences")
            model_name = deployment.get("model_name")
        else:
            prefs = getattr(deployment, "routing_preferences", None)
            model_name = getattr(deployment, "model_name", None)

        if prefs and model_name:
            if isinstance(prefs, list):
                pref_dicts = [
                    p if isinstance(p, dict) else p.model_dump() for p in prefs
                ]
            else:
                pref_dicts = []
            models.append(
                ModelRoutingPreferences(
                    model_name=model_name,
                    routing_preferences=pref_dicts,
                )
            )

    config = llm_router._content_routing_config
    return ContentRoutingPreferencesResponse(
        models=models,
        content_routing_enabled=config.enabled if config else False,
        classifier=config.classifier if config else None,
        default_model=config.default_model if config else None,
    )
