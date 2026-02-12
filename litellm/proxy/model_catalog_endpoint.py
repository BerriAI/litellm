"""
/model_catalog — Stripe-style API for LiteLLM's model pricing & context window data.

Returns structured, paginated model information sourced from
`litellm/model_prices_and_context_window_backup.json`.

This endpoint does NOT require the proxy to have models configured —
it exposes the full upstream catalog that LiteLLM ships with.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(tags=["model catalog"])

# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class ModelCatalogEntry(BaseModel):
    """A single model in the catalog."""

    id: str = Field(description="Canonical model identifier (e.g. 'gpt-4o')")
    object: str = Field(default="model_catalog.entry")
    provider: Optional[str] = Field(
        default=None, description="LiteLLM provider key (e.g. 'openai', 'azure')"
    )
    mode: Optional[str] = Field(
        default=None,
        description="Model mode: chat, embedding, completion, image_generation, etc.",
    )
    max_input_tokens: Optional[int] = Field(default=None)
    max_output_tokens: Optional[int] = Field(default=None)
    max_tokens: Optional[int] = Field(
        default=None, description="Legacy combined token limit"
    )
    input_cost_per_token: Optional[float] = Field(default=None)
    output_cost_per_token: Optional[float] = Field(default=None)
    cache_read_input_token_cost: Optional[float] = Field(default=None)
    input_cost_per_audio_token: Optional[float] = Field(default=None)
    output_cost_per_reasoning_token: Optional[float] = Field(default=None)
    deprecation_date: Optional[str] = Field(
        default=None, description="ISO date when the model is deprecated (YYYY-MM-DD)"
    )

    # Capability flags
    supports_function_calling: Optional[bool] = Field(default=None)
    supports_parallel_function_calling: Optional[bool] = Field(default=None)
    supports_vision: Optional[bool] = Field(default=None)
    supports_audio_input: Optional[bool] = Field(default=None)
    supports_audio_output: Optional[bool] = Field(default=None)
    supports_prompt_caching: Optional[bool] = Field(default=None)
    supports_reasoning: Optional[bool] = Field(default=None)
    supports_response_schema: Optional[bool] = Field(default=None)
    supports_system_messages: Optional[bool] = Field(default=None)
    supports_web_search: Optional[bool] = Field(default=None)

    model_config = ConfigDict(extra="allow")


class ModelCatalogListResponse(BaseModel):
    """Stripe-style list response."""

    object: str = Field(default="list")
    data: List[ModelCatalogEntry]
    total_count: int = Field(description="Total models matching the filters")
    has_more: bool
    page: int
    page_size: int


# ---------------------------------------------------------------------------
# Data loader (cached in-process)
# ---------------------------------------------------------------------------

_catalog_cache: Optional[Dict[str, Any]] = None


def _load_catalog() -> Dict[str, Any]:
    global _catalog_cache
    if _catalog_cache is not None:
        return _catalog_cache

    json_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "model_prices_and_context_window_backup.json",
    )
    try:
        with open(json_path, "r") as f:
            raw: Dict[str, Any] = json.load(f)
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="model_prices_and_context_window_backup.json not found",
        )

    # Strip the sample_spec key — it's documentation, not a model.
    raw.pop("sample_spec", None)
    _catalog_cache = raw
    return _catalog_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAPPED_FIELDS = {
    "litellm_provider": "provider",
}


def _entry_from_raw(model_id: str, raw: Dict[str, Any]) -> ModelCatalogEntry:
    """Convert a raw JSON entry into a typed ModelCatalogEntry."""
    data: Dict[str, Any] = {"id": model_id}
    for k, v in raw.items():
        mapped_key = _MAPPED_FIELDS.get(k, k)
        data[mapped_key] = v
    return ModelCatalogEntry(**data)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/model_catalog",
    response_model=ModelCatalogListResponse,
    summary="List the full LiteLLM model catalog (Stripe-style)",
    description=(
        "Returns model pricing, context window sizes, and capability metadata "
        "from LiteLLM's built-in catalog. Supports filtering by provider, mode, "
        "model name pattern, and pagination."
    ),
)
async def list_model_catalog(
    provider: Optional[str] = Query(
        default=None,
        description="Filter by provider (e.g. 'openai', 'anthropic', 'bedrock'). Case-insensitive.",
    ),
    mode: Optional[str] = Query(
        default=None,
        description="Filter by mode (e.g. 'chat', 'embedding', 'image_generation').",
    ),
    model: Optional[str] = Query(
        default=None,
        description="Filter by model name. Supports substring match or regex (prefix with 're:').",
    ),
    supports_vision: Optional[bool] = Query(default=None),
    supports_function_calling: Optional[bool] = Query(default=None),
    supports_reasoning: Optional[bool] = Query(default=None),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(
        default=50,
        ge=1,
        le=500,
        description="Number of results per page (max 500)",
    ),
) -> ModelCatalogListResponse:
    catalog = _load_catalog()

    # Precompile regex if needed
    model_regex = None
    if model is not None:
        if model.startswith("re:"):
            try:
                model_regex = re.compile(model[3:], re.IGNORECASE)
            except re.error as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid regex in 'model' param: {e}",
                )
        else:
            model_lower = model.lower()

    filtered: List[ModelCatalogEntry] = []
    for model_id, raw_info in catalog.items():
        # --- provider filter ---
        if provider is not None:
            entry_provider = raw_info.get("litellm_provider", "")
            if entry_provider.lower() != provider.lower():
                continue

        # --- mode filter ---
        if mode is not None:
            if raw_info.get("mode", "").lower() != mode.lower():
                continue

        # --- model name filter ---
        if model is not None:
            if model_regex is not None:
                if not model_regex.search(model_id):
                    continue
            else:
                if model_lower not in model_id.lower():
                    continue

        # --- capability filters ---
        if supports_vision is not None:
            if raw_info.get("supports_vision") != supports_vision:
                continue
        if supports_function_calling is not None:
            if raw_info.get("supports_function_calling") != supports_function_calling:
                continue
        if supports_reasoning is not None:
            if raw_info.get("supports_reasoning") != supports_reasoning:
                continue

        filtered.append(_entry_from_raw(model_id, raw_info))

    total_count = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    page_data = filtered[start:end]
    has_more = end < total_count

    return ModelCatalogListResponse(
        data=page_data,
        total_count=total_count,
        has_more=has_more,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/model_catalog/{model_id:path}",
    response_model=ModelCatalogEntry,
    summary="Get a single model from the catalog",
    description="Returns detailed information for a specific model by its exact ID.",
)
async def get_model_catalog_entry(model_id: str) -> ModelCatalogEntry:
    catalog = _load_catalog()
    raw = catalog.get(model_id)
    if raw is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found in catalog",
        )
    return _entry_from_raw(model_id, raw)
