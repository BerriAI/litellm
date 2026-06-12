"""Injected dependencies: the package's only window onto ambient litellm state.

The translation core is pure; anything that would otherwise read the model
map or a ``litellm`` module global enters here as a value, built once at the
dispatch seam (outside this package) and threaded through ``engine.pipeline``.
That keeps providers unit-testable with hand-built deps and keeps the package
free of upward imports.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class TranslationDeps:
    """Ambient inputs every translation needs, captured as values.

    - ``max_tokens_for_model``: model-map max output tokens, ``None`` when the
      model has no entry (v1: ``get_max_tokens``).
    - ``supports_capability``: boolean capability check against the model map
      with provider-prefix stripping, ``False`` when undeclared (v1:
      ``AnthropicModelInfo._supports_model_capability``).
    - ``capability_flag``: raw tri-state capability flag, ``None`` when the
      model map does not declare it so name-based fallbacks apply (v1:
      ``AnthropicModelInfo._get_model_capability``).
    - ``drop_params``: the effective flag (``litellm.drop_params`` OR the
      request-level kwarg), what v1's ``_apply_sampling_param`` honors.
    - ``drop_params_global``: only ``litellm.drop_params``; v1's stop-sequence
      filter and ``output_config`` drop read the global alone.
    - ``modify_params``: the litellm global, read once at the seam. The seam
      only routes to v2 when it is False; carried so serializers can assert
      the contract.
    - ``api_version``: the resolved azure api-version string (v1:
      ``get_optional_params``' azure branch resolves request kwarg ->
      ``litellm.api_version`` -> env -> ``AZURE_DEFAULT_API_VERSION`` before
      passing it into ``map_openai_params``, utils.py:4865-4875). In v1 the
      version reaching ``map_openai_params`` is ALWAYS a string, so ``None``
      here means the seam never wired the field: the azure gates return a
      typed fallback on it, NOT v1's unparseable-STRING passthrough (which
      needs a real string like ``""``). The default exists for the
      providers that never read the field.
    - ``base_model``: the azure deployment's declared base model; v1 detects
      o-series/gpt-5 param families on ``base_model or model``
      (azure.py:245-247). ``None`` for providers without deployment aliasing.
    """

    max_tokens_for_model: Callable[[str], int | None]
    supports_capability: Callable[[str, str], bool]
    capability_flag: Callable[[str, str], bool | None]
    count_response_tokens: Callable[[str], int]
    drop_params: bool
    drop_params_global: bool
    modify_params: bool
    api_version: str | None = None
    base_model: str | None = None
    watsonx_project_id: str | None = None
    """wave-2b-beta: the resolved watsonx project id. v1 resolves it inside
    ``_get_api_params`` (param kwargs -> WATSONX_PROJECT_ID/WX_PROJECT_ID/
    PROJECT_ID env) and ``_prepare_payload`` injects it into the BODY, so it
    is payload, not envelope; the future watsonx seam fork must run the same
    resolution before building deps. ``None`` together with
    ``watsonx_space_id`` means v1 raises WatsonXAIError 401 — the serializer
    falls back so v1 serves its own raise."""
    watsonx_space_id: str | None = None
    """The deployment-space alternative to ``watsonx_project_id`` (same
    resolution chain, WATSONX_DEPLOYMENT_SPACE_ID/... env); v1 only injects
    it when the project id is absent."""
