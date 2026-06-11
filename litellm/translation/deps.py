"""Injected dependencies: the package's only window onto ambient litellm state.

The translation core is pure; anything that would otherwise read the model
map or a ``litellm`` module global enters here as a value, built once at the
dispatch seam (outside this package) and threaded through ``engine.pipeline``.
That keeps providers unit-testable with hand-built deps and keeps the package
free of upward imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


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
    """

    max_tokens_for_model: Callable[[str], Optional[int]]
    supports_capability: Callable[[str, str], bool]
    capability_flag: Callable[[str, str], Optional[bool]]
    drop_params: bool
    drop_params_global: bool
    modify_params: bool
