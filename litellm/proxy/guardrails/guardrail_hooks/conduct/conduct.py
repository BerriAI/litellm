"""Conduct Guard as a LiteLLM guardrail, backed by the ``conduct-litellm-guard`` PyPI package.

Install: ``pip install "conduct-litellm-guard>=0.2.4"``
Source:  https://github.com/sseshachala/conductai/tree/main/packages/conduct-litellm-guard
"""

from __future__ import annotations

from typing import Final

from litellm.integrations.custom_guardrail import CustomGuardrail

MISSING_PACKAGE_MESSAGE: Final = (
    "conduct-litellm-guard is required for the Conduct guardrail. "
    'Install it with: pip install "conduct-litellm-guard>=0.2.4"'
)

try:
    from conduct_litellm_guard import ConductGuard as ConductGuardrail
except ImportError as import_error:
    _import_error: Final = import_error

    class ConductGuardrail(CustomGuardrail):
        def __init__(self, **kwargs: object) -> None:  # kwargs-ok: mirrors the plugin constructor, only raises
            raise ImportError(MISSING_PACKAGE_MESSAGE) from _import_error


__all__ = ["MISSING_PACKAGE_MESSAGE", "ConductGuardrail"]  # mutable-ok: standard Python re-export list
