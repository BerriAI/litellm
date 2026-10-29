"""CollieAi as a LiteLLM guardrail, backed by the ``collieai-litellm-guardrail`` PyPI package.

Install: ``pip install "collieai-litellm-guardrail>=1.0.0"``
Source:  https://github.com/CollieAi/collieai-litellm-guardrail
"""

from __future__ import annotations

import importlib
from typing import Final

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.proxy.guardrails.guardrail_hooks.collieai import CollieAiGuardrailConfigModel

PACKAGE_MODULE: Final = "collieai_guardrail.guardrail"
MISSING_PACKAGE_MESSAGE: Final = (
    "collieai-litellm-guardrail>=1.0.0 is required for the CollieAi guardrail. "
    'Install it with: pip install "collieai-litellm-guardrail>=1.0.0"'
)


def has_readiness_guard(package: object) -> bool:
    """1.0.0 is the first release with the readiness guard, `require_guardrail`, that the package's block mode needs"""
    return getattr(package, "require_guardrail", None) is not None


def package_guardrail(package: object) -> type[CustomGuardrail] | None:
    """The package's guardrail class, or None for a package too old or not shaped as one"""
    guardrail: Final[object] = getattr(package, "CollieAIGuardrail", None)
    if not has_readiness_guard(package) or not isinstance(guardrail, type):
        return None
    return guardrail if issubclass(guardrail, CustomGuardrail) else None


def missing_package(import_error: ImportError | None) -> type[CustomGuardrail]:
    """A stand-in that fails where the proxy builds the guardrail, naming what to install"""

    class MissingPackage(CustomGuardrail):
        def __init__(self, **kwargs: object) -> None:  # kwargs-ok: mirrors the package constructor, only raises
            raise ImportError(MISSING_PACKAGE_MESSAGE) from import_error

    return MissingPackage


def load_base() -> type[CustomGuardrail]:
    try:
        package: Final = importlib.import_module(PACKAGE_MODULE)
    except ImportError as import_error:
        return missing_package(import_error)
    return package_guardrail(package) or missing_package(None)


_Base: Final = load_base()


class CollieAiGuardrail(_Base):
    @staticmethod
    def get_config_model() -> type[CollieAiGuardrailConfigModel]:
        return CollieAiGuardrailConfigModel


__all__ = (
    "MISSING_PACKAGE_MESSAGE",
    "PACKAGE_MODULE",
    "CollieAiGuardrail",
    "has_readiness_guard",
    "load_base",
    "missing_package",
    "package_guardrail",
)
