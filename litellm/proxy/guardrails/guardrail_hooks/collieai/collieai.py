"""CollieAi as a LiteLLM guardrail, backed by the ``collieai-litellm-guardrail`` PyPI package.

Install: ``pip install "collieai-litellm-guardrail>=1.0.0"``
Source:  https://github.com/CollieAi/collieai-litellm-guardrail
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Final

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.collieai import CollieAiGuardrailConfigModel

PACKAGE_MODULE: Final = "collieai_guardrail.guardrail"
MISSING_PACKAGE_MESSAGE: Final = (
    "collieai-litellm-guardrail>=1.0.0 (Python 3.11 or newer) is required for the CollieAi guardrail. "
    'Install it with: pip install "collieai-litellm-guardrail>=1.0.0". '
    "The LiteLLM Docker image does not include it, "
    "see https://docs.collieai.io/integrations/litellm to add it to an image."
)


def has_readiness_guard(package: object) -> bool:
    return getattr(package, "require_guardrail", None) is not None


def package_guardrail(package: object) -> type[CustomGuardrail] | None:
    guardrail: Final[object] = getattr(package, "CollieAIGuardrail", None)
    if not has_readiness_guard(package) or not isinstance(guardrail, type):
        return None
    return guardrail if issubclass(guardrail, CustomGuardrail) else None


def unavailable_package(message: str, cause: BaseException | None) -> type[CustomGuardrail]:
    class UnavailablePackage(CustomGuardrail):
        def __init__(self, **kwargs: object) -> None:  # kwargs-ok: mirrors the package constructor, only raises
            raise ImportError(message) from cause

    return UnavailablePackage


def missing_package(import_error: ImportError | None) -> type[CustomGuardrail]:
    return unavailable_package(MISSING_PACKAGE_MESSAGE, import_error)


def broken_package(load_error: Exception) -> type[CustomGuardrail]:
    return unavailable_package(
        "collieai-litellm-guardrail is installed but failed to load with this LiteLLM version "
        f"({type(load_error).__name__}: {load_error}). "
        'Upgrade it with: pip install -U "collieai-litellm-guardrail>=1.0.0"',
        load_error,
    )


def is_the_package_missing(import_error: ImportError) -> bool:
    package_root: Final = PACKAGE_MODULE.partition(".")[0]
    return isinstance(import_error, ModuleNotFoundError) and (import_error.name or "").partition(".")[0] == package_root


def load_base(import_module: Callable[[str], object] = importlib.import_module) -> type[CustomGuardrail]:
    try:
        package: Final = import_module(PACKAGE_MODULE)
    except ImportError as import_error:
        return missing_package(import_error) if is_the_package_missing(import_error) else broken_package(import_error)
    except Exception as load_error:  # noqa: BLE001  # a broken third-party install must not break the guardrail routes
        return broken_package(load_error)
    return package_guardrail(package) or missing_package(None)


_Base: Final = load_base()


class CollieAiGuardrail(_Base):
    @staticmethod
    def get_config_model() -> type[CollieAiGuardrailConfigModel]:
        return CollieAiGuardrailConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:  # mutable-ok: CustomGuardrail's signature
        return [GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call]


__all__ = (
    "MISSING_PACKAGE_MESSAGE",
    "PACKAGE_MODULE",
    "CollieAiGuardrail",
    "has_readiness_guard",
    "load_base",
    "missing_package",
    "package_guardrail",
)
