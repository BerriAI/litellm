"""Shared fixtures for translation v2 tests.

``real_deps`` builds ``TranslationDeps`` from the same litellm helpers v1
resolves at runtime (model map lookups, capability flags), so differential
tests exercise identical ambient inputs on both sides.
"""

import os
from typing import Optional

import pytest

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from litellm.llms.anthropic.common_utils import AnthropicModelInfo  # noqa: E402
from litellm.utils import get_max_tokens  # noqa: E402

from litellm.translation import TranslationDeps  # noqa: E402


def _max_tokens_for_model(model: str) -> Optional[int]:
    try:
        return get_max_tokens(model)
    except Exception:
        return None


def build_real_deps(
    drop_params: bool = False,
    drop_params_global: bool = False,
    modify_params: bool = False,
) -> TranslationDeps:
    return TranslationDeps(
        max_tokens_for_model=_max_tokens_for_model,
        supports_capability=AnthropicModelInfo._supports_model_capability,
        capability_flag=AnthropicModelInfo._get_model_capability,
        drop_params=drop_params,
        drop_params_global=drop_params_global,
        modify_params=modify_params,
    )


@pytest.fixture()
def real_deps() -> TranslationDeps:
    return build_real_deps()
