"""Shared fixtures for translation v2 tests.

``real_deps`` builds ``TranslationDeps`` from the same litellm helpers v1
resolves at runtime (model map lookups, capability flags), so differential
tests exercise identical ambient inputs on both sides.
"""

import itertools
import os
from typing import Optional

import pytest

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
# Hermetic AWS/anthropic ambient for the bedrock differential gates (mirrors
# the characterization corpus conftest; nothing performs network I/O).
os.environ.setdefault("AWS_ACCESS_KEY_ID", "AKIADIFFTESTKEY00000")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "diff-test-secret")
os.environ.setdefault("AWS_REGION_NAME", "us-east-1")

from litellm.llms.anthropic.common_utils import AnthropicModelInfo  # noqa: E402
from litellm.utils import get_max_tokens, token_counter  # noqa: E402

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
        count_response_tokens=lambda text: token_counter(
            text=text, count_response_tokens=True
        ),
        drop_params=drop_params,
        drop_params_global=drop_params_global,
        modify_params=modify_params,
    )


@pytest.fixture()
def real_deps() -> TranslationDeps:
    return build_real_deps()


@pytest.fixture()
def frozen_ambient(monkeypatch):
    """Freeze uuid/fastuuid/time the way the characterization corpus does:
    v1 mints chatcmpl ids from ``litellm._uuid`` (fastuuid) and stamps
    ``created`` from ``time.time``."""
    import time
    import uuid

    import fastuuid

    import litellm._uuid

    counter = itertools.count(1)

    def fake_uuid4():
        return uuid.UUID(int=next(counter))

    monkeypatch.setattr(uuid, "uuid4", fake_uuid4)
    monkeypatch.setattr(fastuuid, "uuid4", fake_uuid4)
    monkeypatch.setattr(litellm._uuid, "uuid4", fake_uuid4)
    monkeypatch.setattr(time, "time", lambda: 1718064000.0)
    yield
