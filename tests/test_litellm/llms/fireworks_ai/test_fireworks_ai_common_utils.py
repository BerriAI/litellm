import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.fireworks_ai.common_utils import resolve_fireworks_resource_name


@pytest.mark.parametrize(
    "model, expected",
    [
        ("routers/glm-latest", "accounts/fireworks/routers/glm-latest"),
        ("routers/firerouter", "accounts/fireworks/routers/firerouter"),
        ("fireworks_ai/routers/glm-latest", "accounts/fireworks/routers/glm-latest"),
        ("models/glm-4p6", "accounts/fireworks/models/glm-4p6"),
        ("fireworks_ai/models/glm-4p6", "accounts/fireworks/models/glm-4p6"),
        ("glm-4p6", "accounts/fireworks/models/glm-4p6"),
        ("fireworks_ai/glm-4p6", "accounts/fireworks/models/glm-4p6"),
        ("kimi-k2p6-fast", "accounts/fireworks/routers/kimi-k2p6-fast"),
        (
            "accounts/fireworks/routers/glm-latest",
            "accounts/fireworks/routers/glm-latest",
        ),
        (
            "accounts/fireworks/models/glm-4p6",
            "accounts/fireworks/models/glm-4p6",
        ),
        (
            "fireworks_ai/accounts/fireworks/routers/glm-latest",
            "accounts/fireworks/routers/glm-latest",
        ),
        (
            "accounts/fireworks/models/qwen2p5-coder-7b#accounts/gitlab/deployments/2fb7764c",
            "accounts/fireworks/models/qwen2p5-coder-7b#accounts/gitlab/deployments/2fb7764c",
        ),
        (
            "glm-4p6#accounts/gitlab/deployments/2fb7764c",
            "glm-4p6#accounts/gitlab/deployments/2fb7764c",
        ),
    ],
)
def test_resolve_fireworks_resource_name(model, expected):
    assert resolve_fireworks_resource_name(model) == expected
