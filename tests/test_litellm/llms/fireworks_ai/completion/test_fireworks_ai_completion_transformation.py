import os
import sys

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.fireworks_ai.completion.transformation import (
    FireworksAITextCompletionConfig,
)


def test_transform_text_completion_request_routes_router_slug():
    config = FireworksAITextCompletionConfig()

    data = config.transform_text_completion_request(
        model="routers/glm-latest",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        headers={},
    )

    assert data["model"] == "accounts/fireworks/routers/glm-latest"


def test_transform_text_completion_request_bare_slug_stays_model():
    config = FireworksAITextCompletionConfig()

    data = config.transform_text_completion_request(
        model="glm-4p6",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        headers={},
    )

    assert data["model"] == "accounts/fireworks/models/glm-4p6"
