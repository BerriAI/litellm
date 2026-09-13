"""`get_optional_params` must map Cohere params with the config that will
actually transform the request.

`CohereModelInfo.get_cohere_route` sends every model to v2 unless the id asks
for v1, and `ProviderConfigManager` hands the v2 config to the transform. Param
mapping used to be pinned to the v1 config regardless, so a request could be
built by v2 while its params were mapped by v1.

The two configs happen to agree today, so this is not a behaviour change. It
stops them silently disagreeing the moment either one grows a param the other
does not have.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

import pytest

from litellm.llms.cohere.chat.transformation import CohereChatConfig
from litellm.llms.cohere.chat.v2_transformation import CohereV2ChatConfig
from litellm.utils import ProviderConfigManager


@pytest.mark.parametrize("custom_llm_provider", ["cohere", "cohere_chat"])
@pytest.mark.parametrize(
    "model, expected_config",
    [
        ("command-a-03-2025", CohereV2ChatConfig),
        ("cohere_chat/v1/command-r", CohereChatConfig),
    ],
)
def test_param_mapping_uses_the_route_config(model, expected_config, custom_llm_provider, monkeypatch):
    """The config that maps params must be the one that transforms the request."""
    seen = {}

    def record(self, non_default_params, optional_params, model, drop_params):
        seen["config"] = type(self)
        return optional_params

    monkeypatch.setattr(CohereChatConfig, "map_openai_params", record)
    monkeypatch.setattr(CohereV2ChatConfig, "map_openai_params", record)

    from litellm.utils import get_optional_params

    get_optional_params(
        model=model,
        custom_llm_provider=custom_llm_provider,
        drop_params=True,
        temperature=0.5,
    )

    assert seen["config"] is expected_config
    assert seen["config"] is type(ProviderConfigManager._get_cohere_config(model=model))
