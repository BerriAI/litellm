import os
import sys

from pydantic import BaseModel

from litellm.llms.volcengine import VolcEngineConfig
from litellm.utils import get_optional_params


class TestVolcEngineConfig:
    def test_get_optional_params(self):
        config = VolcEngineConfig()
        supported_params = config.get_supported_openai_params(model="doubao-seed-1.6")
        assert "thinking" in supported_params

        mapped_params = config.map_openai_params(
            non_default_params={
                "thinking": {"type": "disabled"},
            },
            optional_params={},
            model="doubao-seed-1.6",
            drop_params=False,
        )

        assert mapped_params == {
            "thinking": {"type": "disabled"},
        }

        e2e_mapped_params = get_optional_params(
            model="doubao-seed-1.6",
            custom_llm_provider="volcengine",
            thinking={"type": "enabled"},
            drop_params=False,
        )

        assert "thinking" in e2e_mapped_params and e2e_mapped_params["thinking"] == {
            "type": "enabled",
        }
