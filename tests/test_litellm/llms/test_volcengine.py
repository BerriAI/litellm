import os
import sys
from unittest.mock import MagicMock, patch

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
            "extra_body": {
                "thinking": {"type": "disabled"},
            }
        }

        e2e_mapped_params = get_optional_params(
            model="doubao-seed-1.6",
            custom_llm_provider="volcengine",
            thinking={"type": "enabled"},
            drop_params=False,
        )

        assert "thinking" in e2e_mapped_params["extra_body"] and e2e_mapped_params[
            "extra_body"
        ]["thinking"] == {
            "type": "enabled",
        }

    def test_e2e_completion(self):
        from openai import OpenAI

        from litellm import completion
        from litellm.types.utils import ModelResponse

        client = OpenAI(api_key="test_api_key")

        mock_raw_response = MagicMock()
        mock_raw_response.headers = {
            "x-request-id": "123",
            "openai-organization": "org-123",
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "99",
        }
        mock_raw_response.parse.return_value = ModelResponse()

        with patch.object(
            client.chat.completions.with_raw_response, "create", mock_raw_response
        ) as mock_create:
            completion(
                model="volcengine/doubao-seed-1.6",
                messages=[
                    {
                        "role": "system",
                        "content": "**Tell me your model detail information.**",
                    }
                ],
                user="guest",
                stream=True,
                thinking={"type": "disabled"},
                client=client,
            )

            mock_create.assert_called_once()
            print(mock_create.call_args.kwargs)
            assert mock_create.call_args.kwargs["extra_body"] == {
                "thinking": {"type": "disabled"},
            }
