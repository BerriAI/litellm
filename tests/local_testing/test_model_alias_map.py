#### What this tests ####
#    This tests the model alias mapping - if user passes in an alias, and has set an alias, set it to the actual value

import traceback

import pytest

import litellm
from litellm import completion, embedding

litellm.set_verbose = True

model_alias_map = {"good-model": "groq/openai/gpt-oss-120b"}


def test_model_alias_map(caplog):
    try:
        litellm.model_alias_map = model_alias_map
        response = completion(
            "good-model",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            top_p=0.1,
            temperature=0.01,
            max_tokens=10,
        )
        print(response.model)

        for rec in caplog.records:
            if rec.levelname == "ERROR" and rec.name.startswith("LiteLLM"):
                pytest.fail(f"Unexpected litellm ERROR log: {rec.getMessage()}")

        assert "gpt-oss-120b" in response.model
    except litellm.ServiceUnavailableError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_model_alias_map()
