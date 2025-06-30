from litellm.utils import get_optional_params


def test_watsonx_text_moderations():
    optional_params = get_optional_params(
        model="ibm/granite-3.3-8b-instruct",
        custom_llm_provider="watsonx_text",
        moderations={
            "hap": {
                "input": {"enabled": True, "threshold": 0.5},
                "output": {"enabled": True, "threshold": 0.5},
            },
        },
    )
    assert optional_params["moderations"] == {
        "hap": {
            "input": {"enabled": True, "threshold": 0.5},
            "output": {"enabled": True, "threshold": 0.5},
        }
    }
