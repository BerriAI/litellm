import json
import os
import sys


def test_databricks_pricing_integrity():
    """
    Verifies that for all Databricks models in model_prices_and_context_window.json:
    USD Price == DBU Price * 0.07
    """
    json_path = os.path.join(
        os.path.dirname(__file__), "../../../../model_prices_and_context_window.json"
    )

    # Verify file exists
    assert os.path.exists(
        json_path
    ), f"Could not find model_prices_and_context_window.json at {json_path}"

    with open(json_path, "r") as f:
        data = json.load(f)

    conversion_rate = 0.07  # 1 DBU = 0.07 USD
    errors = []

    for model, info in data.items():
        if info.get("litellm_provider") == "databricks":
            # Check Input Cost
            input_usd = info.get("input_cost_per_token")
            input_dbu = info.get("input_dbu_cost_per_token")

            if input_usd is not None and input_dbu is not None:
                expected = input_dbu * conversion_rate
                # Allow small floating point difference
                if abs(input_usd - expected) > 1e-9:
                    errors.append(
                        f"{model} input mismatch: USD={input_usd}, DBU={input_dbu}, Expected={expected}"
                    )

            # Check Output Cost
            output_usd = info.get("output_cost_per_token")
            output_dbu = info.get("output_dbu_cost_per_token")

            if output_usd is not None and output_dbu is not None:
                expected = output_dbu * conversion_rate
                if abs(output_usd - expected) > 1e-9:
                    errors.append(
                        f"{model} output mismatch: USD={output_usd}, DBU={output_dbu}, Expected={expected}"
                    )

    assert not errors, "\n" + "\n".join(errors)


def test_databricks_cost_per_token_honors_cached_tokens():
    """
    Regression for the hand-rolled databricks cost calculator that multiplied every prompt
    token by input_cost_per_token and ignored prompt_tokens_details.cached_tokens. After
    delegating to generic_cost_per_token, cached tokens must bill at cache_read_input_token_cost
    (mirrors deepseek / xai / fireworks). A cache-priced databricks model is registered for the
    test since no bundled databricks entry publishes cache rates yet.
    """
    import litellm
    from litellm.llms.databricks.cost_calculator import cost_per_token
    from litellm.types.utils import PromptTokensDetailsWrapper, Usage

    litellm.register_model(
        {
            "databricks/test-cache-model": {
                "litellm_provider": "databricks",
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 2e-6,
                "cache_read_input_token_cost": 1e-7,
                "mode": "chat",
            }
        }
    )

    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=200,
        total_tokens=1200,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=800),
    )

    prompt_cost, completion_cost = cost_per_token("databricks/test-cache-model", usage)

    # non-cached 200 * 1e-6 + cached 800 * 1e-7 = 0.0002 + 0.00008 = 0.00028
    assert abs(prompt_cost - 0.00028) < 1e-12
    assert abs(completion_cost - 200 * 2e-6) < 1e-12
