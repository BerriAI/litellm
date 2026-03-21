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
