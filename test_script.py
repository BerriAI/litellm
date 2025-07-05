import json

mistral_model_cost_map = json.load(open("model_prices_and_context_window.json"))

for model, model_info in mistral_model_cost_map.items():
    if (
        "bedrock" in model_info.get("litellm_provider")
        and model_info.get("mode") == "chat"
        and any(
            m in model
            for m in [
                "mistral.mistral-large-2402-v1:0",
                "mistral.mistral-small-2402-v1:0",
            ]
        )
    ):
        """
        Update all mistral models to supports_response_schema
        """
        del model_info["supports_tool_choice"]
        print(f"Updated {model} to support response schema")

json.dump(
    mistral_model_cost_map, open("model_prices_and_context_window.json", "w"), indent=4
)
