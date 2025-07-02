import json

mistral_model_cost_map = json.load(open("model_prices_and_context_window.json"))

for model, model_info in mistral_model_cost_map.items():
    if (
        (model_info.get("litellm_provider") == "mistral")
        and model_info.get("mode") == "chat"
        and ("codestral-mamba" not in model)
    ):
        """
        Update all mistral models to supports_response_schema
        """
        model_info["supports_response_schema"] = True
        print(f"Updated {model} to support response schema")

json.dump(
    mistral_model_cost_map, open("model_prices_and_context_window.json", "w"), indent=4
)
