import os
import json

gemini_model_cost_map = json.load(open("model_prices_and_context_window.json"))

for model, model_info in gemini_model_cost_map.items():
    if (
        (
            model_info.get("litellm_provider") == "gemini"
            or model_info.get("litellm_provider") == "vertex_ai-language-models"
        )
        and model_info.get("mode") == "chat"
        and ("gemini-2.5" in model and "tts" not in model)
        and model_info.get("supports_pdf_input") is None
    ):
        """
        Update all gemini chat models to support pdf input
        """
        model_info["supports_pdf_input"] = True
        print(f"Updated {model} to support pdf input")

json.dump(
    gemini_model_cost_map, open("model_prices_and_context_window.json", "w"), indent=4
)
