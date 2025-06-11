import os
import json

gemini_model_cost_map = json.load(open("model_prices_and_context_window.json"))

for model, model_info in gemini_model_cost_map.items():
    if model_info.get("litellm_provider") == "gemini" and model_info.get("mode") == "chat" and "gemini-2" in model:
        """
        Update all gemini chat models to support web search
        """
        model_info["supports_web_search"] = True

json.dump(gemini_model_cost_map, open("model_prices_and_context_window.json", "w"), indent=4)