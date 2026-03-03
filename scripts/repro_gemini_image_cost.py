import os
import litellm
from litellm.llms.gemini.image_generation.transformation import GoogleImageGenConfig
from litellm.types.utils import ImageObject, ImageResponse

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
litellm.model_cost = litellm.get_model_cost_map(url="")

model = "gemini/gemini-3-pro-image-preview"
config = GoogleImageGenConfig()

usage_metadata = {
    "promptTokenCount": 200,
    "candidatesTokenCount": 0,
    "totalTokenCount": 200,
    "promptTokensDetails": [
        {"modality": "TEXT", "tokenCount": 10},
        {"modality": "IMAGE", "tokenCount": 90},
        {"modality": "IMAGE", "tokenCount": 100},
    ],
}

parsed_usage = config._transform_image_usage(usage_metadata)
resp = ImageResponse(
    data=[ImageObject(b64_json="fake_image_data")],
    usage=parsed_usage,
)

observed_cost = litellm.completion_cost(
    completion_response=resp,
    model=model,
    custom_llm_provider="gemini",
)

model_info = litellm.get_model_info(model=model, custom_llm_provider="gemini")
input_cost_per_token = model_info["input_cost_per_token"]

expected_prompt_tokens = 10 + 90 + 100
expected_prompt_cost = expected_prompt_tokens * input_cost_per_token

print(f"\n--- Results for {model} ---")
print(f"Input Tokens (Total): {parsed_usage.input_tokens}")
print(f"Image Tokens: {parsed_usage.input_tokens_details.image_tokens}")
print(f"Observed Cost: ${observed_cost:.8f}")
print(f"Expected Cost: ${expected_prompt_cost:.8f}")

if abs(observed_cost - expected_prompt_cost) < 1e-10:
    print("\n✅ SUCCESS: The fix works! Tokens are accumulated correctly.")
else:
    print("\n❌ FAILED: The cost or tokens don't match.")
