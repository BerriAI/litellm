import vertexai
from vertexai.preview.generative_models import GenerativeModel

LITE_LLM_ENDPOINT = "http://localhost:4000"

vertexai.init(
    project="adroit-crow-413218",
    location="us-central1",
    api_endpoint=f"{LITE_LLM_ENDPOINT}/vertex-ai",
    api_transport="rest",
)

model = GenerativeModel(model_name="gemini-1.0-pro")
response = model.generate_content("hi")

print("response", response)
