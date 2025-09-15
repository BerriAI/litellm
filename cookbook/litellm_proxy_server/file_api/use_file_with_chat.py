from openai import OpenAI
import requests
client = OpenAI(
    base_url="http://0.0.0.0:4000",
    api_key="sk-1234",
)
# Download and save the PDF locally
url = (
    "https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf"
)
response = requests.get(url)
response.raise_for_status()

# Save the PDF locally
with open("2403.05530.pdf", "wb") as f:
    f.write(response.content)

VERTEX_AI_MODEL = "vertex_ai/gemini-2.5-flash"

# # Upload file
file_input_file = client.files.create(
    file=open("./2403.05530.pdf", "rb"),
    purpose="user_data",
    extra_body={"target_model_names": VERTEX_AI_MODEL}
)
print(file_input_file)

# Create batch
chat_completion = client.chat.completions.create(
    model=VERTEX_AI_MODEL,
    messages=[
        {"role": "user", "content": "What is this file about?"},
        {"role": "user", "content": [{"type": "file", "file": {"file_id": file_input_file.id}}]},
    ],
)
print(chat_completion)