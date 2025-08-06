import base64
from openai import OpenAI

client = OpenAI(
    base_url="http://0.0.0.0:12635",
    api_key="sk-1234"
)

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Path to your image
image_path = "litellm/proxy/logo.jpg"

# Getting the Base64 string
base64_image = encode_image(image_path)


response = client.responses.create(
    model="bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    input=[
        {
            "role": "user",
            "content": [
                { "type": "input_text", "text": "what's in this image?" },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        }
    ],
)


print(response.output_text)