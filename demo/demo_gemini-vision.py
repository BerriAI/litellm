from litellm import completion
import os
import pprint
import litellm
litellm.set_verbose=True

# response = completion(
#     model="gemini/gemini-pro", 
#     messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}]
# )

# pprint.pprint(response)
# print("------------------")
# print(response.choices[0].message.content)

print("============================")

prompt = 'Describe what you see in the image, focusing on the main subject and the background.'
# Note: You can pass here the URL or Path of image directly.
image_url = 'https://storage.googleapis.com/github-repo/img/gemini/intro/landmark1.jpg'

# Create the messages payload according to the documentation
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {"url": image_url}
            }
        ]
    }
]

# Make the API call to Gemini model
response = completion(
    model="gemini/gemini-pro-vision",
    messages=messages,
)

print(response)
# Extract the response content
content = response.get('choices', [{}])[0].get('message', {}).get('content')

# Print the result
print(content)