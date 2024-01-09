# Gemini - Google AI Studio

## Pre-requisites
* `pip install -q google-generativeai`

# Gemini-Pro
## Sample Usage
```python
import litellm
import os

os.environ['GEMINI_API_KEY'] = ""
response = completion(
    model="gemini/gemini-pro", 
    messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}]
)
```

# Gemini-Pro-Vision
LiteLLM Supports the following image types passed in `url`
- Images with direct links - https://storage.googleapis.com/github-repo/img/gemini/intro/landmark3.jpg
- Image in local storage - ./localimage.jpeg

## Sample Usage
```python
import os
import litellm 
from dotenv import load_dotenv

# Load the environment variables from .env file
load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv('GEMINI_API_KEY')

prompt = 'Describe the image in a few sentences.'
# Note: You can pass here the URL or Path of image directly.
image_url = 'https://storage.googleapis.com/github-repo/img/gemini/intro/landmark3.jpg'

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
response = litellm.completion(
    model="gemini/gemini-pro-vision",
    messages=messages,
)

# Extract the response content
content = response.get('choices', [{}])[0].get('message', {}).get('content')

# Print the result
print(content)
```

## Chat Models
| Model Name       | Function Call                        | Required OS Variables    |
|------------------|--------------------------------------|-------------------------|
| gemini-pro       | `completion('gemini/gemini-pro', messages)` | `os.environ['GEMINI_API_KEY']` |
| gemini-pro-vision       | `completion('gemini/gemini-pro-vision', messages)` | `os.environ['GEMINI_API_KEY']` |
