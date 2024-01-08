# Gemini - Google AI Studio

## Pre-requisites
* `pip install -q google-generativeai`

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

LiteLLM Supports the following image types passed in `url`
- Images with direct links - https://storage.googleapis.com/github-repo/img/gemini/intro/landmark3.jpg
- Image in local storage - ./localimage.jpeg



## Chat Models
| Model Name       | Function Call                        | Required OS Variables    |
|------------------|--------------------------------------|-------------------------|
| gemini-pro       | `completion('gemini/gemini-pro', messages)` | `os.environ['PALM_API_KEY']` |
| gemini-pro-vision       | `completion('gemini/gemini-pro-vision', messages)` | `os.environ['PALM_API_KEY']` |
