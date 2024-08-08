import datetime

import openai
import vertexai
from vertexai.generative_models import Content, Part
from vertexai.preview import caching
from vertexai.preview.generative_models import GenerativeModel

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")
vertexai.init(project="adroit-crow-413218", location="us-central1")
print("creating cached content")
contents_here: list[Content] = [
    Content(role="user", parts=[Part.from_text("huge string of text here" * 10000)])
]
cached_content = caching.CachedContent.create(
    model_name="gemini-1.5-pro-001",
    contents=contents_here,
    expire_time=datetime.datetime(2024, 8, 10),
)

created_Caches = caching.CachedContent.list()

print("created_Caches contents=", created_Caches)

response = client.chat.completions.create(  # type: ignore
    model="gemini-1.5-pro-001",
    max_tokens=8192,
    messages=[
        {
            "role": "user",
            "content": "quote all everything above this message",
        },
    ],
    temperature=0.7,
    extra_body={"cached_content": cached_content.resource_name},
)

print("response from proxy", response)
