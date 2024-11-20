import openai # openai v1.0.0+
client = openai.OpenAI(api_key="sk-1234",base_url="http://0.0.0.0:4000") # set proxy to base_url
# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(model="gpt-3.5-turbo", messages = [
    {
        "role": "user",
        "content": "this is a test request including a message from Adam to Amanda, write a short poem"
    }],
    extra_body={
         "guardrails": ["acuvity-pre-guard", "acuvity-post-guard"]
    },
)

print(response)
