import openai
client = openai.OpenAI(
    api_key="anything",
    # base_url="http://0.0.0.0:8000",
)

try:
    # request sent to model set on litellm proxy, `litellm --model`
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        },
    ])

    print(response)
# except openai.APITimeoutError:
#     print("Got openai Timeout Exception. Good job. The proxy mapped to OpenAI exceptions")
except Exception as e:
    print("\n the proxy did not map to OpenAI exception. Instead got", e)
    print(e.type) # type: ignore 
    print(e.message) # type: ignore 
    print(e.code) # type: ignore 