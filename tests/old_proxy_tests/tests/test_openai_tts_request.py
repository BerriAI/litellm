import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

# # request sent to model set on litellm proxy, `litellm --model`
response = client.audio.speech.create(
    model="vertex-tts",
    input="the quick brown fox jumped over the lazy dogs",
    voice={"languageCode": "en-US", "name": "en-US-Studio-O"},  # type: ignore
)
print("response from proxy", response)  # noqa
