import openai

from messages_list import messages_list

client = openai.OpenAI(
    api_key="sk-1234",  # pass litellm proxy key, if you're using virtual keys
    base_url="http://0.0.0.0:4000",  # litellm-proxy-base url
    max_retries=1,
)


for index, messages in enumerate(messages_list, 0):

    response = client.chat.completions.create(
        model="gemma-3", messages=messages, max_tokens=50
    )

    print("Response is: ", response.choices[0].message.content)


# NOTE to run these tests, run this in the root folder:

"""
// First add a config.yaml to the root folder with the models you want to use with the proxy

# then run the db and stats services
docker-compose up db prometheus

# then run the proxy server
BYTEZ_API_KEY=KEY_GOES_HERE CONFIG_FILE_PATH=config.yaml uvicorn litellm.proxy.proxy_server:app --host localhost --port 4000 --reload

"""
