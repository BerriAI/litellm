
# build docker image
docker build --platform=linux/arm64 -t litellm-raga:latest -f raga.Dockerfile .

# run the docker image
docker run -it --platform=linux/arm64 \
-e VAULT_ADDR="" \
-e VAULT_TOKEN="" \
-p 4000:4000 litellm-raga:latest


# test the openAI API
curl --location 'http://localhost:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "gpt-3.5-turbo",
      "messages": [
        {
          "role": "user",
          "content": "write a poem on pirates and sandworms"
        }
      ],
      "api_key": "sk_1234567890", # replace this
    }
'