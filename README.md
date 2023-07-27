# litellm
a simple, fast, 100 line package to call OpenAI, Azure, Cohere, Anthropic API Endpoints 

# usage
```
messages = [{ "content": "Hello, how are you?","role": "user"}]
# openai call
response = completion(model="gpt-3.5-turbo", messages=messages)

# cohere call
response = completion("command-nightly", messages)

# azure openai call
response = completion("chatgpt-test", messages, azure=True)
```

