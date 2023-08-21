# Debugging UI Tutorial
LiteLLM offers a free hosted debugger UI for your api calls. Useful if you're testing your LiteLLM server and need to see if the API calls were made successfully.

You can enable this setting `lite_debugger` as a callback. 

## Example Usage

```
 import litellm
 from litellm import embedding, completion

 litellm.input_callback = ["lite_debugger"]
 litellm.success_callback = ["lite_debugger"]
 litellm.failure_callback = ["lite_debugger"]

 litellm.set_verbose = True

 user_message = "Hello, how are you?"
 messages = [{ "content": user_message,"role": "user"}]


 # openai call
 response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}])

 # bad request call
 response = completion(model="chatgpt-test", messages=[{"role": "user", "content": "Hi 👋 - i'm a bad request"}])

```

## Requirements

## How to see the UI
