# Local Debugging
There's 2 ways to do local debugging - `litellm.set_verbose=True` and by passing in a custom function `completion(...logger_fn=<your_local_function>)`. Warning: Make sure to not use `set_verbose` in production. It logs API keys, which might end up in log files.

## Set Verbose 

This is good for getting print statements for everything litellm is doing.
```python
import litellm
from litellm import completion

litellm.set_verbose=True # 👈 this is the 1-line change you need to make

## set ENV variables
os.environ["OPENAI_API_KEY"] = "openai key"
os.environ["COHERE_API_KEY"] = "cohere key"

messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion(model="gpt-3.5-turbo", messages=messages)

# cohere call
response = completion("command-nightly", messages)
```

## Logger Function 
But sometimes all you care about is seeing exactly what's getting sent to your api call and what's being returned - e.g. if the api call is failing, why is that happening? what are the exact params being set? 

In that case, LiteLLM allows you to pass in a custom logging function to see / modify the model call Input/Outputs. 

**Note**: We expect you to accept a dict object. 

Your custom function 

```python
def my_custom_logging_fn(model_call_dict):
    print(f"model call details: {model_call_dict}")
```

### Complete Example
```python
from litellm import completion

def my_custom_logging_fn(model_call_dict):
    print(f"model call details: {model_call_dict}")

## set ENV variables
os.environ["OPENAI_API_KEY"] = "openai key"
os.environ["COHERE_API_KEY"] = "cohere key"

messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion(model="gpt-3.5-turbo", messages=messages, logger_fn=my_custom_logging_fn)

# cohere call
response = completion("command-nightly", messages, logger_fn=my_custom_logging_fn)
```

## Still Seeing Issues? 

Text us @ +17708783106 or Join the [Discord](https://discord.com/invite/wuPM9dRgDw). 

We promise to help you in `lite`ning speed ❤️
