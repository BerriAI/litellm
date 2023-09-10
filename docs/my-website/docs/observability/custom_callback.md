# Custom Callback Functions for Completion()

## Defining a Custom Callback Function
Create a custom callback function that takes specific arguments:

```python
def custom_callback(
    kwargs,                 # kwargs to completion
    completion_response,    # response from completion
    start_time, end_time    # start/end time
):
    # Your custom code here
    print("LITELLM: in custom callback function")
    print("kwargs", kwargs)
    print("completion_response", completion_response)
    print("start_time", start_time)
    print("end_time", end_time)
```

## Using Your Custom Callback Function

```python
import litellm
from litellm import completion

# Assign the custom callback function
litellm.success_callback = [custom_callback]

response = completion(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "Hi 👋 - i'm openai"
        }
    ]
)

print(response)

```

### Custom Callback to write to Mixpanel

```python
import mixpanel
import litellm
from litellm import completion

def custom_callback(
    kwargs,                 # kwargs to completion
    completion_response,    # response from completion
    start_time, end_time    # start/end time
):
    # Your custom code here
    mixpanel.track("LLM Response", {"llm_response": completion_response})


# Assign the custom callback function
litellm.success_callback = [custom_callback]

response = completion(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "Hi 👋 - i'm openai"
        }
    ]
)

print(response)

```












