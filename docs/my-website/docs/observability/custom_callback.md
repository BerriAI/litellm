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












