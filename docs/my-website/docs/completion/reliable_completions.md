# Reliability for Completions()

LiteLLM supports `completion_with_retries`. 

You can use this as a drop-in replacement for the `completion()` function to use tenacity retries - by default we retry the call 3 times. 

Here's a quick look at how you can use it: 

```python 
from litellm import completion_with_retries

user_message = "Hello, whats the weather in San Francisco??"
messages = [{"content": user_message, "role": "user"}]

# normal call 
def test_completion_custom_provider_model_name():
    try:
        response = completion_with_retries(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        printf"Error occurred: {e}")
```