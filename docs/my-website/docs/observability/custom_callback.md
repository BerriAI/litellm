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
## Get complete streaming response

LiteLLM will pass you the complete streaming response in the final streaming chunk as part of the kwargs for your custom callback function.

```python
# litellm.set_verbose = False
        def custom_callback(
            kwargs,                 # kwargs to completion
            completion_response,    # response from completion
            start_time, end_time    # start/end time
        ):
            # print(f"streaming response: {completion_response}")
            if "complete_streaming_response" in kwargs: 
                print(f"Complete Streaming Response: {kwargs['complete_streaming_response']}")
        
        # Assign the custom callback function
        litellm.success_callback = [custom_callback]

        response = completion(model="claude-instant-1", messages=messages, stream=True)
        for idx, chunk in enumerate(response): 
            pass
```

## Examples

### Custom Callback to track costs for Streaming + Non-Streaming
```python

def track_cost_callback(
    kwargs,                 # kwargs to completion
    completion_response,    # response from completion
    start_time, end_time    # start/end time
):
    try:
        # init logging config
        logging.basicConfig(
                filename='cost.log',
                level=logging.INFO,
                format='%(asctime)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
        )

        # check if it has collected an entire stream response
        if "complete_streaming_response" in kwargs:
            # for tracking streaming cost we pass the "messages" and the output_text to litellm.completion_cost 
            completion_response=kwargs["complete_streaming_response"]
            input_text = kwargs["messages"]
            output_text = completion_response["choices"][0]["message"]["content"]
            response_cost = litellm.completion_cost(
                model = kwargs["model"],
                messages = input_text,
                completion=output_text
            )
            print("streaming response_cost", response_cost)
            logging.info(f"Model {kwargs['model']} Cost: ${response_cost:.8f}")

        # for non streaming responses
        else:
            # we pass the completion_response obj
            if kwargs["stream"] != True:
                response_cost = litellm.completion_cost(completion_response=completion_response)
                print("regular response_cost", response_cost)
                logging.info(f"Model {completion_response.model} Cost: ${response_cost:.8f}")
    except:
        pass

# Assign the custom callback function
litellm.success_callback = [track_cost_callback]

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












