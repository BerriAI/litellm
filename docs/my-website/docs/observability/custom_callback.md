# Custom Callbacks

## Callback Class
You can create a custom callback class to precisely log events as they occur in litellm. 

```python
from litellm.integrations.custom_logger import CustomLogger

class MyCustomHandler(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs): 
        print(f"Pre-API Call")
    
    def log_post_api_call(self, kwargs, response_obj, start_time, end_time): 
        print(f"Post-API Call")
    
    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Stream")
        
    def log_success_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"On Success")

    def log_failure_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"On Failure")

customHandler = MyCustomHandler()

litellm.callbacks = [customHandler]
response = completion(model="gpt-3.5-turbo", messages=[{ "role": "user", "content": "Hi 👋 - i'm openai"}],
                              stream=True)
for chunk in response: 
    continue
```

## Callback Functions
If you just want to log on a specific event (e.g. on input) - you can use callback functions. 

You can set custom callbacks to trigger for:
- `litellm.input_callback`   - Track inputs/transformed inputs before making the LLM API call
- `litellm.success_callback` - Track inputs/outputs after making LLM API call
- `litellm.failure_callback` - Track inputs/outputs + exceptions for litellm calls

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

### Setting the custom callback function
```python
import litellm
litellm.success_callback = [custom_callback]
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

## What's in kwargs? 

Notice we pass in a kwargs argument to custom callback. 
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

This is a dictionary containing all the model-call details (the params we receive, the values we send to the http endpoint, the response we receive, stacktrace in case of errors, etc.). 

This is all logged in the [model_call_details via our Logger](https://github.com/BerriAI/litellm/blob/fc757dc1b47d2eb9d0ea47d6ad224955b705059d/litellm/utils.py#L246).

Here's exactly what you can expect in the kwargs dictionary:
```shell
### DEFAULT PARAMS ### 
"model": self.model,
"messages": self.messages,
"optional_params": self.optional_params, # model-specific params passed in
"litellm_params": self.litellm_params, # litellm-specific params passed in (e.g. metadata passed to completion call)
"start_time": self.start_time, # datetime object of when call was started

### PRE-API CALL PARAMS ### (check via kwargs["log_event_type"]="pre_api_call")
"input" = input # the exact prompt sent to the LLM API
"api_key" = api_key # the api key used for that LLM API 
"additional_args" = additional_args # any additional details for that API call (e.g. contains optional params sent)

### POST-API CALL PARAMS ### (check via kwargs["log_event_type"]="post_api_call")
"original_response" = original_response # the original http response received (saved via response.text)

### ON-SUCCESS PARAMS ### (check via kwargs["log_event_type"]="successful_api_call")
"complete_streaming_response" = complete_streaming_response # the complete streamed response (only set if `completion(..stream=True)`)
"end_time" = end_time # datetime object of when call was completed

### ON-FAILURE PARAMS ### (check via kwargs["log_event_type"]="failed_api_call")
"exception" = exception # the Exception raised
"traceback_exception" = traceback_exception # the traceback generated via `traceback.format_exc()`
"end_time" = end_time # datetime object of when call was completed
```

### Get complete streaming response

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


### Log additional metadata

LiteLLM accepts a metadata dictionary in the completion call. You can pass additional metadata into your completion call via `completion(..., metadata={"key": "value"})`. 

Since this is a [litellm-specific param](https://github.com/BerriAI/litellm/blob/b6a015404eed8a0fa701e98f4581604629300ee3/litellm/main.py#L235), it's accessible via kwargs["litellm_params"]

```python
from litellm import completion
import os, litellm

## set ENV variables
os.environ["OPENAI_API_KEY"] = "your-api-key"

messages = [{ "content": "Hello, how are you?","role": "user"}]

def custom_callback(
    kwargs,                 # kwargs to completion
    completion_response,    # response from completion
    start_time, end_time    # start/end time
):
    print(kwargs["litellm_params"]["metadata"])
    

# Assign the custom callback function
litellm.success_callback = [custom_callback]

response = litellm.completion(model="gpt-3.5-turbo", messages=messages, metadata={"hello": "world"})
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

### Custom Callback to log transformed Input to LLMs
```python
def get_transformed_inputs(
    kwargs,
):
    params_to_model = kwargs["additional_args"]["complete_input_dict"]
    print("params to model", params_to_model)

litellm.input_callback = [get_transformed_inputs]

def test_chat_openai():
    try:
        response = completion(model="claude-2",
                              messages=[{
                                  "role": "user",
                                  "content": "Hi 👋 - i'm openai"
                              }])

        print(response)

    except Exception as e:
        print(e)
        pass
```

#### Output
```shell
params to model {'model': 'claude-2', 'prompt': "\n\nHuman: Hi 👋 - i'm openai\n\nAssistant: ", 'max_tokens_to_sample': 256}
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












