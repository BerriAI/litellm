import sys, os
import traceback
sys.path.append('..')  # Adds the parent directory to the system path
import main
from main import completion

main.set_verbose = True

user_message = "Hello, whats the weather in San Francisco??"
messages = [{ "content": user_message,"role": "user"}]


################# Test 3 #################
# test on Azure Openai Completion Call 
try:
    response = completion(model="chatgpt-test", messages=messages, azure=True)
    print(response)
except Exception as e:
    print(f"error occurred: {traceback.format_exc()}") 
    raise e

################# Test 1 #################
# test on openai completion call, with model and messages
try:
    response = completion(model="gpt-3.5-turbo", messages=messages)
    print(response)
except Exception as e:
    print(f"error occurred: {traceback.format_exc()}") 
    raise e

################# Test 1.1 #################
# test on openai completion call, with model and messages, optional params
try:
    response = completion(model="gpt-3.5-turbo", messages=messages, temperature=0.5, top_p=0.1, user="ishaan_dev@berri.ai")
    print(response)
except Exception as e:
    print(f"error occurred: {traceback.format_exc()}") 
    raise e

################# Test 1.2 #################
# test on openai completion call, with model and messages, optional params
try:
    response = completion(model="gpt-3.5-turbo", messages=messages, temperature=0.5, top_p=0.1, n=2, max_tokens=150, presence_penalty=0.5, frequency_penalty=-0.5, logit_bias={123:5}, user="ishaan_dev@berri.ai")
    print(response)
except Exception as e:
    print(f"error occurred: {traceback.format_exc()}") 
    raise e



################# Test 1.3 #################
# Test with Stream = True
try:
    response = completion(model="gpt-3.5-turbo", messages=messages, temperature=0.5, top_p=0.1, n=2, max_tokens=150, presence_penalty=0.5, stream=True, frequency_penalty=-0.5, logit_bias={27000:5}, user="ishaan_dev@berri.ai")
    print(response)
except Exception as e:
    print(f"error occurred: {traceback.format_exc()}") 
    raise e

################# Test 2 #################
# test on openai completion call, with functions
function1 = [
    {
      "name": "get_current_weather",
      "description": "Get the current weather in a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"
          },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"]
          }
        },
        "required": ["location"]
      }
    }
  ]
user_message = "Hello, whats the weather in San Francisco??"
messages = [{ "content": user_message,"role": "user"}]
try:
    response = completion(model="gpt-3.5-turbo", messages=messages, functions=function1)
    print(response)
except Exception as e:
    print(f"error occurred: {traceback.format_exc()}") 
    raise e




################# Test 4 #################
# test on Claude Completion Call 
try:
    response = completion(model="claude-instant-1", messages=messages)
    print(response)
except Exception as e:
    print(f"error occurred: {traceback.format_exc()}") 
    raise e

################# Test 5 #################
# test on Cohere Completion Call 
try:
    response = completion(model="command-nightly", messages=messages, max_tokens=500)
    print(response)
except Exception as e:
    print(f"error occurred: {traceback.format_exc()}") 
    raise e

################# Test 6 #################
# test on Replicate llama2 Completion Call 
try:
    model_name = "replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1"
    response = completion(model=model_name, messages=messages, max_tokens=500)
    print(response)
except Exception as e:
    print(f"error occurred: {traceback.format_exc()}") 
    raise e
