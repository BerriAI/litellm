# Rules

Use this to fail a request based on the input or output of an llm api call. 


```python
import litellm 
import os 

# set env vars 
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["OPENROUTER_API_KEY"] = "your-api-key"

def my_custom_rule(input): # receives the model response 
    if "i don't think i can answer" in input: # trigger fallback if the model refuses to answer 
        return False 
    return True 

litellm.post_call_rules = [my_custom_rule] # have these be functions that can be called to fail a call

response = litellm.completion(model="gpt-3.5-turbo", messages=[{"role": "user", 
"content": "Hey, how's it going?"}], fallbacks=["openrouter/gryphe/mythomax-l2-13b"])
```

## Available Endpoints 

* `litellm.pre_call_rules = []` - A list of functions to iterate over before making the api call. Each function is expected to return either True (allow call) or False (fail call).

* `litellm.post_call_rules = []` - List of functions to iterate over before making the api call. Each function is expected to return either True (allow call) or False (fail call).


## Expected format of rule 

```python
def my_custom_rule(input: str) -> bool: # receives the model response 
    if "i don't think i can answer" in input: # trigger fallback if the model refuses to answer 
        return False 
    return True 
```

#### Inputs
* `input`: *str*: The user input or llm response. 

#### Outputs
* `bool`: Return True (allow call) or False (fail call)


## Example Rules 

### Example 1: Fail if user input is too long 

```python
import litellm 
import os 

# set env vars 
os.environ["OPENAI_API_KEY"] = "your-api-key"

def my_custom_rule(input): # receives the model response 
    if len(input) > 10: # fail call if too long
        return False 
    return True 

litellm.pre_call_rules = [my_custom_rule] # have these be functions that can be called to fail a call

response = litellm.completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hey, how's it going?"}])
```

### Example 2: Fallback to uncensored model if llm refuses to answer


```python
import litellm 
import os 

# set env vars 
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["OPENROUTER_API_KEY"] = "your-api-key"

def my_custom_rule(input): # receives the model response 
    if "i don't think i can answer" in input: # trigger fallback if the model refuses to answer 
        return False 
    return True 

litellm.post_call_rules = [my_custom_rule] # have these be functions that can be called to fail a call

response = litellm.completion(model="gpt-3.5-turbo", messages=[{"role": "user", 
"content": "Hey, how's it going?"}], fallbacks=["openrouter/gryphe/mythomax-l2-13b"])
```