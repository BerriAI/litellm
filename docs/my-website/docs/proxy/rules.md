# Post-Call Rules 

Use this to fail a request based on the output of an llm api call.

## Quick Start

### Step 1: Create a file (e.g. post_call_rules.py)

```python
def my_custom_rule(input): # receives the model response 
    if len(input) < 5: # trigger fallback if the model response is too short
         return False 
    return True 
```

### Step 2. Point it to your proxy

```python
litellm_settings:
  post_call_rules: post_call_rules.my_custom_rule
  num_retries: 3
```

### Step 3. Start + test your proxy

```bash
$ litellm /path/to/config.yaml
```

```bash
curl --location 'http://0.0.0.0:8000/v1/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer sk-1234' \
--data '{
  "model": "deepseek-coder",
  "messages": [{"role":"user","content":"What llm are you?"}],
  "temperature": 0.7,
  "max_tokens": 10,
}'
```
---

This will now check if a response is > len 5, and if it fails, it'll retry a call 3 times before failing.