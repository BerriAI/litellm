# Caching

liteLLM implements exact match caching. It can be enabled by setting
1. `litellm.caching`: When set to `True`, enables caching for all responses. Keys are the input `messages` and values store in the cache is the corresponding `response`

2. `litellm.caching_with_models`: When set to `True`, enables caching on a per-model basis.Keys are the input `messages + model` and values store in the cache is the corresponding `response` 

## Usage
1. Caching - cache
Keys in the cache are `model`, the following example will lead to a cache hit
```python
litellm.caching = True

# Make completion calls
response1 = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Tell me a joke."}])
response2 = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Tell me a joke."}])

# response1 == response2, response 1 is cached

# with a diff model
response3 = completion(model="command-nightly", messages=[{"role": "user", "content": "Tell me a joke."}])

# response3 == response1 == response2, since keys are messages
```


2. Caching with Models - caching_with_models
Keys in the cache are `messages + model`, the following example will not lead to a cache hit
```python
litellm.caching_with_models = True

# Make completion calls
response1 = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Tell me a joke."}])
response2 = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Tell me a joke."}])
# response1 == response2, response 1 is cached

# with a diff model, this will call the API since the key is not cached
response3 = completion(model="command-nightly", messages=[{"role": "user", "content": "Tell me a joke."}])

# response3 != response1, since keys are messages + model
```

