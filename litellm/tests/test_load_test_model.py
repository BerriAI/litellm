import sys, os
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import load_test_model, testing_batch_completion

# ## Load Test Model
# model="gpt-3.5-turbo"
# result = load_test_model(model=model, num_calls=5)
# print(result)
# print(len(result["results"]))

# ## Duration Test Model
# model="gpt-3.5-turbo"
# result = load_test_model(model=model, num_calls=5, duration=15, interval=15) # duration test the model for 2 minutes, sending 5 calls every 15s
# print(result)

## Quality Test across Model
models = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "claude-instant-1",
    {
        "model": "replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781",
        "custom_llm_provider": "replicate",
    },
]
messages = [
    [{"role": "user", "content": "What is your name?"}],
    [{"role": "user", "content": "Hey, how's it going?"}],
]
result = testing_batch_completion(models=models, messages=messages)
print(result)
