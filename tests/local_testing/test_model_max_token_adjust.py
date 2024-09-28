# What this tests?
## Tests if max tokens get adjusted, if over limit

import sys, os, time
import traceback, asyncio
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import completion


@pytest.mark.skip(reason="AWS Suspended Account")
def test_completion_sagemaker():
    litellm.set_verbose = True
    litellm.drop_params = True
    response = completion(
        model="sagemaker/berri-benchmarking-Llama-2-70b-chat-hf-4",
        messages=[{"content": "Hello, how are you?", "role": "user"}],
        temperature=0.2,
        max_tokens=80000,
        hf_model_name="meta-llama/Llama-2-70b-chat-hf",
    )
    print(f"response: {response}")


# test_completion_sagemaker()
