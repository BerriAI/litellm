#### What this tests ####
#    This tests error handling + logging (esp. for sentry breadcrumbs)

import sys, os
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
from litellm import completion

messages=[{"role": "user", "content": "Hey, how's it going?"}]

## All your mistral deployments ##
model_list = [{
	"model_name": "mistral-7b-instruct",
	"litellm_params": { # params for litellm completion/embedding call 
        "model": "replicate/mistralai/mistral-7b-instruct-v0.1:83b6a56e7c828e667f21fd596c338fd4f0039b46bcfa18d973e8e70e455fda70", 
        "api_key": os.getenv("REPLICATE_API_KEY"),
    }
}, {
	"model_name": "mistral-7b-instruct",
	"litellm_params": { # params for litellm completion/embedding call 
        "model": "together_ai/mistralai/Mistral-7B-Instruct-v0.1", 
        "api_key": os.getenv("TOGETHERAI_API_KEY"),
    }
}, {
	"model_name": "mistral-7b-instruct",
	"litellm_params": { # params for litellm completion/embedding call 
        "model": "mistral-7b-instruct", 
        "api_base": "https://api.perplexity.ai",
        "api_key": os.getenv("PERPLEXITYAI_API_KEY")
    }
}, {
	"model_name": "mistral-7b-instruct",
	"litellm_params": {
		"model": "deepinfra/mistralai/Mistral-7B-Instruct-v0.1",
		"api_key": os.getenv("DEEPINFRA_API_KEY")
	}
}]

def test_multiple_deployments():
    try: 
        ## LiteLLM completion call ## returns first response 
        response = completion(model="mistral-7b-instruct", messages=messages, model_list=model_list)
        print(f"response: {response}")
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"An exception occurred: {e}")

test_multiple_deployments()