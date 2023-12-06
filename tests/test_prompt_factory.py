#### What this tests ####
#    This tests if prompts are being correctly formatted
import sys
import os
import io

sys.path.insert(0, os.path.abspath('../..'))

# from litellm.llms.prompt_templates.factory import prompt_factory
from litellm import completion

def codellama_prompt_format():
    model = "huggingface/codellama/CodeLlama-7b-Instruct-hf"
    messages = [{"role": "system", "content": "You are a good bot"}, {"role": "user", "content": "Hey, how's it going?"}]
    expected_response = """[INST] <<SYS>>
You are a good bot
<</SYS>>
 [/INST]
[INST] Hey, how's it going? [/INST]"""
    response = completion(model=model, messages=messages)
    print(response)

# codellama_prompt_format()