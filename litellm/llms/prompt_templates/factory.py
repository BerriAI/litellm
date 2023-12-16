from enum import Enum
import requests, traceback
import json
from jinja2 import Template, exceptions, Environment, meta
from typing import Optional, Any

def default_pt(messages):
    return " ".join(message["content"] for message in messages)

# alpaca prompt template - for models like mythomax, etc. 
def alpaca_pt(messages): 
    prompt = custom_prompt(
        role_dict={
            "system": {
                "pre_message": "### Instruction:\n",
                "post_message": "\n\n",
            },
            "user": {
                "pre_message": "### Instruction:\n",
                "post_message": "\n\n",
            },
            "assistant": {
                "pre_message": "### Response:\n",
                "post_message": "\n\n"
            }
        },
        bos_token="<s>",
        eos_token="</s>",
        messages=messages
    )
    return prompt

# Llama2 prompt template
def llama_2_chat_pt(messages):
    prompt = custom_prompt(
        role_dict={
            "system": {
                "pre_message": "[INST] <<SYS>>\n",
                "post_message": "\n<</SYS>>\n [/INST]\n"
            },
            "user": { # follow this format https://github.com/facebookresearch/llama/blob/77062717054710e352a99add63d160274ce670c6/llama/generation.py#L348
                "pre_message": "[INST] ",
                "post_message": " [/INST]\n"
            }, 
            "assistant": {
                "post_message": "\n" # follows this - https://replicate.com/blog/how-to-prompt-llama
            }
        },
        messages=messages,
        bos_token="<s>",
        eos_token="</s>"
    )
    return prompt

def ollama_pt(model, messages): # https://github.com/jmorganca/ollama/blob/af4cf55884ac54b9e637cd71dadfe9b7a5685877/docs/modelfile.md#template
    
    if "instruct" in model: 
        prompt = custom_prompt(
            role_dict={
                "system": {
                    "pre_message": "### System:\n",
                    "post_message": "\n"
                }, 
                "user": {
                    "pre_message": "### User:\n",
                    "post_message": "\n",
                }, 
                "assistant": {
                    "pre_message": "### Response:\n",
                    "post_message": "\n",
                }
            },
            final_prompt_value="### Response:",
            messages=messages
        )
    elif "llava" in model:
        prompt = ""
        images = []
        for message in messages:
            if isinstance(message["content"], str):
                prompt += message["content"]
            elif isinstance(message["content"], list):
                # see https://docs.litellm.ai/docs/providers/openai#openai-vision-models
                for element in message["content"]:
                    if isinstance(element, dict):
                        if element["type"] == "text":
                            prompt += element["text"]
                        elif element["type"] == "image_url":
                            image_url = element["image_url"]["url"]
                            images.append(image_url)
        return {
            "prompt": prompt,
            "images": images
        }
    else: 
        prompt = "".join(m["content"] if isinstance(m['content'], str) is str else "".join(m['content']) for m in messages)
    return prompt

def mistral_instruct_pt(messages): 
    prompt = custom_prompt(
        initial_prompt_value="<s>",
        role_dict={
            "system": {
                "pre_message": "[INST]",
                "post_message": "[/INST]"
            }, 
            "user": {
                "pre_message": "[INST]", 
                "post_message": "[/INST]"
            },
            "assistant": {
                "pre_message": "[INST]",
                "post_message": "[/INST]"
            }
        },
        final_prompt_value="</s>",
        messages=messages
    )
    return prompt

# Falcon prompt template - from https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py#L110
def falcon_instruct_pt(messages):
    prompt = ""
    for message in messages:
        if message["role"] == "system":
            prompt += message["content"]
        else:
            prompt += message['role']+":"+ message["content"].replace("\r\n", "\n").replace("\n\n", "\n")
            prompt += "\n\n"
    
    return prompt

def falcon_chat_pt(messages):
    prompt = ""
    for message in messages:
        if message["role"] == "system":
            prompt += "System: " + message["content"]
        elif message["role"] == "assistant":
            prompt += "Falcon: " + message["content"]
        elif message["role"] == "user":
            prompt += "User: " + message["content"]

    return prompt

# MPT prompt template - from https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py#L110
def mpt_chat_pt(messages):
    prompt = ""
    for message in messages:
        if message["role"] == "system":
            prompt += "<|im_start|>system" + message["content"] + "<|im_end|>" + "\n"
        elif message["role"] == "assistant":
            prompt += "<|im_start|>assistant" + message["content"] + "<|im_end|>" + "\n"
        elif message["role"] == "user":
            prompt += "<|im_start|>user" + message["content"] + "<|im_end|>" + "\n"
    return prompt

# WizardCoder prompt template - https://huggingface.co/WizardLM/WizardCoder-Python-34B-V1.0#prompt-format
def wizardcoder_pt(messages):
    prompt = ""
    for message in messages:
        if message["role"] == "system":
            prompt += message["content"] + "\n\n"
        elif message["role"] == "user": # map to 'Instruction'
            prompt += "### Instruction:\n" + message["content"] + "\n\n"
        elif message["role"] == "assistant": # map to 'Response'
            prompt += "### Response:\n" + message["content"] + "\n\n"
    return prompt
    
# Phind-CodeLlama prompt template - https://huggingface.co/Phind/Phind-CodeLlama-34B-v2#how-to-prompt-the-model
def phind_codellama_pt(messages):
    prompt = ""
    for message in messages:
        if message["role"] == "system":
            prompt += "### System Prompt\n" + message["content"] + "\n\n"
        elif message["role"] == "user":
            prompt += "### User Message\n" + message["content"] + "\n\n"
        elif message["role"] == "assistant":
            prompt += "### Assistant\n" + message["content"] + "\n\n"
    return prompt

def hf_chat_template(model: str, messages: list, chat_template: Optional[Any]=None):
    ## get the tokenizer config from huggingface
    bos_token = ""
    eos_token = ""
    if chat_template is None: 
        def _get_tokenizer_config(hf_model_name):
            url = f"https://huggingface.co/{hf_model_name}/raw/main/tokenizer_config.json"
            # Make a GET request to fetch the JSON data
            response = requests.get(url)
            if response.status_code == 200:
                # Parse the JSON data
                tokenizer_config = json.loads(response.content)
                return {"status": "success", "tokenizer": tokenizer_config}
            else:
                return {"status": "failure"}
        tokenizer_config = _get_tokenizer_config(model)
        if tokenizer_config["status"] == "failure" or "chat_template" not in tokenizer_config["tokenizer"]:
            raise Exception("No chat template found")
        ## read the bos token, eos token and chat template from the json 
        tokenizer_config = tokenizer_config["tokenizer"]
        bos_token = tokenizer_config["bos_token"]
        eos_token = tokenizer_config["eos_token"]
        chat_template = tokenizer_config["chat_template"]

    def raise_exception(message):
        raise Exception(f"Error message - {message}")
    
    # Create a template object from the template text
    env = Environment()
    env.globals['raise_exception'] = raise_exception
    try:
        template = env.from_string(chat_template)
    except Exception as e:
        raise e

    def _is_system_in_template():
        try:
            # Try rendering the template with a system message
            response = template.render(messages=[{"role": "system", "content": "test"}], eos_token= "<eos>", bos_token= "<bos>")
            return True

        # This will be raised if Jinja attempts to render the system message and it can't
        except:
            return False
        
    try: 
        # Render the template with the provided values
        if _is_system_in_template(): 
            rendered_text = template.render(bos_token=bos_token, eos_token=eos_token, messages=messages)
        else: 
            # treat a system message as a user message, if system not in template
            try:
                reformatted_messages = []
                for message in messages: 
                    if message["role"] == "system": 
                        reformatted_messages.append({"role": "user", "content": message["content"]})
                    else:
                        reformatted_messages.append(message)
                rendered_text = template.render(bos_token=bos_token, eos_token=eos_token, messages=reformatted_messages)
            except Exception as e:
                if "Conversation roles must alternate user/assistant" in str(e): 
                    # reformat messages to ensure user/assistant are alternating, if there's either 2 consecutive 'user' messages or 2 consecutive 'assistant' message, add a blank 'user' or 'assistant' message to ensure compatibility
                    new_messages = []
                    for i in range(len(reformatted_messages)-1): 
                        new_messages.append(reformatted_messages[i])
                        if reformatted_messages[i]["role"] == reformatted_messages[i+1]["role"]:
                            if reformatted_messages[i]["role"] == "user":
                                new_messages.append({"role": "assistant", "content": ""})
                            else:
                                new_messages.append({"role": "user", "content": ""})
                    new_messages.append(reformatted_messages[-1])
                    rendered_text = template.render(bos_token=bos_token, eos_token=eos_token, messages=new_messages)
        return rendered_text
    except Exception as e: 
        raise Exception(f"Error rendering template - {str(e)}")

# Anthropic template 
def claude_2_1_pt(messages: list): # format - https://docs.anthropic.com/claude/docs/how-to-use-system-prompts
    """
    Claude v2.1 allows system prompts (no Human: needed), but requires it be followed by Human: 
    - you can't just pass a system message
    - you can't pass a system message and follow that with an assistant message 
    if system message is passed in, you can only do system, human, assistant or system, human

    if a system message is passed in and followed by an assistant message, insert a blank human message between them. 
    """
    class AnthropicConstants(Enum):
        HUMAN_PROMPT = "\n\nHuman: "
        AI_PROMPT = "\n\nAssistant: "
    
    prompt = "" 
    for idx, message in enumerate(messages): 
        if message["role"] == "user":
            prompt += (
                f"{AnthropicConstants.HUMAN_PROMPT.value}{message['content']}"
            )
        elif message["role"] == "system":
            prompt += (
                f"{message['content']}"
            )
        elif message["role"] == "assistant":
            if idx > 0 and messages[idx - 1]["role"] == "system": 
                prompt += f"{AnthropicConstants.HUMAN_PROMPT.value}" # Insert a blank human message
            prompt += (
                f"{AnthropicConstants.AI_PROMPT.value}{message['content']}"
            )
    prompt += f"{AnthropicConstants.AI_PROMPT.value}" # prompt must end with \"\n\nAssistant: " turn
    return prompt

### TOGETHER AI 

def get_model_info(token, model):
    try: 
        headers = {
            'Authorization': f'Bearer {token}'
        }
        response = requests.get('https://api.together.xyz/models/info', headers=headers)
        if response.status_code == 200:
            model_info = response.json()
            for m in model_info: 
                if m["name"].lower().strip() == model.strip(): 
                    return m['config'].get('prompt_format', None), m['config'].get('chat_template', None)
            return None, None
        else:
            return None, None
    except Exception as e: # safely fail a prompt template request
        return None, None

def format_prompt_togetherai(messages, prompt_format, chat_template):
    if prompt_format is None:
        return default_pt(messages)
    
    human_prompt, assistant_prompt = prompt_format.split('{prompt}')

    if chat_template is not None:
        prompt = hf_chat_template(model=None, messages=messages, chat_template=chat_template)
    elif prompt_format is not None: 
        prompt = custom_prompt(role_dict={}, messages=messages, initial_prompt_value=human_prompt, final_prompt_value=assistant_prompt)
    else: 
        prompt = default_pt(messages)
    return prompt 

###

def anthropic_pt(messages: list): # format - https://docs.anthropic.com/claude/reference/complete_post
    class AnthropicConstants(Enum):
        HUMAN_PROMPT = "\n\nHuman: "
        AI_PROMPT = "\n\nAssistant: "
    
    prompt = "" 
    for idx, message in enumerate(messages): # needs to start with `\n\nHuman: ` and end with `\n\nAssistant: `
        if message["role"] == "user":
            prompt += (
                f"{AnthropicConstants.HUMAN_PROMPT.value}{message['content']}"
            )
        elif message["role"] == "system":
            prompt += (
                f"{AnthropicConstants.HUMAN_PROMPT.value}<admin>{message['content']}</admin>"
            )
        else:
            prompt += (
                f"{AnthropicConstants.AI_PROMPT.value}{message['content']}"
            )
        if idx == 0 and message["role"] == "assistant": # ensure the prompt always starts with `\n\nHuman: `
            prompt = f"{AnthropicConstants.HUMAN_PROMPT.value}" + prompt
    prompt += f"{AnthropicConstants.AI_PROMPT.value}"
    return prompt 

# Function call template 
def function_call_prompt(messages: list, functions: list):
    function_prompt = "The following functions are available to you:"
    for function in functions: 
        function_prompt += f"""\n{function}\n"""
    
    function_added_to_prompt = False
    for message in messages: 
        if "system" in message["role"]: 
            message['content'] += f"""{function_prompt}"""
            function_added_to_prompt = True
    
    if function_added_to_prompt == False: 
        messages.append({'role': 'system', 'content': f"""{function_prompt}"""})

    return messages


# Custom prompt template
def custom_prompt(role_dict: dict, messages: list, initial_prompt_value: str="", final_prompt_value: str="", bos_token: str="", eos_token: str=""):
    prompt = bos_token + initial_prompt_value
    bos_open = True
    ## a bos token is at the start of a system / human message
    ## an eos token is at the end of the assistant response to the message
    for message in messages:
        role = message["role"]
        
        if role in ["system", "human"] and not bos_open:
            prompt += bos_token
            bos_open = True
        
        pre_message_str = role_dict[role]["pre_message"] if role in role_dict and "pre_message" in role_dict[role] else "" 
        post_message_str = role_dict[role]["post_message"] if role in role_dict and "post_message" in role_dict[role] else "" 
        prompt += pre_message_str + message["content"] + post_message_str
        
        if role == "assistant":
            prompt += eos_token
            bos_open = False

    prompt += final_prompt_value
    return prompt

def prompt_factory(model: str, messages: list, custom_llm_provider: Optional[str]=None, api_key: Optional[str]=None):
    original_model_name = model
    model = model.lower()
    if custom_llm_provider == "ollama": 
        return ollama_pt(model=model, messages=messages)
    elif custom_llm_provider == "anthropic":
        if "claude-2.1" in model: 
            return claude_2_1_pt(messages=messages)
        else: 
            return anthropic_pt(messages=messages)
    elif custom_llm_provider == "together_ai": 
        prompt_format, chat_template = get_model_info(token=api_key, model=model)
        return format_prompt_togetherai(messages=messages, prompt_format=prompt_format, chat_template=chat_template)
    try:
        if "meta-llama/llama-2" in model and "chat" in model:
            return llama_2_chat_pt(messages=messages)
        elif "tiiuae/falcon" in model: # Note: for the instruct models, it's best to use a User: .., Assistant:.. approach in your prompt template.
            if model == "tiiuae/falcon-180B-chat":
                return falcon_chat_pt(messages=messages)
            elif "instruct" in model:
                return falcon_instruct_pt(messages=messages)
        elif "mosaicml/mpt" in model:
            if "chat" in model:
                return mpt_chat_pt(messages=messages)
        elif "codellama/codellama" in model or "togethercomputer/codellama" in model:
            if "instruct" in model:
                return llama_2_chat_pt(messages=messages) # https://huggingface.co/blog/codellama#conversational-instructions
        elif "wizardlm/wizardcoder" in model:
            return wizardcoder_pt(messages=messages)
        elif "phind/phind-codellama" in model:
            return phind_codellama_pt(messages=messages)
        elif "togethercomputer/llama-2" in model and ("instruct" in model or "chat" in model):
            return llama_2_chat_pt(messages=messages)
        elif model in ["gryphe/mythomax-l2-13b", "gryphe/mythomix-l2-13b", "gryphe/mythologic-l2-13b"]:
            return alpaca_pt(messages=messages) 
        else: 
            return hf_chat_template(original_model_name, messages)
    except:
        return default_pt(messages=messages) # default that covers Bloom, T-5, any non-chat tuned model (e.g. base Llama2)
    
