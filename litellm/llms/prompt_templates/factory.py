def default_pt(messages):
    return " ".join(message["content"] for message in messages)

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
        messages=messages
    )
    return prompt

# Falcon prompt template - from https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py#L110
def falcon_instruct_pt(messages):
    prompt = ""
    for message in messages:
        if message["role"] == "system":
            prompt += messages["content"]
        else:
            prompt += message['role']+":"+ message["content"].replace("\r\n", "\n").replace("\n\n", "\n")
            prompt += "\n\n"

def falcon_chat_pt(messages):
    prompt = ""
    for message in messages:
        if message["role"] == "system":
            prompt += "System: " + messages["content"]
        elif message["role"] == "assistant":
            prompt += "Falcon: " + message["content"]
        elif message["role"] == "user":
            prompt += "User: " + message["content"]


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

# Custom prompt template
def custom_prompt(role_dict: dict, messages: list, initial_prompt_value: str="", final_prompt_value: str=""):
    prompt = initial_prompt_value
    for message in messages:
        role = message["role"]
        pre_message_str = role_dict[role]["pre_message"] if role in role_dict and "pre_message" in role_dict[role] else "" 
        post_message_str = role_dict[role]["post_message"] if role in role_dict and "post_message" in role_dict[role] else "" 
        prompt += pre_message_str + message["content"] + post_message_str
    
    prompt += final_prompt_value
    return prompt

def prompt_factory(model: str, messages: list):
    model = model.lower()
    if "meta-llama/Llama-2" in model:
        if "chat" in model:
            return llama_2_chat_pt(messages=messages)
    elif "tiiuae/falcon" in model: # Note: for the instruct models, it's best to use a User: .., Assistant:.. approach in your prompt template.
        if model == "tiiuae/falcon-180B-chat":
            return falcon_chat_pt(messages=messages)
        elif "instruct" in model:
            return falcon_instruct_pt(messages=messages)
    elif "mosaicml/mpt" in model:
        if "chat" in model:
            return mpt_chat_pt(messages=messages)
    elif "codellama/codellama" in model:
        if "instruct" in model:
            return llama_2_chat_pt(messages=messages) # https://huggingface.co/blog/codellama#conversational-instructions
    elif "wizardlm/wizardcoder" in model:
        return wizardcoder_pt(messages=messages)
    elif "phind/phind-codellama" in model:
        return phind_codellama_pt(messages=messages)
    elif "togethercomputer/llama-2" in model and ("instruct" in model or "chat" in model):
        return llama_2_chat_pt(messages=messages)
    return default_pt(messages=messages) # default that covers Bloom, T-5, any non-chat tuned model (e.g. base Llama2)