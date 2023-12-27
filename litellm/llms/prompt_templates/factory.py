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
            "assistant": {"pre_message": "### Response:\n", "post_message": "\n\n"},
        },
        bos_token="<s>",
        eos_token="</s>",
        messages=messages,
    )
    return prompt


# Llama2 prompt template
def llama_2_chat_pt(messages):
    prompt = custom_prompt(
        role_dict={
            "system": {
                "pre_message": "[INST] <<SYS>>\n",
                "post_message": "\n<</SYS>>\n [/INST]\n",
            },
            "user": {  # follow this format https://github.com/facebookresearch/llama/blob/77062717054710e352a99add63d160274ce670c6/llama/generation.py#L348
                "pre_message": "[INST] ",
                "post_message": " [/INST]\n",
            },
            "assistant": {
                "post_message": "\n"  # follows this - https://replicate.com/blog/how-to-prompt-llama
            },
        },
        messages=messages,
        bos_token="<s>",
        eos_token="</s>",
    )
    return prompt


def ollama_pt(
    model, messages
):  # https://github.com/jmorganca/ollama/blob/af4cf55884ac54b9e637cd71dadfe9b7a5685877/docs/modelfile.md#template
    if "instruct" in model:
        prompt = custom_prompt(
            role_dict={
                "system": {"pre_message": "### System:\n", "post_message": "\n"},
                "user": {
                    "pre_message": "### User:\n",
                    "post_message": "\n",
                },
                "assistant": {
                    "pre_message": "### Response:\n",
                    "post_message": "\n",
                },
            },
            final_prompt_value="### Response:",
            messages=messages,
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
        return {"prompt": prompt, "images": images}
    else:
        prompt = "".join(
            m["content"]
            if isinstance(m["content"], str) is str
            else "".join(m["content"])
            for m in messages
        )
    return prompt


def mistral_instruct_pt(messages):
    prompt = custom_prompt(
        initial_prompt_value="<s>",
        role_dict={
            "system": {"pre_message": "[INST]", "post_message": "[/INST]"},
            "user": {"pre_message": "[INST]", "post_message": "[/INST]"},
            "assistant": {"pre_message": "[INST]", "post_message": "[/INST]"},
        },
        final_prompt_value="</s>",
        messages=messages,
    )
    return prompt


# Falcon prompt template - from https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py#L110
def falcon_instruct_pt(messages):
    prompt = ""
    for message in messages:
        if message["role"] == "system":
            prompt += message["content"]
        else:
            prompt += (
                message["role"]
                + ":"
                + message["content"].replace("\r\n", "\n").replace("\n\n", "\n")
            )
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
        elif message["role"] == "user":  # map to 'Instruction'
            prompt += "### Instruction:\n" + message["content"] + "\n\n"
        elif message["role"] == "assistant":  # map to 'Response'
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


def hf_chat_template(model: str, messages: list, chat_template: Optional[Any] = None):
    ## get the tokenizer config from huggingface
    bos_token = ""
    eos_token = ""
    if chat_template is None:

        def _get_tokenizer_config(hf_model_name):
            url = (
                f"https://huggingface.co/{hf_model_name}/raw/main/tokenizer_config.json"
            )
            # Make a GET request to fetch the JSON data
            response = requests.get(url)
            if response.status_code == 200:
                # Parse the JSON data
                tokenizer_config = json.loads(response.content)
                return {"status": "success", "tokenizer": tokenizer_config}
            else:
                return {"status": "failure"}

        tokenizer_config = _get_tokenizer_config(model)
        if (
            tokenizer_config["status"] == "failure"
            or "chat_template" not in tokenizer_config["tokenizer"]
        ):
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
    env.globals["raise_exception"] = raise_exception
    try:
        template = env.from_string(chat_template)
    except Exception as e:
        raise e

    def _is_system_in_template():
        try:
            # Try rendering the template with a system message
            response = template.render(
                messages=[{"role": "system", "content": "test"}],
                eos_token="<eos>",
                bos_token="<bos>",
            )
            return True

        # This will be raised if Jinja attempts to render the system message and it can't
        except:
            return False

    try:
        # Render the template with the provided values
        if _is_system_in_template():
            rendered_text = template.render(
                bos_token=bos_token, eos_token=eos_token, messages=messages
            )
        else:
            # treat a system message as a user message, if system not in template
            try:
                reformatted_messages = []
                for message in messages:
                    if message["role"] == "system":
                        reformatted_messages.append(
                            {"role": "user", "content": message["content"]}
                        )
                    else:
                        reformatted_messages.append(message)
                rendered_text = template.render(
                    bos_token=bos_token,
                    eos_token=eos_token,
                    messages=reformatted_messages,
                )
            except Exception as e:
                if "Conversation roles must alternate user/assistant" in str(e):
                    # reformat messages to ensure user/assistant are alternating, if there's either 2 consecutive 'user' messages or 2 consecutive 'assistant' message, add a blank 'user' or 'assistant' message to ensure compatibility
                    new_messages = []
                    for i in range(len(reformatted_messages) - 1):
                        new_messages.append(reformatted_messages[i])
                        if (
                            reformatted_messages[i]["role"]
                            == reformatted_messages[i + 1]["role"]
                        ):
                            if reformatted_messages[i]["role"] == "user":
                                new_messages.append(
                                    {"role": "assistant", "content": ""}
                                )
                            else:
                                new_messages.append({"role": "user", "content": ""})
                    new_messages.append(reformatted_messages[-1])
                    rendered_text = template.render(
                        bos_token=bos_token, eos_token=eos_token, messages=new_messages
                    )
        return rendered_text
    except Exception as e:
        raise Exception(f"Error rendering template - {str(e)}")


# Anthropic template
def claude_2_1_pt(
    messages: list,
):  # format - https://docs.anthropic.com/claude/docs/how-to-use-system-prompts
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
            prompt += f"{AnthropicConstants.HUMAN_PROMPT.value}{message['content']}"
        elif message["role"] == "system":
            prompt += f"{message['content']}"
        elif message["role"] == "assistant":
            if idx > 0 and messages[idx - 1]["role"] == "system":
                prompt += f"{AnthropicConstants.HUMAN_PROMPT.value}"  # Insert a blank human message
            prompt += f"{AnthropicConstants.AI_PROMPT.value}{message['content']}"
    prompt += f"{AnthropicConstants.AI_PROMPT.value}"  # prompt must end with \"\n\nAssistant: " turn
    return prompt


### TOGETHER AI


def get_model_info(token, model):
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get("https://api.together.xyz/models/info", headers=headers)
        if response.status_code == 200:
            model_info = response.json()
            for m in model_info:
                if m["name"].lower().strip() == model.strip():
                    return m["config"].get("prompt_format", None), m["config"].get(
                        "chat_template", None
                    )
            return None, None
        else:
            return None, None
    except Exception as e:  # safely fail a prompt template request
        return None, None


def format_prompt_togetherai(messages, prompt_format, chat_template):
    if prompt_format is None:
        return default_pt(messages)

    human_prompt, assistant_prompt = prompt_format.split("{prompt}")

    if chat_template is not None:
        prompt = hf_chat_template(
            model=None, messages=messages, chat_template=chat_template
        )
    elif prompt_format is not None:
        prompt = custom_prompt(
            role_dict={},
            messages=messages,
            initial_prompt_value=human_prompt,
            final_prompt_value=assistant_prompt,
        )
    else:
        prompt = default_pt(messages)
    return prompt


###


def anthropic_pt(
    messages: list,
):  # format - https://docs.anthropic.com/claude/reference/complete_post
    class AnthropicConstants(Enum):
        HUMAN_PROMPT = "\n\nHuman: "
        AI_PROMPT = "\n\nAssistant: "

    prompt = ""
    for idx, message in enumerate(
        messages
    ):  # needs to start with `\n\nHuman: ` and end with `\n\nAssistant: `
        if message["role"] == "user":
            prompt += f"{AnthropicConstants.HUMAN_PROMPT.value}{message['content']}"
        elif message["role"] == "system":
            prompt += f"{AnthropicConstants.HUMAN_PROMPT.value}<admin>{message['content']}</admin>"
        else:
            prompt += f"{AnthropicConstants.AI_PROMPT.value}{message['content']}"
        if (
            idx == 0 and message["role"] == "assistant"
        ):  # ensure the prompt always starts with `\n\nHuman: `
            prompt = f"{AnthropicConstants.HUMAN_PROMPT.value}" + prompt
    prompt += f"{AnthropicConstants.AI_PROMPT.value}"
    return prompt


def gemini_text_image_pt(messages: list):
    """
    {
        "contents":[
            {
            "parts":[
                {"text": "What is this picture?"},
                {
                "inline_data": {
                    "mime_type":"image/jpeg",
                    "data": "'$(base64 -w0 image.jpg)'"
                }
                }
            ]
            }
        ]
    }
    """
    try:
        import google.generativeai as genai
    except:
        raise Exception(
            "Importing google.generativeai failed, please run 'pip install -q google-generativeai"
        )

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

    content = [prompt] + images
    return content


import re, json
from guidance import system, user, assistant, models, select, gen
import guidance


def transform_function_list(functions):
    transformed = {}

    for function in functions:
        func_name = function['name']
        new_func = {
            "description": function['description']
        }

        # Check if parameters type is object
        if function['parameters'].get('type') == 'object':
            properties = function['parameters'].get('properties', {})
            required_params = function['parameters'].get('required', [])

            # Update properties to include 'required' field
            for prop_name, prop_details in properties.items():
                prop_details['required'] = prop_name in required_params

            new_func['properties'] = properties
        else:
            new_func['parameters'] = function['parameters']

        transformed[func_name] = new_func

    return transformed
  
  



@guidance
def function_call(lm,  messages, functions, patterns=False):
    functions = transform_function_list(functions)
    lm +=str(messages)
    func_names = []
    params={}
    with user():
        lm += "The following are the names and descriptions of available functions. Please choose the appropriate function based on previous chat history\n"
        
        for name, details in functions.items():
            func_names.append(name)
            description = details["description"]
            lm += f"Function Name: {name}\nDescription: {description}\n\n"
            
            lm += "None\n\n"
    # func_names.append("None")

    with assistant():
        lm += f'''\
            Now I will choose from one of following options:
            Choice: {select(func_names, name='function_choice')}
        \n\n
        '''
    func_name = lm['function_choice']
    # print(str(functions[func_name]['properties']))

    args =json.loads(json.dumps(functions[func_name]['properties']))
    
    # Starting the prompt engineering section
    desc=functions[func_name]["description"]

    
    usr_msg= f'''\
        Based on this function name and description, generate the appropirate arguments
        Function Name: {func_name}\nDescription: {desc}
        \n
        ''' 
    for arg in args:

        desc = args[arg].get("description", "No description provided")

        if args[arg].get("required") == False:
            with user():
                lm += usr_msg +f'''\
                Note that this argument is not necessary for parsing and execution however it 
                may be helpful to include, please make a decision to include this argument or not
                Argument: {arg}
                description: {desc}
                ''' 
            with assistant():
                lm += f'''\
                Understood I will only generate either to include or not_include this parameter based on the messsages
                Messages:{str(messages)}
                {arg}:{select(name=arg, options=["include","not_include"])}
                and here is my justification for my decision:
                {gen(name="just", max_tokens=50)}
                '''
            
            assert lm[arg] in ["include","not_include"]
            
            if lm[arg] == 'not_include':
                
                continue
        
        if "enum" in args[arg]:
            with user():
                lm += usr_msg +f'''\
                Now please generate the arguments for this function:
                ensuring that the argument is one of the following:
                Argument: {arg}
                description: {desc}
                ''' 
            with assistant():
                lm += f'''\
                Understood I will only generate an option within the provided list
                {arg}:{select(name=arg, options=args[arg]["enum"])}
                '''
            assert lm[arg] in args[arg]["enum"]
            params[arg]=lm[arg]

        elif args[arg]["type"] == "string":
            with user():
                lm += usr_msg + f'''\
                Please ensure that your output is only the relevant string and not 
                any does not contain any unecessary characters or new lines
                Argument Name: {arg}
                description: {desc}
                type: string
                '''
            with assistant():
                lm += f'''\
                Understood I will only generate a string that best matches the provided argument
                {arg}:{gen(name=arg, max_tokens=30)}
                '''
                resp=(lm[arg].rsplit('\n', 1)[0]).strip()
                resp =resp.replace("'", "").replace('"', "")
            params[arg]=resp

        elif args[arg]["type"] == "integer":
            convert_to_int = lambda s: int(''.join(re.findall(r'\d', s))) if re.search(r'\d', s) else 0
            if patterns == True:
                with user():
                    lm += usr_msg + f'''\
                    Now please generate the arguments for this function:
                    ensuring that the argument is one of the following:
                    Argument: {arg}
                    description: {desc}
                    type: int
                    ''' 
                with assistant():
                    lm += f'''\
                    Understood I will only generate an option which is a valid integer
                    {arg}:{gen(name=arg,regex="[0-9]+")}
                    '''
                num = convert_to_int(lm[arg])
            else:
                with user():
                    lm += usr_msg +f'''\
                    Now please generate the arguments for this function:
                    ensuring that the argument is one of the following:
                    Argument: {arg}
                    description: {desc}
                    type: int
                    ''' 
                with assistant():
                    lm += f'''\
                    Understood I will only generate an option which is a valid integer
                    {arg}:{gen(name=arg, max_tokens=10)}
                    '''
            num = convert_to_int((lm[arg].rsplit('\n', 1)[0]))
            
            assert isinstance(num, int)
            params[arg]=num

        elif args[arg]["type"] == "float":

            if patterns == True:
                with assistant():
                    lm += f'''\
                    Now I will generate the arguments for this function,
                    ENSURING THAT THE ARGUMENT IS A FLOAT:
                    Argument: {arg}
                    description: {desc}
                    {arg}:{gen(name=arg,regex="[0-9]+")}       
                    '''
                assert isinstance(float(lm[arg]), float)

            else:
                with assistant():
                    lm += f'''\
                    Now I will generate the arguments for this function,
                    ENSURING THAT THE ARGUMENT IS A FLOAT:
                    Argument: {arg}
                    description: {desc}
                    {arg}:{gen(name=arg)}        
                    '''
                assert isinstance(float(lm[arg]), float)
            params[arg]=float(lm[arg])
            with user():
                lm += "good now please generate the next argument"


        elif args[arg]["type"] == "boolean":
            with user():
                lm += f'''\
                Now please generate the arguments for this function:
                ensuring that the argument which is a boolean
                Argument: {arg}
                description: {desc}
                type: boolean
                ''' 

            with assistant():
                lm += f'''\
                Now I will generate the arguments for this function:
                Understood I will only generate an option which is either true or false
                Argument: {arg}
                description: {desc}
                Choice: {select(name=arg, options=["true","false"])}
                '''
            choice =bool(lm[arg])
            assert isinstance(choice, bool)
            params[arg]=choice

              

    results ={"function_call":{"name": func_name, "arguments": params}}

    lm = lm.set('return_value', results)

    # Return the updated lm
    return lm



# Function call template
def function_call_prompt(messages: list, functions: list, model_response):


    from litellm.utils import Choices, Message, FunctionCall, Usage
    from datetime import datetime
    import time

    llm =guidance.models.MistralChat("./models/mixtral-8x7b-instruct-v0.1.Q4_0.gguf",echo=False, n_ctx=4096)

    lm =llm+function_call(messages,functions ,patterns=True)
    results =(lm['return_value'])


    model_response.id = "chatcmpl-8aTreRFE763VO0d88d4kUBrqea1tz" #todo
    current_unix_timestamp = int(time.time())


    function_call_arguments = results['function_call']['arguments']
    function_call_name = results['function_call']['name']

    # Serialize arguments to a JSON string if it is a dictionary
    if isinstance(function_call_arguments, dict):
        function_call_arguments = json.dumps(function_call_arguments)

    # Ensure arguments and name are passed as strings to FunctionCall
    model_response.choices = [
        Choices(
            finish_reason='function_call',
            index=0,
            message=Message(
                content=None, 
                role='assistant',
                function_call={
                    'arguments': function_call_arguments,
                    'name': function_call_name
                }
            ),
            created=current_unix_timestamp,
            model='random_text', #todo
            object='chat.completion',
            system_fingerprint=None,
            usage=Usage( #todo
                completion_tokens=18,
                prompt_tokens=82,
                total_tokens=100
            )
        )
    ]

# Custom prompt template
def custom_prompt(
    role_dict: dict,
    messages: list,
    initial_prompt_value: str = "",
    final_prompt_value: str = "",
    bos_token: str = "",
    eos_token: str = "",
):
    prompt = bos_token + initial_prompt_value
    bos_open = True
    ## a bos token is at the start of a system / human message
    ## an eos token is at the end of the assistant response to the message
    for message in messages:
        role = message["role"]

        if role in ["system", "human"] and not bos_open:
            prompt += bos_token
            bos_open = True

        pre_message_str = (
            role_dict[role]["pre_message"]
            if role in role_dict and "pre_message" in role_dict[role]
            else ""
        )
        post_message_str = (
            role_dict[role]["post_message"]
            if role in role_dict and "post_message" in role_dict[role]
            else ""
        )
        prompt += pre_message_str + message["content"] + post_message_str

        if role == "assistant":
            prompt += eos_token
            bos_open = False

    prompt += final_prompt_value
    return prompt


def prompt_factory(
    model: str,
    messages: list,
    custom_llm_provider: Optional[str] = None,
    api_key: Optional[str] = None,
):
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
        return format_prompt_togetherai(
            messages=messages, prompt_format=prompt_format, chat_template=chat_template
        )
    elif custom_llm_provider == "gemini":
        return gemini_text_image_pt(messages=messages)
    try:
        if "meta-llama/llama-2" in model and "chat" in model:
            return llama_2_chat_pt(messages=messages)
        elif (
            "tiiuae/falcon" in model
        ):  # Note: for the instruct models, it's best to use a User: .., Assistant:.. approach in your prompt template.
            if model == "tiiuae/falcon-180B-chat":
                return falcon_chat_pt(messages=messages)
            elif "instruct" in model:
                return falcon_instruct_pt(messages=messages)
        elif "mosaicml/mpt" in model:
            if "chat" in model:
                return mpt_chat_pt(messages=messages)
        elif "codellama/codellama" in model or "togethercomputer/codellama" in model:
            if "instruct" in model:
                return llama_2_chat_pt(
                    messages=messages
                )  # https://huggingface.co/blog/codellama#conversational-instructions
        elif "wizardlm/wizardcoder" in model:
            return wizardcoder_pt(messages=messages)
        elif "phind/phind-codellama" in model:
            return phind_codellama_pt(messages=messages)
        elif "togethercomputer/llama-2" in model and (
            "instruct" in model or "chat" in model
        ):
            return llama_2_chat_pt(messages=messages)
        elif model in [
            "gryphe/mythomax-l2-13b",
            "gryphe/mythomix-l2-13b",
            "gryphe/mythologic-l2-13b",
        ]:
            return alpaca_pt(messages=messages)
        else:
            return hf_chat_template(original_model_name, messages)
    except Exception as e:
        return default_pt(
            messages=messages
        )  # default that covers Bloom, T-5, any non-chat tuned model (e.g. base Llama2)
