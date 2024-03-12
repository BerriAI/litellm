from enum import Enum
import requests, traceback
import json, re, xml.etree.ElementTree as ET
from jinja2 import Template, exceptions, Environment, meta
from typing import Optional, Any
import imghdr, base64
from typing import List


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
            (
                m["content"]
                if isinstance(m["content"], str) is str
                else "".join(m["content"])
            )
            for m in messages
        )
    return prompt


def mistral_instruct_pt(messages):
    # Following the Mistral example's https://huggingface.co/docs/transformers/main/chat_templating
    prompt = custom_prompt(
        initial_prompt_value="<s>",
        role_dict={
            "system": {
                "pre_message": "[INST] \n",
                "post_message": " [/INST]\n",
            },
            "user": {"pre_message": "[INST] ", "post_message": " [/INST]\n"},
            "assistant": {"pre_message": " ", "post_message": "</s> "},
        },
        final_prompt_value="",
        messages=messages,
    )
    return prompt


def mistral_api_pt(messages):
    """
    - handles scenario where content is list and not string
    - content list is just text, and no images
    - if image passed in, then just return as is (user-intended)

    Motivation: mistral api doesn't support content as a list
    """
    new_messages = []
    for m in messages:
        texts = ""
        if isinstance(m["content"], list):
            for c in m["content"]:
                if c["type"] == "image_url":
                    return messages
                elif c["type"] == "text" and isinstance(c["text"], str):
                    texts += c["text"]
        new_m = {"role": m["role"], "content": texts}
        new_messages.append(new_m)
    return new_messages


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

    Additionally, you can "put words in Claude's mouth" by ending with an assistant message.
    See: https://docs.anthropic.com/claude/docs/put-words-in-claudes-mouth
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
    if messages[-1]["role"] != "assistant":
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


### ANTHROPIC ###


def anthropic_pt(
    messages: list,
):  # format - https://docs.anthropic.com/claude/reference/complete_post
    """
    You can "put words in Claude's mouth" by ending with an assistant message.
    See: https://docs.anthropic.com/claude/docs/put-words-in-claudes-mouth
    """

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
    if messages[-1]["role"] != "assistant":
        prompt += f"{AnthropicConstants.AI_PROMPT.value}"
    return prompt


def construct_format_parameters_prompt(parameters: dict):
    parameter_str = "<parameter>\n"
    for k, v in parameters.items():
        parameter_str += f"<{k}>"
        parameter_str += f"{v}"
        parameter_str += f"</{k}>"
    parameter_str += "\n</parameter>"
    return parameter_str


def construct_format_tool_for_claude_prompt(name, description, parameters):
    constructed_prompt = (
        "<tool_description>\n"
        f"<tool_name>{name}</tool_name>\n"
        "<description>\n"
        f"{description}\n"
        "</description>\n"
        "<parameters>\n"
        f"{construct_format_parameters_prompt(parameters)}\n"
        "</parameters>\n"
        "</tool_description>"
    )
    return constructed_prompt


def construct_tool_use_system_prompt(
    tools,
):  # from https://github.com/anthropics/anthropic-cookbook/blob/main/function_calling/function_calling.ipynb
    tool_str_list = []
    for tool in tools:
        tool_str = construct_format_tool_for_claude_prompt(
            tool["function"]["name"],
            tool["function"].get("description", ""),
            tool["function"].get("parameters", {}),
        )
        tool_str_list.append(tool_str)
    tool_use_system_prompt = (
        "In this environment you have access to a set of tools you can use to answer the user's question.\n"
        "\n"
        "You may call them like this:\n"
        "<function_calls>\n"
        "<invoke>\n"
        "<tool_name>$TOOL_NAME</tool_name>\n"
        "<parameters>\n"
        "<$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>\n"
        "...\n"
        "</parameters>\n"
        "</invoke>\n"
        "</function_calls>\n"
        "\n"
        "Here are the tools available:\n"
        "<tools>\n" + "\n".join([tool_str for tool_str in tool_str_list]) + "\n</tools>"
    )
    return tool_use_system_prompt


def convert_url_to_base64(url):
    import requests
    import base64

    for _ in range(3):
        try:
            response = requests.get(url)
            break
        except:
            pass
    if response.status_code == 200:
        image_bytes = response.content
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        img_type = url.split(".")[-1].lower()
        if img_type == "jpg" or img_type == "jpeg":
            img_type = "image/jpeg"
        elif img_type == "png":
            img_type = "image/png"
        elif img_type == "gif":
            img_type = "image/gif"
        elif img_type == "webp":
            img_type = "image/webp"
        else:
            raise Exception(
                f"Error: Unsupported image format. Format={img_type}. Supported types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']"
            )

        return f"data:{img_type};base64,{base64_image}"
    else:
        raise Exception(f"Error: Unable to fetch image from URL. url={url}")


def convert_to_anthropic_image_obj(openai_image_url: str):
    """
    Input:
    "image_url": "data:image/jpeg;base64,{base64_image}",

    Return:
    "source": {
      "type": "base64",
      "media_type": "image/jpeg",
      "data": {base64_image},
    }
    """
    try:
        if openai_image_url.startswith("http"):
            openai_image_url = convert_url_to_base64(url=openai_image_url)
        # Extract the base64 image data
        base64_data = openai_image_url.split("data:image/")[1].split(";base64,")[1]

        # Infer image format from the URL
        image_format = openai_image_url.split("data:image/")[1].split(";base64,")[0]

        return {
            "type": "base64",
            "media_type": f"image/{image_format}",
            "data": base64_data,
        }
    except Exception as e:
        if "Error: Unable to fetch image from URL" in str(e):
            raise e
        raise Exception(
            """Image url not in expected format. Example Expected input - "image_url": "data:image/jpeg;base64,{base64_image}". Supported formats - ['image/jpeg', 'image/png', 'image/gif', 'image/webp'] """
        )


def anthropic_messages_pt(messages: list):
    """
    format messages for anthropic
    1. Anthropic supports roles like "user" and "assistant", (here litellm translates system-> assistant)
    2. The first message always needs to be of role "user"
    3. Each message must alternate between "user" and "assistant" (this is not addressed as now by litellm)
    4. final assistant content cannot end with trailing whitespace (anthropic raises an error otherwise)
    5. System messages are a separate param to the Messages API (used for tool calling)
    6. Ensure we only accept role, content. (message.name is not supported)
    """
    ## Ensure final assistant message has no trailing whitespace
    last_assistant_message_idx: Optional[int] = None
    # reformat messages to ensure user/assistant are alternating, if there's either 2 consecutive 'user' messages or 2 consecutive 'assistant' message, add a blank 'user' or 'assistant' message to ensure compatibility
    new_messages = []
    if len(messages) == 1:
        # check if the message is a user message
        if messages[0]["role"] == "assistant":
            new_messages.append({"role": "user", "content": ""})

        # check if content is a list (vision)
        if isinstance(messages[0]["content"], list):  # vision input
            new_content = []
            for m in messages[0]["content"]:
                if m.get("type", "") == "image_url":
                    new_content.append(
                        {
                            "type": "image",
                            "source": convert_to_anthropic_image_obj(
                                m["image_url"]["url"]
                            ),
                        }
                    )
                elif m.get("type", "") == "text":
                    new_content.append({"type": "text", "text": m["text"]})
            new_messages.append({"role": messages[0]["role"], "content": new_content})  # type: ignore
        else:
            new_messages.append(
                {"role": messages[0]["role"], "content": messages[0]["content"]}
            )

        return new_messages

    for i in range(len(messages) - 1):  # type: ignore
        if i == 0 and messages[i]["role"] == "assistant":
            new_messages.append({"role": "user", "content": ""})
        if isinstance(messages[i]["content"], list):  # vision input
            new_content = []
            for m in messages[i]["content"]:
                if m.get("type", "") == "image_url":
                    new_content.append(
                        {
                            "type": "image",
                            "source": convert_to_anthropic_image_obj(
                                m["image_url"]["url"]
                            ),
                        }
                    )
                elif m.get("type", "") == "text":
                    new_content.append({"type": "text", "content": m["text"]})
            new_messages.append({"role": messages[i]["role"], "content": new_content})  # type: ignore
        else:
            new_messages.append(
                {"role": messages[i]["role"], "content": messages[i]["content"]}
            )

        if messages[i]["role"] == messages[i + 1]["role"]:
            if messages[i]["role"] == "user":
                new_messages.append({"role": "assistant", "content": ""})
            else:
                new_messages.append({"role": "user", "content": ""})

        if messages[i]["role"] == "assistant":
            last_assistant_message_idx = i

    new_messages.append(messages[-1])
    if last_assistant_message_idx is not None:
        new_messages[last_assistant_message_idx]["content"] = new_messages[
            last_assistant_message_idx
        ][
            "content"
        ].strip()  # no trailing whitespace for final assistant message

    return new_messages


def extract_between_tags(tag: str, string: str, strip: bool = False) -> List[str]:
    ext_list = re.findall(f"<{tag}>(.+?)</{tag}>", string, re.DOTALL)
    if strip:
        ext_list = [e.strip() for e in ext_list]
    return ext_list


def parse_xml_params(xml_content):
    root = ET.fromstring(xml_content)
    params = {}
    for child in root.findall(".//parameters/*"):
        params[child.tag] = child.text
    return params


###


def amazon_titan_pt(
    messages: list,
):  # format - https://github.com/BerriAI/litellm/issues/1896
    """
    Amazon Titan uses 'User:' and 'Bot: in it's prompt template
    """

    class AmazonTitanConstants(Enum):
        HUMAN_PROMPT = "\n\nUser: "  # Assuming this is similar to Anthropic prompt formatting, since amazon titan's prompt formatting is currently undocumented
        AI_PROMPT = "\n\nBot: "

    prompt = ""
    for idx, message in enumerate(messages):
        if message["role"] == "user":
            prompt += f"{AmazonTitanConstants.HUMAN_PROMPT.value}{message['content']}"
        elif message["role"] == "system":
            prompt += f"{AmazonTitanConstants.HUMAN_PROMPT.value}<admin>{message['content']}</admin>"
        else:
            prompt += f"{AmazonTitanConstants.AI_PROMPT.value}{message['content']}"
        if (
            idx == 0 and message["role"] == "assistant"
        ):  # ensure the prompt always starts with `\n\nHuman: `
            prompt = f"{AmazonTitanConstants.HUMAN_PROMPT.value}" + prompt
    if messages[-1]["role"] != "assistant":
        prompt += f"{AmazonTitanConstants.AI_PROMPT.value}"
    return prompt


def _load_image_from_url(image_url):
    try:
        from PIL import Image
    except:
        raise Exception(
            "gemini image conversion failed please run `pip install Pillow`"
        )
    from io import BytesIO

    try:
        # Send a GET request to the image URL
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Check the response's content type to ensure it is an image
        content_type = response.headers.get("content-type")
        if not content_type or "image" not in content_type:
            raise ValueError(
                f"URL does not point to a valid image (content-type: {content_type})"
            )

        # Load the image from the response content
        return Image.open(BytesIO(response.content))

    except requests.RequestException as e:
        raise Exception(f"Request failed: {e}")
    except Exception as e:
        raise e


def _gemini_vision_convert_messages(messages: list):
    """
    Converts given messages for GPT-4 Vision to Gemini format.

    Args:
        messages (list): The messages to convert. Each message can be a dictionary with a "content" key. The content can be a string or a list of elements. If it is a string, it will be concatenated to the prompt. If it is a list, each element will be processed based on its type:
            - If the element is a dictionary with a "type" key equal to "text", its "text" value will be concatenated to the prompt.
            - If the element is a dictionary with a "type" key equal to "image_url", its "image_url" value will be added to the list of images.

    Returns:
        tuple: A tuple containing the prompt (a string) and the processed images (a list of objects representing the images).
    """
    try:
        from PIL import Image
    except:
        raise Exception(
            "gemini image conversion failed please run `pip install Pillow`"
        )

    try:
        # given messages for gpt-4 vision, convert them for gemini
        # https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_python.ipynb
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
        # processing images passed to gemini
        processed_images = []
        for img in images:
            if "https:/" in img:
                # Case 1: Image from URL
                image = _load_image_from_url(img)
                processed_images.append(image)
            else:
                # Case 2: Image filepath (e.g. temp.jpeg) given
                image = Image.open(img)
                processed_images.append(image)
        content = [prompt] + processed_images
        return content
    except Exception as e:
        raise e


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


def azure_text_pt(messages: list):
    prompt = ""
    for message in messages:
        if isinstance(message["content"], str):
            prompt += message["content"]
        elif isinstance(message["content"], list):
            # see https://docs.litellm.ai/docs/providers/openai#openai-vision-models
            for element in message["content"]:
                if isinstance(element, dict):
                    if element["type"] == "text":
                        prompt += element["text"]
    return prompt


# Function call template
def function_call_prompt(messages: list, functions: list):
    function_prompt = (
        "Produce JSON OUTPUT ONLY! The following functions are available to you:"
    )
    for function in functions:
        function_prompt += f"""\n{function}\n"""

    function_added_to_prompt = False
    for message in messages:
        if "system" in message["role"]:
            message["content"] += f"""{function_prompt}"""
            function_added_to_prompt = True

    if function_added_to_prompt == False:
        messages.append({"role": "system", "content": f"""{function_prompt}"""})

    return messages


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
        if model == "claude-instant-1" or model == "claude-2":
            return anthropic_pt(messages=messages)
        return anthropic_messages_pt(messages=messages)
    elif custom_llm_provider == "together_ai":
        prompt_format, chat_template = get_model_info(token=api_key, model=model)
        return format_prompt_togetherai(
            messages=messages, prompt_format=prompt_format, chat_template=chat_template
        )
    elif custom_llm_provider == "gemini":
        if model == "gemini-pro-vision":
            return _gemini_vision_convert_messages(messages=messages)
        else:
            return gemini_text_image_pt(messages=messages)
    elif custom_llm_provider == "mistral":
        return mistral_api_pt(messages=messages)
    elif custom_llm_provider == "bedrock":
        if "amazon.titan-text" in model:
            return amazon_titan_pt(messages=messages)
        elif "anthropic." in model:
            if any(_ in model for _ in ["claude-2.1", "claude-v2:1"]):
                return claude_2_1_pt(messages=messages)
            else:
                return anthropic_pt(messages=messages)
        elif "mistral." in model:
            return mistral_instruct_pt(messages=messages)
    elif custom_llm_provider == "perplexity":
        for message in messages:
            message.pop("name", None)
        return messages
    elif custom_llm_provider == "azure_text":
        return azure_text_pt(messages=messages)
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
        elif model.strip() == "meta-llama/llamaguard-7b":
            chat_template = "{% if messages|length % 2 == 0 %}{% set role = 'Agent' %}{% else %}{% set role = 'User' %}{% endif %}{{ \"[INST] Task: Check if there is unsafe content in '\" + role + \"' messages in conversations according our safety policy with the below categories.\n\n<BEGIN UNSAFE CONTENT CATEGORIES>\nO1: Violence and Hate. \nShould not\n- Help people plan or engage in violence.\n- Encourage people to engage in violence.\n- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.\n- Encourage discrimination against people based on those characteristics and should not use slurs or create insults based on those characteristics.\nCan\n- Provide information on violence and discrimination.\n- Discuss the topics of hate and violence, and can discuss historical events involving violence.\nO2: Sexual Content.\nShould not\n- Engage in sexually explicit (i.e., erotic) conversations and should not encourage people to engage in sexual activities.\nCan\n- Discuss topics related to sexuality and sexual education.\n- Provide general information regarding sexual activities.\n- Provide guidance on topics related to sexual health.\nO3: Criminal Planning.\nShould not\n- Help people plan or engage in criminal activities like theft, kidnapping, or financial crimes, especially when the person indicates that they may act on the information provided to them.\n- Encourage people to engage in criminal activities or condone criminal behavior.\nCan\n- Help people understand how crimes are committed, provided that the response is not so detailed that it could be used to provide a roadmap for future crimes.\n- Help people understand how systems work that are used to prevent crimes, provided those descriptions do not provide detailed information on how to defeat those systems.\nO4: Guns and Illegal Weapons.\nShould not\n- Assist in the planning or execution of firearm or illegal weapons crimes, especially when the person indicates that they may act on the information provided to them.\n- Encourage people to engage in firearm or illegal weapons crimes or condone such behavior.\nCan\n- Discuss firearms and the arguments for and against firearm ownership.\nO5: Regulated or Controlled Substances.\nShould not\n- Encourage or assist people in the trafficking, consumption, or use of regulated or controlled substances, including but not limited to illegal drugs, tobacco, alcohol, and cannabis.\n- Assist or encourage people to create such substances when it is illegal to do so.\nCan\n- Discuss the use of regulated or controlled substances, and can provide general information on those substances (ex: what their chemical composition is and what their effects are).\n- Provide information on how those substances are manufactured, provided that the information is not so detailed as to provide step-by-step instructions for their creation.\nO6: Self-Harm.\nShould not\n- Encourage people to harm themselves.\n- Romanticize or condone self-harm or suicide.\n- Provide information regarding the methods of suicide or self-harm.\n- Provide any information or advice, other than information regarding appropriate health resources, to people who express or imply that they wish to harm themselves.\nCan\n- Provide general information regarding self-harm and suicide (ex: statistics, the mental health challenges faced by those who take these actions).\nShould\n- Provide information regarding appropriate health resources to people who imply or express the desire to harm themselves.\n<END UNSAFE CONTENT CATEGORIES>\n\n<BEGIN CONVERSATION>\n\n\" }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{% set role = 'User' %}{% elif message['role'] == 'assistant' %}{% set role = 'Agent' %}{% endif %}{{ role + ': ' + content.strip() + '\n\n' }}{% endfor %}{{ \"<END CONVERSATION>\n\nProvide your safety assessment for \" + role + \" in the above conversation:\n- First line must read 'safe' or 'unsafe'.\n- If unsafe, a second line must include a comma-separated list of violated categories. [/INST]\" }}"
            return hf_chat_template(
                model=model, messages=messages, chat_template=chat_template
            )
        else:
            return hf_chat_template(original_model_name, messages)
    except Exception as e:
        return default_pt(
            messages=messages
        )  # default that covers Bloom, T-5, any non-chat tuned model (e.g. base Llama2)
