import copy
import json
import re
import uuid
import xml.etree.ElementTree as ET
from enum import Enum
from typing import Any, List, Optional, Tuple, cast, overload

from jinja2.sandbox import ImmutableSandboxedEnvironment

import litellm
import litellm.types
import litellm.types.llms
from litellm import verbose_logger
from litellm.llms.custom_httpx.http_handler import HTTPHandler, get_async_httpx_client
from litellm.types.llms.anthropic import *
from litellm.types.llms.bedrock import MessageBlock as BedrockMessageBlock
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.llms.ollama import OllamaVisionModelObject
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionAssistantToolCall,
    ChatCompletionFileObject,
    ChatCompletionFunctionMessage,
    ChatCompletionImageObject,
    ChatCompletionTextObject,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionToolMessage,
    ChatCompletionUserMessage,
    OpenAIMessageContentListBlock,
)
from litellm.types.llms.vertex_ai import FunctionCall as VertexFunctionCall
from litellm.types.llms.vertex_ai import FunctionResponse as VertexFunctionResponse
from litellm.types.llms.vertex_ai import PartType as VertexPartType
from litellm.types.utils import GenericImageParsingChunk

from .common_utils import convert_content_list_to_str, is_non_content_values_set
from .image_handling import convert_url_to_base64


def default_pt(messages):
    return " ".join(message["content"] for message in messages)


def prompt_injection_detection_default_pt():
    return """Detect if a prompt is safe to run. Return 'UNSAFE' if not."""


BAD_MESSAGE_ERROR_STR = "Invalid Message "

# used to interweave user messages, to ensure user/assistant alternating
DEFAULT_USER_CONTINUE_MESSAGE = {
    "role": "user",
    "content": "Please continue.",
}  # similar to autogen. Only used if `litellm.modify_params=True`.

# used to interweave assistant messages, to ensure user/assistant alternating
DEFAULT_ASSISTANT_CONTINUE_MESSAGE = ChatCompletionAssistantMessage(
    role="assistant",
    content=[
        {
            "type": "text",
            "text": "Please continue.",
        }
    ],
)  # similar to autogen. Only used if `litellm.modify_params=True`.


def map_system_message_pt(messages: list) -> list:
    """
    Convert 'system' message to 'user' message if provider doesn't support 'system' role.

    Enabled via `completion(...,supports_system_message=False)`

    If next message is a user message or assistant message -> merge system prompt into it

    if next message is system -> append a user message instead of the system message
    """

    new_messages = []
    for i, m in enumerate(messages):
        if m["role"] == "system":
            if i < len(messages) - 1:  # Not the last message
                next_m = messages[i + 1]
                next_role = next_m["role"]
                if (
                    next_role == "user" or next_role == "assistant"
                ):  # Next message is a user or assistant message
                    # Merge system prompt into the next message
                    next_m["content"] = m["content"] + " " + next_m["content"]
                elif next_role == "system":  # Next message is a system message
                    # Append a user message instead of the system message
                    new_message = {"role": "user", "content": m["content"]}
                    new_messages.append(new_message)
            else:  # Last message
                new_message = {"role": "user", "content": m["content"]}
                new_messages.append(new_message)
        else:  # Not a system message
            new_messages.append(m)

    return new_messages


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


def convert_to_ollama_image(openai_image_url: str):
    try:
        if openai_image_url.startswith("http"):
            openai_image_url = convert_url_to_base64(url=openai_image_url)

        if openai_image_url.startswith("data:image/"):
            # Extract the base64 image data
            base64_data = openai_image_url.split("data:image/")[1].split(";base64,")[1]
        else:
            base64_data = openai_image_url

        return base64_data
    except Exception as e:
        if "Error: Unable to fetch image from URL" in str(e):
            raise e
        raise Exception(
            """Image url not in expected format. Example Expected input - "image_url": "data:image/jpeg;base64,{base64_image}". """
        )


def _handle_ollama_system_message(
    messages: list, prompt: str, msg_i: int
) -> Tuple[str, int]:
    system_content_str = ""
    ## MERGE CONSECUTIVE SYSTEM CONTENT ##
    while msg_i < len(messages) and messages[msg_i]["role"] == "system":
        msg_content = convert_content_list_to_str(messages[msg_i])
        system_content_str += msg_content

        msg_i += 1

    return system_content_str, msg_i


def ollama_pt(
    model: str, messages: list
) -> Union[
    str, OllamaVisionModelObject
]:  # https://github.com/ollama/ollama/blob/af4cf55884ac54b9e637cd71dadfe9b7a5685877/docs/modelfile.md#template
    user_message_types = {"user", "tool", "function"}
    msg_i = 0
    images = []
    prompt = ""
    while msg_i < len(messages):
        init_msg_i = msg_i
        user_content_str = ""
        ## MERGE CONSECUTIVE USER CONTENT ##
        while msg_i < len(messages) and messages[msg_i]["role"] in user_message_types:
            msg_content = messages[msg_i].get("content")
            if msg_content:
                if isinstance(msg_content, list):
                    for m in msg_content:
                        if m.get("type", "") == "image_url":
                            if isinstance(m["image_url"], str):
                                images.append(m["image_url"])
                            elif isinstance(m["image_url"], dict):
                                images.append(m["image_url"]["url"])
                        elif m.get("type", "") == "text":
                            user_content_str += m["text"]
                else:
                    # Tool message content will always be a string
                    user_content_str += msg_content

            msg_i += 1

        if user_content_str:
            prompt += f"### User:\n{user_content_str}\n\n"

        system_content_str, msg_i = _handle_ollama_system_message(
            messages, prompt, msg_i
        )
        if system_content_str:
            prompt += f"### System:\n{system_content_str}\n\n"

        assistant_content_str = ""
        ## MERGE CONSECUTIVE ASSISTANT CONTENT ##
        while msg_i < len(messages) and messages[msg_i]["role"] == "assistant":
            assistant_content_str += convert_content_list_to_str(messages[msg_i])
            msg_i += 1

            tool_calls = messages[msg_i].get("tool_calls")
            ollama_tool_calls = []
            if tool_calls:
                for call in tool_calls:
                    call_id: str = call["id"]
                    function_name: str = call["function"]["name"]
                    arguments = json.loads(call["function"]["arguments"])

                    ollama_tool_calls.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": function_name,
                                "arguments": arguments,
                            },
                        }
                    )

            if ollama_tool_calls:
                assistant_content_str += (
                    f"Tool Calls: {json.dumps(ollama_tool_calls, indent=2)}"
                )

            msg_i += 1

        if assistant_content_str:
            prompt += f"### Assistant:\n{assistant_content_str}\n\n"

        if msg_i == init_msg_i:  # prevent infinite loops
            raise litellm.BadRequestError(
                message=BAD_MESSAGE_ERROR_STR + f"passed in {messages[msg_i]}",
                model=model,
                llm_provider="ollama",
            )

    response_dict: OllamaVisionModelObject = {
        "prompt": prompt,
        "images": images,
    }

    return response_dict


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


def hf_chat_template(  # noqa: PLR0915
    model: str, messages: list, chat_template: Optional[Any] = None
):
    # Define Jinja2 environment
    env = ImmutableSandboxedEnvironment()

    def raise_exception(message):
        raise Exception(f"Error message - {message}")

    # Create a template object from the template text
    env.globals["raise_exception"] = raise_exception

    ## get the tokenizer config from huggingface
    bos_token = ""
    eos_token = ""
    if chat_template is None:

        def _get_tokenizer_config(hf_model_name):
            try:
                url = f"https://huggingface.co/{hf_model_name}/raw/main/tokenizer_config.json"
                # Make a GET request to fetch the JSON data
                client = HTTPHandler(concurrent_limit=1)

                response = client.get(url)
            except Exception as e:
                raise e
            if response.status_code == 200:
                # Parse the JSON data
                tokenizer_config = json.loads(response.content)
                return {"status": "success", "tokenizer": tokenizer_config}
            else:
                return {"status": "failure"}

        if model in litellm.known_tokenizer_config:
            tokenizer_config = litellm.known_tokenizer_config[model]
        else:
            tokenizer_config = _get_tokenizer_config(model)
            litellm.known_tokenizer_config.update({model: tokenizer_config})

        if (
            tokenizer_config["status"] == "failure"
            or "chat_template" not in tokenizer_config["tokenizer"]
        ):
            raise Exception("No chat template found")
        ## read the bos token, eos token and chat template from the json
        tokenizer_config = tokenizer_config["tokenizer"]  # type: ignore

        bos_token = tokenizer_config["bos_token"]  # type: ignore
        if bos_token is not None and not isinstance(bos_token, str):
            if isinstance(bos_token, dict):
                bos_token = bos_token.get("content", None)
        eos_token = tokenizer_config["eos_token"]  # type: ignore
        if eos_token is not None and not isinstance(eos_token, str):
            if isinstance(eos_token, dict):
                eos_token = eos_token.get("content", None)
        chat_template = tokenizer_config["chat_template"]  # type: ignore
    try:
        template = env.from_string(chat_template)  # type: ignore
    except Exception as e:
        raise e

    def _is_system_in_template():
        try:
            # Try rendering the template with a system message
            template.render(
                messages=[{"role": "system", "content": "test"}],
                eos_token="<eos>",
                bos_token="<bos>",
            )
            return True

        # This will be raised if Jinja attempts to render the system message and it can't
        except Exception:
            return False

    try:
        rendered_text = ""
        # Render the template with the provided values
        if _is_system_in_template():
            rendered_text = template.render(
                bos_token=bos_token,
                eos_token=eos_token,
                messages=messages,
                add_generation_prompt=True,
            )
        else:
            # treat a system message as a user message, if system not in template
            reformatted_messages = []
            try:
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
                    add_generation_prompt=True,
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
        raise Exception(
            f"Error rendering template - {str(e)}"
        )  # don't use verbose_logger.exception, if exception is raised


def deepseek_r1_pt(messages):
    return hf_chat_template(
        model="deepseek-r1/deepseek-r1-7b-instruct", messages=messages
    )


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
        client = HTTPHandler(concurrent_limit=1)
        response = client.get("https://api.together.xyz/models/info", headers=headers)
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
    except Exception:  # safely fail a prompt template request
        return None, None


## OLD TOGETHER AI FLOW
# def format_prompt_togetherai(messages, prompt_format, chat_template):
#     if prompt_format is None:
#         return default_pt(messages)

#     human_prompt, assistant_prompt = prompt_format.split("{prompt}")

#     if chat_template is not None:
#         prompt = hf_chat_template(
#             model=None, messages=messages, chat_template=chat_template
#         )
#     elif prompt_format is not None:
#         prompt = custom_prompt(
#             role_dict={},
#             messages=messages,
#             initial_prompt_value=human_prompt,
#             final_prompt_value=assistant_prompt,
#         )
#     else:
#         prompt = default_pt(messages)
#     return prompt


### IBM Granite


def ibm_granite_pt(messages: list):
    """
    IBM's Granite models uses the template:
    <|system|> {system_message} <|user|> {user_message} <|assistant|> {assistant_message}

    See: https://www.ibm.com/docs/en/watsonx-as-a-service?topic=solutions-supported-foundation-models
    """
    return custom_prompt(
        messages=messages,
        role_dict={
            "system": {
                "pre_message": "<|system|>\n",
                "post_message": "\n",
            },
            "user": {
                "pre_message": "<|user|>\n",
                # Assistant tag is needed in the prompt after the user message
                # to avoid the model completing the users sentence before it answers
                # https://www.ibm.com/docs/en/watsonx/w-and-w/2.0.x?topic=models-granite-13b-chat-v2-prompting-tips#chat
                "post_message": "\n<|assistant|>\n",
            },
            "assistant": {
                "pre_message": "",
                "post_message": "\n",
            },
        },
    ).strip()


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
        tool_function = get_attribute_or_key(tool, "function")
        tool_str = construct_format_tool_for_claude_prompt(
            get_attribute_or_key(tool_function, "name"),
            get_attribute_or_key(tool_function, "description", ""),
            get_attribute_or_key(tool_function, "parameters", {}),
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


def convert_generic_image_chunk_to_openai_image_obj(
    image_chunk: GenericImageParsingChunk,
) -> str:
    """
    Convert a generic image chunk to an OpenAI image object.

    Input:
    GenericImageParsingChunk(
        type="base64",
        media_type="image/jpeg",
        data="...",
    )

    Return:
    "data:image/jpeg;base64,{base64_image}"
    """
    media_type = image_chunk["media_type"]
    return "data:{};{},{}".format(media_type, image_chunk["type"], image_chunk["data"])


def convert_to_anthropic_image_obj(
    openai_image_url: str, format: Optional[str]
) -> GenericImageParsingChunk:
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
        # Extract the media type and base64 data
        media_type, base64_data = openai_image_url.split("data:")[1].split(";base64,")

        if format:
            media_type = format
        else:
            media_type = media_type.replace("\\/", "/")

        return GenericImageParsingChunk(
            type="base64",
            media_type=media_type,
            data=base64_data,
        )
    except Exception as e:
        if "Error: Unable to fetch image from URL" in str(e):
            raise e
        raise Exception(
            """Image url not in expected format. Example Expected input - "image_url": "data:image/jpeg;base64,{base64_image}". Supported formats - ['image/jpeg', 'image/png', 'image/gif', 'image/webp']."""
        )


# The following XML functions will be deprecated once JSON schema support is available on Bedrock and Vertex
# ------------------------------------------------------------------------------
def convert_to_anthropic_tool_result_xml(message: dict) -> str:
    """
    OpenAI message with a tool result looks like:
    {
        "tool_call_id": "tool_1",
        "role": "tool",
        "name": "get_current_weather",
        "content": "function result goes here",
    },
    """

    """
    Anthropic tool_results look like:

    [Successful results]
    <function_results>
    <result>
    <tool_name>get_current_weather</tool_name>
    <stdout>
    function result goes here
    </stdout>
    </result>
    </function_results>

    [Error results]
    <function_results>
    <error>
    error message goes here
    </error>
    </function_results>
    """
    name = message.get("name")
    content = message.get("content", "")
    content = content.replace("<", "&lt;").replace(">", "&gt;").replace("&", "&amp;")

    # We can't determine from openai message format whether it's a successful or
    # error call result so default to the successful result template
    anthropic_tool_result = (
        "<function_results>\n"
        "<result>\n"
        f"<tool_name>{name}</tool_name>\n"
        "<stdout>\n"
        f"{content}\n"
        "</stdout>\n"
        "</result>\n"
        "</function_results>"
    )

    return anthropic_tool_result


def convert_to_anthropic_tool_invoke_xml(tool_calls: list) -> str:
    invokes = ""
    for tool in tool_calls:
        if get_attribute_or_key(tool, "type") != "function":
            continue

        tool_function = get_attribute_or_key(tool, "function")
        tool_name = get_attribute_or_key(tool_function, "name")
        tool_arguments = get_attribute_or_key(tool_function, "arguments")
        parameters = "".join(
            f"<{param}>{val}</{param}>\n"
            for param, val in json.loads(tool_arguments).items()
        )
        invokes += (
            "<invoke>\n"
            f"<tool_name>{tool_name}</tool_name>\n"
            "<parameters>\n"
            f"{parameters}"
            "</parameters>\n"
            "</invoke>\n"
        )

    anthropic_tool_invoke = f"<function_calls>\n{invokes}</function_calls>"

    return anthropic_tool_invoke


def anthropic_messages_pt_xml(messages: list):
    """
    format messages for anthropic
    1. Anthropic supports roles like "user" and "assistant", (here litellm translates system-> assistant)
    2. The first message always needs to be of role "user"
    3. Each message must alternate between "user" and "assistant" (this is not addressed as now by litellm)
    4. final assistant content cannot end with trailing whitespace (anthropic raises an error otherwise)
    5. System messages are a separate param to the Messages API (used for tool calling)
    6. Ensure we only accept role, content. (message.name is not supported)
    """
    # add role=tool support to allow function call result/error submission
    user_message_types = {"user", "tool"}
    # reformat messages to ensure user/assistant are alternating, if there's either 2 consecutive 'user' messages or 2 consecutive 'assistant' message, merge them.
    new_messages = []
    msg_i = 0
    while msg_i < len(messages):
        user_content = []
        ## MERGE CONSECUTIVE USER CONTENT ##
        while msg_i < len(messages) and messages[msg_i]["role"] in user_message_types:
            if isinstance(messages[msg_i]["content"], list):
                for m in messages[msg_i]["content"]:
                    if m.get("type", "") == "image_url":
                        format = m["image_url"].get("format")
                        user_content.append(
                            {
                                "type": "image",
                                "source": convert_to_anthropic_image_obj(
                                    m["image_url"]["url"], format=format
                                ),
                            }
                        )
                    elif m.get("type", "") == "text":
                        user_content.append({"type": "text", "text": m["text"]})
            else:
                # Tool message content will always be a string
                user_content.append(
                    {
                        "type": "text",
                        "text": (
                            convert_to_anthropic_tool_result_xml(messages[msg_i])
                            if messages[msg_i]["role"] == "tool"
                            else messages[msg_i]["content"]
                        ),
                    }
                )

            msg_i += 1

        if user_content:
            new_messages.append({"role": "user", "content": user_content})

        assistant_content = []
        ## MERGE CONSECUTIVE ASSISTANT CONTENT ##
        while msg_i < len(messages) and messages[msg_i]["role"] == "assistant":
            assistant_text = (
                messages[msg_i].get("content") or ""
            )  # either string or none
            if messages[msg_i].get(
                "tool_calls", []
            ):  # support assistant tool invoke conversion
                assistant_text += convert_to_anthropic_tool_invoke_xml(  # type: ignore
                    messages[msg_i]["tool_calls"]
                )

            assistant_content.append({"type": "text", "text": assistant_text})
            msg_i += 1

        if assistant_content:
            new_messages.append({"role": "assistant", "content": assistant_content})

    if not new_messages or new_messages[0]["role"] != "user":
        if litellm.modify_params:
            new_messages.insert(
                0, {"role": "user", "content": [{"type": "text", "text": "."}]}
            )
        else:
            raise Exception(
                "Invalid first message. Should always start with 'role'='user' for Anthropic. System prompt is sent separately for Anthropic. set 'litellm.modify_params = True' or 'litellm_settings:modify_params = True' on proxy, to insert a placeholder user message - '.' as the first message, "
            )

    if new_messages[-1]["role"] == "assistant":
        for content in new_messages[-1]["content"]:
            if isinstance(content, dict) and content["type"] == "text":
                content["text"] = content[
                    "text"
                ].rstrip()  # no trailing whitespace for final assistant message

    return new_messages


# ------------------------------------------------------------------------------


def _azure_tool_call_invoke_helper(
    function_call_params: ChatCompletionToolCallFunctionChunk,
) -> Optional[ChatCompletionToolCallFunctionChunk]:
    """
    Azure requires 'arguments' to be a string.
    """
    if function_call_params.get("arguments") is None:
        function_call_params["arguments"] = ""
    return function_call_params


def convert_to_azure_openai_messages(
    messages: List[AllMessageValues],
) -> List[AllMessageValues]:
    for m in messages:
        if m["role"] == "assistant":
            function_call = m.get("function_call", None)
            if function_call is not None:
                m["function_call"] = _azure_tool_call_invoke_helper(function_call)
    return messages


# ------------------------------------------------------------------------------


def infer_protocol_value(
    value: Any,
) -> Literal[
    "string_value",
    "number_value",
    "bool_value",
    "struct_value",
    "list_value",
    "null_value",
    "unknown",
]:
    if value is None:
        return "null_value"
    if isinstance(value, int) or isinstance(value, float):
        return "number_value"
    if isinstance(value, str):
        return "string_value"
    if isinstance(value, bool):
        return "bool_value"
    if isinstance(value, dict):
        return "struct_value"
    if isinstance(value, list):
        return "list_value"

    return "unknown"


def _gemini_tool_call_invoke_helper(
    function_call_params: ChatCompletionToolCallFunctionChunk,
) -> Optional[VertexFunctionCall]:
    name = function_call_params.get("name", "") or ""
    arguments = function_call_params.get("arguments", "")
    arguments_dict = json.loads(arguments)
    function_call = VertexFunctionCall(
        name=name,
        args=arguments_dict,
    )
    return function_call


def convert_to_gemini_tool_call_invoke(
    message: ChatCompletionAssistantMessage,
) -> List[VertexPartType]:
    """
    OpenAI tool invokes:
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_abc123",
          "type": "function",
          "function": {
            "name": "get_current_weather",
            "arguments": "{\n\"location\": \"Boston, MA\"\n}"
          }
        }
      ]
    },
    """
    """
    Gemini tool call invokes:
    {
      "role": "model",
      "parts": [
        {
          "functionCall": {
            "name": "get_current_weather",
            "args": {
              "unit": "fahrenheit",
              "predicted_temperature": 45,
              "location": "Boston, MA",
            }
          }
        }
      ]
    }
    """

    """
    - json.load the arguments
    """
    try:
        _parts_list: List[VertexPartType] = []
        tool_calls = message.get("tool_calls", None)
        function_call = message.get("function_call", None)
        if tool_calls is not None:
            for tool in tool_calls:
                if "function" in tool:
                    gemini_function_call: Optional[
                        VertexFunctionCall
                    ] = _gemini_tool_call_invoke_helper(
                        function_call_params=tool["function"]
                    )
                    if gemini_function_call is not None:
                        _parts_list.append(
                            VertexPartType(function_call=gemini_function_call)
                        )
                    else:  # don't silently drop params. Make it clear to user what's happening.
                        raise Exception(
                            "function_call missing. Received tool call with 'type': 'function'. No function call in argument - {}".format(
                                tool
                            )
                        )
        elif function_call is not None:
            gemini_function_call = _gemini_tool_call_invoke_helper(
                function_call_params=function_call
            )
            if gemini_function_call is not None:
                _parts_list.append(VertexPartType(function_call=gemini_function_call))
            else:  # don't silently drop params. Make it clear to user what's happening.
                raise Exception(
                    "function_call missing. Received tool call with 'type': 'function'. No function call in argument - {}".format(
                        message
                    )
                )
        return _parts_list
    except Exception as e:
        raise Exception(
            "Unable to convert openai tool calls={} to gemini tool calls. Received error={}".format(
                message, str(e)
            )
        )


def convert_to_gemini_tool_call_result(
    message: Union[ChatCompletionToolMessage, ChatCompletionFunctionMessage],
    last_message_with_tool_calls: Optional[dict],
) -> VertexPartType:
    """
    OpenAI message with a tool result looks like:
    {
        "tool_call_id": "tool_1",
        "role": "tool",
        "content": "function result goes here",
    },

    # NOTE: Function messages have been deprecated
    OpenAI message with a function call result looks like:
    {
        "role": "function",
        "name": "get_current_weather",
        "content": "function result goes here",
    }
    """
    content_str: str = ""
    if isinstance(message["content"], str):
        content_str = message["content"]
    elif isinstance(message["content"], List):
        content_list = message["content"]
        for content in content_list:
            if content["type"] == "text":
                content_str += content["text"]
    name: Optional[str] = message.get("name", "")  # type: ignore

    # Recover name from last message with tool calls
    if last_message_with_tool_calls:
        tools = last_message_with_tool_calls.get("tool_calls", [])
        msg_tool_call_id = message.get("tool_call_id", None)
        for tool in tools:
            prev_tool_call_id = tool.get("id", None)
            if (
                msg_tool_call_id
                and prev_tool_call_id
                and msg_tool_call_id == prev_tool_call_id
            ):
                name = tool.get("function", {}).get("name", "")

    if not name:
        raise Exception(
            "Missing corresponding tool call for tool response message. Received - message={}, last_message_with_tool_calls={}".format(
                message, last_message_with_tool_calls
            )
        )

    # We can't determine from openai message format whether it's a successful or
    # error call result so default to the successful result template
    _function_response = VertexFunctionResponse(
        name=name, response={"content": content_str}  # type: ignore
    )

    _part = VertexPartType(function_response=_function_response)

    return _part


def convert_to_anthropic_tool_result(
    message: Union[ChatCompletionToolMessage, ChatCompletionFunctionMessage]
) -> AnthropicMessagesToolResultParam:
    """
    OpenAI message with a tool result looks like:
    {
        "tool_call_id": "tool_1",
        "role": "tool",
        "name": "get_current_weather",
        "content": "function result goes here",
    },

    OpenAI message with a function call result looks like:
    {
        "role": "function",
        "name": "get_current_weather",
        "content": "function result goes here",
    }
    """

    """
    Anthropic tool_results look like:
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
                "content": "ConnectionError: the weather service API is not available (HTTP 500)",
                # "is_error": true
            }
        ]
    }
    """
    anthropic_content: Union[
        str,
        List[Union[AnthropicMessagesToolResultContent, AnthropicMessagesImageParam]],
    ] = ""
    if isinstance(message["content"], str):
        anthropic_content = message["content"]
    elif isinstance(message["content"], List):
        content_list = message["content"]
        anthropic_content_list: List[
            Union[AnthropicMessagesToolResultContent, AnthropicMessagesImageParam]
        ] = []
        for content in content_list:
            if content["type"] == "text":
                anthropic_content_list.append(
                    AnthropicMessagesToolResultContent(
                        type="text",
                        text=content["text"],
                    )
                )
            elif content["type"] == "image_url":
                if isinstance(content["image_url"], str):
                    image_chunk = convert_to_anthropic_image_obj(
                        content["image_url"], format=None
                    )
                else:
                    format = content["image_url"].get("format")
                    image_chunk = convert_to_anthropic_image_obj(
                        content["image_url"]["url"], format=format
                    )
                anthropic_content_list.append(
                    AnthropicMessagesImageParam(
                        type="image",
                        source=AnthropicContentParamSource(
                            type="base64",
                            media_type=image_chunk["media_type"],
                            data=image_chunk["data"],
                        ),
                    )
                )

        anthropic_content = anthropic_content_list
    anthropic_tool_result: Optional[AnthropicMessagesToolResultParam] = None
    ## PROMPT CACHING CHECK ##
    cache_control = message.get("cache_control", None)
    if message["role"] == "tool":
        tool_message: ChatCompletionToolMessage = message
        tool_call_id: str = tool_message["tool_call_id"]

        # We can't determine from openai message format whether it's a successful or
        # error call result so default to the successful result template
        anthropic_tool_result = AnthropicMessagesToolResultParam(
            type="tool_result", tool_use_id=tool_call_id, content=anthropic_content
        )

    if message["role"] == "function":
        function_message: ChatCompletionFunctionMessage = message
        tool_call_id = function_message.get("tool_call_id") or str(uuid.uuid4())
        anthropic_tool_result = AnthropicMessagesToolResultParam(
            type="tool_result", tool_use_id=tool_call_id, content=anthropic_content
        )

    if anthropic_tool_result is None:
        raise Exception(f"Unable to parse anthropic tool result for message: {message}")
    if cache_control is not None:
        anthropic_tool_result["cache_control"] = cache_control  # type: ignore
    return anthropic_tool_result


def convert_function_to_anthropic_tool_invoke(
    function_call: Union[dict, ChatCompletionToolCallFunctionChunk],
) -> List[AnthropicMessagesToolUseParam]:
    try:
        _name = get_attribute_or_key(function_call, "name") or ""
        _arguments = get_attribute_or_key(function_call, "arguments")
        anthropic_tool_invoke = [
            AnthropicMessagesToolUseParam(
                type="tool_use",
                id=str(uuid.uuid4()),
                name=_name,
                input=json.loads(_arguments) if _arguments else {},
            )
        ]
        return anthropic_tool_invoke
    except Exception as e:
        raise e


def convert_to_anthropic_tool_invoke(
    tool_calls: List[ChatCompletionAssistantToolCall],
) -> List[AnthropicMessagesToolUseParam]:
    """
    OpenAI tool invokes:
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_abc123",
          "type": "function",
          "function": {
            "name": "get_current_weather",
            "arguments": "{\n\"location\": \"Boston, MA\"\n}"
          }
        }
      ]
    },
    """

    """
    Anthropic tool invokes:
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "<thinking>To answer this question, I will: 1. Use the get_weather tool to get the current weather in San Francisco. 2. Use the get_time tool to get the current time in the America/Los_Angeles timezone, which covers San Francisco, CA.</thinking>"
        },
        {
          "type": "tool_use",
          "id": "toolu_01A09q90qw90lq917835lq9",
          "name": "get_weather",
          "input": {"location": "San Francisco, CA"}
        }
      ]
    }
    """
    anthropic_tool_invoke = []

    for tool in tool_calls:
        if not get_attribute_or_key(tool, "type") == "function":
            continue

        _anthropic_tool_use_param = AnthropicMessagesToolUseParam(
            type="tool_use",
            id=cast(str, get_attribute_or_key(tool, "id")),
            name=cast(
                str,
                get_attribute_or_key(get_attribute_or_key(tool, "function"), "name"),
            ),
            input=json.loads(
                get_attribute_or_key(
                    get_attribute_or_key(tool, "function"), "arguments"
                )
            ),
        )

        _content_element = add_cache_control_to_content(
            anthropic_content_element=_anthropic_tool_use_param,
            orignal_content_element=dict(tool),
        )

        if "cache_control" in _content_element:
            _anthropic_tool_use_param["cache_control"] = _content_element[
                "cache_control"
            ]

        anthropic_tool_invoke.append(_anthropic_tool_use_param)

    return anthropic_tool_invoke


def add_cache_control_to_content(
    anthropic_content_element: Union[
        dict,
        AnthropicMessagesImageParam,
        AnthropicMessagesTextParam,
        AnthropicMessagesDocumentParam,
        AnthropicMessagesToolUseParam,
        ChatCompletionThinkingBlock,
    ],
    orignal_content_element: Union[dict, AllMessageValues],
):
    cache_control_param = orignal_content_element.get("cache_control")
    if cache_control_param is not None and isinstance(cache_control_param, dict):
        transformed_param = ChatCompletionCachedContent(**cache_control_param)  # type: ignore

        anthropic_content_element["cache_control"] = transformed_param

    return anthropic_content_element


def _anthropic_content_element_factory(
    image_chunk: GenericImageParsingChunk,
) -> Union[AnthropicMessagesImageParam, AnthropicMessagesDocumentParam]:
    if image_chunk["media_type"] == "application/pdf":
        _anthropic_content_element: Union[
            AnthropicMessagesDocumentParam, AnthropicMessagesImageParam
        ] = AnthropicMessagesDocumentParam(
            type="document",
            source=AnthropicContentParamSource(
                type="base64",
                media_type=image_chunk["media_type"],
                data=image_chunk["data"],
            ),
        )
    else:
        _anthropic_content_element = AnthropicMessagesImageParam(
            type="image",
            source=AnthropicContentParamSource(
                type="base64",
                media_type=image_chunk["media_type"],
                data=image_chunk["data"],
            ),
        )

    return _anthropic_content_element


def anthropic_messages_pt(  # noqa: PLR0915
    messages: List[AllMessageValues],
    model: str,
    llm_provider: str,
) -> List[
    Union[
        AnthropicMessagesUserMessageParam,
        AnthopicMessagesAssistantMessageParam,
    ]
]:
    """
    format messages for anthropic
    1. Anthropic supports roles like "user" and "assistant" (system prompt sent separately)
    2. The first message always needs to be of role "user"
    3. Each message must alternate between "user" and "assistant" (this is not addressed as now by litellm)
    4. final assistant content cannot end with trailing whitespace (anthropic raises an error otherwise)
    5. System messages are a separate param to the Messages API
    6. Ensure we only accept role, content. (message.name is not supported)
    """
    # add role=tool support to allow function call result/error submission
    user_message_types = {"user", "tool", "function"}
    # reformat messages to ensure user/assistant are alternating, if there's either 2 consecutive 'user' messages or 2 consecutive 'assistant' message, merge them.
    new_messages: List[
        Union[
            AnthropicMessagesUserMessageParam,
            AnthopicMessagesAssistantMessageParam,
        ]
    ] = []
    msg_i = 0
    while msg_i < len(messages):
        user_content: List[AnthropicMessagesUserMessageValues] = []
        init_msg_i = msg_i
        if isinstance(messages[msg_i], BaseModel):
            messages[msg_i] = dict(messages[msg_i])  # type: ignore
        ## MERGE CONSECUTIVE USER CONTENT ##
        while msg_i < len(messages) and messages[msg_i]["role"] in user_message_types:
            user_message_types_block: Union[
                ChatCompletionToolMessage,
                ChatCompletionUserMessage,
                ChatCompletionFunctionMessage,
            ] = messages[
                msg_i
            ]  # type: ignore
            if user_message_types_block["role"] == "user":
                if isinstance(user_message_types_block["content"], list):
                    for m in user_message_types_block["content"]:
                        if m.get("type", "") == "image_url":
                            m = cast(ChatCompletionImageObject, m)
                            format: Optional[str] = None
                            if isinstance(m["image_url"], str):
                                image_chunk = convert_to_anthropic_image_obj(
                                    openai_image_url=m["image_url"], format=None
                                )
                            else:
                                format = m["image_url"].get("format")
                                image_chunk = convert_to_anthropic_image_obj(
                                    openai_image_url=m["image_url"]["url"],
                                    format=format,
                                )

                            _anthropic_content_element = (
                                _anthropic_content_element_factory(image_chunk)
                            )
                            _content_element = add_cache_control_to_content(
                                anthropic_content_element=_anthropic_content_element,
                                orignal_content_element=dict(m),
                            )

                            if "cache_control" in _content_element:
                                _anthropic_content_element[
                                    "cache_control"
                                ] = _content_element["cache_control"]
                            user_content.append(_anthropic_content_element)
                        elif m.get("type", "") == "text":
                            m = cast(ChatCompletionTextObject, m)
                            _anthropic_text_content_element = (
                                AnthropicMessagesTextParam(
                                    type="text",
                                    text=m["text"],
                                )
                            )
                            _content_element = add_cache_control_to_content(
                                anthropic_content_element=_anthropic_text_content_element,
                                orignal_content_element=dict(m),
                            )
                            _content_element = cast(
                                AnthropicMessagesTextParam, _content_element
                            )

                            user_content.append(_content_element)
                        elif m.get("type", "") == "document":
                            user_content.append(cast(AnthropicMessagesDocumentParam, m))
                        elif m.get("type", "") == "file":
                            file_message = cast(ChatCompletionFileObject, m)
                            file_data = file_message["file"].get("file_data")
                            if file_data:
                                image_chunk = convert_to_anthropic_image_obj(
                                    openai_image_url=file_data,
                                    format=file_message["file"].get("format"),
                                )
                                anthropic_document_param = (
                                    AnthropicMessagesDocumentParam(
                                        type="document",
                                        source=AnthropicContentParamSource(
                                            type="base64",
                                            media_type=image_chunk["media_type"],
                                            data=image_chunk["data"],
                                        ),
                                    )
                                )
                                user_content.append(anthropic_document_param)
                elif isinstance(user_message_types_block["content"], str):
                    _anthropic_content_text_element: AnthropicMessagesTextParam = {
                        "type": "text",
                        "text": user_message_types_block["content"],
                    }
                    _content_element = add_cache_control_to_content(
                        anthropic_content_element=_anthropic_content_text_element,
                        orignal_content_element=dict(user_message_types_block),
                    )

                    if "cache_control" in _content_element:
                        _anthropic_content_text_element[
                            "cache_control"
                        ] = _content_element["cache_control"]

                    user_content.append(_anthropic_content_text_element)

            elif (
                user_message_types_block["role"] == "tool"
                or user_message_types_block["role"] == "function"
            ):
                # OpenAI's tool message content will always be a string
                user_content.append(
                    convert_to_anthropic_tool_result(user_message_types_block)
                )

            msg_i += 1

        if user_content:
            new_messages.append({"role": "user", "content": user_content})

        assistant_content: List[AnthropicMessagesAssistantMessageValues] = []
        ## MERGE CONSECUTIVE ASSISTANT CONTENT ##
        while msg_i < len(messages) and messages[msg_i]["role"] == "assistant":
            assistant_content_block: ChatCompletionAssistantMessage = messages[msg_i]  # type: ignore

            thinking_blocks = assistant_content_block.get("thinking_blocks", None)
            if (
                thinking_blocks is not None
            ):  # IMPORTANT: ADD THIS FIRST, ELSE ANTHROPIC WILL RAISE AN ERROR
                assistant_content.extend(thinking_blocks)
            if "content" in assistant_content_block and isinstance(
                assistant_content_block["content"], list
            ):
                for m in assistant_content_block["content"]:
                    # handle thinking blocks
                    thinking_block = cast(str, m.get("thinking", ""))
                    text_block = cast(str, m.get("text", ""))
                    if (
                        m.get("type", "") == "thinking" and len(thinking_block) > 0
                    ):  # don't pass empty text blocks. anthropic api raises errors.
                        anthropic_message: Union[
                            ChatCompletionThinkingBlock,
                            AnthropicMessagesTextParam,
                        ] = cast(ChatCompletionThinkingBlock, m)
                        assistant_content.append(anthropic_message)
                    # handle text
                    elif (
                        m.get("type", "") == "text" and len(text_block) > 0
                    ):  # don't pass empty text blocks. anthropic api raises errors.
                        anthropic_message = AnthropicMessagesTextParam(
                            type="text", text=text_block
                        )
                        _cached_message = add_cache_control_to_content(
                            anthropic_content_element=anthropic_message,
                            orignal_content_element=dict(m),
                        )

                        assistant_content.append(
                            cast(AnthropicMessagesTextParam, _cached_message)
                        )
            elif (
                "content" in assistant_content_block
                and isinstance(assistant_content_block["content"], str)
                and assistant_content_block[
                    "content"
                ]  # don't pass empty text blocks. anthropic api raises errors.
            ):
                _anthropic_text_content_element = AnthropicMessagesTextParam(
                    type="text",
                    text=assistant_content_block["content"],
                )

                _content_element = add_cache_control_to_content(
                    anthropic_content_element=_anthropic_text_content_element,
                    orignal_content_element=dict(assistant_content_block),
                )

                if "cache_control" in _content_element:
                    _anthropic_text_content_element["cache_control"] = _content_element[
                        "cache_control"
                    ]

                assistant_content.append(_anthropic_text_content_element)

            assistant_tool_calls = assistant_content_block.get("tool_calls")
            if (
                assistant_tool_calls is not None
            ):  # support assistant tool invoke conversion
                assistant_content.extend(
                    convert_to_anthropic_tool_invoke(assistant_tool_calls)
                )

            assistant_function_call = assistant_content_block.get("function_call")

            if assistant_function_call is not None:
                assistant_content.extend(
                    convert_function_to_anthropic_tool_invoke(assistant_function_call)
                )

            msg_i += 1

        if assistant_content:
            new_messages.append({"role": "assistant", "content": assistant_content})

        if msg_i == init_msg_i:  # prevent infinite loops
            raise litellm.BadRequestError(
                message=BAD_MESSAGE_ERROR_STR + f"passed in {messages[msg_i]}",
                model=model,
                llm_provider=llm_provider,
            )

    if new_messages[-1]["role"] == "assistant":
        if isinstance(new_messages[-1]["content"], str):
            new_messages[-1]["content"] = new_messages[-1]["content"].rstrip()
        elif isinstance(new_messages[-1]["content"], list):
            for content in new_messages[-1]["content"]:
                if isinstance(content, dict) and content["type"] == "text":
                    content["text"] = content[
                        "text"
                    ].rstrip()  # no trailing whitespace for final assistant message

    return new_messages


def extract_between_tags(tag: str, string: str, strip: bool = False) -> List[str]:
    ext_list = re.findall(f"<{tag}>(.+?)</{tag}>", string, re.DOTALL)
    if strip:
        ext_list = [e.strip() for e in ext_list]
    return ext_list


def contains_tag(tag: str, string: str) -> bool:
    return bool(re.search(f"<{tag}>(.+?)</{tag}>", string, re.DOTALL))


def parse_xml_params(xml_content, json_schema: Optional[dict] = None):
    """
    Compare the xml output to the json schema

    check if a value is a list - if so, get it's child elements
    """
    root = ET.fromstring(xml_content)
    params = {}

    if json_schema is not None:  # check if we have a json schema for this function call
        # iterate over all properties in the schema
        for prop in json_schema["properties"]:
            # If property is an array, get the nested items
            _element = root.find(f"parameters/{prop}")
            if json_schema["properties"][prop]["type"] == "array":
                items = []
                if _element is not None:
                    for value in _element:
                        try:
                            if value.text is not None:
                                _value = json.loads(value.text)
                            else:
                                continue
                        except json.JSONDecodeError:
                            _value = value.text
                        items.append(_value)
                    params[prop] = items
            # If property is not an array, append the value directly
            elif _element is not None and _element.text is not None:
                try:
                    _value = json.loads(_element.text)
                except json.JSONDecodeError:
                    _value = _element.text
                params[prop] = _value
    else:
        for child in root.findall(".//parameters/*"):
            if child is not None and child.text is not None:
                try:
                    # Attempt to decode the element's text as JSON
                    params[child.tag] = json.loads(child.text)  # type: ignore
                except json.JSONDecodeError:
                    # If JSON decoding fails, use the original text
                    params[child.tag] = child.text  # type: ignore

    return params


### GEMINI HELPER FUNCTIONS ###


def get_system_prompt(messages):
    system_prompt_indices = []
    system_prompt = ""
    for idx, message in enumerate(messages):
        if message["role"] == "system":
            system_prompt += message["content"]
            system_prompt_indices.append(idx)
    if len(system_prompt_indices) > 0:
        for idx in reversed(system_prompt_indices):
            messages.pop(idx)
    return system_prompt, messages


from litellm.types.llms.cohere import (
    CallObject,
    ChatHistory,
    ChatHistoryChatBot,
    ChatHistorySystem,
    ChatHistoryToolResult,
    ChatHistoryUser,
    ToolCallObject,
    ToolResultObject,
)


def convert_openai_message_to_cohere_tool_result(
    message: Union[ChatCompletionToolMessage, ChatCompletionFunctionMessage],
    tool_calls: List,
) -> ToolResultObject:
    """
    OpenAI message with a tool result looks like:
    {
            "tool_call_id": "tool_1",
            "role": "tool",
            "content": {"location": "San Francisco, CA", "unit": "fahrenheit", "temperature": "72"},
    },
    """
    """
    OpenAI message with a function call looks like:
    {
        "role": "function",
        "name": "get_current_weather",
        "content": "function result goes here",
    }
    """

    """
    Cohere tool_results look like:
    {
       "call": {
           "name": "query_daily_sales_report",
           "parameters": {
               "day": "2023-09-29"
           },
       },
       "outputs": [
           {
               "date": "2023-09-29",
               "summary": "Total Sales Amount: 10000, Total Units Sold: 250"
           }
       ]
   },
    """

    content_str: str = ""
    if isinstance(message["content"], str):
        content_str = message["content"]
    elif isinstance(message["content"], List):
        content_list = message["content"]
        for content in content_list:
            if content["type"] == "text":
                content_str += content["text"]
    if len(content_str) > 0:
        try:
            content = json.loads(content_str)
        except json.JSONDecodeError:
            content = {"result": content_str}
    else:
        content = {}
    name = ""
    arguments = {}
    # Recover name from last message with tool calls
    if len(tool_calls) > 0:
        tools = tool_calls
        msg_tool_call_id = message.get("tool_call_id", None)
        for tool in tools:
            prev_tool_call_id = tool.get("id", None)
            if (
                msg_tool_call_id
                and prev_tool_call_id
                and msg_tool_call_id == prev_tool_call_id
            ):
                name = tool.get("function", {}).get("name", "")
                arguments_str = tool.get("function", {}).get("arguments", "")
                if arguments_str is not None and len(arguments_str) > 0:
                    arguments = json.loads(arguments_str)

    if message["role"] == "function":
        function_message: ChatCompletionFunctionMessage = message
        name = function_message["name"]
        cohere_tool_result: ToolResultObject = {
            "call": CallObject(name=name, parameters=arguments),
            "outputs": [content],
        }
        return cohere_tool_result
    else:
        # We can't determine from openai message format whether it's a successful or
        # error call result so default to the successful result template

        cohere_tool_result = {
            "call": CallObject(name=name, parameters=arguments),
            "outputs": [content],
        }
        return cohere_tool_result


def get_all_tool_calls(messages: List) -> List:
    """
    Returns extracted list of `tool_calls`.

    Done to handle openai no longer returning tool call 'name' in tool results.
    """
    tool_calls: List = []
    for m in messages:
        if m.get("tool_calls", None) is not None:
            if isinstance(m["tool_calls"], list):
                tool_calls.extend(m["tool_calls"])

    return tool_calls


def convert_to_cohere_tool_invoke(tool_calls: list) -> List[ToolCallObject]:
    """
    OpenAI tool invokes:
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_abc123",
          "type": "function",
          "function": {
            "name": "get_current_weather",
            "arguments": "{\n\"location\": \"Boston, MA\"\n}"
          }
        }
      ]
    },
    """

    """
    Cohere tool invokes:
    {
      "role": "CHATBOT",
      "tool_calls": [{"name": "get_weather", "parameters": {"location": "San Francisco, CA"}}]
    }
    """

    cohere_tool_invoke: List[ToolCallObject] = [
        {
            "name": get_attribute_or_key(
                get_attribute_or_key(tool, "function"), "name"
            ),
            "parameters": json.loads(
                get_attribute_or_key(
                    get_attribute_or_key(tool, "function"), "arguments"
                )
            ),
        }
        for tool in tool_calls
        if get_attribute_or_key(tool, "type") == "function"
    ]

    return cohere_tool_invoke


def cohere_messages_pt_v2(  # noqa: PLR0915
    messages: List,
    model: str,
    llm_provider: str,
) -> Tuple[Union[str, ToolResultObject], ChatHistory]:
    """
    Returns a tuple(Union[tool_result, message], chat_history)

    - if last message is tool result -> return 'tool_result'
    - if last message is text -> return message (str)

    - return preceding messages as 'chat_history'

    Note:
    - cannot specify message if the last entry in chat history contains tool results
    - message must be at least 1 token long or tool results must be specified.
    - cannot specify tool_results if the last entry in chat history contains a user message
    """
    tool_calls: List = get_all_tool_calls(messages=messages)

    ## GET MOST RECENT MESSAGE
    most_recent_message = messages.pop(-1)
    returned_message: Union[ToolResultObject, str] = ""
    if (
        most_recent_message.get("role", "") is not None
        and most_recent_message["role"] == "tool"
    ):
        # tool result
        returned_message = convert_openai_message_to_cohere_tool_result(
            most_recent_message, tool_calls
        )
    else:
        content: Union[str, List] = most_recent_message.get("content")
        if isinstance(content, str):
            returned_message = content
        else:
            for chunk in content:
                if chunk.get("type") == "text":
                    returned_message += chunk.get("text")

    ## CREATE CHAT HISTORY
    user_message_types = {"user"}
    tool_message_types = {"tool", "function"}
    # reformat messages to ensure user/assistant are alternating, if there's either 2 consecutive 'user' messages or 2 consecutive 'assistant' message, merge them.
    new_messages: ChatHistory = []
    msg_i = 0

    while msg_i < len(messages):
        user_content: str = ""
        init_msg_i = msg_i
        ## MERGE CONSECUTIVE USER CONTENT ##
        while msg_i < len(messages) and messages[msg_i]["role"] in user_message_types:
            if isinstance(messages[msg_i]["content"], list):
                for m in messages[msg_i]["content"]:
                    if m.get("type", "") == "text":
                        user_content += m["text"]
            else:
                user_content += messages[msg_i]["content"]
            msg_i += 1

        if len(user_content) > 0:
            new_messages.append(ChatHistoryUser(role="USER", message=user_content))

        system_content: str = ""
        ## MERGE CONSECUTIVE SYSTEM CONTENT ##
        while msg_i < len(messages) and messages[msg_i]["role"] == "system":
            if isinstance(messages[msg_i]["content"], list):
                for m in messages[msg_i]["content"]:
                    if m.get("type", "") == "text":
                        system_content += m["text"]
            else:
                system_content += messages[msg_i]["content"]
            msg_i += 1

        if len(system_content) > 0:
            new_messages.append(
                ChatHistorySystem(role="SYSTEM", message=system_content)
            )

        assistant_content: str = ""
        assistant_tool_calls: List[ToolCallObject] = []
        ## MERGE CONSECUTIVE ASSISTANT CONTENT ##
        while msg_i < len(messages) and messages[msg_i]["role"] == "assistant":
            if messages[msg_i].get("content", None) is not None and isinstance(
                messages[msg_i]["content"], list
            ):
                for m in messages[msg_i]["content"]:
                    if m.get("type", "") == "text":
                        assistant_content += m["text"]
            elif messages[msg_i].get("content") is not None and isinstance(
                messages[msg_i]["content"], str
            ):
                assistant_content += messages[msg_i]["content"]
            if messages[msg_i].get(
                "tool_calls", []
            ):  # support assistant tool invoke conversion
                assistant_tool_calls.extend(
                    convert_to_cohere_tool_invoke(messages[msg_i]["tool_calls"])
                )

            if messages[msg_i].get("function_call"):
                assistant_tool_calls.extend(
                    convert_to_cohere_tool_invoke(messages[msg_i]["function_call"])
                )

            msg_i += 1

        if len(assistant_content) > 0:
            new_messages.append(
                ChatHistoryChatBot(
                    role="CHATBOT",
                    message=assistant_content,
                    tool_calls=assistant_tool_calls,
                )
            )

        ## MERGE CONSECUTIVE TOOL RESULTS
        tool_results: List[ToolResultObject] = []
        while msg_i < len(messages) and messages[msg_i]["role"] in tool_message_types:
            tool_results.append(
                convert_openai_message_to_cohere_tool_result(
                    messages[msg_i], tool_calls
                )
            )

            msg_i += 1

        if len(tool_results) > 0:
            new_messages.append(
                ChatHistoryToolResult(role="TOOL", tool_results=tool_results)
            )

        if msg_i == init_msg_i:  # prevent infinite loops
            raise litellm.BadRequestError(
                message=BAD_MESSAGE_ERROR_STR + f"passed in {messages[msg_i]}",
                model=model,
                llm_provider=llm_provider,
            )

    return returned_message, new_messages


def cohere_message_pt(messages: list):
    tool_calls: List = get_all_tool_calls(messages=messages)
    prompt = ""
    tool_results = []
    for message in messages:
        # check if this is a tool_call result
        if message["role"] == "tool":
            tool_result = convert_openai_message_to_cohere_tool_result(
                message, tool_calls=tool_calls
            )
            tool_results.append(tool_result)
        elif message.get("content"):
            prompt += message["content"] + "\n\n"
    prompt = prompt.rstrip()
    return prompt, tool_results


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
    except Exception:
        raise Exception("image conversion failed please run `pip install Pillow`")
    from io import BytesIO

    try:
        # Send a GET request to the image URL
        client = HTTPHandler(concurrent_limit=1)
        response = client.get(image_url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Check the response's content type to ensure it is an image
        content_type = response.headers.get("content-type")
        if not content_type or "image" not in content_type:
            raise ValueError(
                f"URL does not point to a valid image (content-type: {content_type})"
            )

        # Load the image from the response content
        return Image.open(BytesIO(response.content))

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
                try:
                    from PIL import Image
                except Exception:
                    raise Exception(
                        "gemini image conversion failed please run `pip install Pillow`"
                    )

                if "base64" in img:
                    # Case 2: Base64 image data
                    import base64
                    import io

                    # Extract the base64 image data
                    base64_data = img.split("base64,")[1]

                    # Decode the base64 image data
                    image_data = base64.b64decode(base64_data)

                    # Load the image from the decoded data
                    image = Image.open(io.BytesIO(image_data))
                else:
                    # Case 3: Image filepath (e.g. temp.jpeg) given
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
        pass  # type: ignore
    except Exception:
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


###### AZURE AI #######
def stringify_json_tool_call_content(messages: List) -> List:
    """

    - Check 'content' in tool role -> convert to dict (if not) -> stringify

    Done for azure_ai/cohere calls to handle results of a tool call
    """

    for m in messages:
        if m["role"] == "tool" and isinstance(m["content"], str):
            # check if content is a valid json object
            try:
                json.loads(m["content"])
            except json.JSONDecodeError:
                m["content"] = json.dumps({"result": m["content"]})

    return messages


###### AMAZON BEDROCK #######

import base64
import mimetypes
from email.message import Message

import httpx

from litellm.types.llms.bedrock import (
    BedrockConverseReasoningContentBlock,
    BedrockConverseReasoningTextBlock,
)
from litellm.types.llms.bedrock import ContentBlock as BedrockContentBlock
from litellm.types.llms.bedrock import DocumentBlock as BedrockDocumentBlock
from litellm.types.llms.bedrock import ImageBlock as BedrockImageBlock
from litellm.types.llms.bedrock import SourceBlock as BedrockSourceBlock
from litellm.types.llms.bedrock import ToolBlock as BedrockToolBlock
from litellm.types.llms.bedrock import (
    ToolInputSchemaBlock as BedrockToolInputSchemaBlock,
)
from litellm.types.llms.bedrock import ToolResultBlock as BedrockToolResultBlock
from litellm.types.llms.bedrock import (
    ToolResultContentBlock as BedrockToolResultContentBlock,
)
from litellm.types.llms.bedrock import ToolSpecBlock as BedrockToolSpecBlock
from litellm.types.llms.bedrock import ToolUseBlock as BedrockToolUseBlock


def _parse_content_type(content_type: str) -> str:
    m = Message()
    m["content-type"] = content_type
    return m.get_content_type()


class BedrockImageProcessor:
    """Handles both sync and async image processing for Bedrock conversations."""

    @staticmethod
    def _post_call_image_processing(response: httpx.Response) -> Tuple[str, str]:
        # Check the response's content type to ensure it is an image
        content_type = response.headers.get("content-type")
        if not content_type:
            raise ValueError(
                f"URL does not contain content-type (content-type: {content_type})"
            )
        content_type = _parse_content_type(content_type)

        # Convert the image content to base64 bytes
        base64_bytes = base64.b64encode(response.content).decode("utf-8")

        return base64_bytes, content_type

    @staticmethod
    async def get_image_details_async(image_url) -> Tuple[str, str]:
        try:
            client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.PromptFactory,
                params={"concurrent_limit": 1},
            )
            # Send a GET request to the image URL
            response = await client.get(image_url, follow_redirects=True)
            response.raise_for_status()  # Raise an exception for HTTP errors

            return BedrockImageProcessor._post_call_image_processing(response)

        except Exception as e:
            raise e

    @staticmethod
    def get_image_details(image_url) -> Tuple[str, str]:
        try:
            client = HTTPHandler(concurrent_limit=1)
            # Send a GET request to the image URL
            response = client.get(image_url, follow_redirects=True)
            response.raise_for_status()  # Raise an exception for HTTP errors

            return BedrockImageProcessor._post_call_image_processing(response)

        except Exception as e:
            raise e

    @staticmethod
    def _parse_base64_image(image_url: str) -> Tuple[str, str, str]:
        """Parse base64 encoded image data."""
        image_metadata, img_without_base_64 = image_url.split(",")

        # Extract MIME type using regular expression
        mime_type_match = re.match(r"data:(.*?);base64", image_metadata)
        if mime_type_match:
            mime_type = mime_type_match.group(1)
            image_format = mime_type.split("/")[1]
        else:
            mime_type = "image/jpeg"
            image_format = "jpeg"

        return img_without_base_64, mime_type, image_format

    @staticmethod
    def _validate_format(mime_type: str, image_format: str) -> str:
        """Validate image format and mime type for both images and documents."""

        supported_image_formats = (
            litellm.AmazonConverseConfig().get_supported_image_types()
        )
        supported_doc_formats = (
            litellm.AmazonConverseConfig().get_supported_document_types()
        )

        document_types = ["application", "text"]
        is_document = any(mime_type.startswith(doc_type) for doc_type in document_types)

        if is_document:
            potential_extensions = mimetypes.guess_all_extensions(mime_type)
            valid_extensions = [
                ext[1:]
                for ext in potential_extensions
                if ext[1:] in supported_doc_formats
            ]

            if not valid_extensions:
                raise ValueError(
                    f"No supported extensions for MIME type: {mime_type}. Supported formats: {supported_doc_formats}"
                )

            # Use first valid extension instead of provided image_format
            return valid_extensions[0]
        else:
            if image_format not in supported_image_formats:
                raise ValueError(
                    f"Unsupported image format: {image_format}. Supported formats: {supported_image_formats}"
                )
            return image_format

    @staticmethod
    def _create_bedrock_block(
        image_bytes: str, mime_type: str, image_format: str
    ) -> BedrockContentBlock:
        """Create appropriate Bedrock content block based on mime type."""
        _blob = BedrockSourceBlock(bytes=image_bytes)

        document_types = ["application", "text"]
        is_document = any(mime_type.startswith(doc_type) for doc_type in document_types)

        if is_document:
            return BedrockContentBlock(
                document=BedrockDocumentBlock(
                    source=_blob,
                    format=image_format,
                    name=f"DocumentPDFmessages_{str(uuid.uuid4())}",
                )
            )
        else:
            return BedrockContentBlock(
                image=BedrockImageBlock(source=_blob, format=image_format)
            )

    @classmethod
    def process_image_sync(
        cls, image_url: str, format: Optional[str] = None
    ) -> BedrockContentBlock:
        """Synchronous image processing."""

        if "base64" in image_url:
            img_bytes, mime_type, image_format = cls._parse_base64_image(image_url)
        elif "http://" in image_url or "https://" in image_url:
            img_bytes, mime_type = BedrockImageProcessor.get_image_details(image_url)
            image_format = mime_type.split("/")[1]
        else:
            raise ValueError(
                "Unsupported image type. Expected either image url or base64 encoded string"
            )

        if format:
            mime_type = format
            image_format = mime_type.split("/")[1]

        image_format = cls._validate_format(mime_type, image_format)
        return cls._create_bedrock_block(img_bytes, mime_type, image_format)

    @classmethod
    async def process_image_async(
        cls, image_url: str, format: Optional[str]
    ) -> BedrockContentBlock:
        """Asynchronous image processing."""

        if "base64" in image_url:
            img_bytes, mime_type, image_format = cls._parse_base64_image(image_url)
        elif "http://" in image_url or "https://" in image_url:
            img_bytes, mime_type = await BedrockImageProcessor.get_image_details_async(
                image_url
            )
            image_format = mime_type.split("/")[1]
        else:
            raise ValueError(
                "Unsupported image type. Expected either image url or base64 encoded string"
            )

        if format:  # override with user-defined params
            mime_type = format
            image_format = mime_type.split("/")[1]

        image_format = cls._validate_format(mime_type, image_format)
        return cls._create_bedrock_block(img_bytes, mime_type, image_format)


def _convert_to_bedrock_tool_call_invoke(
    tool_calls: list,
) -> List[BedrockContentBlock]:
    """
    OpenAI tool invokes:
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_abc123",
          "type": "function",
          "function": {
            "name": "get_current_weather",
            "arguments": "{\n\"location\": \"Boston, MA\"\n}"
          }
        }
      ]
    },
    """
    """
    Bedrock tool invokes: 
    [   
        {
            "role": "assistant",
            "toolUse": {
                "input": {"location": "Boston, MA", ..},
                "name": "get_current_weather",
                "toolUseId": "call_abc123"
            }
        }
    ]
    """
    """
    - json.loads argument
    - extract name 
    - extract id
    """

    try:
        _parts_list: List[BedrockContentBlock] = []
        for tool in tool_calls:
            if "function" in tool:
                id = tool["id"]
                name = tool["function"].get("name", "")
                arguments = tool["function"].get("arguments", "")
                arguments_dict = json.loads(arguments)
                bedrock_tool = BedrockToolUseBlock(
                    input=arguments_dict, name=name, toolUseId=id
                )
                bedrock_content_block = BedrockContentBlock(toolUse=bedrock_tool)
                _parts_list.append(bedrock_content_block)
        return _parts_list
    except Exception as e:
        raise Exception(
            "Unable to convert openai tool calls={} to bedrock tool calls. Received error={}".format(
                tool_calls, str(e)
            )
        )


def _convert_to_bedrock_tool_call_result(
    message: Union[ChatCompletionToolMessage, ChatCompletionFunctionMessage]
) -> BedrockContentBlock:
    """
    OpenAI message with a tool result looks like:
    {
        "tool_call_id": "tool_1",
        "role": "tool",
        "name": "get_current_weather",
        "content": "function result goes here",
    },

    OpenAI message with a function call result looks like:
    {
        "role": "function",
        "name": "get_current_weather",
        "content": "function result goes here",
    }
    """
    """
    Bedrock result looks like this: 
    {
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "toolUseId": "tooluse_kZJMlvQmRJ6eAyJE5GIl7Q",
                    "content": [
                        {
                            "json": {
                                "song": "Elemental Hotel",
                                "artist": "8 Storey Hike"
                            }
                        }
                    ]
                }
            }
        ]
    }
    """
    """
    - 
    """
    content_str: str = ""
    if isinstance(message["content"], str):
        content_str = message["content"]
    elif isinstance(message["content"], List):
        content_list = message["content"]
        for content in content_list:
            if content["type"] == "text":
                content_str += content["text"]
    message.get("name", "")
    id = str(message.get("tool_call_id", str(uuid.uuid4())))

    tool_result_content_block = BedrockToolResultContentBlock(text=content_str)
    tool_result = BedrockToolResultBlock(
        content=[tool_result_content_block],
        toolUseId=id,
    )
    content_block = BedrockContentBlock(toolResult=tool_result)

    return content_block


def _insert_assistant_continue_message(
    messages: List[BedrockMessageBlock],
    assistant_continue_message: Optional[
        Union[str, ChatCompletionAssistantMessage]
    ] = None,
) -> List[BedrockMessageBlock]:
    """
    Add dummy message between user/tool result blocks.

    Conversation blocks and tool result blocks cannot be provided in the same turn. Issue: https://github.com/BerriAI/litellm/issues/6053
    """
    if assistant_continue_message is not None:
        if isinstance(assistant_continue_message, str):
            messages.append(
                BedrockMessageBlock(
                    role="assistant",
                    content=[BedrockContentBlock(text=assistant_continue_message)],
                )
            )
        elif isinstance(assistant_continue_message, dict):
            text = convert_content_list_to_str(assistant_continue_message)
            messages.append(
                BedrockMessageBlock(
                    role="assistant",
                    content=[BedrockContentBlock(text=text)],
                )
            )
    elif litellm.modify_params:
        text = convert_content_list_to_str(
            cast(ChatCompletionAssistantMessage, DEFAULT_ASSISTANT_CONTINUE_MESSAGE)
        )
        messages.append(
            BedrockMessageBlock(
                role="assistant",
                content=[
                    BedrockContentBlock(text=text),
                ],
            )
        )
    return messages


def get_user_message_block_or_continue_message(
    message: ChatCompletionUserMessage,
    user_continue_message: Optional[ChatCompletionUserMessage] = None,
) -> ChatCompletionUserMessage:
    """
    Returns the user content block
    if content block is an empty string, then return the default continue message

    Relevant Issue: https://github.com/BerriAI/litellm/issues/7169
    """
    content_block = message.get("content", None)

    # Handle None case
    if content_block is None or (
        user_continue_message is None and litellm.modify_params is False
    ):
        return skip_empty_text_blocks(message=message)

    # Handle string case
    if isinstance(content_block, str):
        # check if content is empty
        if content_block.strip():
            return message
        else:
            return ChatCompletionUserMessage(
                **(user_continue_message or DEFAULT_USER_CONTINUE_MESSAGE)  # type: ignore
            )

    # Handle list case
    if isinstance(content_block, list):
        """
        CHECK FOR
            "content": [
                {
                "type": "text",
                "text": ""
                }
            ],
        """
        if not content_block:
            return ChatCompletionUserMessage(
                **(user_continue_message or DEFAULT_USER_CONTINUE_MESSAGE)  # type: ignore
            )
        # Create a copy of the message to avoid modifying the original
        modified_content_block = content_block.copy()

        for item in modified_content_block:
            # Check if the list is empty
            if item["type"] == "text":
                if not item["text"].strip():
                    # Replace empty text with continue message
                    _user_continue_message = ChatCompletionUserMessage(
                        **(user_continue_message or DEFAULT_USER_CONTINUE_MESSAGE)  # type: ignore
                    )
                    text = convert_content_list_to_str(_user_continue_message)
                    item["text"] = text
                    break
        modified_message = message.copy()
        modified_message["content"] = modified_content_block
        return modified_message

    # Handle unsupported type
    raise ValueError(f"Unsupported content type: {type(content_block)}")


def return_assistant_continue_message(
    assistant_continue_message: Optional[
        Union[str, ChatCompletionAssistantMessage]
    ] = None
) -> ChatCompletionAssistantMessage:
    if assistant_continue_message and isinstance(assistant_continue_message, str):
        return ChatCompletionAssistantMessage(
            role="assistant",
            content=assistant_continue_message,
        )
    elif assistant_continue_message and isinstance(assistant_continue_message, dict):
        return ChatCompletionAssistantMessage(**assistant_continue_message)
    else:
        return DEFAULT_ASSISTANT_CONTINUE_MESSAGE


def _skip_empty_dict_blocks(blocks: List[dict]) -> List[dict]:
    """
    Filter out empty text blocks from a list of dictionaries.

    Args:
        blocks: List of dictionaries representing message content blocks

    Returns:
        Filtered list of non-empty text blocks
    """
    return [
        item
        for item in blocks
        if not (item.get("type") == "text" and not item.get("text", "").strip())
    ]


@overload
def skip_empty_text_blocks(
    message: ChatCompletionAssistantMessage,
) -> ChatCompletionAssistantMessage:
    pass


@overload
def skip_empty_text_blocks(
    message: ChatCompletionUserMessage,
) -> ChatCompletionUserMessage:
    pass


def skip_empty_text_blocks(
    message: Union[ChatCompletionAssistantMessage, ChatCompletionUserMessage],
) -> Union[ChatCompletionAssistantMessage, ChatCompletionUserMessage]:
    """
    Skips empty text blocks in message content text blocks.

    Do not insert content here. This is a helper function, which can also be used in base case.
    """
    content_block = message.get("content", None)
    if content_block is None:
        return message
    if (
        isinstance(content_block, str)
        and not content_block.strip()
        and is_non_content_values_set(message)
        and message["role"] == "assistant"
    ):
        modified_message = message.copy()
        modified_message["content"] = None  # user message content cannot be None
        return modified_message
    elif isinstance(content_block, list):
        modified_content_block = _skip_empty_dict_blocks(
            cast(List[dict], content_block)
        )

        # If no content remains and it's an assistant message, set content to None
        if not modified_content_block and message["role"] == "assistant":
            modified_message = message.copy()
            modified_message["content"] = None
            return modified_message

        modified_message_alt = message.copy()

        # Type-specific casting based on message role
        if message["role"] == "assistant":
            modified_message_alt["content"] = cast(  # type: ignore
                Optional[List[OpenAIMessageContentListBlock]],
                modified_content_block or None,
            )
        elif message["role"] == "user" and modified_content_block is not None:
            modified_message_alt["content"] = cast(  # type: ignore
                Optional[List[ChatCompletionTextObject]], modified_content_block
            )

        return modified_message_alt

    return message


def process_empty_text_blocks(
    message: ChatCompletionAssistantMessage,
    assistant_continue_message: Optional[
        Union[str, ChatCompletionAssistantMessage]
    ] = None,
) -> ChatCompletionAssistantMessage:
    modified_content_block = message.get("content", None)
    ## BASE CASE ##
    if modified_content_block is None or not isinstance(modified_content_block, list):
        return message

    # Check if all items are empty text blocks
    if all(
        item["type"] == "text" and not item["text"].strip()
        for item in modified_content_block
    ):
        # Replace with a single continue message
        _assistant_continue_message = return_assistant_continue_message(
            assistant_continue_message
        )
        modified_content_block = [
            {
                "type": "text",
                "text": convert_content_list_to_str(_assistant_continue_message),
            }
        ]
    else:
        # Filter out only empty text blocks, keeping non-empty text and other block types
        modified_content_block = [
            item
            for item in modified_content_block
            if not (item["type"] == "text" and not item["text"].strip())
        ]

    modified_message = message.copy()
    modified_message["content"] = modified_content_block
    return modified_message


def get_assistant_message_block_or_continue_message(
    message: ChatCompletionAssistantMessage,
    assistant_continue_message: Optional[
        Union[str, ChatCompletionAssistantMessage]
    ] = None,
) -> ChatCompletionAssistantMessage:
    """
    Returns the user content block
    if content block is an empty string, then return the default continue message

    Relevant Issue: https://github.com/BerriAI/litellm/issues/7169
    """
    content_block = message.get("content", None)

    # Handle Base case
    if content_block is None or (
        assistant_continue_message is None and litellm.modify_params is False
    ):
        return skip_empty_text_blocks(message=message)

    # Handle string case
    if isinstance(content_block, str):
        # check if content is empty
        if content_block.strip():
            return message
        else:
            if is_non_content_values_set(message):
                modified_message = message.copy()
                modified_message["content"] = None
                return modified_message
            return return_assistant_continue_message(assistant_continue_message)

    # Handle list case
    if isinstance(content_block, list):
        """
        CHECK FOR
            "content": [
                {
                "type": "text",
                "text": ""
                }
            ],
        """
        return process_empty_text_blocks(
            message=message, assistant_continue_message=assistant_continue_message
        )

    # Handle unsupported type
    raise ValueError(f"Unsupported content type: {type(content_block)}")


class BedrockConverseMessagesProcessor:
    @staticmethod
    def _initial_message_setup(
        messages: List,
        user_continue_message: Optional[ChatCompletionUserMessage] = None,
    ) -> List:
        if messages[0].get("role") is not None and messages[0]["role"] == "assistant":
            if user_continue_message is not None:
                messages.insert(0, user_continue_message)
            elif litellm.modify_params:
                messages.insert(0, DEFAULT_USER_CONTINUE_MESSAGE)

        # if final message is assistant message
        if messages[-1].get("role") is not None and messages[-1]["role"] == "assistant":
            if user_continue_message is not None:
                messages.append(user_continue_message)
            elif litellm.modify_params:
                messages.append(DEFAULT_USER_CONTINUE_MESSAGE)
        return messages

    @staticmethod
    async def _bedrock_converse_messages_pt_async(  # noqa: PLR0915
        messages: List,
        model: str,
        llm_provider: str,
        user_continue_message: Optional[ChatCompletionUserMessage] = None,
        assistant_continue_message: Optional[
            Union[str, ChatCompletionAssistantMessage]
        ] = None,
    ) -> List[BedrockMessageBlock]:
        contents: List[BedrockMessageBlock] = []
        msg_i = 0

        ## BASE CASE ##
        if len(messages) == 0:
            raise litellm.BadRequestError(
                message=BAD_MESSAGE_ERROR_STR
                + "bedrock requires at least one non-system message",
                model=model,
                llm_provider=llm_provider,
            )

        # if initial message is assistant message
        messages = BedrockConverseMessagesProcessor._initial_message_setup(
            messages, user_continue_message
        )

        while msg_i < len(messages):
            user_content: List[BedrockContentBlock] = []
            init_msg_i = msg_i
            ## MERGE CONSECUTIVE USER CONTENT ##
            while msg_i < len(messages) and messages[msg_i]["role"] == "user":
                message_block = get_user_message_block_or_continue_message(
                    message=messages[msg_i],
                    user_continue_message=user_continue_message,
                )
                if isinstance(message_block["content"], list):
                    _parts: List[BedrockContentBlock] = []
                    for element in message_block["content"]:
                        if isinstance(element, dict):
                            if element["type"] == "text":
                                _part = BedrockContentBlock(text=element["text"])
                                _parts.append(_part)
                            elif element["type"] == "image_url":
                                format: Optional[str] = None
                                if isinstance(element["image_url"], dict):
                                    image_url = element["image_url"]["url"]
                                    format = element["image_url"].get("format")
                                else:
                                    image_url = element["image_url"]
                                _part = await BedrockImageProcessor.process_image_async(  # type: ignore
                                    image_url=image_url, format=format
                                )
                                _parts.append(_part)  # type: ignore
                            elif element["type"] == "file":
                                _part = await BedrockConverseMessagesProcessor._async_process_file_message(
                                    message=cast(ChatCompletionFileObject, element)
                                )
                                _parts.append(_part)
                            _cache_point_block = (
                                litellm.AmazonConverseConfig()._get_cache_point_block(
                                    message_block=cast(
                                        OpenAIMessageContentListBlock, element
                                    ),
                                    block_type="content_block",
                                )
                            )
                            if _cache_point_block is not None:
                                _parts.append(_cache_point_block)
                    user_content.extend(_parts)
                elif message_block["content"] and isinstance(
                    message_block["content"], str
                ):
                    _part = BedrockContentBlock(text=messages[msg_i]["content"])
                    _cache_point_block = (
                        litellm.AmazonConverseConfig()._get_cache_point_block(
                            message_block, block_type="content_block"
                        )
                    )
                    user_content.append(_part)
                    if _cache_point_block is not None:
                        user_content.append(_cache_point_block)

                msg_i += 1
            if user_content:
                if len(contents) > 0 and contents[-1]["role"] == "user":
                    if (
                        assistant_continue_message is not None
                        or litellm.modify_params is True
                    ):
                        # if last message was a 'user' message, then add a dummy assistant message (bedrock requires alternating roles)
                        contents = _insert_assistant_continue_message(
                            messages=contents,
                            assistant_continue_message=assistant_continue_message,
                        )
                        contents.append(
                            BedrockMessageBlock(role="user", content=user_content)
                        )
                    else:
                        verbose_logger.warning(
                            "Potential consecutive user/tool blocks. Trying to merge. If error occurs, please set a 'assistant_continue_message' or set 'modify_params=True' to insert a dummy assistant message for bedrock calls."
                        )
                        contents[-1]["content"].extend(user_content)
                else:
                    contents.append(
                        BedrockMessageBlock(role="user", content=user_content)
                    )

            ## MERGE CONSECUTIVE TOOL CALL MESSAGES ##
            tool_content: List[BedrockContentBlock] = []
            while msg_i < len(messages) and messages[msg_i]["role"] == "tool":
                tool_call_result = _convert_to_bedrock_tool_call_result(messages[msg_i])

                tool_content.append(tool_call_result)
                msg_i += 1
            if tool_content:
                # if last message was a 'user' message, then add a blank assistant message (bedrock requires alternating roles)
                if len(contents) > 0 and contents[-1]["role"] == "user":
                    if (
                        assistant_continue_message is not None
                        or litellm.modify_params is True
                    ):
                        # if last message was a 'user' message, then add a dummy assistant message (bedrock requires alternating roles)
                        contents = _insert_assistant_continue_message(
                            messages=contents,
                            assistant_continue_message=assistant_continue_message,
                        )
                        contents.append(
                            BedrockMessageBlock(role="user", content=tool_content)
                        )
                    else:
                        verbose_logger.warning(
                            "Potential consecutive user/tool blocks. Trying to merge. If error occurs, please set a 'assistant_continue_message' or set 'modify_params=True' to insert a dummy assistant message for bedrock calls."
                        )
                        contents[-1]["content"].extend(tool_content)
                else:
                    contents.append(
                        BedrockMessageBlock(role="user", content=tool_content)
                    )
            assistant_content: List[BedrockContentBlock] = []
            ## MERGE CONSECUTIVE ASSISTANT CONTENT ##
            while msg_i < len(messages) and messages[msg_i]["role"] == "assistant":
                assistant_message_block = (
                    get_assistant_message_block_or_continue_message(
                        message=messages[msg_i],
                        assistant_continue_message=assistant_continue_message,
                    )
                )
                _assistant_content = assistant_message_block.get("content", None)

                if _assistant_content is not None and isinstance(
                    _assistant_content, list
                ):
                    assistants_parts: List[BedrockContentBlock] = []
                    for element in _assistant_content:
                        if isinstance(element, dict):
                            if element["type"] == "thinking":
                                thinking_block = BedrockConverseMessagesProcessor.translate_thinking_blocks_to_reasoning_content_blocks(
                                    thinking_blocks=[
                                        cast(ChatCompletionThinkingBlock, element)
                                    ]
                                )
                                assistants_parts.extend(thinking_block)
                            elif element["type"] == "text":
                                assistants_part = BedrockContentBlock(
                                    text=element["text"]
                                )
                                assistants_parts.append(assistants_part)
                            elif element["type"] == "image_url":
                                if isinstance(element["image_url"], dict):
                                    image_url = element["image_url"]["url"]
                                else:
                                    image_url = element["image_url"]
                                assistants_part = await BedrockImageProcessor.process_image_async(  # type: ignore
                                    image_url=image_url
                                )
                                assistants_parts.append(assistants_part)
                    assistant_content.extend(assistants_parts)
                elif _assistant_content is not None and isinstance(
                    _assistant_content, str
                ):
                    assistant_content.append(
                        BedrockContentBlock(text=_assistant_content)
                    )
                _tool_calls = assistant_message_block.get("tool_calls", [])
                if _tool_calls:
                    assistant_content.extend(
                        _convert_to_bedrock_tool_call_invoke(_tool_calls)
                    )

                msg_i += 1

            if assistant_content:
                contents.append(
                    BedrockMessageBlock(role="assistant", content=assistant_content)
                )

            if msg_i == init_msg_i:  # prevent infinite loops
                raise litellm.BadRequestError(
                    message=BAD_MESSAGE_ERROR_STR + f"passed in {messages[msg_i]}",
                    model=model,
                    llm_provider=llm_provider,
                )

        return contents

    @staticmethod
    def translate_thinking_blocks_to_reasoning_content_blocks(
        thinking_blocks: List[ChatCompletionThinkingBlock],
    ) -> List[BedrockContentBlock]:
        reasoning_content_blocks: List[BedrockContentBlock] = []
        for thinking_block in thinking_blocks:
            reasoning_text = thinking_block.get("thinking")
            reasoning_signature = thinking_block.get("signature")
            text_block = BedrockConverseReasoningTextBlock(
                text=reasoning_text or "",
            )
            if reasoning_signature is not None:
                text_block["signature"] = reasoning_signature
            reasoning_content_block = BedrockConverseReasoningContentBlock(
                reasoningText=text_block,
            )
            bedrock_content_block = BedrockContentBlock(
                reasoningContent=reasoning_content_block
            )
            reasoning_content_blocks.append(bedrock_content_block)
        return reasoning_content_blocks

    @staticmethod
    def _process_file_message(message: ChatCompletionFileObject) -> BedrockContentBlock:
        file_message = message["file"]
        file_data = file_message.get("file_data")
        file_id = file_message.get("file_id")

        if file_data is None and file_id is None:
            raise litellm.BadRequestError(
                message="file_data and file_id cannot both be None. Got={}".format(
                    message
                ),
                model="",
                llm_provider="bedrock",
            )
        format = file_message.get("format")
        return BedrockImageProcessor.process_image_sync(
            image_url=cast(str, file_id or file_data), format=format
        )

    @staticmethod
    async def _async_process_file_message(
        message: ChatCompletionFileObject,
    ) -> BedrockContentBlock:
        file_message = message["file"]
        file_data = file_message.get("file_data")
        file_id = file_message.get("file_id")
        format = file_message.get("format")
        if file_data is None and file_id is None:
            raise litellm.BadRequestError(
                message="file_data and file_id cannot both be None. Got={}".format(
                    message
                ),
                model="",
                llm_provider="bedrock",
            )
        return await BedrockImageProcessor.process_image_async(
            image_url=cast(str, file_id or file_data), format=format
        )


def _bedrock_converse_messages_pt(  # noqa: PLR0915
    messages: List,
    model: str,
    llm_provider: str,
    user_continue_message: Optional[ChatCompletionUserMessage] = None,
    assistant_continue_message: Optional[
        Union[str, ChatCompletionAssistantMessage]
    ] = None,
) -> List[BedrockMessageBlock]:
    """
    Converts given messages from OpenAI format to Bedrock format

    - Roles must alternate b/w 'user' and 'model' (same as anthropic -> merge consecutive roles)
    - Please ensure that function response turn comes immediately after a function call turn
    - Conversation blocks and tool result blocks cannot be provided in the same turn. Issue: https://github.com/BerriAI/litellm/issues/6053
    """

    contents: List[BedrockMessageBlock] = []
    msg_i = 0

    ## BASE CASE ##
    if len(messages) == 0:
        raise litellm.BadRequestError(
            message=BAD_MESSAGE_ERROR_STR
            + "bedrock requires at least one non-system message",
            model=model,
            llm_provider=llm_provider,
        )

    # if initial message is assistant message
    if messages[0].get("role") is not None and messages[0]["role"] == "assistant":
        if user_continue_message is not None:
            messages.insert(0, user_continue_message)
        elif litellm.modify_params:
            messages.insert(0, DEFAULT_USER_CONTINUE_MESSAGE)

    # if final message is assistant message
    if messages[-1].get("role") is not None and messages[-1]["role"] == "assistant":
        if user_continue_message is not None:
            messages.append(user_continue_message)
        elif litellm.modify_params:
            messages.append(DEFAULT_USER_CONTINUE_MESSAGE)

    while msg_i < len(messages):
        user_content: List[BedrockContentBlock] = []
        init_msg_i = msg_i
        ## MERGE CONSECUTIVE USER CONTENT ##
        while msg_i < len(messages) and messages[msg_i]["role"] == "user":
            message_block = get_user_message_block_or_continue_message(
                message=messages[msg_i],
                user_continue_message=user_continue_message,
            )
            if isinstance(message_block["content"], list):
                _parts: List[BedrockContentBlock] = []
                for element in message_block["content"]:
                    if isinstance(element, dict):
                        if element["type"] == "text":
                            _part = BedrockContentBlock(text=element["text"])
                            _parts.append(_part)
                        elif element["type"] == "image_url":
                            format: Optional[str] = None
                            if isinstance(element["image_url"], dict):
                                image_url = element["image_url"]["url"]
                                format = element["image_url"].get("format")
                            else:
                                image_url = element["image_url"]
                            _part = BedrockImageProcessor.process_image_sync(  # type: ignore
                                image_url=image_url,
                                format=format,
                            )
                            _parts.append(_part)  # type: ignore
                        elif element["type"] == "file":
                            _part = (
                                BedrockConverseMessagesProcessor._process_file_message(
                                    message=cast(ChatCompletionFileObject, element)
                                )
                            )
                            _parts.append(_part)
                        _cache_point_block = (
                            litellm.AmazonConverseConfig()._get_cache_point_block(
                                message_block=cast(
                                    OpenAIMessageContentListBlock, element
                                ),
                                block_type="content_block",
                            )
                        )
                        if _cache_point_block is not None:
                            _parts.append(_cache_point_block)
                user_content.extend(_parts)
            elif message_block["content"] and isinstance(message_block["content"], str):
                _part = BedrockContentBlock(text=messages[msg_i]["content"])
                _cache_point_block = (
                    litellm.AmazonConverseConfig()._get_cache_point_block(
                        message_block, block_type="content_block"
                    )
                )
                user_content.append(_part)
                if _cache_point_block is not None:
                    user_content.append(_cache_point_block)

            msg_i += 1
        if user_content:
            if len(contents) > 0 and contents[-1]["role"] == "user":
                if (
                    assistant_continue_message is not None
                    or litellm.modify_params is True
                ):
                    # if last message was a 'user' message, then add a dummy assistant message (bedrock requires alternating roles)
                    contents = _insert_assistant_continue_message(
                        messages=contents,
                        assistant_continue_message=assistant_continue_message,
                    )
                    contents.append(
                        BedrockMessageBlock(role="user", content=user_content)
                    )
                else:
                    verbose_logger.warning(
                        "Potential consecutive user/tool blocks. Trying to merge. If error occurs, please set a 'assistant_continue_message' or set 'modify_params=True' to insert a dummy assistant message for bedrock calls."
                    )
                    contents[-1]["content"].extend(user_content)
            else:
                contents.append(BedrockMessageBlock(role="user", content=user_content))

        ## MERGE CONSECUTIVE TOOL CALL MESSAGES ##
        tool_content: List[BedrockContentBlock] = []
        while msg_i < len(messages) and messages[msg_i]["role"] == "tool":
            tool_call_result = _convert_to_bedrock_tool_call_result(messages[msg_i])

            tool_content.append(tool_call_result)
            msg_i += 1
        if tool_content:
            # if last message was a 'user' message, then add a blank assistant message (bedrock requires alternating roles)
            if len(contents) > 0 and contents[-1]["role"] == "user":
                if (
                    assistant_continue_message is not None
                    or litellm.modify_params is True
                ):
                    # if last message was a 'user' message, then add a dummy assistant message (bedrock requires alternating roles)
                    contents = _insert_assistant_continue_message(
                        messages=contents,
                        assistant_continue_message=assistant_continue_message,
                    )
                    contents.append(
                        BedrockMessageBlock(role="user", content=tool_content)
                    )
                else:
                    verbose_logger.warning(
                        "Potential consecutive user/tool blocks. Trying to merge. If error occurs, please set a 'assistant_continue_message' or set 'modify_params=True' to insert a dummy assistant message for bedrock calls."
                    )
                    contents[-1]["content"].extend(tool_content)
            else:
                contents.append(BedrockMessageBlock(role="user", content=tool_content))
        assistant_content: List[BedrockContentBlock] = []
        ## MERGE CONSECUTIVE ASSISTANT CONTENT ##
        while msg_i < len(messages) and messages[msg_i]["role"] == "assistant":
            assistant_message_block = get_assistant_message_block_or_continue_message(
                message=messages[msg_i],
                assistant_continue_message=assistant_continue_message,
            )
            _assistant_content = assistant_message_block.get("content", None)
            thinking_blocks = cast(
                Optional[List[ChatCompletionThinkingBlock]],
                assistant_message_block.get("thinking_blocks"),
            )

            if thinking_blocks is not None:
                assistant_content.extend(
                    BedrockConverseMessagesProcessor.translate_thinking_blocks_to_reasoning_content_blocks(
                        thinking_blocks
                    )
                )

            if _assistant_content is not None and isinstance(_assistant_content, list):
                assistants_parts: List[BedrockContentBlock] = []
                for element in _assistant_content:
                    if isinstance(element, dict):
                        if element["type"] == "thinking":
                            thinking_block = BedrockConverseMessagesProcessor.translate_thinking_blocks_to_reasoning_content_blocks(
                                thinking_blocks=[
                                    cast(ChatCompletionThinkingBlock, element)
                                ]
                            )
                            assistants_parts.extend(thinking_block)
                        elif element["type"] == "text":
                            assistants_part = BedrockContentBlock(text=element["text"])
                            assistants_parts.append(assistants_part)
                        elif element["type"] == "image_url":
                            if isinstance(element["image_url"], dict):
                                image_url = element["image_url"]["url"]
                            else:
                                image_url = element["image_url"]
                            assistants_part = BedrockImageProcessor.process_image_sync(  # type: ignore
                                image_url=image_url
                            )
                            assistants_parts.append(assistants_part)
                assistant_content.extend(assistants_parts)
            elif _assistant_content is not None and isinstance(_assistant_content, str):
                assistant_content.append(BedrockContentBlock(text=_assistant_content))
            _tool_calls = assistant_message_block.get("tool_calls", [])
            if _tool_calls:
                assistant_content.extend(
                    _convert_to_bedrock_tool_call_invoke(_tool_calls)
                )

            msg_i += 1

        if assistant_content:
            contents.append(
                BedrockMessageBlock(role="assistant", content=assistant_content)
            )

        if msg_i == init_msg_i:  # prevent infinite loops
            raise litellm.BadRequestError(
                message=BAD_MESSAGE_ERROR_STR + f"passed in {messages[msg_i]}",
                model=model,
                llm_provider=llm_provider,
            )

    return contents


def make_valid_bedrock_tool_name(input_tool_name: str) -> str:
    """
    Replaces any invalid characters in the input tool name with underscores
    and ensures the resulting string is a valid identifier for Bedrock tools
    """

    def replace_invalid(char):
        """
        Bedrock tool names only supports alpha-numeric characters and underscores
        """
        if char.isalnum() or char == "_":
            return char
        return "_"

    # If the string is empty, return a default valid identifier
    if input_tool_name is None or len(input_tool_name) == 0:
        return input_tool_name
    bedrock_tool_name = copy.copy(input_tool_name)
    # If it doesn't start with a letter, prepend 'a'
    if not bedrock_tool_name[0].isalpha():
        bedrock_tool_name = "a" + bedrock_tool_name

    # Replace any invalid characters with underscores
    valid_string = "".join(replace_invalid(char) for char in bedrock_tool_name)

    if input_tool_name != valid_string:
        # passed tool name was formatted to become valid
        # store it internally so we can use for the response
        litellm.bedrock_tool_name_mappings.set_cache(
            key=valid_string, value=input_tool_name
        )

    return valid_string


def _bedrock_tools_pt(tools: List) -> List[BedrockToolBlock]:
    """
    OpenAI tools looks like:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            }
        }
    ]
    """
    """
    Bedrock toolConfig looks like: 
    "tools": [
        {
            "toolSpec": {
                "name": "top_song",
                "description": "Get the most popular song played on a radio station.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "sign": {
                                "type": "string",
                                "description": "The call sign for the radio station for which you want the most popular song. Example calls signs are WZPZ, and WKRP."
                            }
                        },
                        "required": [
                            "sign"
                        ]
                    }
                }
            }
        }
    ]
    """
    from litellm.litellm_core_utils.prompt_templates.common_utils import unpack_defs

    tool_block_list: List[BedrockToolBlock] = []
    for tool in tools:
        parameters = tool.get("function", {}).get(
            "parameters", {"type": "object", "properties": {}}
        )
        name = tool.get("function", {}).get("name", "")

        # related issue: https://github.com/BerriAI/litellm/issues/5007
        # Bedrock tool names must satisfy regular expression pattern: [a-zA-Z][a-zA-Z0-9_]* ensure this is true
        name = make_valid_bedrock_tool_name(input_tool_name=name)
        description = tool.get("function", {}).get(
            "description", name
        )  # converse api requires a description

        defs = parameters.pop("$defs", {})
        defs_copy = copy.deepcopy(defs)
        # flatten the defs
        for _, value in defs_copy.items():
            unpack_defs(value, defs_copy)
        unpack_defs(parameters, defs_copy)
        tool_input_schema = BedrockToolInputSchemaBlock(json=parameters)
        tool_spec = BedrockToolSpecBlock(
            inputSchema=tool_input_schema, name=name, description=description
        )
        tool_block = BedrockToolBlock(toolSpec=tool_spec)
        tool_block_list.append(tool_block)

    return tool_block_list


# Function call template
def function_call_prompt(messages: list, functions: list):
    function_prompt = """Produce JSON OUTPUT ONLY! Adhere to this format {"name": "function_name", "arguments":{"argument_name": "argument_value"}} The following functions are available to you:"""
    for function in functions:
        function_prompt += f"""\n{function}\n"""

    function_added_to_prompt = False
    for message in messages:
        if "system" in message["role"]:
            message["content"] += f""" {function_prompt}"""
            function_added_to_prompt = True

    if function_added_to_prompt is False:
        messages.append({"role": "system", "content": f"""{function_prompt}"""})

    return messages


def response_schema_prompt(model: str, response_schema: dict) -> str:
    """
    Decides if a user-defined custom prompt or default needs to be used

    Returns the prompt str that's passed to the model as a user message
    """
    custom_prompt_details: Optional[dict] = None
    response_schema_as_message = [
        {"role": "user", "content": "{}".format(response_schema)}
    ]
    if f"{model}/response_schema_prompt" in litellm.custom_prompt_dict:
        custom_prompt_details = litellm.custom_prompt_dict[
            f"{model}/response_schema_prompt"
        ]  # allow user to define custom response schema prompt by model
    elif "response_schema_prompt" in litellm.custom_prompt_dict:
        custom_prompt_details = litellm.custom_prompt_dict["response_schema_prompt"]

    if custom_prompt_details is not None:
        return custom_prompt(
            role_dict=custom_prompt_details["roles"],
            initial_prompt_value=custom_prompt_details["initial_prompt_value"],
            final_prompt_value=custom_prompt_details["final_prompt_value"],
            messages=response_schema_as_message,
        )
    else:
        return default_response_schema_prompt(response_schema=response_schema)


def default_response_schema_prompt(response_schema: dict) -> str:
    """
    Used if provider/model doesn't support 'response_schema' param.

    This is the default prompt. Allow user to override this with a custom_prompt.
    """
    prompt_str = """Use this JSON schema: 
    ```json 
    {}
    ```""".format(
        response_schema
    )
    return prompt_str


# Custom prompt template
def custom_prompt(
    role_dict: dict,
    messages: list,
    initial_prompt_value: str = "",
    final_prompt_value: str = "",
    bos_token: str = "",
    eos_token: str = "",
) -> str:
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
        if isinstance(message["content"], str):
            prompt += pre_message_str + message["content"] + post_message_str
        elif isinstance(message["content"], list):
            text_str = ""
            for content in message["content"]:
                if content.get("text", None) is not None and isinstance(
                    content["text"], str
                ):
                    text_str += content["text"]
            prompt += pre_message_str + text_str + post_message_str

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
        if litellm.AnthropicTextConfig._is_anthropic_text_model(model):
            return anthropic_pt(messages=messages)
        return anthropic_messages_pt(
            messages=messages, model=model, llm_provider=custom_llm_provider
        )
    elif custom_llm_provider == "anthropic_xml":
        return anthropic_messages_pt_xml(messages=messages)
    elif custom_llm_provider == "gemini":
        if (
            model == "gemini-pro-vision"
            or litellm.supports_vision(model=model)
            or litellm.supports_vision(model=custom_llm_provider + "/" + model)
        ):
            return _gemini_vision_convert_messages(messages=messages)
        else:
            return gemini_text_image_pt(messages=messages)
    elif custom_llm_provider == "mistral":
        return litellm.MistralConfig()._transform_messages(
            messages=messages, model=model
        )
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
        elif "llama2" in model and "chat" in model:
            return llama_2_chat_pt(messages=messages)
        elif "llama3" in model and "instruct" in model:
            return hf_chat_template(
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                messages=messages,
            )

    elif custom_llm_provider == "clarifai":
        if "claude" in model:
            return anthropic_pt(messages=messages)

    elif custom_llm_provider == "perplexity":
        for message in messages:
            message.pop("name", None)
        return messages
    elif custom_llm_provider == "azure_text":
        return azure_text_pt(messages=messages)
    elif custom_llm_provider == "watsonx":
        if "granite" in model and "chat" in model:
            # granite-13b-chat-v1 and granite-13b-chat-v2 use a specific prompt template
            return ibm_granite_pt(messages=messages)
        elif "ibm-mistral" in model and "instruct" in model:
            # models like ibm-mistral/mixtral-8x7b-instruct-v01-q use the mistral instruct prompt template
            return mistral_instruct_pt(messages=messages)
        elif "meta-llama/llama-3" in model and "instruct" in model:
            # https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
            return custom_prompt(
                role_dict={
                    "system": {
                        "pre_message": "<|start_header_id|>system<|end_header_id|>\n",
                        "post_message": "<|eot_id|>",
                    },
                    "user": {
                        "pre_message": "<|start_header_id|>user<|end_header_id|>\n",
                        "post_message": "<|eot_id|>",
                    },
                    "assistant": {
                        "pre_message": "<|start_header_id|>assistant<|end_header_id|>\n",
                        "post_message": "<|eot_id|>",
                    },
                },
                messages=messages,
                initial_prompt_value="<|begin_of_text|>",
                final_prompt_value="<|start_header_id|>assistant<|end_header_id|>\n",
            )
    try:
        if "meta-llama/llama-2" in model and "chat" in model:
            return llama_2_chat_pt(messages=messages)
        elif (
            "meta-llama/llama-3" in model or "meta-llama-3" in model
        ) and "instruct" in model:
            return hf_chat_template(
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                messages=messages,
            )
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
    except Exception:
        return default_pt(
            messages=messages
        )  # default that covers Bloom, T-5, any non-chat tuned model (e.g. base Llama2)


def get_attribute_or_key(tool_or_function, attribute, default=None):
    if hasattr(tool_or_function, attribute):
        return getattr(tool_or_function, attribute)
    return tool_or_function.get(attribute, default)
