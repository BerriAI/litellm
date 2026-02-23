import json
from datetime import datetime
from typing import Any, Dict, Union

from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.llms.custom_http import httpxSpecialProvider


def strftime_now(fmt: str) -> str:
    """
    Custom function for templates that need current date/time formatting (e.g., gpt-oss)
    
    Args:
        fmt: Format string for datetime.now().strftime()

    Returns:
        Formatted string
    """
    return datetime.now().strftime(fmt)


def _get_tokenizer_config(hf_model_name: str) -> Dict[str, Any]:
    """
    Fetch tokenizer_config.json from HuggingFace (sync)
    
    Args:
        hf_model_name: HuggingFace model name (e.g., 'openai/gpt-oss-120b')
        
    Returns:
        Dict with 'status' and optionally 'tokenizer' keys
    """
    try:
        url = f"https://huggingface.co/{hf_model_name}/raw/main/tokenizer_config.json"
        client = _get_httpx_client()
        response = client.get(url=url)
    except Exception as e:
        raise e
    if response.status_code == 200:
        tokenizer_config = json.loads(response.content)
        return {"status": "success", "tokenizer": tokenizer_config}
    else:
        return {"status": "failure"}


async def _aget_tokenizer_config(hf_model_name: str) -> Dict[str, Any]:
    """
    Fetch tokenizer_config.json from HuggingFace (async)
    
    Args:
        hf_model_name: HuggingFace model name (e.g., 'openai/gpt-oss-120b')
        
    Returns:
        Dict with 'status' and optionally 'tokenizer' keys
    """
    try:
        url = f"https://huggingface.co/{hf_model_name}/raw/main/tokenizer_config.json"
        client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.PromptFactory,
        )
        response = await client.get(url=url)
    except Exception as e:
        raise e
    if response.status_code == 200:
        tokenizer_config = json.loads(response.content)
        return {"status": "success", "tokenizer": tokenizer_config}
    else:
        return {"status": "failure"}


def _get_chat_template_file(hf_model_name: str) -> Dict[str, Any]:
    """
    Fetch chat template from separate .jinja file (sync)
    
    Args:
        hf_model_name: HuggingFace model name (e.g., 'openai/gpt-oss-120b')
        
    Returns:
        Dict with 'status' and optionally 'chat_template' keys
    """
    template_filenames = ["chat_template.jinja", "chat_template.jinja2"]
    client = _get_httpx_client()
    
    for filename in template_filenames:
        try:
            url = f"https://huggingface.co/{hf_model_name}/raw/main/{filename}"
            response = client.get(url=url)
            if response.status_code == 200:
                return {"status": "success", "chat_template": response.content.decode("utf-8")}
        except Exception:
            continue
    
    return {"status": "failure"}


async def _aget_chat_template_file(hf_model_name: str) -> Dict[str, Any]:
    """
    Fetch chat template from separate .jinja file (async)
    
    Args:
        hf_model_name: HuggingFace model name (e.g., 'openai/gpt-oss-120b')
        
    Returns:
        Dict with 'status' and optionally 'chat_template' keys
    """
    template_filenames = ["chat_template.jinja", "chat_template.jinja2"]
    client = get_async_httpx_client(
        llm_provider=httpxSpecialProvider.PromptFactory,
    )
    
    for filename in template_filenames:
        try:
            url = f"https://huggingface.co/{hf_model_name}/raw/main/{filename}"
            response = await client.get(url=url)
            if response.status_code == 200:
                return {"status": "success", "chat_template": response.content.decode("utf-8")}
        except Exception:
            continue
    
    return {"status": "failure"}


def _extract_token_value(token_value: Union[None, str, Dict[str, Any]]) -> str:
    """
    Extract token string from various formats (string, dict, etc.)
    
    Args:
        token_value: Token value in various formats (None, str, or dict with 'content' key)
        
    Returns:
        Extracted token string
    """
    if token_value is None or isinstance(token_value, str):
        return token_value or ""
    if isinstance(token_value, dict):
        return token_value.get("content", "")
    return ""