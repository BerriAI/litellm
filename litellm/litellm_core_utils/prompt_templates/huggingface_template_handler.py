import json
from datetime import datetime
from typing import Any, Dict, Union

from litellm.llms.custom_httpx.http_handler import HTTPHandler


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
    Fetch tokenizer_config.json from HuggingFace
    
    Args:
        hf_model_name: HuggingFace model name (e.g., 'openai/gpt-oss-120b')
        
    Returns:
        Dict with 'status' and optionally 'tokenizer' keys
    """
    try:
        url = f"https://huggingface.co/{hf_model_name}/raw/main/tokenizer_config.json"
        client = HTTPHandler(concurrent_limit=1)
        response = client.get(url)
    except Exception as e:
        raise e
    if response.status_code == 200:
        tokenizer_config = json.loads(response.content)
        return {"status": "success", "tokenizer": tokenizer_config}
    else:
        return {"status": "failure"}


def _get_chat_template_file(hf_model_name: str) -> Dict[str, Any]:
    """
    Fetch chat template from separate .jinja file (for models like gpt-oss)
    
    Args:
        hf_model_name: HuggingFace model name (e.g., 'openai/gpt-oss-120b')
        
    Returns:
        Dict with 'status' and optionally 'chat_template' keys
    """
    template_filenames = ["chat_template.jinja", "chat_template.jinja2"]
    client = HTTPHandler(concurrent_limit=1)
    
    for filename in template_filenames:
        try:
            url = f"https://huggingface.co/{hf_model_name}/raw/main/{filename}"
            response = client.get(url)
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