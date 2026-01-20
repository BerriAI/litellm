"""
Constants and helpers for Qwen OAuth.
"""
import platform
from typing import Optional, Union

import httpx

from litellm._version import version as litellm_version
from litellm.llms.base_llm.chat.transformation import BaseLLMException

QWEN_OAUTH_BASE_URL = "https://chat.qwen.ai"
QWEN_OAUTH_DEVICE_CODE_ENDPOINT = f"{QWEN_OAUTH_BASE_URL}/api/v1/oauth2/device/code"
QWEN_OAUTH_TOKEN_ENDPOINT = f"{QWEN_OAUTH_BASE_URL}/api/v1/oauth2/token"
QWEN_OAUTH_CLIENT_ID = "f0304373b74a44d2b584a3fb70ca9e56"
QWEN_OAUTH_SCOPE = "openid profile email model.completion"
QWEN_OAUTH_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
QWEN_DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_TOKEN_DIRNAME = ".qwen"
QWEN_OAUTH_CREDENTIAL_FILE = "oauth_creds.json"


class QwenAIError(BaseLLMException):
    def __init__(
        self,
        status_code,
        message,
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
        headers: Optional[Union[httpx.Headers, dict]] = None,
        body: Optional[dict] = None,
    ):
        super().__init__(
            status_code=status_code,
            message=message,
            request=request,
            response=response,
            headers=headers,
            body=body,
        )


class GetDeviceCodeError(QwenAIError):
    pass


class GetAccessTokenError(QwenAIError):
    pass


class RefreshAccessTokenError(QwenAIError):
    pass


class CredentialsClearRequiredError(QwenAIError):
    pass


def normalize_qwen_api_base(api_base: Optional[str]) -> Optional[str]:
    if not api_base:
        return None
    base = api_base.strip()
    if not base:
        return None
    if not base.startswith(("http://", "https://")):
        base = f"https://{base}"
    base = base.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def _safe_header_value(value: str) -> str:
    if not value:
        return ""
    return "".join(ch if 32 <= ord(ch) <= 126 else "_" for ch in value)


def get_qwen_user_agent() -> str:
    os_type = platform.system() or "Unknown"
    os_version = platform.release() or "0"
    arch = platform.machine() or "unknown"
    candidate = f"litellm/{litellm_version} ({os_type} {os_version}; {arch})"
    return _safe_header_value(candidate) or "litellm/unknown"
