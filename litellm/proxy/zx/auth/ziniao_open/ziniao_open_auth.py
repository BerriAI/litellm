"""
Ziniao Open API authentication utilities for MCP OpenAPI servers.

This module provides utilities for generating RSA2 signatures for Ziniao Open API requests,
following the same pattern as the OpenClient implementation.
"""

import json
import time
import requests
from typing import Any, Dict, Optional

from litellm._logging import verbose_logger
from litellm.caching.caching import DualCache
from . import SignUtil

token_cache = DualCache(
    default_in_memory_ttl = 2 * 60 * 60 - 30
)

def request_sign(client_id, client_secret, base_url, method, path, url, params, json_body, headers):
    if client_id and client_secret:
        # Build Ziniao parameters
        ziniao_params = _build_params(
            client_id=client_id,
            client_secret=client_secret,
            path=path,
            params_model=params,
            biz_model=json_body,
            app_token=get_token(client_id, client_secret, base_url)
        )

        # For GET requests, add to query params
        if method == "get":
            params = ziniao_params
        else:
            json_body = ziniao_params
    return base_url, params, json_body, headers

def get_token(client_id, client_secret, base_url):
    token: Optional[str] = token_cache.get_cache(key=client_id)
    if token:
        return token

    path = '/auth/get_app_token'
    all_params = _build_params(client_id, client_secret, path, {}, {}, None)
    resp = requests.request('POST', base_url, json=all_params)
    response_dict = json.loads(resp.text)
    if response_dict.get('error_response') is None and response_dict['code'] == '0' and response_dict['msg'] == "SUCCESS":
        token = response_dict['data']['appAuthToken']
        expiresIn = int(response_dict['data']['expiresIn'])
        token_cache.set_cache(client_id, token, local_only=True, ttl=expiresIn - 60)
        return token
    raise ValueError(f"ziniao open api get token error: {resp.text}")

def _build_params(client_id, client_secret, path, biz_model, params_model, app_token = None):
    """构建所有的请求参数

    :param request: 请求对象
    :type request: request.BaseRequest

    :param params: 业务请求参数
    :type params: dict

    :param token: token
    :type token: str

    :return: 返回请求参数
    :rtype: str
    """
    all_params = {
        'app_id': client_id,
        'method': path,
        'charset': 'UTF-8',
        'sign_type': 'RSA2',
        'timestamp': int(round(time.time() * 1000)),
        'version': '1.0',
        'sdk_version': '1.0'
    }

    if app_token is not None:
        all_params['app_auth_token'] = app_token

    # biz_model = request.biz_model
    # params_model = request.params_model

    if biz_model is None:
        biz_model = {}
    if isinstance(biz_model, str):
        biz_str = biz_model
    else:
        biz_str = json.dumps(biz_model)

    if params_model is None:
        params_model = {}
    if isinstance(params_model, str):
        params_str = params_model
    else:
        params_str = json.dumps(params_model)

    # 添加业务参数
    if biz_str is not None:
        all_params['biz_content'] = biz_str
    # url携带参数(GET请求以外有效)
    if params_str is not None:
        all_params['params_content'] = params_str

    # 构建sign
    sign = SignUtil.create_sign(all_params, client_secret, 'RSA2')
    all_params['sign'] = sign
    return all_params


def get_sign_content(params: Dict[str, Any]) -> str:
    """
    Build signature content from parameters.

    Sorts parameters by key and creates a string in the format: key1=value1&key2=value2&...

    Args:
        params: Dictionary of parameters

    Returns:
        String to be signed
    """
    keys = sorted(params.keys())
    result = []
    for key in keys:
        value = str(params.get(key, ""))
        if len(value) > 0:
            result.append(f"{key}={value}")

    return "&".join(result)


def create_sign(params: Dict[str, Any], private_key: str) -> str:
    """
    Create RSA2 signature for parameters.

    Args:
        params: Dictionary of parameters to sign
        private_key: RSA private key as string

    Returns:
        Base64-encoded signature

    Raises:
        ImportError: If Crypto library is not available
        Exception: If signature creation fails
    """
    try:
        from Crypto.Hash import SHA256
        from Crypto.PublicKey import RSA
        from Crypto.Signature import PKCS1_v1_5

        import base64
    except ImportError as e:
        verbose_logger.error(f"Crypto library not found: {e}")
        raise ImportError(
            "PyCryptodome is required for ziniao_open authentication. "
            "Install it with: pip install pycryptodome"
        ) from e

    # Format private key with PEM markers if needed
    private_key = _format_private_key(private_key)

    # Get content to sign
    sign_content = get_sign_content(params)

    # Create signature
    try:
        key = RSA.importKey(private_key)
        hash_value = SHA256.new(bytes(sign_content, encoding="utf-8"))
        signer = PKCS1_v1_5.new(key)
        signature = signer.sign(hash_value)
        return str(base64.b64encode(signature), encoding="utf-8")
    except Exception as e:
        verbose_logger.error(f"Failed to create signature: {e}")
        raise Exception(f"Failed to create ziniao_open signature: {str(e)}") from e


def _format_private_key(private_key: str) -> str:
    """
    Format private key with PEM markers if needed.

    Args:
        private_key: RSA private key string

    Returns:
        Formatted private key with PEM markers
    """
    pem_begin = "-----BEGIN RSA PRIVATE KEY-----\n"
    pem_end = "\n-----END RSA PRIVATE KEY-----"

    if not private_key.startswith(pem_begin):
        private_key = pem_begin + private_key
    if not private_key.endswith(pem_end):
        private_key = private_key + pem_end

    return private_key
