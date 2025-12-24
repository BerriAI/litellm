"""
Ziniao API Gateway authentication utilities for MCP OpenAPI servers.
"""

from .zx_sign import ZXSign

def request_sign(client_id, client_secret, base_url, method, path, url, params, json_body, headers):
    if client_id and client_secret:
        sign_header = ZXSign.get_sign_header(client_id, client_secret, None)
        headers = (headers or {}) | sign_header
    return url, params, json_body, headers
