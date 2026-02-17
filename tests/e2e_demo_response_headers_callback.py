"""
Demo CustomLogger that injects custom response headers.

Shows how to:
1. Echo an incoming request header (e.g., APIGEE request ID) into the response
2. Inject headers on both success and failure paths
3. Works for /chat/completions, /embeddings, and /responses

Usage:
    litellm --config tests/e2e_demo_response_headers_config.yaml

Test commands:
    # /chat/completions (non-streaming)
    curl -s -D- http://localhost:4000/chat/completions \
      -H "Authorization: Bearer sk-1234" \
      -H "x-apigee-request-id: apigee-req-001" \
      -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}'

    # /chat/completions (streaming)
    curl -s -D- http://localhost:4000/chat/completions \
      -H "Authorization: Bearer sk-1234" \
      -H "x-apigee-request-id: apigee-req-002" \
      -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}],"stream":true}'

    # /embeddings
    curl -s -D- http://localhost:4000/embeddings \
      -H "Authorization: Bearer sk-1234" \
      -H "x-apigee-request-id: apigee-req-003" \
      -d '{"model":"text-embedding-3-small","input":"hello"}'

    # /v1/responses (non-streaming)
    curl -s -D- http://localhost:4000/v1/responses \
      -H "Authorization: Bearer sk-1234" \
      -H "x-apigee-request-id: apigee-req-004" \
      -d '{"model":"gpt-4o-mini","input":"hi"}'

    # /v1/responses (streaming)
    curl -s -D- http://localhost:4000/v1/responses \
      -H "Authorization: Bearer sk-1234" \
      -H "x-apigee-request-id: apigee-req-005" \
      -d '{"model":"gpt-4o-mini","input":"hi","stream":true}'

    # Failure path (bad model → headers still injected)
    curl -s -D- http://localhost:4000/chat/completions \
      -H "Authorization: Bearer sk-1234" \
      -H "x-apigee-request-id: apigee-req-006" \
      -d '{"model":"nonexistent-model","messages":[{"role":"user","content":"hi"}]}'

Expected: All responses contain x-apigee-request-id, x-custom-header, and x-litellm-hook-model.
"""

from typing import Any, Dict, Optional

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth


class ResponseHeaderInjector(CustomLogger):
    """
    Demonstrates injecting custom HTTP response headers via the proxy hook.

    Key features:
    - Echoes the incoming x-apigee-request-id header back in the response
    - Adds a static custom header and the model name
    - Works for success (streaming + non-streaming) and failure responses
    - Works for all endpoints: /chat/completions, /embeddings, /responses
    """

    async def async_post_call_response_headers_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, str]]:
        headers: Dict[str, str] = {
            "x-custom-header": "hello-from-hook",
            "x-litellm-hook-model": data.get("model", "unknown"),
        }

        # Echo the APIGEE request ID from the incoming request into the response
        if request_headers:
            apigee_id = request_headers.get("x-apigee-request-id")
            if apigee_id:
                headers["x-apigee-request-id"] = apigee_id

        return headers


response_header_injector = ResponseHeaderInjector()
