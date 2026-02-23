import json
from typing import Optional
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral
from litellm.proxy.utils import ProxyLogging

def _get_call_type_from_endpoint(endpoint: str) -> CallTypesLiteral:
    """Map batch JSONL endpoint to CallTypesLiteral."""
    if "chat/completions" in endpoint:
        return "acompletion"
    elif "embeddings" in endpoint:
        return "aembedding"
    else:
        return "acompletion"  # default fallback

async def run_pre_call_guardrails_on_batch_file(
    file_content: bytes,
    proxy_logging_obj: ProxyLogging,
    user_api_key_dict: UserAPIKeyAuth,
) -> None:
    """
    Parse a batch JSONL file and run pre-call guardrails on each request.

    Raises an exception if any guardrail rejects a request.
    """
    lines = file_content.decode("utf-8").strip().splitlines()

    for line_num, line in enumerate(lines, start=1):
        if not line.strip():
            continue
            
        try:
            json_obj = json.loads(line)
        except json.JSONDecodeError:
            continue
            
        body = json_obj.get("body", {})

        if not body or "messages" not in body:
            continue

        # Determine call_type from the endpoint in the JSONL line
        endpoint = json_obj.get("url", "")
        call_type = _get_call_type_from_endpoint(endpoint)

        try:
            # Run pre_call_hook (which triggers all guardrails)
            await proxy_logging_obj.pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                data=body,
                call_type=call_type,
            )
        except Exception as e:
            custom_id = json_obj.get("custom_id", f"line_{line_num}")
            # Reraise exception to abort the upload, appending item information
            raise Exception(f"Guardrail rejected batch item {custom_id} (line {line_num}): {str(e)}") from e
