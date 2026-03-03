import json

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import CallTypesLiteral


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
    user_api_key_cache: DualCache,
    user_api_key_dict: UserAPIKeyAuth,
) -> None:
    """
    Parse a batch JSONL file and run pre-call guardrails on each request.

    Directly invokes guardrail callbacks instead of going through
    pre_call_hook() which requires internal proxy state (logging objects,
    pipelines, etc.) that batch file items don't have.

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

        endpoint = json_obj.get("url", "")
        call_type = _get_call_type_from_endpoint(endpoint)
        custom_id = json_obj.get("custom_id", f"line_{line_num}")

        try:
            for callback in litellm.callbacks:
                if isinstance(callback, CustomGuardrail):
                    if (
                        callback.should_run_guardrail(
                            data=body, event_type=GuardrailEventHooks.pre_call
                        )
                        is not True
                    ):
                        continue

                    response = await callback.async_pre_call_hook(
                        user_api_key_dict=user_api_key_dict,
                        cache=user_api_key_cache,
                        data=body,
                        call_type=call_type,
                    )
                    if response is not None and isinstance(response, dict):
                        body = response

                elif (
                    isinstance(callback, CustomLogger)
                    and "async_pre_call_hook" in vars(callback.__class__)
                    and callback.__class__.async_pre_call_hook
                    != CustomLogger.async_pre_call_hook
                ):
                    await callback.async_pre_call_hook(
                        user_api_key_dict=user_api_key_dict,
                        cache=user_api_key_cache,
                        data=body,
                        call_type=call_type,
                    )
        except Exception as e:
            raise Exception(
                f"Guardrail rejected batch item '{custom_id}' (line {line_num}): {str(e)}"
            ) from e
