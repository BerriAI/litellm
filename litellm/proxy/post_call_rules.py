from collections.abc import Mapping

import litellm


def post_response_rule(input: str) -> Mapping[str, bool | str]:
    print(  # noqa: T201  # legacy stdout diagnostic
        "post_response_rule:input=redacted-by-litellm"
        if litellm.redact_messages_in_debug_logs
        else f"post_response_rule:input={input}"
    )
    if len(input) < 200:
        return {
            "decision": False,
            "message": "This violates LiteLLM Proxy Rules. Response too short",
        }
    return {"decision": True}  # message not required since, request will pass
