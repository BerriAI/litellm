from litellm._logging import verbose_proxy_logger


def post_response_rule(input: str):  # receives the model response
    verbose_proxy_logger.debug("post_response_rule:input=%s", input)
    if len(input) < 200:
        return {
            "decision": False,
            "message": "This violates LiteLLM Proxy Rules. Response too short",
        }
    return {"decision": True}  # message not required since, request will pass
