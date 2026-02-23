def post_response_rule(input):  # receives the model response
    print(f"post_response_rule:input={input}")  # noqa
    if len(input) < 200:
        return {
            "decision": False,
            "message": "This violates LiteLLM Proxy Rules. Response too short",
        }
    return {"decision": True}  # message not required since, request will pass
