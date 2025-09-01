from typing import Optional

import litellm


class Rules:
    """
    Fail calls based on the input or llm api output

    Example usage:
    import litellm
    def my_custom_rule(input): # receives the model response
            if "i don't think i can answer" in input: # trigger fallback if the model refuses to answer
                    return False
            return True

    litellm.post_call_rules = [my_custom_rule] # have these be functions that can be called to fail a call

    response = litellm.completion(model="gpt-3.5-turbo", messages=[{"role": "user",
        "content": "Hey, how's it going?"}], fallbacks=["openrouter/mythomax"])
    """

    def __init__(self) -> None:
        pass

    def pre_call_rules(self, input: str, model: str):
        for rule in litellm.pre_call_rules:
            if callable(rule):
                decision = rule(input)
                if decision is False:
                    raise litellm.APIResponseValidationError(message="LLM Response failed post-call-rule check", llm_provider="", model=model)  # type: ignore
        return True

    def post_call_rules(self, input: Optional[str], model: str) -> bool:
        if input is None:
            return True
        for rule in litellm.post_call_rules:
            if callable(rule):
                decision = rule(input)
                if isinstance(decision, bool):
                    if decision is False:
                        raise litellm.APIResponseValidationError(message="LLM Response failed post-call-rule check", llm_provider="", model=model)  # type: ignore
                elif isinstance(decision, dict):
                    decision_val = decision.get("decision", True)
                    decision_message = decision.get(
                        "message", "LLM Response failed post-call-rule check"
                    )
                    if decision_val is False:
                        raise litellm.APIResponseValidationError(message=decision_message, llm_provider="", model=model)  # type: ignore
        return True
