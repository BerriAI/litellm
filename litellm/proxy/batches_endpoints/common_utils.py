from litellm.proxy._types import ProxyException


def validate_batch_list_limit(limit: int | None) -> None:
    if limit is None or 0 <= limit <= 100:
        return
    bound, expected, openai_code = (
        ("below minimum", ">= 0", "integer_below_min_value")
        if limit < 0
        else ("above maximum", "<= 100", "integer_above_max_value")
    )
    raise ProxyException(
        message=f"Invalid 'limit': integer {bound} value. Expected a value {expected}, but got {limit} instead.",
        type="invalid_request_error",
        param="limit",
        code=400,
        openai_code=openai_code,
    )
