import re


def is_opencode_api_base(api_base: str | None) -> bool:
    return bool(
        api_base
        and re.fullmatch(
            r"https://opencode\.ai/(?:zen(?:/go)?|inference/(?:go/)?(?:openai|anthropic))(?:/v1)?/?", api_base
        )
    )
