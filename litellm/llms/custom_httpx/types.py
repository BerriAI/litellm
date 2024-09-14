from enum import Enum

import litellm


class httpxSpecialProvider(str, Enum):
    LoggingCallback = "logging_callback"
    GuardrailCallback = "guardrail_callback"
    Caching = "caching"
    Oauth2Check = "oauth2_check"
