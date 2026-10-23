import os


def is_offline_mode() -> bool:
    """True when ``LITELLM_OFFLINE_MODE=true``. Every request LiteLLM makes on its own, rather than to a
    destination the operator configured, must check this and use its bundled copy or skip instead"""
    return os.getenv("LITELLM_OFFLINE_MODE", "").strip().lower() == "true"
