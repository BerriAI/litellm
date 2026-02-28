import litellm

def exception_handler(status_code, message, model=None):
    """
    Surgical mapping for DeepSeek-specific infrastructure errors.
    Ensures 'Server Overloaded' (503) triggers automatic retries in the Router.
    """
    if status_code == 503 or "server overloaded" in message.lower():
        return litellm.ServiceUnavailableError(
            message="DeepSeek is currently overloaded. LiteLLM triggering automatic failover.",
            model=model or "deepseek-v3",
            llm_provider="deepseek"
        )
    return None
