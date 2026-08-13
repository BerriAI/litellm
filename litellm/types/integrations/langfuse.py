from typing_extensions import TypedDict


class LangfuseLoggingConfig(TypedDict):
    langfuse_secret: str | None
    langfuse_public_key: str | None
    langfuse_host: str | None


class LangfuseUsageDetails(TypedDict):
    input: int | None
    output: int | None
    total: int | None
    cache_creation_input_tokens: int | None
    cache_read_input_tokens: int | None
