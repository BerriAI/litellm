from typing import Optional, TypedDict


class LangfuseLoggingConfig(TypedDict):
    langfuse_secret: Optional[str]
    langfuse_public_key: Optional[str]
    langfuse_host: Optional[str]
