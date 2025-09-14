from typing import Optional

from typing_extensions import TypedDict


class LangfuseLoggingConfig(TypedDict):
    langfuse_secret: Optional[str]
    langfuse_public_key: Optional[str]
    langfuse_host: Optional[str]
