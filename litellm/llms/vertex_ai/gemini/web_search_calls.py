import hashlib
import hmac
import secrets
from collections.abc import Sequence
from typing import Final

from openai.types.responses.response_function_web_search import ResponseFunctionWebSearch

from litellm.types.llms.gemini import GEMINI_SEARCH_TOOL_ID_PREFIX
from litellm.types.responses.main import build_web_search_call

_SEARCH_CALL_ID_KEY: Final = secrets.token_bytes(32)


def _search_call_id(response_id: str, query: str) -> str:
    digest: Final = hmac.new(_SEARCH_CALL_ID_KEY, f"{response_id}\0{query}".encode(), hashlib.sha256).hexdigest()[:16]
    return f"{GEMINI_SEARCH_TOOL_ID_PREFIX}{response_id}_{digest}"


def gemini_web_search_calls(response_id: str, queries: Sequence[str]) -> tuple[ResponseFunctionWebSearch, ...]:
    return tuple(
        build_web_search_call(_search_call_id(response_id, query), {"query": query}, None, status="completed")
        for query in queries
    )
