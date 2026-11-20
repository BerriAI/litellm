import hashlib
from collections.abc import Sequence

from openai.types.responses.response_function_web_search import ResponseFunctionWebSearch

from litellm.types.responses.main import build_web_search_call


def _search_call_id(response_id: str, query: str) -> str:
    return f"{response_id}_{hashlib.sha256(query.encode()).hexdigest()[:16]}"


def gemini_web_search_calls(response_id: str, queries: Sequence[str]) -> tuple[ResponseFunctionWebSearch, ...]:
    return tuple(
        build_web_search_call(_search_call_id(response_id, query), {"query": query}, None, status="completed")
        for query in queries
    )
