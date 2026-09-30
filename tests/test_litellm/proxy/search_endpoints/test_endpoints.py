from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.route_llm_request import ProxyMissingRequiredParamError
from litellm.proxy.search_endpoints.endpoints import search


def _json_request(body: dict[str, object]) -> MagicMock:
    request = MagicMock()
    request.body = AsyncMock(return_value=orjson.dumps(body))
    return request


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [{"query": "litellm"}, {"query": "litellm", "search_tool_name": ""}])
async def test_search_without_search_tool_name_or_model_is_a_400(body):
    with pytest.raises(ProxyMissingRequiredParamError) as exc_info:
        await search(
            request=_json_request(body),
            fastapi_response=MagicMock(),
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
        )

    assert exc_info.value.code == "400"
    assert exc_info.value.param == "search_tool_name"
    assert exc_info.value.message == "/search: Missing required parameter: 'search_tool_name'."
