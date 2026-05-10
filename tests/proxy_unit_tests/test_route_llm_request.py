import pytest

from litellm.proxy.route_llm_request import route_request


class _RouterSettings:
    pass_through_all_models = False


class _PatternRouter:
    patterns = []


class _FakeRouter:
    model_names = ["qwen3.6"]
    model_group_alias = None
    default_deployment = None
    deployment_names = []
    router_general_settings = _RouterSettings()
    pattern_router = _PatternRouter()

    def __init__(self):
        self.kwargs = None

    def has_model_id(self, model):
        return False

    def map_team_model(self, model, team_id):
        return None

    def acompletion(self, **kwargs):
        self.kwargs = kwargs
        return kwargs


@pytest.mark.asyncio
async def test_route_request_drops_empty_chat_tools():
    router = _FakeRouter()
    data = {
        "model": "qwen3.6",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [],
        "tool_choice": "auto",
    }

    result = await route_request(
        data=data,
        route_type="acompletion",
        llm_router=router,
        user_model=None,
    )

    assert result == router.kwargs
    assert "tools" not in router.kwargs
    assert "tool_choice" not in router.kwargs


@pytest.mark.asyncio
async def test_route_request_preserves_non_empty_chat_tools():
    router = _FakeRouter()
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Lookup a value",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    data = {
        "model": "qwen3.6",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [tool],
        "tool_choice": "auto",
    }

    result = await route_request(
        data=data,
        route_type="acompletion",
        llm_router=router,
        user_model=None,
    )

    assert result == router.kwargs
    assert router.kwargs["tools"] == [tool]
    assert router.kwargs["tool_choice"] == "auto"
