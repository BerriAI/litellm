import pytest

from litellm.router_strategy.complexity_router import request_models
from litellm.types.management_endpoints import auto_router_endpoints


@pytest.mark.parametrize("name", ["RequestComplexityRouterConfig", "AutoRouterRoutingTestRequest"])
def test_old_import_path_returns_the_moved_request_models(name: str) -> None:
    assert getattr(auto_router_endpoints, name) is getattr(request_models, name)


def test_old_import_path_still_raises_for_unknown_attributes() -> None:
    with pytest.raises(AttributeError):
        getattr(auto_router_endpoints, "NotARequestModel")
