from typing import Final

from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from tests._support.provider import SharedProvider
from tests._support.wire import Request


def assert_served_natively(
    server: SharedProvider,
    *responses: object,
    count: int = 1,
    include_model_discovery: bool = False,
) -> tuple[Request, ...]:
    requests: Final = server.received(include_model_discovery=include_model_discovery)
    assert len(requests) == max(len(responses), count)
    if not responses:
        assert not any(request.headers.get("user-agent", "").startswith("python-httpx") for request in requests)
    for response in responses:
        additional_headers: Final = get_hidden_params_dict(response).get("additional_headers")
        assert isinstance(additional_headers, dict)
        assert additional_headers.get("x-litellm-rust") == "true"
    return requests
