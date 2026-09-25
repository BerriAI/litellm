from collections.abc import Mapping
from typing import Final, Protocol

from mcp.client.session import ClientRequestContext
from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    CreateMessageResultWithTools,
    ElicitRequestParams,
    ElicitResult,
    ErrorData,
)

from litellm.proxy._experimental.mcp_server.contracts import OperationContext
from litellm.proxy._types import UserAPIKeyAuth


class SamplingCallback(Protocol):
    async def __call__(
        self, context: ClientRequestContext, params: CreateMessageRequestParams, /
    ) -> CreateMessageResult | CreateMessageResultWithTools | ErrorData: ...


class ElicitationCallback(Protocol):
    async def __call__(self, context: object, params: ElicitRequestParams, /) -> ElicitResult | ErrorData: ...


def create_sampling_callback(
    user_api_key_auth: UserAPIKeyAuth | None = None,
    raw_headers: Mapping[str, str] | None = None,
    client_ip: str | None = None,
    operation_context: OperationContext | None = None,
) -> SamplingCallback:
    from litellm.proxy._experimental.mcp_server.server import get_active_auth_context

    auth: Final = get_active_auth_context() if operation_context is None and user_api_key_auth is None else None
    captured: Final = (
        operation_context
        if operation_context is not None
        else OperationContext(
            _caller=user_api_key_auth if user_api_key_auth is not None else (auth.user_api_key_auth if auth else None),
            raw_headers=raw_headers if raw_headers is not None else (auth.raw_headers if auth else None),
            client_ip=client_ip if client_ip is not None else (auth.client_ip if auth else None),
        )
    )

    async def callback(
        context: ClientRequestContext, params: CreateMessageRequestParams
    ) -> CreateMessageResult | CreateMessageResultWithTools | ErrorData:
        import litellm
        from litellm.proxy._experimental.mcp_server.sampling_handler import handle_sampling_create_message

        return await handle_sampling_create_message(
            context=context,
            params=params,
            default_model=getattr(litellm, "default_mcp_sampling_model", None),
            user_api_key_auth=captured.user_api_key_auth,
            raw_headers=dict(captured.raw_headers) if captured.raw_headers is not None else None,
            client_ip=captured.client_ip,
        )

    return callback


def create_elicitation_callback() -> ElicitationCallback:
    from litellm.proxy._experimental.mcp_server.server import get_active_mcp_session

    downstream_session: Final = get_active_mcp_session()
    downstream_capabilities: Final = getattr(downstream_session, "capabilities", None)

    async def callback(context: object, params: ElicitRequestParams) -> ElicitResult | ErrorData:
        from litellm.proxy._experimental.mcp_server.elicitation_handler import handle_elicitation_request

        return await handle_elicitation_request(
            context=context,
            params=params,
            downstream_session=downstream_session,
            downstream_capabilities=downstream_capabilities,
        )

    return callback
