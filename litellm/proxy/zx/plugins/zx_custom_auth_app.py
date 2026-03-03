import time
from typing import Union, Optional
from fastapi import Request

from litellm.caching.caching import DualCache
from litellm.proxy._types import ProxyException, UserAPIKeyAuth

from litellm.proxy.auth.auth_checks import (
    get_key_object,
)
from litellm.proxy.proxy_server import (
    proxy_logging_obj,
    user_api_key_cache,
)
from ..zx_security_validator import security_validator

ttl = 24 * 60 * 60
user_key_hashed_token_cache = DualCache(
    default_in_memory_ttl = ttl
)

tmp_prisma_client = None

async def user_api_key_auth(
    request: Request, api_key: str
) -> Union[UserAPIKeyAuth, str]:
    try:
        if api_key.startswith("sk-zx-u-"):
            signature = api_key[8:]

            key = f'user_key_hashed_token:{signature}'
            hashed_token: Optional[str] = await user_key_hashed_token_cache.async_get_cache(key=key)
            if hashed_token is None:
                client_id = request.headers.get('zx-client-id')
                client_app_id = request.headers.get('zx-client-app-id')
                email = request.headers.get('zx-user-email')
                data = email or client_app_id
                timestamp  = request.headers.get('zx-timestamp')
                if client_id is None :
                    raise ProxyException(
                        message="Invalid API key: zx-client-id not found",
                        type="invalid_request_error",
                        param="api_key",
                        code=401,
                    )
                if data is None :
                    raise ProxyException(
                        message="Invalid API key: zx-client-app-id or zx_user_email not found",
                        type="invalid_request_error",
                        param="api_key",
                        code=401,
                    )
                if timestamp is None:
                    raise ProxyException(
                        message="Invalid API key: zx-timestamp not found",
                        type="invalid_request_error",
                        param="api_key",
                        code=401,
                    )
                try:
                    int_num = int(timestamp)
                except ValueError as e:
                    raise ProxyException(
                        message=f"Invalid API key: timestamp[{timestamp}] error",
                        type="invalid_request_error",
                        param="api_key",
                        code=401,
                    )
                # 半小时内有效
                if time.time() - int_num > ttl:
                    raise ProxyException(
                        message=f"Invalid API key: timestamp[{timestamp}] expired",
                        type="invalid_request_error",
                        param="api_key",
                        code=401,
                    )

                if not security_validator.validate(client_id, signature, f"{data}:{timestamp}"):
                    raise ProxyException(
                        message="Invalid API key: signature error",
                        type="invalid_request_error",
                        param="api_key",
                        code=401,
                    )
            
                global tmp_prisma_client
                if tmp_prisma_client is None:
                    from litellm.proxy.proxy_server import prisma_client
                    tmp_prisma_client = prisma_client

                if tmp_prisma_client:
                    key_alias = email or f"{client_id}_{client_app_id}"
                    response = await tmp_prisma_client.get_generic_data(key='key_alias', value=key_alias, table_name='keys')
                    hashed_token = getattr(response, 'token', None)
                    if hashed_token:
                        await user_key_hashed_token_cache.async_set_cache(
                            key=key,
                            value=hashed_token,
                            local_only=True
                        )

            if hashed_token is None:
                raise ProxyException(
                    message="Invalid API key: user not found",
                    type="invalid_request_error",
                    param="api_key",
                    code=401,
                )

            return await get_key_object(
                hashed_token=hashed_token,
                prisma_client=tmp_prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=None,
                proxy_logging_obj=proxy_logging_obj,
            )
        else:
            return api_key
    except Exception as e:
        if isinstance(e, ProxyException):
            raise e
        raise Exception("Invalid API key")