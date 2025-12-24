import time
from typing import Union, Optional
from fastapi import Request

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth

from litellm.proxy.auth.auth_checks import (
    get_key_object,
)
from litellm.proxy.proxy_server import (
    proxy_logging_obj,
    user_api_key_cache,
)
from .zx_security_validator import security_validator

user_key_hashed_token_cache = DualCache(
    default_in_memory_ttl = 24 * 60 * 60
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
                client_id = request.headers.get('zx_client_id')
                client_app_id = request.headers.get('zx_client_app_id')
                email = request.headers.get('zx_user_email')
                data = email or client_app_id
                timestamp  = request.headers.get('zx_timestamp')
                if client_id is None :
                    raise Exception("Invalid API key: zx_client_id not found")
                if data is None :
                    raise Exception("Invalid API key: zx_client_app_id or zx_user_email not found")
                if timestamp is None:
                    raise Exception("Invalid API key: zx_timestamp not found")
                try:
                    int_num = int(timestamp)
                except ValueError as e:
                    raise Exception(f"Invalid API key: timestamp[{timestamp}] error")
                # 半小时内有效
                if time.time() - int_num > 1800:
                    raise Exception(f"Invalid API key: timestamp[{timestamp}] expired")

                if not security_validator.validate(client_id, signature, f"{data}:{timestamp}"):
                    raise Exception("Invalid API key: signature error")
            
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
                raise Exception("Invalid API key: user not found")

            return await get_key_object(
                hashed_token=hashed_token,
                prisma_client=tmp_prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=None,
                proxy_logging_obj=proxy_logging_obj,
            )
            # return UserAPIKeyAuth(
            #     api_key=api_key,
            #     user_id="team_user_456",
            #     user_email="user@company.com",
            #     user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
            #     team_id="dev_team",
            #     team_alias="Development Team",
            #     max_budget=100.0,
            #     tpm_limit=1000,
            #     rpm_limit=20,
            #     models=["gpt-3.5-turbo", "claude-3-haiku"],
            #     team_member_tpm_limit=500,  # Limit within team
            #     end_user_tpm_limit=100,     # Per end-user limit
            #     metadata={"project": "chatbot_v2"}
            # )
        else:
            return api_key
    except Exception:
        raise Exception("Invalid API key")