import dataclasses
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal, Optional, Dict
import time
import hashlib
from fastapi import Request
from litellm.proxy.management_endpoints import internal_user_endpoints
from litellm.proxy.management_endpoints import key_management_endpoints
from litellm.proxy.management_endpoints import team_endpoints
from litellm.proxy._types import (
    NewTeamRequest,
    LitellmUserRoles,
    UserAPIKeyAuth,
    NewUserRequest,
    GenerateKeyRequest,
    LiteLLM_UserTableWithKeyCount,
    LiteLLMKeyType,
    UpdateKeyRequest,
)

logger = logging.getLogger()

TokenType = Literal["cli", "app", "client_iframe", "api"]
TokenStatus = Literal["pending", "success", "failed"]


@dataclass
class TokenStore:
    type: TokenType
    token: str
    login: bool
    status: TokenStatus
    expire_time: float
    auth_key: Optional[str]
    data: dict


# In-memory fallback store (single-node only)
token_stores: Dict[str, TokenStore] = {}

_REDIS_KEY_PREFIX = "litellm:zx:token"
_REDIS_AUTH_PREFIX = "litellm:zx:auth"
_redis_client: Optional[Any] = None


async def _get_redis() -> Optional[Any]:
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        return None
    try:
        import redis.asyncio as aioredis
        _redis_client = aioredis.from_url(redis_url, decode_responses=True)
        return _redis_client
    except ImportError:
        logger.warning("redis package not available, falling back to in-memory token store")
        return None


class ClientError(Exception):
    pass


async def set_store(
    type: TokenType,
    token: str,
    auth_key: str | None = None,
    data: dict | None = None,
    timeout: int = 30,
) -> "TokenStore":
    ts = TokenStore(
        type=type,
        token=token,
        login=False,
        status="pending",
        expire_time=time.time() + timeout * 60,
        auth_key=auth_key,
        data=data or {},
    )
    redis = await _get_redis()
    if redis:
        ttl = timeout * 60
        payload = json.dumps(dataclasses.asdict(ts))
        await redis.setex(f"{_REDIS_KEY_PREFIX}:{type}:{token}", ttl, payload)
        if auth_key:
            await redis.setex(f"{_REDIS_AUTH_PREFIX}:{auth_key}", ttl, f"{type}:{token}")
    else:
        token_stores[f"{type}:{token}"] = ts
    return ts


async def save_store(ts: "TokenStore") -> None:
    """Persist a mutated TokenStore back to the store (required for Redis path)."""
    redis = await _get_redis()
    if redis:
        ttl = max(1, int(ts.expire_time - time.time()))
        payload = json.dumps(dataclasses.asdict(ts))
        await redis.setex(f"{_REDIS_KEY_PREFIX}:{ts.type}:{ts.token}", ttl, payload)
    else:
        token_stores[f"{ts.type}:{ts.token}"] = ts


async def get_store(
    type: TokenType | None = None,
    token: str | None = None,
    auth_key: str | None = None,
    check_login: bool = True,
    remove: bool = False,
) -> Optional["TokenStore"]:
    redis = await _get_redis()
    if redis:
        if auth_key:
            ptr = await redis.get(f"{_REDIS_AUTH_PREFIX}:{auth_key}")
            if not ptr:
                return None
            raw = await redis.get(f"{_REDIS_KEY_PREFIX}:{ptr}")
        elif type and token:
            raw = await redis.get(f"{_REDIS_KEY_PREFIX}:{type}:{token}")
        else:
            return None
        if not raw:
            return None
        ts = TokenStore(**json.loads(raw))
        if remove:
            await redis.delete(f"{_REDIS_KEY_PREFIX}:{ts.type}:{ts.token}")
            if ts.auth_key:
                await redis.delete(f"{_REDIS_AUTH_PREFIX}:{ts.auth_key}")
        if check_login and not ts.login:
            return None
        return ts
    else:
        for k, v in list(token_stores.items()):
            if v.expire_time < time.time():
                del token_stores[k]

        store = None
        if auth_key:
            store = next((x for x in token_stores.values() if x.auth_key == auth_key), None)
        elif type and token:
            store = token_stores.get(f"{type}:{token}", None)
        if not store:
            return None
        if remove:
            del token_stores[f"{store.type}:{store.token}"]
        if check_login and not store.login:
            return None
        return store


async def create_or_get_user_key(
    provider: str,
    user_id: str,
    user_name: str,
    org_email: str,
    dept_id: Optional[str],
    user_api_key_dict: Optional[UserAPIKeyAuth] = None,
    key_alias: Optional[str] = None,
    key_metadata: Optional[dict] = {},
) -> tuple[bool, str]:
    if dept_id is None or dept_id.strip() == "":
        dept_id = "none_dept_id"
    if user_api_key_dict is None:
        user_api_key_dict = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    if key_alias is None:
        key_alias = org_email

    keys = await key_management_endpoints.list_keys(
        Request(
            {
                "type": "http",
                "query_string": "",
            }
        ),
        page=1,
        size=25,
        key_alias=key_alias,
        user_api_key_dict=user_api_key_dict,
        user_id=user_id,
        team_id=None,
        organization_id=None,
        key_hash=None,
        return_full_object=False,
        include_team_keys=True,
        include_created_by_keys=False,
        sort_by=None,
        sort_order="desc",
        expand=None,
        status=None,
        project_id=None,
        access_group_id=None,
    )

    key_total_count = keys.get("total_count", 0) or 0
    if key_total_count > 1:
        raise ClientError(
            f"user [{org_email}] 的 key[{key_alias}] 有多个匹配的 key，无法确定: {key_total_count}"
        )
    if key_total_count == 1:
        key_token = keys.get('keys', [])[0]
        if key_token and isinstance(key_token, str):
            return (False, key_token)

    teams = await team_endpoints.list_team(
        Request(
            {
                "type": "http",
                "query_string": "",
            }
        ),
        user_id=None,
        organization_id=None,
        user_api_key_dict=user_api_key_dict,
    )
    team = next(
        (x for x in teams if x.team_alias and x.team_alias.endswith(f"__{dept_id}")),
        None,
    )
    if team is None:
        logger.info(f"user[{user_id}:{org_email}] create team start")
        team_data = NewTeamRequest(
            team_alias=f"__{dept_id}", models=["all-proxy-models"]
        )
        team_res = await team_endpoints.new_team(
            team_data,
            Request(
                {
                    "type": "http",
                    "query_string": "",
                }
            ),
            litellm_changed_by=provider,
            user_api_key_dict=user_api_key_dict,
        )
        logger.info(f"user[{user_id}:{org_email}] create team end")
        team_id = team_res["team_id"]
    else:
        team_id = team.team_id

    users = await internal_user_endpoints.get_users(
        user_email=org_email,
        role=None,
        user_ids=None,
        sso_user_ids=None,
        team=None,
        page=1,
        page_size=100,
        sort_by=None,
        sort_order="asc",
        organization_ids=None,
        user_api_key_dict=user_api_key_dict,
    )
    # get_users 使用模糊匹配，需要过滤出精确匹配的用户
    exact_users = [
        u for u in (users.get("users") or [])
        if u.user_email == org_email
    ]
    user_total_count = len(exact_users)
    if user_total_count > 1:
        raise ClientError(f"user [{org_email}]在系统中重复: {user_total_count}")
    uid = None
    if user_total_count == 1:
        user: LiteLLM_UserTableWithKeyCount = exact_users[0]
        uid = user.user_id
    else:
        metadata = {"provider": provider}
        logger.info(f"user[{user_id}:{org_email}] create start")
        user_data = NewUserRequest(
            auto_create_key=False,
            user_id=user_id,
            user_alias=user_name,
            user_email=org_email,
            user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
            team_id=team_id,
            metadata=metadata,
        )
        user_res = await internal_user_endpoints.new_user(
            user_data, user_api_key_dict=user_api_key_dict
        )
        logger.info(f"user[{user_id}:{org_email}] create end")
        uid = user_res.user_id

    logger.info(f"user[{user_id}:{org_email}] key[{key_alias}] create start")
    key_data = GenerateKeyRequest(
        user_id=uid,
        key_alias=key_alias,
        key_type=LiteLLMKeyType.LLM_API,
        models=["all-team-models"],
        metadata=(key_metadata or {}) | {"org_email": org_email},
    )
    key_res = await key_management_endpoints.generate_key_fn(
        key_data, user_api_key_dict=user_api_key_dict
    )
    logger.info(f"user[{user_id}:{org_email}] key[{key_alias}] create start")

    return (True, key_res.key)
