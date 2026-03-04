import logging
from dataclasses import dataclass
from typing import Literal, Optional, Dict
import time
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


token_stores: Dict[str, TokenStore] = {}


def set_store(
    type: TokenType,
    token: str,
    auth_key: str | None = None,
    data: dict | None = None,
    timeout: int = 30,
):
    ts = TokenStore(
        type=type,
        token=token,
        login=False,
        status="pending",
        expire_time=time.time() + timeout * 60,
        auth_key=auth_key,
        data=data or {},
    )
    token_stores[f"{type}:{token}"] = ts
    return ts


def get_store(
    type: TokenType | None = None,
    token: str | None = None,
    auth_key: str | None = None,
    check_login: bool = True,
    remove: bool = False,
):
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
        key_alias=None,
        user_api_key_dict=user_api_key_dict,
        user_id=user_id,
        team_id=None,
        organization_id=None,
        key_hash=None,
        return_full_object=True,
        include_team_keys=True,
        include_created_by_keys=True,
        sort_by=None,
        sort_order="desc",
        expand=None,
        status=None,
    )
    key_total_count = keys.get("total_count", 0) or 0
    if key_total_count > 0:
        key_infos: list[UserAPIKeyAuth] = keys.get("keys", [])  # type: ignore
        # 按 key_alias 过滤
        matching_keys = [k for k in key_infos if k.key_alias == key_alias]

        if len(matching_keys) > 1:
            # 如果有 device_id，按 metadata 进一步过滤
            if key_metadata and "device_id" in key_metadata:
                device_id = key_metadata["device_id"]
                device_matching_keys = [
                    k
                    for k in matching_keys
                    if k.metadata and k.metadata.get("device_id") == device_id
                ]
                if device_matching_keys:
                    matching_keys = device_matching_keys

            if len(matching_keys) > 1:
                raise RuntimeError(
                    f"key_alias [{key_alias}] 有多个匹配的 key，无法确定: {len(matching_keys)}"
                )

        if len(matching_keys) == 1:
            key_info = matching_keys[0]
            if key_info and key_info.token:
                # 如果新 metadata 中有 device_id，需要合并到旧 key 的 metadata 中
                if key_metadata and "device_id" in key_metadata:
                    try:
                        # 保留旧 metadata 中的所有字段，只补充新的 device_id 和 device_name
                        merged_metadata = (
                            key_info.metadata.copy() if key_info.metadata else {}
                        )
                        merged_metadata["device_id"] = key_metadata.get("device_id")
                        merged_metadata["device_name"] = key_metadata.get(
                            "device_name", "unknown"
                        )

                        # 更新旧 key 的 metadata
                        update_request = UpdateKeyRequest(
                            key=key_info.token, metadata=merged_metadata
                        )
                        await key_management_endpoints.update_key_fn(
                            request=Request({"type": "http", "query_string": ""}),
                            data=update_request,
                            user_api_key_dict=user_api_key_dict,
                            litellm_changed_by=provider,
                        )
                        logger.info(
                            f"user[{user_id}:{org_email}] key[{key_info.token}] metadata merged with device_id"
                        )
                    except Exception as e:
                        logger.warning(
                            f"user[{user_id}:{org_email}] failed to merge key metadata: {e}"
                        )
                        # 不阻塞返回，继续返回现有 key

                return (False, key_info.token)

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
        page_size=25,
        sort_by=None,
        sort_order="asc",
    )
    user_total_count = users.get("total", 0) or 0
    if user_total_count > 1:
        raise RuntimeError(f"user [{org_email}]在系统中重复: {user_total_count}")
    uid = None
    if user_total_count == 1:
        user: LiteLLM_UserTableWithKeyCount = users.get("users", [None])[0]
        if user is None:
            raise RuntimeError(f"user [{org_email}]不存在")
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
        metadata=key_metadata,
    )
    key_res = await key_management_endpoints.generate_key_fn(
        key_data, user_api_key_dict=user_api_key_dict
    )
    logger.info(f"user[{user_id}:{org_email}] key[{key_alias}] create start")

    return (True, key_res.key)
