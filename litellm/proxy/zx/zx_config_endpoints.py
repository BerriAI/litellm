import os
import logging
from dataclasses import dataclass
from typing import Annotated, Literal, Optional, Dict
import json
import time
import uuid
from fastapi import APIRouter, Depends, HTTPException, Header, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
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
)
from . import zixun_auth
from . import zx_development_data
from . import zx_security_validator

auth = zixun_auth.ZixunAuth(
    os.environ.get('ZX_AUTH_HOST'),
    os.environ.get('ZX_AUTH_API_HOST'),
    os.environ.get('ZX_AUTH_APP_KEY'),
    os.environ.get('ZX_AUTH_APP_SECRET'),
    )

logger = logging.getLogger()

router = APIRouter()

continue_plugin_dev_data_enabled = 'true' == os.environ.get('ZX_CONTINUE_PLUGIN_DEV_DATA_ENABLED', 'false').strip()

security_validator = zx_security_validator.SecurityValidator('ZX_APP_CLIENT_CREDENTIALS_')
add_user_allow_email_domain = '@'+os.environ.get('ZX_ADD_USER_ALLOW_EMAIL_DOMAIN', 'fzzixun.com').strip()

TokenType = Literal['cli', 'app', 'client_iframe', 'api']
TokenStatus = Literal['pending', 'success', 'failed']

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

def set_store(type: TokenType, token: str, auth_key: str | None = None, data: dict | None = None, timeout: int = 30):
    ts = TokenStore(
        type=type,
        token=token,
        login=False, 
        status="pending", 
        expire_time=time.time() + timeout * 60, 
        auth_key=auth_key, 
        data=data or {}
    )
    token_stores[f"{type}:{token}"] = ts
    return ts

def get_store(type: TokenType | None = None, token: str | None = None, auth_key: str | None = None, check_login: bool = True, remove: bool = False):
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
        user_api_key_dict: Optional[UserAPIKeyAuth] = None
) -> tuple[bool, str]:
    if dept_id is None or dept_id.strip() == '':
        dept_id = "none_dept_id"
    if user_api_key_dict is None:
        user_api_key_dict = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    
    keys = await key_management_endpoints.list_keys(
        Request({'type': 'http', 'query_string': '',}),
        page=1, 
        size=25, 
        key_alias=org_email, 
        user_api_key_dict=user_api_key_dict,
        user_id=None,
        team_id=None,
        organization_id=None,
        key_hash=None,
        return_full_object=False,
        include_team_keys=True,
        include_created_by_keys=True,
        sort_by=None,
        sort_order="desc"
    )
    key_total_count = keys.get('total_count', 0) or 0
    if key_total_count > 1:
        raise RuntimeError(f"key_alias [{org_email}]在系统中重复: {key_total_count}")
    if key_total_count == 1:
        key_id = keys.get('keys', [None])[0]
        if isinstance(key_id, str):
            return (False, key_id)
    
    logger.info(f'user[{user_id}:{org_email}] create start...')
    teams = await team_endpoints.list_team(Request({'type': 'http', 'query_string': '',}), user_id=None, organization_id=None, user_api_key_dict=user_api_key_dict)
    team = next((x for x in teams if x.team_alias and x.team_alias.endswith(f'__{dept_id}')), None)
    if team is None:
        logger.info(f'user[{user_id}:{org_email}] create team')
        team_data = NewTeamRequest(team_alias=f'__{dept_id}', models=["all-proxy-models"])
        team_res = await team_endpoints.new_team(team_data, Request({'type': 'http', 'query_string': '',}), litellm_changed_by=provider, user_api_key_dict=user_api_key_dict)
        team_id = team_res['team_id']
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
        sort_order='asc'
    )
    user_total_count = users.get('total', 0) or 0
    if user_total_count > 1:
        raise RuntimeError(f"user [{org_email}]在系统中重复: {user_total_count}")
    if user_total_count == 1:
        user: LiteLLM_UserTableWithKeyCount = users.get('users', [None])[0]
        if user is None:
            raise RuntimeError(f"user [{org_email}]不存在")
        key_data = GenerateKeyRequest(user_id=user.user_id, key_alias=org_email, team_id=team_id, models=["all-team-models"])
        key_res = await key_management_endpoints.generate_key_fn(key_data, user_api_key_dict=user_api_key_dict)
    else:
        metadata = {
            'provider': provider
        }
        user_data = NewUserRequest(key_alias=org_email, auto_create_key=True, user_id=user_id, user_alias=user_name, user_email=org_email, user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY, team_id=team_id, metadata=metadata)
        logger.info(f'user[{user_id}:{org_email}] create end...')
        key_res = await internal_user_endpoints.new_user(user_data, user_api_key_dict=user_api_key_dict)

    return (True, key_res.key)


@router.get(
    "/zx/job",
    tags=["ZX"],
    dependencies=[Depends(user_api_key_auth)],
)
async def zx_job(request: Request, start_date: str | None = None, end_date: str | None = None, user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),):
    """
    定时任务
    """

    if user_api_key_dict.user_role is None or LitellmUserRoles.PROXY_ADMIN not in user_api_key_dict.user_role:
        return HTMLResponse(status_code=400, content="""
<!doctype html>
<html lang="en">
    <body>
        No admin access
    </body>
</html>
""")

    from . import zx_job
    _ = await zx_job.ai_usage_to_nocobase(start_date, end_date)
    # return [request.headers, request.url, request.client]
    return HTMLResponse(content="""
<!doctype html>
<html lang="en">
    <body>
        OK
    </body>
</html>
""")

@router.get(
    "/zx/cli_login",
    tags=["ZX"],
    # include_in_schema=False,
)
async def cli_login(token: str, request: Request):
    """
    处理CLI 登录请求
    """
    auth_key = str(uuid.uuid4())
    scheme = request.headers.get("X-Forwarded-Proto", request.url.scheme)
    host = request.headers.get("X-Forwarded-Host", request.url.netloc)
    url = auth.generate_oauth_url(f'{scheme}://{host}/zx/auth_callback?auth_key={auth_key}')
    set_store(type='cli', token=token, auth_key=auth_key, timeout=5)
    return RedirectResponse(url, status_code=status.HTTP_302_FOUND)


@router.get(
    "/zx/auth_callback",
    tags=["ZX"],
)
async def auth_callback(auth_key: str, code: str, request: Request):
    """
    处理统一登录
    """
    # 从token_store[code]中匹配auth_key
    store = get_store(auth_key=auth_key, check_login=False)
    if store is None:
        raise RuntimeError(f"回调错误，auth_key [{auth_key}]不正确")

    try:
        access_token = auth.get_access_token(code)
        user_info = auth.get_user_info(access_token)

        store.status = 'success'
        store.login = True
        store.data['user_info'] = user_info
    except Exception as e:
        store.status = 'failed'
        raise RuntimeError(f"获取用户信息失败，auth_key [{auth_key}] 错误: {e}")

    return RedirectResponse('/zx/auth_success', status_code=status.HTTP_302_FOUND)


@router.get(
    "/zx/auth_success",
    tags=["ZX"],
)
async def auth_success(request: Request):
    """
    登录成功
    """

    # return [request.headers, request.url, request.client]
    return HTMLResponse(content="""
<!doctype html>
<html lang="en">
    <body>
        用户登录完成，请手动关闭本页面并返回命令行查看配置结果。
    </body>
</html>
""")


@router.get(
    "/zx/cli_check_token",
    tags=["ZX"],
)
async def cli_check_token(token: Annotated[str | None, Header()] = None):
    """
    处理CLI 校验Token
    """
    store = get_store(type='cli', token=token)
    if store is None:
        return False
    return True


@router.get(
    "/zx/cli_get_key",
    tags=["ZX"],
)
async def cli_get_key(token: Annotated[str | None, Header()] = None):
    """
    处理CLI 获取Key
    """
    store = get_store(type='cli', token=token)
    if store is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    user_api_key_dict = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    
    user_info = store.data.get('user_info', {})
    user_id = user_info['userId']
    user_name = user_info['name']
    org_email = user_info['orgEmail']
    dept_id = None
    if user_info.get('deptIdList'):
        dept_id = user_info.get('deptIdList')[0]
    (created, key_or_key_id) = await create_or_get_user_key('ai_developer', user_id, user_name, org_email, dept_id, user_api_key_dict)
    key = None
    if created:
        key = key_or_key_id
    else:
        logger.info(f'user[{user_id}] recreate key')
        key_res = await key_management_endpoints.regenerate_key_fn(key_or_key_id, litellm_changed_by='ai_developer', user_api_key_dict=user_api_key_dict)
        if key_res is not None:
            key = key_res.key

    return {
        'key': key
    }


@router.get(
    "/zx/cli_get_config",
    tags=["ZX"],
)
async def cli_get_config(token: Annotated[str | None, Header()] = None):
    """
    处理CLI 获取配置
    """
    store = get_store(type='cli', token=token)
    if store is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    with open(os.path.join('/app/zx', 'llm_config.json'), 'r') as file:
        llm_config = json.load(file)
    return llm_config


@router.get(
    "/zx/cli_get_config_yaml",
    tags=["ZX"],
)
async def cli_get_config_yaml(token: Annotated[str | None, Header()] = None):
    """
    处理CLI 获取Yaml配置
    """
    store = get_store(type='cli', token=token)
    if store is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    user_info = store.data.get('user_info', {})
    org_email = user_info['orgEmail']
    
    from litellm.proxy.proxy_server import prisma_client
    litellm_user_info = await prisma_client.db.litellm_usertable.find_first( # type: ignore
        where={'user_email': org_email}
    )
    if not litellm_user_info:
        logger.warning(f"User information not found for org_email: {org_email}")
        return {
            "error": f"获取用户信息不存在，org_email[{org_email}]"
        }
    litellm_user_id = litellm_user_info.user_id

    with open(os.path.join('/app/zx', 'llm_ide_continue_config_zixun.yaml'), 'r', encoding='utf-8') as file:
        content = file.read()
    return {
        "data": content.replace('<UTOKEN>', litellm_user_id)
    }


@router.post(
    "/zx/continue_plugin/dev_data",
    tags=["ZX"],
)
async def continue_plugin_dev_data(request: Request, utoken: Annotated[str | None, Header()] = None):
    """
    接收Continue插件使用数据
    """
    
    if continue_plugin_dev_data_enabled:
        body = await request.body()
        event = body.decode('utf-8')
        user_event = f'{event[0]}"litellm_user_id":"{utoken}",{event[1:]}'
        zx_development_data.add_continue_plugin_event(user_event)

    return {'success': 'true'}


@router.post(
    "/zx/provider/user_add",
    tags=["ZX"],
)
async def provider_user_add(client_id: str, signature: str, timestamp: str, request: Request,):
    """
    动态添加用户
    """

    try:
        int_num = int(timestamp)
    except ValueError as e:
        raise Exception(f"Invalid API key: timestamp[{timestamp}] error")
    # 半小时内有效
    if time.time() - int_num > 1800:
        raise Exception(f"Invalid API key: timestamp[{timestamp}] expired")

    body = await request.body()
    data = body.decode('utf-8')

    if not security_validator.validate(client_id, signature, f"{data}:{timestamp}"):
        raise HTTPException(status_code=401, detail="Invalid token")

    user_info = json.loads(data)
    org_email = user_info['orgEmail']
    if not org_email.endswith(add_user_allow_email_domain):
        return {'success': 'false', 'email': org_email, 'error': 'Invalid email domain' }

    user_id = user_info.get('userId') or f"{client_id}_{uuid.uuid4()}"
    user_name = user_info['name']
    dept_id = user_info.get('deptId')
    if dept_id is None and user_info.get('deptIdList'):
        dept_id = user_info.get('deptIdList')[0]
    (created, res) = await create_or_get_user_key(client_id, user_id, user_name, org_email, dept_id)

    return {'success': 'true', 'email': org_email, 'created': created }
