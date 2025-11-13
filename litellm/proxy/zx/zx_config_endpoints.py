import os
import logging
from typing import Annotated
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
)
from .import zixun_auth
from . import zx_development_data

auth = zixun_auth.ZixunAuth(
    os.environ.get('ZX_AUTH_HOST'),
    os.environ.get('ZX_AUTH_API_HOST'),
    os.environ.get('ZX_AUTH_APP_KEY'),
    os.environ.get('ZX_AUTH_APP_SECRET'),
    )

logger = logging.getLogger()

router = APIRouter()

token_store = {}

def get_store(token: str | None):
    for k, v in list(token_store.items()):
        if v['expire_time'] < time.time():
            del token_store[k]

    store = token_store.get(token, None)
    if not store:
        return None
    # if store['expire_time'] < time.time():
    #     del token_store[token]
    #     return None
    if not store['login']:
        return None
    return store


@router.get(
    "/zx/job",
    tags=["ZX"],
    dependencies=[Depends(user_api_key_auth)],
)
async def zx_job(request: Request, start_date: str | None = None, end_date: str | None = None):
    """
    定时任务
    """

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
    token_store[token] = {
        'login': False,
        'status': "pending",
        # 过期时间
        'expire_time': time.time() + 30 * 60,
        'auth_key': auth_key,
    }
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
    store = next((x for x in token_store.values() if x['auth_key'] == auth_key), None)
    if store is None:
        raise RuntimeError(f"回调错误，auth_key [{auth_key}]不正确")

    try:
        access_token = auth.get_access_token(code)
        user_info = auth.get_user_info(access_token)

        store['user_info'] = user_info
        store['status'] = 'success'
        store['login'] = True
    except Exception as e:
        store['status'] = 'failed'
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
    store = get_store(token)
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
    store = get_store(token)
    if store is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    user_api_key_dict = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    
    user_info = store.get('user_info', {})
    user_id = user_info['userId']
    user_name = user_info['name']
    org_email = user_info['orgEmail']
    dept_id = ""
    if user_info.get('deptIdList'):
        dept_id = user_info.get('deptIdList')[0]
    if not dept_id:
        dept_id = "none_dept_id"
    keys = await key_management_endpoints.list_keys(
        Request({'type': 'http', 'query_string': '',}),
        page=1, 
        size=1, 
        key_alias=org_email, 
        user_api_key_dict=user_api_key_dict,
        user_id=None,
        team_id=None,
        organization_id=None,
        key_hash=None,
        return_full_object=False,
        include_team_keys=False,
        include_created_by_keys=False,
        sort_by=None,
        sort_order="desc"
    )
    key_id = None
    if keys:
        key_id = keys.get('key', [None])[0]
    if key_id is None:
        logger.info(f'user[{user_id}] create key')
        teams = await team_endpoints.list_team(Request({'type': 'http', 'query_string': '',}), user_id=None, organization_id=None, user_api_key_dict=user_api_key_dict)
        team = next((x for x in teams if x.team_alias and x.team_alias.endswith(f'__{dept_id}')), None)
        if team is None:
            logger.info(f'user[{user_id}] create team')
            team_data = NewTeamRequest(team_alias=f'__{dept_id}', models=["all-proxy-models"])
            team_res = await team_endpoints.new_team(team_data, Request({'type': 'http', 'query_string': '',}), litellm_changed_by='ai_developer', user_api_key_dict=user_api_key_dict)
            team_id = team_res['team_id']
        else:
            team_id = team.team_id
        user_data = NewUserRequest(key_alias=org_email, auto_create_key=True, user_id=user_id, user_alias=user_name, user_email=org_email, user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY, team_id=team_id)
        logger.info(f'user[{user_id}] create user')
        key_res = await internal_user_endpoints.new_user(user_data, user_api_key_dict=user_api_key_dict)
        # key_data = GenerateKeyRequest(user_id='default_user_id', key_alias=org_email, team_id=team_id, models=["all-team-models"])
        # key_res = await generate_key_fn(key_data, user_api_key_dict=user_api_key_dict)
    else:
        logger.info(f'user[{user_id}] recreate key')
        key_res = await key_management_endpoints.regenerate_key_fn(key_id, litellm_changed_by='ai_developer', user_api_key_dict=user_api_key_dict)

    new_key = None
    if key_res is not None:
        new_key = key_res.key

    return {
        'key': new_key
    }


@router.get(
    "/zx/cli_get_config",
    tags=["ZX"],
)
async def cli_get_config(token: Annotated[str | None, Header()] = None):
    """
    处理CLI 获取配置
    """
    store = get_store(token)
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
    store = get_store(token)
    if store is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    user_info = store.get('user_info', {})
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

    with open(os.path.join('/app/zx', 'llm_config_zixun.yaml'), 'r', encoding='utf-8') as file:
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
    # store = get_store(token)
    # if store is None:
    #     raise HTTPException(status_code=401, detail="Invalid token")

    body = await request.body()
    event = body.decode('utf-8')
    user_event = f'{event[0]}"litellm_user_id":"{utoken}",{event[1:]}'
    zx_development_data.add_continue_plugin_event(user_event)
    return {'success': 'true'}
