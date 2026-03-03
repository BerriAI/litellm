import os
import logging
from typing import Annotated, Literal, Optional
import json
import uuid
from fastapi import APIRouter, Depends, HTTPException, Header, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints import key_management_endpoints
from litellm.proxy._types import (
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from . import zixun_auth
from . import zx_development_data
from .token_util import (
    set_store,
    get_store,
    create_or_get_user_key
)

auth = zixun_auth.ZixunAuth(
    os.environ.get('ZX_AUTH_HOST'),
    os.environ.get('ZX_AUTH_API_HOST'),
    os.environ.get('ZX_AUTH_APP_KEY'),
    os.environ.get('ZX_AUTH_APP_SECRET'),
    )

logger = logging.getLogger()

router = APIRouter()

continue_plugin_dev_data_enabled = 'true' == os.environ.get('ZX_CONTINUE_PLUGIN_DEV_DATA_ENABLED', 'false').strip()

add_user_allow_email_domain = '@'+os.environ.get('ZX_ADD_USER_ALLOW_EMAIL_DOMAIN', 'fzzixun.com').strip()


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
async def cli_get_key(type: Optional[str] = None, token: Annotated[str | None, Header()] = None):
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
    key_alias = None
    if type is not None and type.strip().startswith("assistant-"):
        key_alias = f"{type.strip()}--{org_email.split('@')[0]}"
    (created, key_or_key_id) = await create_or_get_user_key('ai_developer', user_id, user_name, org_email, dept_id, user_api_key_dict, key_alias)
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
