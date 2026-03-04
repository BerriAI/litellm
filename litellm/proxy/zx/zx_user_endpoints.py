import os
import logging
import json
import time
import uuid
from fastapi import APIRouter, HTTPException, Request
from .zx_security_validator import security_validator
from .token_util import create_or_get_user_key

logger = logging.getLogger()

router = APIRouter()

add_user_allow_email_domain = (
    "@" + os.environ.get("ZX_ADD_USER_ALLOW_EMAIL_DOMAIN", "fzzixun.com").strip()
)


@router.post(
    "/zx/api/provider/user_add",
    tags=["ZX"],
)
async def provider_user_add(
    client_id: str,
    signature: str,
    timestamp: str,
    request: Request,
):
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
    data = body.decode("utf-8")

    if not security_validator.validate(client_id, signature, f"{data}:{timestamp}"):
        raise HTTPException(status_code=401, detail="Invalid token")

    user_info = json.loads(data)
    org_email = user_info["orgEmail"]
    if not org_email.endswith(add_user_allow_email_domain):
        return {"success": "false", "email": org_email, "error": "Invalid email domain"}

    user_id = user_info.get("userId") or f"{client_id}_{uuid.uuid4()}"
    user_name = user_info["name"]
    dept_id = user_info.get("deptId")
    if dept_id is None and user_info.get("deptIdList"):
        dept_id = user_info.get("deptIdList")[0]
    (created, res) = await create_or_get_user_key(
        client_id, user_id, user_name, org_email, dept_id
    )

    return {"success": "true", "email": org_email, "created": created}
