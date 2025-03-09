"""
Functions to create audit logs for LiteLLM Proxy
"""

import json

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LiteLLM_AuditLogs


async def create_audit_log_for_update(request_data: LiteLLM_AuditLogs):
    from litellm.proxy.proxy_server import premium_user, prisma_client

    if premium_user is not True:
        return

    if litellm.store_audit_logs is not True:
        return
    if prisma_client is None:
        raise Exception("prisma_client is None, no DB connected")

    verbose_proxy_logger.debug("creating audit log for %s", request_data)

    if isinstance(request_data.updated_values, dict):
        request_data.updated_values = json.dumps(request_data.updated_values)

    if isinstance(request_data.before_value, dict):
        request_data.before_value = json.dumps(request_data.before_value)

    _request_data = request_data.model_dump(exclude_none=True)

    try:
        await prisma_client.db.litellm_auditlog.create(
            data={
                **_request_data,  # type: ignore
            }
        )
    except Exception as e:
        # [Non-Blocking Exception. Do not allow blocking LLM API call]
        verbose_proxy_logger.error(f"Failed Creating audit log {e}")

    return
