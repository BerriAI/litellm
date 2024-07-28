#### CRUD ENDPOINTS for UI Settings #####
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import fastapi
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


class IPAddress(BaseModel):
    ip: str


@router.get(
    "/get/allowed_ips",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def get_allowed_ips():
    from litellm.proxy.proxy_server import general_settings

    _allowed_ip = general_settings.get("allowed_ips")
    return {"data": _allowed_ip}


@router.post(
    "/add/allowed_ip",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
)
async def add_allowed_ip(ip_address: IPAddress):
    from litellm.proxy.proxy_server import (
        general_settings,
        prisma_client,
        proxy_config,
        store_model_in_db,
    )

    _allowed_ips: List = general_settings.get("allowed_ips", [])
    if ip_address.ip not in _allowed_ips:
        _allowed_ips.append(ip_address.ip)
        general_settings["allowed_ips"] = _allowed_ips
    else:
        raise HTTPException(status_code=400, detail="IP address already exists")

    if prisma_client is None:
        raise Exception("No DB Connected")

    if store_model_in_db is not True:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Set `'STORE_MODEL_IN_DB='True'` in your env to enable this feature."
            },
        )

    # Load existing config
    config = await proxy_config.get_config()
    verbose_proxy_logger.debug("Loaded config: %s", config)
    if "general_settings" not in config:
        config["general_settings"] = {}

    if "allowed_ips" not in config["general_settings"]:
        config["general_settings"]["allowed_ips"] = []

    if ip_address.ip not in config["general_settings"]["allowed_ips"]:
        config["general_settings"]["allowed_ips"].append(ip_address.ip)

    await proxy_config.save_config(new_config=config)

    return {
        "message": f"IP {ip_address.ip} address added successfully",
        "status": "success",
    }


@router.post(
    "/delete/allowed_ip",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_allowed_ip(ip_address: IPAddress):
    from litellm.proxy.proxy_server import general_settings, proxy_config

    _allowed_ips: List = general_settings.get("allowed_ips", [])
    if ip_address.ip in _allowed_ips:
        _allowed_ips.remove(ip_address.ip)
        general_settings["allowed_ips"] = _allowed_ips
    else:
        raise HTTPException(status_code=404, detail="IP address not found")

    # Load existing config
    config = await proxy_config.get_config()
    verbose_proxy_logger.debug("Loaded config: %s", config)
    if "general_settings" not in config:
        config["general_settings"] = {}

    if "allowed_ips" not in config["general_settings"]:
        config["general_settings"]["allowed_ips"] = []

    if ip_address.ip in config["general_settings"]["allowed_ips"]:
        config["general_settings"]["allowed_ips"].remove(ip_address.ip)

    await proxy_config.save_config(new_config=config)

    return {"message": f"IP {ip_address.ip} deleted successfully", "status": "success"}
