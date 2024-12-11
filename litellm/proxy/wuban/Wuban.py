import asyncio
import copy
import os
import traceback
from datetime import datetime, timedelta
from typing import Literal, Optional, Union

import fastapi
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    AlertType,
    CallInfo,
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
    WebhookEvent,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.health_check import perform_health_check
from litellm.proxy.proxy_server import model_list
#### Health ENDPOINTS ####

router = APIRouter()


@router.get(
    "/api/models",
    dependencies=[Depends(user_api_key_auth)],
    tags=["model management"]
)
async def models(user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth)):
    originData = model_list(user_api_key_dict)
    ## todo transform data
    print(originData)
    # outData = []
    return originData


@router.get(
    "/api/endpoints",
    dependencies=[Depends(user_api_key_auth)],
    tags=["model management"]
)
async def endpoints(user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth)):
    originData = model_list(user_api_key_dict)
    ## todo transform data
    print(originData)
    # outData = []
    return originData


@router.get(
    "/api/keys",
    dependencies=[Depends(user_api_key_auth)],
    tags=["model management"]
)
async def keys(name):
    return ""

@router.post(
    "/api/files/images",
    dependencies=[Depends(user_api_key_auth)],
    tags=["model management"]
)
async def image():
    return ""


@router.post(
    "/api/files",
    dependencies=[Depends(user_api_key_auth)],
    tags=["model management"]
)
async def file():
    return ""
