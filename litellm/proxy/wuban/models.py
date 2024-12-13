from json import dumps
from typing import Callable

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


router = APIRouter()
model_list: Callable

def set_model_list_def(model_list_from_proxy: Callable):
    global model_list
    model_list = model_list_from_proxy

@router.get(
    "/api/models",
    dependencies=[Depends(user_api_key_auth)],
    tags=["model management"]
)
async def models(user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth)):
    originData = await model_list(user_api_key_dict)
    ## todo transform data
    print(originData)
    # "data": [
    #     {"id": "deepseek/deepseek-chat",
    #      "object": "model",
    #      "created": 1677610602,
    #      "owned_by": "openai"}
    # ],
    data = originData['data']
    if data is not None:
        if len(data) > 0:
            outData = {}
            for item in data:
                group = outData[item["owned_by"]]
                if group is None:
                    group = []
                    outData[item["owned_by"]] = group
                group.append(item["id"])
            return outData
    ## mock data
    outData = {}
    outData["openAI"] = ["chatgpt-4o-latest", "gpt-4o-2024-08-06"]
    outData["deepseek"] = ["deepseek/chat", "deepseek/code"]
    return outData


@router.get(
    "/api/endpoints",
    dependencies=[Depends(user_api_key_auth)],
    tags=["model management"]
)
async def endpoints(user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth)):
    originData = await model_list(user_api_key_dict)
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
    map = {}
    map.expiresAt = "2034-11-17T06:58:35.462Z"
    return dumps(map)


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
