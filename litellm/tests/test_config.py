# What is this?
## Unit tests for ProxyConfig class


import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
import io
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

import litellm
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value
from litellm.proxy.proxy_server import ProxyConfig
from litellm.proxy.utils import DualCache, ProxyLogging
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo


class DBModel(BaseModel):
    model_id: str
    model_name: str
    model_info: dict
    litellm_params: dict

    model_config = ConfigDict(protected_namespaces=())


@pytest.mark.asyncio
async def test_delete_deployment():
    """
    - Ensure the global llm router is not being reset
    - Ensure invalid model is deleted
    - Check if model id != model_info["id"], the model_info["id"] is picked
    """
    import base64

    litellm_params = LiteLLM_Params(
        model="azure/chatgpt-v-2",
        api_key=os.getenv("AZURE_API_KEY"),
        api_base=os.getenv("AZURE_API_BASE"),
        api_version=os.getenv("AZURE_API_VERSION"),
    )
    encrypted_litellm_params = litellm_params.dict(exclude_none=True)

    master_key = "sk-1234"

    setattr(litellm.proxy.proxy_server, "master_key", master_key)

    for k, v in encrypted_litellm_params.items():
        if isinstance(v, str):
            encrypted_value = encrypt_value(v, master_key)
            encrypted_litellm_params[k] = base64.b64encode(encrypted_value).decode(
                "utf-8"
            )

    deployment = Deployment(model_name="gpt-3.5-turbo", litellm_params=litellm_params)
    deployment_2 = Deployment(
        model_name="gpt-3.5-turbo-2", litellm_params=litellm_params
    )

    llm_router = litellm.Router(
        model_list=[
            deployment.to_json(exclude_none=True),
            deployment_2.to_json(exclude_none=True),
        ]
    )
    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)
    print(f"llm_router: {llm_router}")

    pc = ProxyConfig()

    db_model = DBModel(
        model_id=deployment.model_info.id,
        model_name="gpt-3.5-turbo",
        litellm_params=encrypted_litellm_params,
        model_info={"id": deployment.model_info.id},
    )

    db_models = [db_model]
    deleted_deployments = await pc._delete_deployment(db_models=db_models)

    assert deleted_deployments == 1
    assert len(llm_router.model_list) == 1

    """
    Scenario 2 - if model id != model_info["id"]
    """

    llm_router = litellm.Router(
        model_list=[
            deployment.to_json(exclude_none=True),
            deployment_2.to_json(exclude_none=True),
        ]
    )
    print(f"llm_router: {llm_router}")
    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)
    pc = ProxyConfig()

    db_model = DBModel(
        model_id=deployment.model_info.id,
        model_name="gpt-3.5-turbo",
        litellm_params=encrypted_litellm_params,
        model_info={"id": deployment.model_info.id},
    )

    db_models = [db_model]
    deleted_deployments = await pc._delete_deployment(db_models=db_models)

    assert deleted_deployments == 1
    assert len(llm_router.model_list) == 1


@pytest.mark.asyncio
async def test_add_existing_deployment():
    """
    - Only add new models
    - don't re-add existing models
    """
    import base64

    litellm_params = LiteLLM_Params(
        model="gpt-3.5-turbo",
        api_key=os.getenv("AZURE_API_KEY"),
        api_base=os.getenv("AZURE_API_BASE"),
        api_version=os.getenv("AZURE_API_VERSION"),
    )
    deployment = Deployment(model_name="gpt-3.5-turbo", litellm_params=litellm_params)
    deployment_2 = Deployment(
        model_name="gpt-3.5-turbo-2", litellm_params=litellm_params
    )

    llm_router = litellm.Router(
        model_list=[
            deployment.to_json(exclude_none=True),
            deployment_2.to_json(exclude_none=True),
        ]
    )

    init_len_list = len(llm_router.model_list)
    print(f"llm_router: {llm_router}")
    master_key = "sk-1234"
    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)
    setattr(litellm.proxy.proxy_server, "master_key", master_key)
    pc = ProxyConfig()

    encrypted_litellm_params = litellm_params.dict(exclude_none=True)

    for k, v in encrypted_litellm_params.items():
        if isinstance(v, str):
            encrypted_value = encrypt_value(v, master_key)
            encrypted_litellm_params[k] = base64.b64encode(encrypted_value).decode(
                "utf-8"
            )
    db_model = DBModel(
        model_id=deployment.model_info.id,
        model_name="gpt-3.5-turbo",
        litellm_params=encrypted_litellm_params,
        model_info={"id": deployment.model_info.id},
    )

    db_models = [db_model]
    num_added = pc._add_deployment(db_models=db_models)

    assert init_len_list == len(llm_router.model_list)


litellm_params = LiteLLM_Params(
    model="azure/chatgpt-v-2",
    api_key=os.getenv("AZURE_API_KEY"),
    api_base=os.getenv("AZURE_API_BASE"),
    api_version=os.getenv("AZURE_API_VERSION"),
)

deployment = Deployment(model_name="gpt-3.5-turbo", litellm_params=litellm_params)
deployment_2 = Deployment(model_name="gpt-3.5-turbo-2", litellm_params=litellm_params)


def _create_model_list(flag_value: Literal[0, 1], master_key: str):
    """
    0 - empty list
    1 - list with an element
    """
    import base64

    new_litellm_params = LiteLLM_Params(
        model="azure/chatgpt-v-2-3",
        api_key=os.getenv("AZURE_API_KEY"),
        api_base=os.getenv("AZURE_API_BASE"),
        api_version=os.getenv("AZURE_API_VERSION"),
    )

    encrypted_litellm_params = new_litellm_params.dict(exclude_none=True)

    for k, v in encrypted_litellm_params.items():
        if isinstance(v, str):
            encrypted_value = encrypt_value(v, master_key)
            encrypted_litellm_params[k] = base64.b64encode(encrypted_value).decode(
                "utf-8"
            )
    db_model = DBModel(
        model_id="12345",
        model_name="gpt-3.5-turbo",
        litellm_params=encrypted_litellm_params,
        model_info={"id": "12345"},
    )

    db_models = [db_model]

    if flag_value == 0:
        return []
    elif flag_value == 1:
        return db_models


@pytest.mark.parametrize(
    "llm_router",
    [
        None,
        litellm.Router(),
        litellm.Router(
            model_list=[
                deployment.to_json(exclude_none=True),
                deployment_2.to_json(exclude_none=True),
            ]
        ),
    ],
)
@pytest.mark.parametrize(
    "model_list_flag_value",
    [0, 1],
)
@pytest.mark.asyncio
async def test_add_and_delete_deployments(llm_router, model_list_flag_value):
    """
    Test add + delete logic in 3 scenarios
    - when router is none
    - when router is init but empty
    - when router is init and not empty
    """

    master_key = "sk-1234"
    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)
    setattr(litellm.proxy.proxy_server, "master_key", master_key)
    pc = ProxyConfig()
    pl = ProxyLogging(DualCache())

    async def _monkey_patch_get_config(*args, **kwargs):
        print(f"ENTERS MP GET CONFIG")
        if llm_router is None:
            return {}
        else:
            print(f"llm_router.model_list: {llm_router.model_list}")
            return {"model_list": llm_router.model_list}

    pc.get_config = _monkey_patch_get_config

    model_list = _create_model_list(
        flag_value=model_list_flag_value, master_key=master_key
    )

    if llm_router is None:
        prev_llm_router_val = None
    else:
        prev_llm_router_val = len(llm_router.model_list)

    await pc._update_llm_router(new_models=model_list, proxy_logging_obj=pl)

    llm_router = getattr(litellm.proxy.proxy_server, "llm_router")

    if model_list_flag_value == 0:
        if prev_llm_router_val is None:
            assert prev_llm_router_val == llm_router
        else:
            assert prev_llm_router_val == len(llm_router.model_list)
    else:
        if prev_llm_router_val is None:
            assert len(llm_router.model_list) == len(model_list)
        else:
            assert len(llm_router.model_list) == len(model_list) + prev_llm_router_val
