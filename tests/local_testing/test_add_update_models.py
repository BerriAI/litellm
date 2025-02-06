import sys, os
import traceback
from dotenv import load_dotenv
from fastapi import Request
from datetime import datetime

load_dotenv()
import os, io, time

# this file is to test litellm/proxy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest, logging, asyncio
import litellm, asyncio
from litellm.proxy.proxy_server import add_new_model, update_model, LitellmUserRoles
from litellm._logging import verbose_proxy_logger
from litellm.proxy.utils import PrismaClient, ProxyLogging

verbose_proxy_logger.setLevel(level=logging.DEBUG)
from litellm.caching.caching import DualCache
from litellm.router import (
    Deployment,
    LiteLLM_Params,
)
from litellm.types.router import ModelInfo, updateDeployment, updateLiteLLMParams

from litellm.proxy._types import (
    UserAPIKeyAuth,
)

proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())


@pytest.fixture
def prisma_client():
    from litellm.proxy.proxy_cli import append_query_params

    ### add connection pool + pool timeout args
    params = {"connection_limit": 100, "pool_timeout": 60}
    database_url = os.getenv("DATABASE_URL")
    modified_url = append_query_params(database_url, params)
    os.environ["DATABASE_URL"] = modified_url
    os.environ["STORE_MODEL_IN_DB"] = "true"

    # Assuming PrismaClient is a class that needs to be instantiated
    prisma_client = PrismaClient(
        database_url=os.environ["DATABASE_URL"], proxy_logging_obj=proxy_logging_obj
    )

    # Reset litellm.proxy.proxy_server.prisma_client to None
    litellm.proxy.proxy_server.litellm_proxy_budget_name = (
        f"litellm-proxy-budget-{time.time()}"
    )
    litellm.proxy.proxy_server.user_custom_key_generate = None

    return prisma_client


@pytest.mark.asyncio
@pytest.mark.skip(reason="new feature, tests passing locally")
async def test_add_new_model(prisma_client):
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    setattr(litellm.proxy.proxy_server, "store_model_in_db", True)

    await litellm.proxy.proxy_server.prisma_client.connect()
    from litellm.proxy.proxy_server import user_api_key_cache
    import uuid

    _new_model_id = f"local-test-{uuid.uuid4().hex}"

    await add_new_model(
        model_params=Deployment(
            model_name="test_model",
            litellm_params=LiteLLM_Params(
                model="azure/gpt-3.5-turbo",
                api_key="test_api_key",
                api_base="test_api_base",
                rpm=1000,
                tpm=1000,
            ),
            model_info=ModelInfo(
                id=_new_model_id,
            ),
        ),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN.value,
            api_key="sk-1234",
            user_id="1234",
        ),
    )

    _new_models = await prisma_client.db.litellm_proxymodeltable.find_many()
    print("_new_models: ", _new_models)

    _new_model_in_db = None
    for model in _new_models:
        print("current model: ", model)
        if model.model_info["id"] == _new_model_id:
            print("FOUND MODEL: ", model)
            _new_model_in_db = model

    assert _new_model_in_db is not None


@pytest.mark.parametrize(
    "team_id, key_team_id, user_role, expected_result",
    [
        ("1234", "1234", LitellmUserRoles.PROXY_ADMIN.value, True),
        (
            "1234",
            "1235",
            LitellmUserRoles.PROXY_ADMIN.value,
            True,
        ),  # proxy admin can add models for any team
        (None, "1234", LitellmUserRoles.PROXY_ADMIN.value, True),
        (None, None, LitellmUserRoles.PROXY_ADMIN.value, True),
        (
            "1234",
            "1234",
            LitellmUserRoles.INTERNAL_USER.value,
            True,
        ),  # internal users can add models for their team
        ("1234", "1235", LitellmUserRoles.INTERNAL_USER.value, False),
        (None, "1234", LitellmUserRoles.INTERNAL_USER.value, False),
        (
            None,
            None,
            LitellmUserRoles.INTERNAL_USER.value,
            False,
        ),  # internal users cannot add models by default
    ],
)
def test_can_add_model(team_id, key_team_id, user_role, expected_result):
    from litellm.proxy.proxy_server import check_if_team_id_matches_key

    args = {
        "team_id": team_id,
        "user_api_key_dict": UserAPIKeyAuth(
            user_role=user_role,
            api_key="sk-1234",
            team_id=key_team_id,
        ),
    }

    assert check_if_team_id_matches_key(**args) is expected_result


@pytest.mark.asyncio
@pytest.mark.skip(reason="new feature, tests passing locally")
async def test_add_update_model(prisma_client):
    # test that existing litellm_params are not updated
    # only new / updated params get updated
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    setattr(litellm.proxy.proxy_server, "store_model_in_db", True)

    await litellm.proxy.proxy_server.prisma_client.connect()
    from litellm.proxy.proxy_server import user_api_key_cache
    import uuid

    _new_model_id = f"local-test-{uuid.uuid4().hex}"

    await add_new_model(
        model_params=Deployment(
            model_name="test_model",
            litellm_params=LiteLLM_Params(
                model="azure/gpt-3.5-turbo",
                api_key="test_api_key",
                api_base="test_api_base",
                rpm=1000,
                tpm=1000,
            ),
            model_info=ModelInfo(
                id=_new_model_id,
            ),
        ),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN.value,
            api_key="sk-1234",
            user_id="1234",
        ),
    )

    _new_models = await prisma_client.db.litellm_proxymodeltable.find_many()
    print("_new_models: ", _new_models)

    _new_model_in_db = None
    for model in _new_models:
        print("current model: ", model)
        if model.model_info["id"] == _new_model_id:
            print("FOUND MODEL: ", model)
            _new_model_in_db = model

    assert _new_model_in_db is not None

    _original_model = _new_model_in_db
    _original_litellm_params = _new_model_in_db.litellm_params
    print("_original_litellm_params: ", _original_litellm_params)
    print("now updating the tpm for model")
    # run update to update "tpm"
    await update_model(
        model_params=updateDeployment(
            litellm_params=updateLiteLLMParams(tpm=123456),
            model_info=ModelInfo(
                id=_new_model_id,
            ),
        ),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN.value,
            api_key="sk-1234",
            user_id="1234",
        ),
    )

    _new_models = await prisma_client.db.litellm_proxymodeltable.find_many()

    _new_model_in_db = None
    for model in _new_models:
        if model.model_info["id"] == _new_model_id:
            print("\nFOUND MODEL: ", model)
            _new_model_in_db = model

    # assert all other litellm params are identical to _original_litellm_params
    for key, value in _original_litellm_params.items():
        if key == "tpm":
            # assert that tpm actually got updated
            assert _new_model_in_db.litellm_params[key] == 123456
        else:
            assert _new_model_in_db.litellm_params[key] == value

    assert _original_model.model_id == _new_model_in_db.model_id
    assert _original_model.model_name == _new_model_in_db.model_name
    assert _original_model.model_info == _new_model_in_db.model_info
