import os
import sys
import pytest


sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import pytest
import sys
from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch
from litellm.proxy.prompts import prompt_endpoints as mod
from litellm.proxy._types import LitellmUserRoles
from litellm.types.prompts.init_prompts import PromptLiteLLMParams, PromptInfo, PromptSpec


@pytest.fixture
def client():
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(mod.router)
    return TestClient(app)


@pytest.fixture
def admin_user():
    return MagicMock(user_role=LitellmUserRoles.PROXY_ADMIN, metadata=None)


@pytest.fixture
def normal_user():
    return MagicMock(user_role="USER", metadata=None)


def make_prompt_spec(pid="p1"):
    return PromptSpec(
        prompt_id=pid,
        litellm_params=PromptLiteLLMParams(
            prompt_id=pid, prompt_integration="gitlab", model_config={"model": "gpt-4"}
        ),
        prompt_info=PromptInfo(prompt_type="db"),
    )


# ---------------------------------------------------------------------
# list endpoints
# ---------------------------------------------------------------------
def test_list_prompts_returns_admin_prompts(monkeypatch, client, admin_user):
    fake_prompt = make_prompt_spec("p1")
    fake_hub = MagicMock()
    fake_hub.IN_MEMORY_PROMPTS = {"p1": fake_prompt}
    monkeypatch.setitem(
        sys.modules, "litellm.proxy.prompts.prompt_registry", MagicMock(PROMPT_HUB=fake_hub)
    )

    with patch.object(mod, "user_api_key_auth", lambda: admin_user):
        resp = client.get("/prompts/list")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["prompts"]) == 1
    assert data["prompts"][0]["prompt_id"] == "p1"


@pytest.mark.asyncio
async def test_list_prompts_non_admin_empty(monkeypatch):
    user = MagicMock(user_role="USER", metadata={"prompts": None})
    fake_prompt = make_prompt_spec("p1")
    fake_hub = MagicMock()
    fake_hub.IN_MEMORY_PROMPTS = {"p1": fake_prompt}
    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.prompts.prompt_registry",
        MagicMock(PROMPT_HUB=fake_hub),
    )

    result = await mod.list_prompts(user)
    assert result.prompts == []





# ---------------------------------------------------------------------
# get_prompt_info
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_prompt_info_admin_with_gitlab(monkeypatch):
    fake_prompt = make_prompt_spec("pid")

    class FakeGitLabPromptManager:
        def __init__(self):
            self.integration_name = "gitlab"
            self.prompt_manager = MagicMock()
            self.prompt_manager.get_all_prompts_as_json.return_value = {
                "pid": {"content": "hi", "metadata": {"m": 1}}
            }

    fake_cb = FakeGitLabPromptManager()
    fake_hub = MagicMock()
    fake_hub.get_prompt_by_id.return_value = fake_prompt
    fake_hub.get_prompt_callback_by_id.return_value = fake_cb
    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.prompts.prompt_registry",
        MagicMock(PROMPT_HUB=fake_hub),
    )
    # ensure isinstance() passes
    monkeypatch.setitem(
        sys.modules,
        "litellm.integrations.gitlab",
        MagicMock(GitLabPromptManager=FakeGitLabPromptManager),
    )

    user = MagicMock(user_role=LitellmUserRoles.PROXY_ADMIN, metadata=None)
    result = await mod.get_prompt_info("pid", user)
    assert result.raw_prompt_template.content == "hi"



# ---------------------------------------------------------------------
# create / update / patch / delete
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_create_prompt_success(monkeypatch):
    req = mod.Prompt(
        prompt_id="p1",
        litellm_params=PromptLiteLLMParams(
            prompt_id="p1", prompt_integration="gitlab", model_config={"model": "gpt-4"}
        ),
        prompt_info=PromptInfo(prompt_type="db"),
    )
    admin = MagicMock(user_role=LitellmUserRoles.PROXY_ADMIN)

    fake_db_entry = MagicMock()
    fake_db_entry.model_dump.return_value = make_prompt_spec("p1").model_dump()

    fake_prisma = MagicMock()
    fake_prisma.db.litellm_prompttable.create = AsyncMock(return_value=fake_db_entry)
    monkeypatch.setitem(
        sys.modules, "litellm.proxy.proxy_server", MagicMock(prisma_client=fake_prisma)
    )

    fake_reg = MagicMock()
    fake_reg.get_prompt_by_id.return_value = None
    fake_reg.initialize_prompt.return_value = {"ok": True}
    monkeypatch.setitem(
        sys.modules, "litellm.proxy.prompts.prompt_registry",
        MagicMock(IN_MEMORY_PROMPT_REGISTRY=fake_reg)
    )

    resp = await mod.create_prompt(req, admin)
    assert resp == {"ok": True}


@pytest.mark.asyncio
async def test_patch_prompt_success(monkeypatch):
    admin = MagicMock(user_role=LitellmUserRoles.PROXY_ADMIN)
    fake_prompt = make_prompt_spec("p1")

    fake_prisma = MagicMock()
    fake_entry = MagicMock()
    fake_entry.model_dump.return_value = make_prompt_spec("p1").model_dump()
    fake_prisma.db.litellm_prompttable.update = AsyncMock(return_value=fake_entry)
    monkeypatch.setitem(
        sys.modules, "litellm.proxy.proxy_server", MagicMock(prisma_client=fake_prisma)
    )

    fake_reg = MagicMock()
    fake_reg.get_prompt_by_id.return_value = fake_prompt
    fake_reg.IN_MEMORY_PROMPTS = {"p1": fake_prompt}
    fake_reg.prompt_id_to_custom_prompt = {}
    fake_reg.initialize_prompt.return_value = {"patched": True}
    monkeypatch.setitem(
        sys.modules, "litellm.proxy.prompts.prompt_registry",
        MagicMock(IN_MEMORY_PROMPT_REGISTRY=fake_reg)
    )

    req = mod.PatchPromptRequest()
    resp = await mod.patch_prompt("p1", req, admin)
    assert resp == {"patched": True}
