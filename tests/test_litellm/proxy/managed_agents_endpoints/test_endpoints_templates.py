from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.managed_agents_endpoints.endpoints import router


@pytest.fixture
def admin():
    return UserAPIKeyAuth(
        api_key="sk-admin", user_id="a1", user_role=LitellmUserRoles.PROXY_ADMIN
    )


@pytest.fixture
def user():
    return UserAPIKeyAuth(
        api_key="sk-user", user_id="u1", user_role=LitellmUserRoles.INTERNAL_USER
    )


@pytest.fixture
def app_factory():
    def make(auth_user):
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[user_api_key_auth] = lambda: auth_user
        return TestClient(app)

    return make


@pytest.fixture
def fake_prisma():
    p = MagicMock()
    table = MagicMock()
    table.create = AsyncMock()
    table.find_unique = AsyncMock()
    table.find_many = AsyncMock(return_value=[])
    table.update = AsyncMock()
    table.delete = AsyncMock()
    p.db.litellm_managedagentsandboxtemplatetable = table
    agents_t = MagicMock()
    agents_t.count = AsyncMock(return_value=0)
    p.db.litellm_managedagenttable = agents_t
    cred_t = MagicMock()
    cred_t.create = AsyncMock(return_value=SimpleNamespace(credential_id="cred-1"))
    p.db.litellm_credentialstable = cred_t
    return p


def _public_body():
    return {
        "name": "tpl-1",
        "dockerfile_id": "opencode",
        "repo_url": "https://github.com/x/y",
        "default_branch": "main",
        "visibility": "public",
        "git_token": None,
    }


def test_dockerfiles_lists(app_factory, user):
    client = app_factory(user)
    with patch(
        "litellm.proxy.managed_agents_endpoints.endpoints.list_dockerfiles",
        return_value=[SimpleNamespace(dockerfile_id="opencode", container_port=4096)],
    ):
        resp = client.get("/v1/managed_agents/dockerfiles")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["id"] == "opencode"
    assert data[0]["container_port"] == 4096


def test_template_create_non_admin_403(app_factory, user, fake_prisma):
    client = app_factory(user)
    with patch("litellm.proxy.proxy_server.prisma_client", fake_prisma):
        resp = client.post("/v1/managed_agents/sandbox-templates", json=_public_body())
    assert resp.status_code == 403


def test_template_create_unknown_dockerfile_id_400(app_factory, admin, fake_prisma):
    client = app_factory(admin)
    body = _public_body()
    body["dockerfile_id"] = "nope"
    with (
        patch("litellm.proxy.proxy_server.prisma_client", fake_prisma),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints.get_dockerfile",
            side_effect=KeyError("nope"),
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints.list_dockerfiles",
            return_value=[
                SimpleNamespace(dockerfile_id="opencode", container_port=4096)
            ],
        ),
    ):
        resp = client.post("/v1/managed_agents/sandbox-templates", json=body)
    assert resp.status_code == 400
    assert "available" in str(resp.json()["detail"]).lower()


def test_template_create_happy_path(app_factory, admin, fake_prisma):
    client = app_factory(admin)

    created_row = SimpleNamespace(
        template_id="t1",
        template_name=None,
        dockerfile_id="opencode",
        container_port=4096,
        repo_url="https://github.com/x/y",
        default_branch="main",
        visibility="public",
        image_uri=None,
        task_def_arn=None,
        build_status="pending",
        build_error=None,
    )
    updated_row = SimpleNamespace(
        template_id="t1",
        template_name=None,
        dockerfile_id="opencode",
        container_port=4096,
        repo_url="https://github.com/x/y",
        default_branch="main",
        visibility="public",
        image_uri="img:abc",
        task_def_arn="arn:td",
        build_status="ready",
        build_error=None,
    )
    fake_prisma.db.litellm_managedagentsandboxtemplatetable.create = AsyncMock(
        return_value=created_row
    )
    fake_prisma.db.litellm_managedagentsandboxtemplatetable.update = AsyncMock(
        return_value=updated_row
    )

    provisioned = SimpleNamespace(
        image_uri="img:abc",
        task_def_arn="arn:td",
        image_hash="abc",
        container_port=4096,
        cluster_arn="arn:cluster",
        security_group_id="sg-1",
        subnet_ids=["sub-1"],
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", fake_prisma),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints.get_dockerfile",
            return_value=SimpleNamespace(
                dockerfile_id="opencode",
                container_port=4096,
                path="/x",
                context_dir="/x",
                build_platform="linux/amd64",
            ),
        ),
        patch("litellm.proxy.managed_agents_endpoints.endpoints.validate_repo_branch"),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints.provision_template",
            AsyncMock(return_value=provisioned),
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.config_loader.MANAGED_AGENTS_CONFIG",
            SimpleNamespace(aws_region="us-west-2", aws=SimpleNamespace(cluster=None)),
        ),
        patch("boto3.client", return_value=MagicMock()),
    ):
        resp = client.post("/v1/managed_agents/sandbox-templates", json=_public_body())

    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["image_uri"] == "img:abc"
    assert payload["build_status"] == "ready"
    fake_prisma.db.litellm_managedagentsandboxtemplatetable.update.assert_called()


def test_template_create_private_requires_token_400(app_factory, admin, fake_prisma):
    client = app_factory(admin)
    body = _public_body()
    body["visibility"] = "private"
    body["git_token"] = None

    with (
        patch("litellm.proxy.proxy_server.prisma_client", fake_prisma),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints.get_dockerfile",
            return_value=SimpleNamespace(
                dockerfile_id="opencode",
                container_port=4096,
                path="/x",
                context_dir="/x",
                build_platform="linux/amd64",
            ),
        ),
    ):
        resp = client.post("/v1/managed_agents/sandbox-templates", json=body)
    assert resp.status_code == 400


def _template_row(
    template_id="t1",
    visibility="public",
    created_by="u1",
):
    return SimpleNamespace(
        template_id=template_id,
        template_name=None,
        dockerfile_id="opencode",
        container_port=4096,
        repo_url="https://github.com/x/y",
        default_branch="main",
        visibility=visibility,
        image_uri=None,
        task_def_arn=None,
        build_status="ready",
        build_error=None,
        created_by=created_by,
    )


def test_list_templates_filters_private_for_non_owner(app_factory, user, fake_prisma):
    """Non-admin caller passes a where-clause that excludes private templates
    they don't own; assert the where-clause is correct."""
    client = app_factory(user)
    fake_prisma.db.litellm_managedagentsandboxtemplatetable.find_many = AsyncMock(
        return_value=[]
    )
    with patch("litellm.proxy.proxy_server.prisma_client", fake_prisma):
        resp = client.get("/v1/managed_agents/sandbox-templates")
    assert resp.status_code == 200
    where = fake_prisma.db.litellm_managedagentsandboxtemplatetable.find_many.call_args.kwargs[
        "where"
    ]
    assert "OR" in where
    assert {"visibility": "public"} in where["OR"]
    assert {"created_by": "u1"} in where["OR"]


def test_list_templates_admin_sees_all(app_factory, admin, fake_prisma):
    client = app_factory(admin)
    fake_prisma.db.litellm_managedagentsandboxtemplatetable.find_many = AsyncMock(
        return_value=[]
    )
    with patch("litellm.proxy.proxy_server.prisma_client", fake_prisma):
        resp = client.get("/v1/managed_agents/sandbox-templates")
    assert resp.status_code == 200
    where = fake_prisma.db.litellm_managedagentsandboxtemplatetable.find_many.call_args.kwargs[
        "where"
    ]
    assert where == {}


def test_get_private_template_returns_404_for_non_owner(app_factory, user, fake_prisma):
    client = app_factory(user)
    fake_prisma.db.litellm_managedagentsandboxtemplatetable.find_unique = AsyncMock(
        return_value=_template_row(visibility="private", created_by="u-other")
    )
    with patch("litellm.proxy.proxy_server.prisma_client", fake_prisma):
        resp = client.get("/v1/managed_agents/sandbox-templates/t1")
    assert resp.status_code == 404


def test_get_private_template_visible_to_owner(app_factory, user, fake_prisma):
    client = app_factory(user)
    fake_prisma.db.litellm_managedagentsandboxtemplatetable.find_unique = AsyncMock(
        return_value=_template_row(visibility="private", created_by="u1")
    )
    with patch("litellm.proxy.proxy_server.prisma_client", fake_prisma):
        resp = client.get("/v1/managed_agents/sandbox-templates/t1")
    assert resp.status_code == 200


def test_get_private_template_visible_to_admin(app_factory, admin, fake_prisma):
    client = app_factory(admin)
    fake_prisma.db.litellm_managedagentsandboxtemplatetable.find_unique = AsyncMock(
        return_value=_template_row(visibility="private", created_by="u-other")
    )
    with patch("litellm.proxy.proxy_server.prisma_client", fake_prisma):
        resp = client.get("/v1/managed_agents/sandbox-templates/t1")
    assert resp.status_code == 200


def test_template_delete_with_agents_409(app_factory, admin, fake_prisma):
    client = app_factory(admin)
    fake_prisma.db.litellm_managedagentsandboxtemplatetable.find_unique = AsyncMock(
        return_value=SimpleNamespace(
            template_id="t1",
            template_name=None,
            dockerfile_id="opencode",
            container_port=4096,
            repo_url="https://github.com/x/y",
            default_branch="main",
            visibility="public",
            image_uri=None,
            task_def_arn=None,
            build_status="ready",
            build_error=None,
        )
    )
    fake_prisma.db.litellm_managedagenttable.count = AsyncMock(return_value=1)

    with patch("litellm.proxy.proxy_server.prisma_client", fake_prisma):
        resp = client.delete("/v1/managed_agents/sandbox-templates/t1")
    assert resp.status_code == 409


def test_template_delete_uses_litellm_agents_cluster_default(
    app_factory, admin, fake_prisma
):
    """Default cluster name on template delete must match _resolve_cluster
    elsewhere ('litellm-agents'); a mismatch leaves orphan Fargate tasks."""
    client = app_factory(admin)
    fake_prisma.db.litellm_managedagentsandboxtemplatetable.find_unique = AsyncMock(
        return_value=_template_row(template_id="t1")
    )
    fake_prisma.db.litellm_managedagenttable.count = AsyncMock(return_value=0)
    stop_mock = AsyncMock(return_value=0)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", fake_prisma),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints.stop_sessions_for_template",
            new=stop_mock,
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.config_loader.MANAGED_AGENTS_CONFIG",
            SimpleNamespace(aws_region="us-west-2", aws=SimpleNamespace(cluster=None)),
        ),
        patch("boto3.client", return_value=MagicMock()),
    ):
        resp = client.delete("/v1/managed_agents/sandbox-templates/t1")
    assert resp.status_code == 200
    _, kwargs = stop_mock.call_args
    assert kwargs["cluster"] == "litellm-agents"


def test_template_create_validate_repo_branch_runs_off_thread(
    app_factory, admin, fake_prisma
):
    """validate_repo_branch must be wrapped in asyncio.to_thread to avoid
    blocking the event loop on the 15s subprocess."""
    import asyncio as _asyncio

    client = app_factory(admin)
    fake_prisma.db.litellm_managedagentsandboxtemplatetable.create = AsyncMock(
        return_value=_template_row(template_id="t1", visibility="public")
    )
    fake_prisma.db.litellm_managedagentsandboxtemplatetable.update = AsyncMock(
        return_value=_template_row(template_id="t1", visibility="public")
    )
    provisioned = SimpleNamespace(
        image_uri="img:abc",
        task_def_arn="arn:td",
        image_hash="abc",
    )

    captured: dict = {"called_in_thread": None}

    def fake_validate(*args, **kwargs):
        try:
            running_loop = _asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        captured["called_in_thread"] = running_loop is None

    with (
        patch("litellm.proxy.proxy_server.prisma_client", fake_prisma),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints.get_dockerfile",
            return_value=SimpleNamespace(
                dockerfile_id="opencode",
                container_port=4096,
                path="/x",
                context_dir="/x",
                build_platform="linux/amd64",
            ),
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints.validate_repo_branch",
            side_effect=fake_validate,
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints.provision_template",
            AsyncMock(return_value=provisioned),
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.config_loader.MANAGED_AGENTS_CONFIG",
            SimpleNamespace(aws_region="us-west-2", aws=SimpleNamespace(cluster=None)),
        ),
        patch("boto3.client", return_value=MagicMock()),
    ):
        resp = client.post("/v1/managed_agents/sandbox-templates", json=_public_body())
    assert resp.status_code == 200, resp.text
    # validate_repo_branch was invoked from a worker thread (no running loop)
    assert captured["called_in_thread"] is True
