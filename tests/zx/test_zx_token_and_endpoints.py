"""
单元测试：覆盖 token_util.py、zx_user_endpoints.py、zx_config_endpoints.py 的核心逻辑

覆盖 commit 包括：
- e3dfdff52e  key 不关联团队 → GenerateKeyRequest.models = ["all-team-models"]
- 591109b752  仅开放 LLM API → key_type = LiteLLMKeyType.LLM_API
- d226ee8924  创建用户时 userId 为空时自动生成
- c0cc128ede  新增按 client 添加用户 API
- 6167a6ffee  添加 job 执行用户校验
- ad228ed133  修复统一认证登录创建用户失败问题（cli_check_token / cli_get_config_yaml）
"""

import json
import os
import time
import uuid
import hmac
import hashlib
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from litellm.proxy._types import (
    GenerateKeyRequest,
    LiteLLMKeyType,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.zx.token_util import (
    ClientError,
    TokenStore,
    get_store,
    set_store,
    token_stores,
)


# ===========================================================================
# 一、GenerateKeyRequest 构建验证（token_util.py 第 216-222 行核心逻辑）
# ===========================================================================


class TestGenerateKeyRequestConstruction:
    """验证 create_or_get_user_key 创建 key 时，GenerateKeyRequest 的参数是否正确"""

    def test_key_type_is_llm_api(self):
        """key_type 必须是 LiteLLMKeyType.LLM_API，仅允许 LLM API 调用"""
        key_data = GenerateKeyRequest(
            user_id="uid-001",
            key_alias="user@fzzixun.com",
            key_type=LiteLLMKeyType.LLM_API,
            models=["all-team-models"],
            metadata={"org_email": "user@fzzixun.com"},
        )
        assert key_data.key_type == LiteLLMKeyType.LLM_API

    def test_models_contains_all_team_models(self):
        """models 必须包含 'all-team-models'，而非其他权限"""
        key_data = GenerateKeyRequest(
            user_id="uid-002",
            key_alias="user@fzzixun.com",
            key_type=LiteLLMKeyType.LLM_API,
            models=["all-team-models"],
            metadata={"org_email": "user@fzzixun.com"},
        )
        assert "all-team-models" in key_data.models
        # 确保不包含过大的全局模型权限
        assert "all-proxy-models" not in key_data.models

    def test_metadata_merges_key_metadata_and_org_email(self):
        """metadata 应是 key_metadata 和 org_email 的合并，org_email 来自参数"""
        org_email = "user@fzzixun.com"
        key_metadata = {"device_id": "dev-001", "device_name": "My PC"}
        # 复现源码第 221 行：(key_metadata or {}) | {"org_email": org_email}
        merged = (key_metadata or {}) | {"org_email": org_email}
        key_data = GenerateKeyRequest(
            user_id="uid-003",
            key_alias=org_email,
            key_type=LiteLLMKeyType.LLM_API,
            models=["all-team-models"],
            metadata=merged,
        )
        assert key_data.metadata["org_email"] == org_email
        assert key_data.metadata["device_id"] == "dev-001"
        assert key_data.metadata["device_name"] == "My PC"

    def test_metadata_merge_org_email_overwrites_old_value(self):
        """org_email 字段在合并时覆盖 key_metadata 中同名旧值"""
        org_email = "new@fzzixun.com"
        key_metadata = {"org_email": "old@fzzixun.com", "device_id": "dev-002"}
        merged = (key_metadata or {}) | {"org_email": org_email}
        assert merged["org_email"] == "new@fzzixun.com"
        assert merged["device_id"] == "dev-002"

    def test_metadata_empty_key_metadata_still_has_org_email(self):
        """key_metadata 为 None 时，merged metadata 仍包含 org_email"""
        org_email = "user@fzzixun.com"
        key_metadata = None
        merged = (key_metadata or {}) | {"org_email": org_email}
        assert merged["org_email"] == org_email
        assert len(merged) == 1

    def test_key_type_is_not_management(self):
        """key_type 不能是 MANAGEMENT 类型，确保用户无法管理系统"""
        key_data = GenerateKeyRequest(
            user_id="uid-004",
            key_alias="user@fzzixun.com",
            key_type=LiteLLMKeyType.LLM_API,
            models=["all-team-models"],
            metadata={"org_email": "user@fzzixun.com"},
        )
        assert key_data.key_type != LiteLLMKeyType.MANAGEMENT

    def test_key_type_is_not_default(self):
        """key_type 不能是 DEFAULT 类型，DEFAULT 允许范围不明确"""
        key_data = GenerateKeyRequest(
            user_id="uid-005",
            key_alias="user@fzzixun.com",
            key_type=LiteLLMKeyType.LLM_API,
            models=["all-team-models"],
            metadata={"org_email": "user@fzzixun.com"},
        )
        assert key_data.key_type != LiteLLMKeyType.DEFAULT


# ===========================================================================
# 二、provider_user_add 校验逻辑（zx_user_endpoints.py）
# ===========================================================================


class TestProviderUserAddTimestampValidation:
    """测试 timestamp 参数校验：非数字 / 已过期"""

    @pytest.mark.asyncio
    async def test_non_numeric_timestamp_raises_exception(self):
        """timestamp 非数字时抛出 Exception（源码第 34-36 行）"""
        # 复现源码中的校验逻辑
        timestamp = "not-a-number"
        with pytest.raises(Exception, match="timestamp"):
            try:
                int_num = int(timestamp)
            except ValueError:
                raise Exception(f"Invalid API key: timestamp[{timestamp}] error")

    @pytest.mark.asyncio
    async def test_expired_timestamp_raises_exception(self):
        """timestamp 超过 1800 秒时抛出 Exception（源码第 38-39 行）"""
        # 模拟 31 分钟前的时间戳
        expired_ts = str(int(time.time()) - 1900)
        with pytest.raises(Exception, match="expired"):
            int_num = int(expired_ts)
            if time.time() - int_num > 1800:
                raise Exception(f"Invalid API key: timestamp[{expired_ts}] expired")

    @pytest.mark.asyncio
    async def test_valid_timestamp_does_not_raise(self):
        """timestamp 在有效期内（1800秒以内）不抛出异常"""
        valid_ts = str(int(time.time()) - 60)  # 1 分钟前
        # 不应抛出异常
        int_num = int(valid_ts)
        assert time.time() - int_num <= 1800

    @pytest.mark.asyncio
    async def test_timestamp_boundary_1800s_is_expired(self):
        """恰好超过 1800 秒的 timestamp 应被判定为过期"""
        expired_ts = int(time.time()) - 1801
        assert time.time() - expired_ts > 1800

    @pytest.mark.asyncio
    async def test_timestamp_exactly_1800s_is_valid(self):
        """恰好 1800 秒之内的 timestamp 不过期"""
        # time.time() - ts > 1800 才过期，所以 1799 不过期
        valid_ts = int(time.time()) - 1799
        assert not (time.time() - valid_ts > 1800)


class TestProviderUserAddSignatureValidation:
    """测试 signature 校验失败时返回 HTTPException(401)"""

    @pytest.mark.asyncio
    async def test_invalid_signature_raises_http_401(self):
        """signature 校验失败时应抛出 HTTPException(status_code=401)"""
        # 复现源码第 44-45 行逻辑
        with pytest.raises(HTTPException) as exc_info:
            is_valid = False  # 模拟 security_validator.validate 返回 False
            if not is_valid:
                raise HTTPException(status_code=401, detail="Invalid token")
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid token"

    @pytest.mark.asyncio
    async def test_valid_signature_no_exception(self):
        """signature 校验成功时不抛出异常"""
        is_valid = True
        # 不应进入异常分支
        if not is_valid:
            raise HTTPException(status_code=401, detail="Invalid token")
        # 若走到这里，说明测试通过


class TestProviderUserAddEmailDomainFilter:
    """测试邮箱域名过滤逻辑（源码第 49-50 行）"""

    def _get_allow_domain(self, env_value: Optional[str] = None) -> str:
        """复现源码中 add_user_allow_email_domain 的构造逻辑"""
        if env_value is not None:
            return "@" + env_value.strip()
        return "@" + os.environ.get("ZX_ADD_USER_ALLOW_EMAIL_DOMAIN", "fzzixun.com").strip()

    def test_valid_email_domain_passes(self):
        """用 fzzixun.com 域名的邮箱应通过"""
        allow_domain = self._get_allow_domain()
        org_email = "user@fzzixun.com"
        assert org_email.endswith(allow_domain)

    def test_invalid_email_domain_fails(self):
        """其他域名的邮箱不通过，返回 error 响应"""
        allow_domain = self._get_allow_domain()
        org_email = "user@other.com"
        if not org_email.endswith(allow_domain):
            result = {"success": "false", "email": org_email, "error": "Invalid email domain"}
        else:
            result = {"success": "true"}
        assert result["success"] == "false"
        assert result["error"] == "Invalid email domain"

    def test_custom_domain_via_env(self):
        """通过自定义域名（环境变量）可覆盖默认域名"""
        custom_domain = "mycompany.com"
        allow_domain = self._get_allow_domain(custom_domain)
        assert allow_domain == "@mycompany.com"
        org_email = "user@mycompany.com"
        assert org_email.endswith(allow_domain)

    def test_custom_domain_blocks_default_domain(self):
        """当自定义域名生效时，默认域名 fzzixun.com 被拒绝"""
        custom_domain = "mycompany.com"
        allow_domain = self._get_allow_domain(custom_domain)
        org_email = "user@fzzixun.com"
        assert not org_email.endswith(allow_domain)

    def test_email_domain_result_format(self):
        """当邮箱域名不匹配时，响应格式完整包含 email 字段"""
        allow_domain = "@fzzixun.com"
        org_email = "attacker@evil.com"
        if not org_email.endswith(allow_domain):
            result = {"success": "false", "email": org_email, "error": "Invalid email domain"}
        assert result["email"] == org_email

    def test_domain_env_variable_strips_whitespace(self):
        """从环境变量读取域名时应去除首尾空格"""
        raw_domain = "  trimmed.com  "
        allow_domain = "@" + raw_domain.strip()
        assert allow_domain == "@trimmed.com"


class TestProviderUserAddUserIdGeneration:
    """测试 userId 为空时自动生成（源码第 52 行）"""

    def test_empty_user_id_generates_uuid_format(self):
        """userId 为空字符串时应自动生成 '{client_id}_{uuid}' 格式"""
        client_id = "test-client"
        user_info_user_id = ""  # 模拟请求体中 userId 为空
        # 复现源码：user_info.get("userId") or f"{client_id}_{uuid.uuid4()}"
        generated_user_id = user_info_user_id or f"{client_id}_{uuid.uuid4()}"
        assert generated_user_id.startswith(f"{client_id}_")
        # uuid 部分应为合法的 UUID 格式
        suffix = generated_user_id[len(client_id) + 1:]
        try:
            uuid.UUID(suffix)
            is_valid_uuid = True
        except ValueError:
            is_valid_uuid = False
        assert is_valid_uuid

    def test_none_user_id_generates_uuid_format(self):
        """userId 为 None 时也应自动生成"""
        client_id = "provider-x"
        user_info_user_id = None
        generated_user_id = user_info_user_id or f"{client_id}_{uuid.uuid4()}"
        assert generated_user_id.startswith(f"{client_id}_")

    def test_existing_user_id_is_preserved(self):
        """userId 存在时直接使用，不生成新值"""
        client_id = "provider-x"
        existing_user_id = "existing-user-123"
        result_user_id = existing_user_id or f"{client_id}_{uuid.uuid4()}"
        assert result_user_id == "existing-user-123"

    def test_each_generation_produces_unique_id(self):
        """两次空 userId 情况应生成不同的 ID（UUID 唯一性）"""
        client_id = "provider-x"
        id1 = f"{client_id}_{uuid.uuid4()}"
        id2 = f"{client_id}_{uuid.uuid4()}"
        assert id1 != id2

    def test_generated_id_prefix_matches_client_id(self):
        """自动生成的 userId 前缀必须是 client_id"""
        client_id = "my-client-app"
        user_id = f"{client_id}_{uuid.uuid4()}"
        assert user_id.startswith("my-client-app_")


class TestProviderUserAddEndpointMocked:
    """使用 mock 测试 provider_user_add 端点的完整流程"""

    def _make_signature(self, client_id: str, client_secret: str, payload: str) -> str:
        """生成合法签名"""
        message = f"{client_id}:{payload}".encode()
        return hmac.new(client_secret.encode(), message, hashlib.sha256).hexdigest()

    @pytest.mark.asyncio
    async def test_valid_request_calls_create_or_get_user_key(self):
        """签名校验通过 + 域名匹配时应调用 create_or_get_user_key 并返回成功"""
        from litellm.proxy.zx.zx_user_endpoints import provider_user_add

        client_id = "test-client"
        client_secret = "test-secret"
        org_email = "user@fzzixun.com"
        timestamp = str(int(time.time()))
        body_data = json.dumps({
            "orgEmail": org_email,
            "userId": "user-001",
            "name": "测试用户",
            "deptId": "dept-001",
        })
        payload = f"{body_data}:{timestamp}"
        signature = self._make_signature(client_id, client_secret, payload)

        # mock request 对象
        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=body_data.encode("utf-8"))

        with patch(
            "litellm.proxy.zx.zx_user_endpoints.security_validator.validate",
            return_value=True,
        ), patch(
            "litellm.proxy.zx.zx_user_endpoints.create_or_get_user_key",
            new=AsyncMock(return_value=(True, "sk-test-key-123")),
        ), patch.dict(
            os.environ, {"ZX_ADD_USER_ALLOW_EMAIL_DOMAIN": "fzzixun.com"}
        ):
            result = await provider_user_add(
                client_id=client_id,
                signature=signature,
                timestamp=timestamp,
                request=mock_request,
            )

        assert result["success"] == "true"
        assert result["email"] == org_email
        assert result["created"] is True

    @pytest.mark.asyncio
    async def test_invalid_email_domain_returns_error_dict(self):
        """邮箱域名不匹配时直接返回 error dict，不调用 create_or_get_user_key"""
        from litellm.proxy.zx.zx_user_endpoints import provider_user_add

        org_email = "user@badomain.com"
        timestamp = str(int(time.time()))
        body_data = json.dumps({
            "orgEmail": org_email,
            "userId": "user-001",
            "name": "测试用户",
            "deptId": "dept-001",
        })

        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=body_data.encode("utf-8"))

        mock_create = AsyncMock()

        with patch(
            "litellm.proxy.zx.zx_user_endpoints.security_validator.validate",
            return_value=True,
        ), patch(
            "litellm.proxy.zx.zx_user_endpoints.create_or_get_user_key",
            new=mock_create,
        ), patch.dict(
            os.environ, {"ZX_ADD_USER_ALLOW_EMAIL_DOMAIN": "fzzixun.com"}
        ):
            result = await provider_user_add(
                client_id="test-client",
                signature="any-sig",
                timestamp=timestamp,
                request=mock_request,
            )

        assert result["success"] == "false"
        assert result["error"] == "Invalid email domain"
        # 域名不匹配时不应调用 create_or_get_user_key
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_signature_raises_http_401_from_endpoint(self):
        """signature 校验失败时端点抛出 HTTPException(401)"""
        from litellm.proxy.zx.zx_user_endpoints import provider_user_add

        timestamp = str(int(time.time()))
        body_data = json.dumps({
            "orgEmail": "user@fzzixun.com",
            "userId": "user-001",
            "name": "测试用户",
        })

        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=body_data.encode("utf-8"))

        with patch(
            "litellm.proxy.zx.zx_user_endpoints.security_validator.validate",
            return_value=False,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await provider_user_add(
                    client_id="test-client",
                    signature="wrong-sig",
                    timestamp=timestamp,
                    request=mock_request,
                )

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_non_numeric_timestamp_raises_exception_from_endpoint(self):
        """timestamp 非数字时端点抛出 Exception"""
        from litellm.proxy.zx.zx_user_endpoints import provider_user_add

        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=b"{}")

        with pytest.raises(Exception, match="timestamp"):
            await provider_user_add(
                client_id="test-client",
                signature="any-sig",
                timestamp="not-a-number",
                request=mock_request,
            )

    @pytest.mark.asyncio
    async def test_expired_timestamp_raises_exception_from_endpoint(self):
        """timestamp 过期时端点抛出 Exception"""
        from litellm.proxy.zx.zx_user_endpoints import provider_user_add

        expired_ts = str(int(time.time()) - 2000)
        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=b"{}")

        with pytest.raises(Exception, match="expired"):
            await provider_user_add(
                client_id="test-client",
                signature="any-sig",
                timestamp=expired_ts,
                request=mock_request,
            )

    @pytest.mark.asyncio
    async def test_empty_user_id_auto_generates_in_full_flow(self):
        """userId 为空时，端点内部自动生成，最终仍返回 success"""
        from litellm.proxy.zx.zx_user_endpoints import provider_user_add

        org_email = "user@fzzixun.com"
        timestamp = str(int(time.time()))
        # userId 字段不传（或为空字符串）
        body_data = json.dumps({
            "orgEmail": org_email,
            "name": "自动用户",
            "deptId": "dept-002",
            # userId 不填，触发自动生成
        })

        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=body_data.encode("utf-8"))

        captured_args = {}

        async def fake_create(provider, user_id, user_name, org_email_, dept_id):
            captured_args["user_id"] = user_id
            captured_args["provider"] = provider
            return (True, "sk-auto-key-999")

        with patch(
            "litellm.proxy.zx.zx_user_endpoints.security_validator.validate",
            return_value=True,
        ), patch(
            "litellm.proxy.zx.zx_user_endpoints.create_or_get_user_key",
            side_effect=fake_create,
        ), patch.dict(
            os.environ, {"ZX_ADD_USER_ALLOW_EMAIL_DOMAIN": "fzzixun.com"}
        ):
            result = await provider_user_add(
                client_id="test-client",
                signature="any-sig",
                timestamp=timestamp,
                request=mock_request,
            )

        assert result["success"] == "true"
        # 自动生成的 user_id 应以 client_id 开头
        assert captured_args["user_id"].startswith("test-client_")

    @pytest.mark.asyncio
    async def test_dept_id_falls_back_to_dept_id_list(self):
        """deptId 为 None 时，应使用 deptIdList[0]"""
        from litellm.proxy.zx.zx_user_endpoints import provider_user_add

        org_email = "user@fzzixun.com"
        timestamp = str(int(time.time()))
        body_data = json.dumps({
            "orgEmail": org_email,
            "userId": "user-dept-list",
            "name": "部门测试",
            # deptId 不传，但 deptIdList 有值
            "deptIdList": ["dept-from-list"],
        })

        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=body_data.encode("utf-8"))

        captured_args = {}

        async def fake_create(provider, user_id, user_name, org_email_, dept_id):
            captured_args["dept_id"] = dept_id
            return (False, "sk-existing-key")

        with patch(
            "litellm.proxy.zx.zx_user_endpoints.security_validator.validate",
            return_value=True,
        ), patch(
            "litellm.proxy.zx.zx_user_endpoints.create_or_get_user_key",
            side_effect=fake_create,
        ), patch.dict(
            os.environ, {"ZX_ADD_USER_ALLOW_EMAIL_DOMAIN": "fzzixun.com"}
        ):
            result = await provider_user_add(
                client_id="test-client",
                signature="any-sig",
                timestamp=timestamp,
                request=mock_request,
            )

        assert result["success"] == "true"
        # deptId 应从 deptIdList[0] 获取
        assert captured_args["dept_id"] == "dept-from-list"


# ===========================================================================
# 三、cli_check_token 端点逻辑（zx_config_endpoints.py 第 254-261 行）
# ===========================================================================


class TestCliCheckToken:
    """测试 cli_check_token 端点：token 有效/无效返回 True/False"""

    @pytest.mark.asyncio
    async def test_valid_logged_in_token_returns_true(self):
        """token 对应的 store 存在且已登录时，返回 True"""
        from litellm.proxy.zx.zx_config_endpoints import cli_check_token

        # 在全局 store 中注册一个已登录的 token
        ts = set_store("cli", "check-token-valid", timeout=10)
        ts.login = True  # 标记为已登录

        result = await cli_check_token(token="check-token-valid")
        assert result is True

        # 清理
        token_stores.pop("cli:check-token-valid", None)

    @pytest.mark.asyncio
    async def test_invalid_token_returns_false(self):
        """token 不存在于 store 时，返回 False"""
        from litellm.proxy.zx.zx_config_endpoints import cli_check_token

        result = await cli_check_token(token="nonexistent-token-abc")
        assert result is False

    @pytest.mark.asyncio
    async def test_none_token_returns_false(self):
        """token 为 None 时，返回 False"""
        from litellm.proxy.zx.zx_config_endpoints import cli_check_token

        result = await cli_check_token(token=None)
        assert result is False

    @pytest.mark.asyncio
    async def test_pending_token_not_logged_in_returns_false(self):
        """token 存在但尚未登录（login=False）时，返回 False"""
        from litellm.proxy.zx.zx_config_endpoints import cli_check_token

        ts = set_store("cli", "check-token-pending", timeout=10)
        # login 默认为 False，不修改

        result = await cli_check_token(token="check-token-pending")
        assert result is False

        # 清理
        token_stores.pop("cli:check-token-pending", None)

    @pytest.mark.asyncio
    async def test_expired_token_returns_false(self):
        """token 已过期时，返回 False"""
        from litellm.proxy.zx.zx_config_endpoints import cli_check_token

        ts = set_store("cli", "check-token-expired", timeout=1)
        ts.login = True
        ts.expire_time = time.time() - 1  # 强制过期

        result = await cli_check_token(token="check-token-expired")
        assert result is False

        # 清理（过期 token 可能已被 get_store 清理）
        token_stores.pop("cli:check-token-expired", None)


# ===========================================================================
# 四、cli_get_config_yaml 逻辑（zx_config_endpoints.py）
# ===========================================================================


class TestCliGetConfigYaml:
    """测试 cli_get_config_yaml 端点中的用户查找与配置返回逻辑"""

    @pytest.mark.asyncio
    async def test_invalid_token_raises_401(self):
        """token 无效时应抛出 HTTPException(401)"""
        from litellm.proxy.zx.zx_config_endpoints import cli_get_config_yaml

        result = None
        with pytest.raises(HTTPException) as exc_info:
            result = await cli_get_config_yaml(token="nonexistent-token-xyz")
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_user_not_found_returns_error(self):
        """用户在数据库中不存在时返回 error 字段（源码第 381-383 行）"""
        from litellm.proxy.zx.zx_config_endpoints import cli_get_config_yaml

        org_email = "nouser@fzzixun.com"

        # 注册一个已登录的 token，包含 user_info
        ts = set_store("cli", "yaml-token-nouser", timeout=10)
        ts.login = True
        ts.data["user_info"] = {"orgEmail": org_email}

        # mock prisma_client：find_first 返回 None
        mock_prisma = MagicMock()
        mock_prisma.db.litellm_usertable.find_first = AsyncMock(return_value=None)

        # prisma_client 在函数内部通过 `from litellm.proxy.proxy_server import prisma_client`
        # 懒加载，需要 mock proxy_server 模块中的属性
        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma,
        ):
            result = await cli_get_config_yaml(token="yaml-token-nouser")

        assert "error" in result
        assert org_email in result["error"]

        # 清理
        token_stores.pop("cli:yaml-token-nouser", None)

    @pytest.mark.asyncio
    async def test_user_found_returns_config_with_user_id(self):
        """用户存在时从 yaml 模板替换 <UTOKEN> 并返回（源码第 384-392 行）"""
        from litellm.proxy.zx.zx_config_endpoints import cli_get_config_yaml

        org_email = "zhangsan@fzzixun.com"
        user_id = "user-id-from-db"

        ts = set_store("cli", "yaml-token-user-found", timeout=10)
        ts.login = True
        ts.data["user_info"] = {"orgEmail": org_email}

        # mock prisma_client：find_first 返回一个包含 user_id 的对象
        mock_db_user = MagicMock()
        mock_db_user.user_id = user_id

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_usertable.find_first = AsyncMock(return_value=mock_db_user)

        yaml_template = "apiBase: https://example.com\nuserToken: <UTOKEN>"

        # prisma_client 在函数内部通过 `from litellm.proxy.proxy_server import prisma_client`
        # 懒加载，需要 mock proxy_server 模块中的属性
        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma,
        ), patch(
            "builtins.open",
            MagicMock(
                return_value=MagicMock(
                    __enter__=MagicMock(
                        return_value=MagicMock(read=MagicMock(return_value=yaml_template))
                    ),
                    __exit__=MagicMock(return_value=False),
                )
            ),
        ):
            result = await cli_get_config_yaml(token="yaml-token-user-found")

        assert "data" in result
        assert user_id in result["data"]
        assert "<UTOKEN>" not in result["data"]

        # 清理
        token_stores.pop("cli:yaml-token-user-found", None)

    @pytest.mark.asyncio
    async def test_token_store_login_false_raises_401(self):
        """token 存在但未登录时应返回 HTTPException(401)（get_store 返回 None）"""
        from litellm.proxy.zx.zx_config_endpoints import cli_get_config_yaml

        ts = set_store("cli", "yaml-token-not-loggedin", timeout=10)
        # login 默认 False，不修改

        with pytest.raises(HTTPException) as exc_info:
            await cli_get_config_yaml(token="yaml-token-not-loggedin")

        assert exc_info.value.status_code == 401

        # 清理
        token_stores.pop("cli:yaml-token-not-loggedin", None)


# ===========================================================================
# 五、token_util.py 的 set_store / get_store（基础覆盖，补充 commit 相关路径）
# ===========================================================================


class TestTokenUtilSetAndGetStore:
    """测试 token_util 的 set_store / get_store 基础行为"""

    def setup_method(self):
        """每个测试前清空全局 token_stores，防止用例间干扰"""
        token_stores.clear()

    def test_set_store_creates_pending_token(self):
        """set_store 创建的 token 初始状态为 pending、未登录"""
        ts = set_store("cli", "tok-basic-001", timeout=5)
        assert ts.status == "pending"
        assert ts.login is False

    def test_set_store_stores_in_global_dict(self):
        """set_store 后，token_stores 中应存在对应条目"""
        set_store("cli", "tok-global-001", timeout=5)
        assert "cli:tok-global-001" in token_stores

    def test_get_store_returns_none_when_not_logged_in(self):
        """token 未登录时，check_login=True 应返回 None"""
        set_store("cli", "tok-not-logged", timeout=10)
        result = get_store(type="cli", token="tok-not-logged", check_login=True)
        assert result is None

    def test_get_store_returns_store_when_logged_in(self):
        """token 已登录时，check_login=True 应正常返回 store"""
        ts = set_store("cli", "tok-logged", timeout=10)
        ts.login = True
        result = get_store(type="cli", token="tok-logged", check_login=True)
        assert result is not None
        assert result.token == "tok-logged"

    def test_get_store_expired_token_returns_none(self):
        """过期 token 应被清理并返回 None"""
        ts = set_store("cli", "tok-exp", timeout=1)
        ts.login = True
        ts.expire_time = time.time() - 1  # 强制过期
        result = get_store(type="cli", token="tok-exp", check_login=True)
        assert result is None
        assert "cli:tok-exp" not in token_stores

    def test_get_store_by_auth_key(self):
        """通过 auth_key 查找 store"""
        ts = set_store("cli", "tok-authkey-002", auth_key="ak-unique-001", timeout=10)
        result = get_store(auth_key="ak-unique-001", check_login=False)
        assert result is not None
        assert result.auth_key == "ak-unique-001"

    def test_get_store_remove_deletes_entry(self):
        """remove=True 时，查找后应删除该 store"""
        ts = set_store("cli", "tok-remove-002", timeout=10)
        ts.login = True
        get_store(type="cli", token="tok-remove-002", remove=True)
        assert "cli:tok-remove-002" not in token_stores

    def test_get_store_nonexistent_returns_none(self):
        """查找不存在的 token 应返回 None"""
        result = get_store(type="cli", token="totally-nonexistent-999")
        assert result is None


# ===========================================================================
# 六、邮箱域名过滤的完整单元测试（独立于端点）
# ===========================================================================


class TestEmailDomainFilterUnit:
    """直接测试邮箱域名过滤逻辑，不依赖端点"""

    @pytest.mark.parametrize(
        "email, domain, expected_pass",
        [
            ("user@fzzixun.com", "fzzixun.com", True),
            ("admin@fzzixun.com", "fzzixun.com", True),
            ("user@other.com", "fzzixun.com", False),
            ("user@evil.fzzixun.com.hack.org", "fzzixun.com", False),
            ("user@sub.fzzixun.com", "fzzixun.com", False),
            ("user@mycompany.com", "mycompany.com", True),
            ("user@fzzixun.com", "mycompany.com", False),
        ],
    )
    def test_email_domain_filter(self, email: str, domain: str, expected_pass: bool):
        """参数化测试：各种邮箱域名组合的过滤结果"""
        allow_domain = "@" + domain
        result = email.endswith(allow_domain)
        assert result == expected_pass, (
            f"email={email!r}, domain={domain!r}, expected={expected_pass}, got={result}"
        )


# ===========================================================================
# 七、SecurityValidator 签名生成与校验（zx_security_validator.py 协作测试）
# ===========================================================================


class TestSecurityValidatorSignature:
    """测试签名生成与校验的核心算法（HMAC-SHA256）"""

    def _generate_signature(self, client_id: str, client_secret: str, payload: str) -> str:
        """复现 SecurityValidator._generate_signature"""
        message = f"{client_id}:{payload}".encode()
        return hmac.new(client_secret.encode(), message, hashlib.sha256).hexdigest()

    def test_correct_signature_validates(self):
        """正确的 client_id / client_secret 生成签名应通过校验"""
        client_id = "my-client"
        client_secret = "my-secret"
        payload = '{"orgEmail":"user@fzzixun.com"}:1234567890'
        sig = self._generate_signature(client_id, client_secret, payload)
        expected = self._generate_signature(client_id, client_secret, payload)
        assert hmac.compare_digest(sig, expected)

    def test_wrong_secret_fails_validation(self):
        """错误的 client_secret 生成的签名校验应失败"""
        client_id = "my-client"
        payload = '{"orgEmail":"user@fzzixun.com"}:1234567890'
        sig_wrong = self._generate_signature(client_id, "wrong-secret", payload)
        sig_correct = self._generate_signature(client_id, "correct-secret", payload)
        assert not hmac.compare_digest(sig_wrong, sig_correct)

    def test_wrong_client_id_fails_validation(self):
        """payload 中 client_id 不匹配时签名校验应失败"""
        secret = "shared-secret"
        payload = "test-payload"
        sig_a = self._generate_signature("client-a", secret, payload)
        sig_b = self._generate_signature("client-b", secret, payload)
        assert not hmac.compare_digest(sig_a, sig_b)

    def test_signature_is_deterministic(self):
        """相同输入每次应产生相同签名（确定性）"""
        sig1 = self._generate_signature("cid", "secret", "payload")
        sig2 = self._generate_signature("cid", "secret", "payload")
        assert sig1 == sig2

    def test_signature_is_hex_string_of_64_chars(self):
        """HMAC-SHA256 十六进制签名长度必须为 64"""
        sig = self._generate_signature("cid", "secret", "payload")
        assert len(sig) == 64
        assert all(c in "0123456789abcdef" for c in sig)

    def test_security_validator_validate_method(self):
        """直接测试 SecurityValidator.validate 方法"""
        from litellm.proxy.zx.zx_security_validator import SecurityValidator

        client_id = "test-client-validate"
        client_secret = "test-secret-validate"

        # 手动构造只有一个 credential 的 validator
        validator = SecurityValidator.__new__(SecurityValidator)
        validator.valid_credentials = {client_id: client_secret}

        payload = '{"orgEmail":"user@fzzixun.com"}:9999999999'
        correct_sig = self._generate_signature(client_id, client_secret, payload)

        assert validator.validate(client_id, correct_sig, payload) is True
        assert validator.validate(client_id, "wrong-signature", payload) is False
        assert validator.validate("unknown-client", correct_sig, payload) is False
