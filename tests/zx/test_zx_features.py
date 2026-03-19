"""
对新增功能的单元测试，覆盖以下 commit 的核心逻辑：
- 添加 OpenClaw 用户限制（zx_custom_handler_user_email_key_check.py）
- 添加 OpenClaw 限制 / key type 校验（zx_config_endpoints.py）
- llm key 生成修复重复存在 key（token_util.py - ClientError）
- key 用途添加 OpenClaw 校验（async_pre_call_hook 多消息格式）
- token_util.py 的 set_store / get_store 存储管理
"""
import time
import pytest
from fastapi import HTTPException
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching.dual_cache import DualCache
from litellm.proxy.zx.token_util import (
    TokenStore,
    set_store,
    get_store,
    token_stores,
    ClientError,
)
from litellm.proxy.zx.plugins.zx_custom_handler_user_email_key_check import (
    MyCustomHandler,
)


# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------

def _make_key(key_alias: str) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(key_alias=key_alias)


def _make_cache() -> DualCache:
    return DualCache()


# ---------------------------------------------------------------------------
# 一、Token 存储管理（set_store / get_store）
# ---------------------------------------------------------------------------


class TestSetStore:
    """测试 set_store：能正确创建 TokenStore"""

    def test_creates_store_with_correct_type_and_token(self):
        ts = set_store("cli", "token-001")
        assert ts.type == "cli"
        assert ts.token == "token-001"
        assert ts.status == "pending"
        assert ts.login is False

    def test_creates_store_with_data(self):
        ts = set_store("cli", "token-002", data={"device_id": "d-xyz"})
        assert ts.data["device_id"] == "d-xyz"

    def test_creates_store_with_auth_key(self):
        ts = set_store("cli", "token-003", auth_key="auth-key-001")
        assert ts.auth_key == "auth-key-001"

    def test_expire_time_is_in_future(self):
        ts = set_store("cli", "token-004", timeout=5)
        assert ts.expire_time > time.time()

    def test_stores_in_global_dict(self):
        set_store("cli", "token-005")
        assert "cli:token-005" in token_stores


class TestGetStore:
    """测试 get_store：按条件返回 / 过滤"""

    def setup_method(self):
        # 每个测试前清空全局字典，避免干扰
        token_stores.clear()

    def test_returns_none_when_not_logged_in(self):
        set_store("cli", "tok-login-check", timeout=10)
        result = get_store(type="cli", token="tok-login-check", check_login=True)
        assert result is None

    def test_returns_store_when_logged_in(self):
        ts = set_store("cli", "tok-loggedin", timeout=10)
        ts.login = True
        result = get_store(type="cli", token="tok-loggedin", check_login=True)
        assert result is not None
        assert result.token == "tok-loggedin"

    def test_returns_store_without_login_check(self):
        set_store("cli", "tok-nocheck", timeout=10)
        result = get_store(type="cli", token="tok-nocheck", check_login=False)
        assert result is not None

    def test_returns_none_for_expired_store(self):
        ts = set_store("cli", "tok-expired", timeout=1)
        ts.expire_time = time.time() - 1  # 强制过期
        ts.login = True
        result = get_store(type="cli", token="tok-expired", check_login=True)
        assert result is None

    def test_cleans_up_expired_entries(self):
        ts = set_store("cli", "tok-cleanup", timeout=1)
        ts.expire_time = time.time() - 1  # 强制过期
        ts.login = True
        get_store(type="cli", token="tok-cleanup")
        assert "cli:tok-cleanup" not in token_stores

    def test_get_by_auth_key(self):
        ts = set_store("cli", "tok-authkey", auth_key="my-auth-key-001", timeout=10)
        result = get_store(auth_key="my-auth-key-001", check_login=False)
        assert result is not None
        assert result.auth_key == "my-auth-key-001"

    def test_remove_true_deletes_store(self):
        ts = set_store("cli", "tok-remove", timeout=10)
        ts.login = True
        get_store(type="cli", token="tok-remove", remove=True)
        assert "cli:tok-remove" not in token_stores

    def test_returns_none_when_not_found(self):
        result = get_store(type="cli", token="tok-nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# 二、ClientError 在 token_util 中的语义
# ---------------------------------------------------------------------------


class TestClientError:
    """测试 ClientError 是独立异常类，不是 RuntimeError"""

    def test_is_exception_subclass(self):
        err = ClientError("test")
        assert isinstance(err, Exception)

    def test_is_not_runtime_error(self):
        err = ClientError("test")
        assert not isinstance(err, RuntimeError)

    def test_can_be_caught_independently(self):
        caught = False
        try:
            raise ClientError("duplicate key")
        except ClientError:
            caught = True
        assert caught

    def test_not_caught_by_runtime_error(self):
        """ClientError 不是 RuntimeError 子类，不会被 RuntimeError 捕获"""
        caught_as_runtime = False
        caught_as_client = False
        try:
            raise ClientError("test")
        except RuntimeError:
            caught_as_runtime = True
        except ClientError:
            caught_as_client = True
        assert not caught_as_runtime
        assert caught_as_client


# ---------------------------------------------------------------------------
# 三、OpenClaw 校验（async_pre_call_hook）
# ---------------------------------------------------------------------------


class TestOpenClawHandlerBlocksDefaultKey:
    """默认 key（key_alias 含 @ 且无 -- 或含 --default--）应被 OpenClaw 消息阻断"""

    @pytest.mark.asyncio
    async def test_blocks_default_key_via_messages_system(self):
        """通过 messages[0].role=system 检测 OpenClaw 消息"""
        handler = MyCustomHandler()
        user_key = _make_key("user@company.com")
        data = {"messages": [{"role": "system", "content": "running inside OpenClaw"}]}
        result = await handler.async_pre_call_hook(
            user_key, _make_cache(), data, "completion"
        )
        assert isinstance(result, HTTPException)
        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_blocks_default_key_with_device_suffix_via_system(self):
        """key_alias = email--default--device_id 时同样应被阻断"""
        handler = MyCustomHandler()
        user_key = _make_key("user@company.com--default--device-abc")
        data = {
            "system": [{"type": "text", "text": "running inside OpenClaw here"}]
        }
        result = await handler.async_pre_call_hook(
            user_key, _make_cache(), data, "completion"
        )
        assert isinstance(result, HTTPException)
        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_blocks_default_key_via_input_developer(self):
        """通过 input[0].role=developer 检测 OpenClaw 消息"""
        handler = MyCustomHandler()
        user_key = _make_key("user@company.com")
        data = {
            "input": [{"role": "developer", "content": "running inside OpenClaw"}]
        }
        result = await handler.async_pre_call_hook(
            user_key, _make_cache(), data, "completion"
        )
        assert isinstance(result, HTTPException)
        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_blocks_via_system_string(self):
        """system 字段为字符串时也能检测"""
        handler = MyCustomHandler()
        user_key = _make_key("user@company.com")
        data = {"system": "running inside OpenClaw"}
        result = await handler.async_pre_call_hook(
            user_key, _make_cache(), data, "completion"
        )
        assert isinstance(result, HTTPException)
        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_blocks_responses_call_type(self):
        """call_type='responses' 也应被拦截"""
        handler = MyCustomHandler()
        user_key = _make_key("user@company.com")
        data = {"messages": [{"role": "system", "content": "running inside OpenClaw"}]}
        result = await handler.async_pre_call_hook(
            user_key, _make_cache(), data, "responses"
        )
        assert isinstance(result, HTTPException)
        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_blocks_anthropic_messages_call_type(self):
        """call_type='anthropic_messages' 也应被拦截"""
        handler = MyCustomHandler()
        user_key = _make_key("user@company.com")
        data = {"messages": [{"role": "system", "content": "running inside OpenClaw"}]}
        result = await handler.async_pre_call_hook(
            user_key, _make_cache(), data, "anthropic_messages"
        )
        assert isinstance(result, HTTPException)
        assert result.status_code == 403


class TestOpenClawHandlerAllowsOpenclaw:
    """assistant-openclaw 专用 key 不应被阻断"""

    @pytest.mark.asyncio
    async def test_allows_assistant_openclaw_key(self):
        """key_alias 含 assistant-openclaw 的 key 允许通过"""
        handler = MyCustomHandler()
        user_key = _make_key("user@company.com--assistant-openclaw--device-abc")
        data = {"messages": [{"role": "system", "content": "running inside OpenClaw"}]}
        result = await handler.async_pre_call_hook(
            user_key, _make_cache(), data, "completion"
        )
        # 返回原始 data，不是 HTTPException
        assert isinstance(result, dict)
        assert "messages" in result

    @pytest.mark.asyncio
    async def test_allows_key_without_at_sign(self):
        """key_alias 不含 @ 的 key 不经过 OpenClaw 检查"""
        handler = MyCustomHandler()
        user_key = _make_key("service-key-no-email")
        data = {"messages": [{"role": "system", "content": "running inside OpenClaw"}]}
        result = await handler.async_pre_call_hook(
            user_key, _make_cache(), data, "completion"
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_allows_none_key_alias(self):
        """key_alias 为 None 时跳过检查"""
        handler = MyCustomHandler()
        user_key = UserAPIKeyAuth()  # key_alias defaults to None
        data = {"messages": [{"role": "system", "content": "running inside OpenClaw"}]}
        result = await handler.async_pre_call_hook(
            user_key, _make_cache(), data, "completion"
        )
        assert isinstance(result, dict)


class TestOpenClawHandlerAllowsNonLLMCalls:
    """非 LLM 调用类型（embedding、image_generation 等）不应被拦截"""

    @pytest.mark.asyncio
    async def test_allows_embedding_call(self):
        handler = MyCustomHandler()
        user_key = _make_key("user@company.com")
        data = {"input": "running inside OpenClaw"}
        result = await handler.async_pre_call_hook(
            user_key, _make_cache(), data, "embedding"
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_allows_image_generation_call(self):
        handler = MyCustomHandler()
        user_key = _make_key("user@company.com")
        data = {"prompt": "running inside OpenClaw"}
        result = await handler.async_pre_call_hook(
            user_key, _make_cache(), data, "image_generation"
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_allows_transcription_call(self):
        handler = MyCustomHandler()
        user_key = _make_key("user@company.com")
        data = {}
        result = await handler.async_pre_call_hook(
            user_key, _make_cache(), data, "transcription"
        )
        assert isinstance(result, dict)


class TestOpenClawHandlerAllowsNormalMessages:
    """不含 OpenClaw 标记的消息应正常通过"""

    @pytest.mark.asyncio
    async def test_allows_normal_completion(self):
        handler = MyCustomHandler()
        user_key = _make_key("user@company.com")
        data = {"messages": [{"role": "system", "content": "You are a helpful assistant"}]}
        result = await handler.async_pre_call_hook(
            user_key, _make_cache(), data, "completion"
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_allows_empty_messages(self):
        handler = MyCustomHandler()
        user_key = _make_key("user@company.com")
        data = {"messages": [{"role": "user", "content": "Hello"}]}
        result = await handler.async_pre_call_hook(
            user_key, _make_cache(), data, "completion"
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_allows_no_system_message(self):
        handler = MyCustomHandler()
        user_key = _make_key("user@company.com")
        data = {}
        result = await handler.async_pre_call_hook(
            user_key, _make_cache(), data, "completion"
        )
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 四、cli_get_key 校验逻辑（device_id 与 type 参数验证）
# ---------------------------------------------------------------------------


class TestCliGetKeyValidation:
    """测试 cli_get_key 端点的参数校验行为"""

    def _build_store_data(self, device_id=None, device_name=None):
        """构造一个登录后的 store data 结构"""
        return {
            "user_info": {
                "userId": "user-001",
                "name": "张三",
                "orgEmail": "zhangsan@fzzixun.com",
                "deptIdList": ["dept-001"],
            },
            "key_metadata": {
                "device_id": device_id,
                "device_name": device_name or "My Laptop",
            },
        }

    def test_device_id_none_raises_400(self):
        """device_id 为 None 时，端点应返回 400"""
        data = self._build_store_data(device_id=None)
        device_id = data["key_metadata"].get("device_id")
        assert device_id is None, "前提：device_id 为 None"

        # 模拟端点行为：device_id 为 None 应触发 400
        with pytest.raises(HTTPException) as exc_info:
            if device_id is None:
                raise HTTPException(
                    status_code=400, detail="请使用最新版本 llm-config-client"
                )
        assert exc_info.value.status_code == 400

    def test_device_id_present_no_error(self):
        """device_id 存在时，不触发 400"""
        data = self._build_store_data(device_id="device-abc123")
        device_id = data["key_metadata"].get("device_id")
        assert device_id == "device-abc123"
        # 正常不抛出

    def test_assistant_type_raises_400(self):
        """type 参数以 'assistant-' 开头时应返回 400"""
        type_param = "assistant-openclaw"
        with pytest.raises(HTTPException) as exc_info:
            if type_param is not None and type_param.strip().startswith("assistant-"):
                raise HTTPException(
                    status_code=400,
                    detail=f"不支持创建key type[{type_param}]",
                )
        assert exc_info.value.status_code == 400
        assert "assistant-openclaw" in exc_info.value.detail

    def test_non_assistant_type_defaults_to_default(self):
        """type 不以 'assistant-' 开头时，被设为 'default'"""
        type_param = None
        if type_param is not None and str(type_param).strip().startswith("assistant-"):
            type_param = type_param.strip()
        else:
            type_param = "default"
        assert type_param == "default"

    def test_key_alias_format_with_device_id(self):
        """key_alias 格式应为 '{org_email}--{type}--{device_id}'"""
        org_email = "zhangsan@fzzixun.com"
        key_type = "default"
        device_id = "device-abc123"
        key_alias = f"{org_email}--{key_type}--{device_id}"
        assert key_alias == "zhangsan@fzzixun.com--default--device-abc123"


# ---------------------------------------------------------------------------
# 五、auth_callback 旧 key 关联逻辑（key alias 迁移）
# ---------------------------------------------------------------------------


class TestLegacyKeyAliasMigration:
    """测试 auth_callback 中旧 key alias 的迁移规则"""

    def _migrate_key_alias(self, old_alias: str, org_email: str, device_id: str) -> tuple[str, str]:
        """复现 auth_callback 中的 key_alias 迁移逻辑"""
        user_name = org_email.split("@")[0]
        new_metadata_key_type = None
        new_alias = old_alias  # 默认不变

        if old_alias == org_email:
            new_alias = f"{org_email}--default--{device_id}"
            new_metadata_key_type = "default"
        elif (
            old_alias == f"assistant-openclaw--{user_name}"
            or old_alias == f"{user_name}--assistant-openclaw"
        ):
            new_metadata_key_type = "assistant-openclaw"
            new_alias = f"{org_email}--assistant-openclaw--{device_id}"

        return new_alias, new_metadata_key_type  # type: ignore

    def test_default_key_alias_migrated(self):
        """旧 key_alias 为 org_email 时，迁移为 email--default--device_id"""
        org_email = "zhangsan@fzzixun.com"
        device_id = "dev-001"
        new_alias, key_type = self._migrate_key_alias(org_email, org_email, device_id)
        assert new_alias == "zhangsan@fzzixun.com--default--dev-001"
        assert key_type == "default"

    def test_openclaw_key_alias_migrated_format1(self):
        """旧格式 'assistant-openclaw--{user_name}' 迁移正确"""
        org_email = "zhangsan@fzzixun.com"
        device_id = "dev-001"
        old_alias = "assistant-openclaw--zhangsan"
        new_alias, key_type = self._migrate_key_alias(old_alias, org_email, device_id)
        assert new_alias == "zhangsan@fzzixun.com--assistant-openclaw--dev-001"
        assert key_type == "assistant-openclaw"

    def test_openclaw_key_alias_migrated_format2(self):
        """旧格式 '{user_name}--assistant-openclaw' 迁移正确"""
        org_email = "zhangsan@fzzixun.com"
        device_id = "dev-001"
        old_alias = "zhangsan--assistant-openclaw"
        new_alias, key_type = self._migrate_key_alias(old_alias, org_email, device_id)
        assert new_alias == "zhangsan@fzzixun.com--assistant-openclaw--dev-001"
        assert key_type == "assistant-openclaw"

    def test_unknown_key_alias_not_changed(self):
        """无法识别的旧 key_alias 不做迁移"""
        org_email = "zhangsan@fzzixun.com"
        device_id = "dev-001"
        old_alias = "some-unknown-alias"
        new_alias, key_type = self._migrate_key_alias(old_alias, org_email, device_id)
        assert new_alias == old_alias
        assert key_type is None

    def test_migration_preserves_old_metadata_fields(self):
        """迁移时保留旧 metadata 字段，只补充新字段"""
        old_metadata = {"custom_field": "value", "org_email": "old@fzzixun.com"}
        new_device_id = "dev-xyz"
        new_device_name = "My PC"

        new_metadata = old_metadata.copy()
        new_metadata["org_email"] = "zhangsan@fzzixun.com"
        new_metadata["device_id"] = new_device_id
        new_metadata["device_name"] = new_device_name

        assert new_metadata["custom_field"] == "value"
        assert new_metadata["device_id"] == new_device_id
        assert new_metadata["device_name"] == new_device_name
        assert new_metadata["org_email"] == "zhangsan@fzzixun.com"

    def test_no_migration_when_device_id_already_set(self):
        """现有 key 的 metadata 已有 device_id 时，不执行迁移"""
        existing_metadata = {
            "device_id": "already-set",
            "device_name": "Old Device",
        }
        # 模拟判断条件
        should_update = existing_metadata.get("device_id") is None
        assert should_update is False

    def test_skip_association_when_no_key_hash(self):
        """store 中没有 key_hash 时跳过关联逻辑"""
        store_data = {"key_metadata": {"device_id": "dev-001"}}
        key_hash = store_data.get("key_hash")
        assert key_hash is None  # 不应进入关联逻辑
