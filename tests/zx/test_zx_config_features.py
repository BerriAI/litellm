"""
单元测试：zx_config_endpoints.py 的补充功能

涉及 commit：
  - 6c20588a43  修复用户自助申请 key 失败问题（provider_user_add）
  - ad228ed133  修复统一认证登录创建用户失败问题（auth_callback）
  - 456220b552  添加开发辅助配置下发（cli_get_config / cli_get_config_yaml）
  - 6167a6ffee  添加 job 执行用户校验

测试覆盖点：
  1. cli_get_config 端点 - token 无效时返回 401
  2. cli_get_config_yaml 端点 - token 无效时返回 401
  3. auth_callback 中 key_hash 为 None 时跳过旧 key 关联逻辑（不尝试查询数据库）
  4. auth_callback 中 device_id 为 None 时跳过旧 key 关联（即使有 key_hash）
  5. continue_plugin_dev_data - continue_plugin_dev_data_enabled=False 时不处理数据
  6. continue_plugin_dev_data - 开关开启时正确注入 utoken 到事件数据

约束：
  - 所有外部依赖（DB、文件系统、key_management_endpoints）必须 mock
  - 不修改任何源码
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 初始化 mock，避免导入时触发真实外部依赖
# ---------------------------------------------------------------------------

_proxy_server_mock = MagicMock()
_proxy_server_mock.general_settings = {}
_proxy_server_mock.master_key = None
# prisma_client 在部分端点中被使用（cli_get_config_yaml 等）
_proxy_server_mock.prisma_client = MagicMock()

sys.modules.setdefault("litellm.proxy.proxy_server", _proxy_server_mock)


from litellm.proxy.zx import zx_config_endpoints  # noqa: E402
from litellm.proxy.zx import zx_development_data  # noqa: E402
from litellm.proxy.zx.token_util import set_store, token_stores  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures：每个测试前清空 token_stores，并重置 mock
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_token_stores():
    """每个测试前清空 token_stores，防止测试间互相干扰。"""
    token_stores.clear()
    yield
    token_stores.clear()


@pytest.fixture(autouse=True)
def reset_proxy_mock():
    """每个测试前重置 proxy_server mock 的 prisma_client，确保隔离。"""
    _proxy_server_mock.prisma_client = MagicMock()
    yield


# ---------------------------------------------------------------------------
# 辅助函数：构造已登录的 TokenStore
# ---------------------------------------------------------------------------


def _make_logged_in_store(
    token: str,
    user_id: str = "user-001",
    org_email: str = "test@fzzixun.com",
    key_hash: str = None,
    device_id: str = "device-001",
    device_name: str = "My PC",
):
    """创建已完成登录的 TokenStore（login=True, status=success），方便测试复用。"""
    ts = set_store(
        "cli",
        token,
        auth_key=f"auth-{token}",
        timeout=5,
        data={
            "user_info": {
                "userId": user_id,
                "name": "Test User",
                "orgEmail": org_email,
                "deptIdList": ["dept-001"],
            },
            "key_metadata": {
                "device_id": device_id,
                "device_name": device_name,
            },
            "key_hash": key_hash,
        },
    )
    ts.login = True
    ts.status = "success"
    return ts


# ---------------------------------------------------------------------------
# 测试类1：cli_get_config 端点鉴权
# ---------------------------------------------------------------------------


class TestCliGetConfig:
    """测试 cli_get_config 端点的 Token 鉴权逻辑。"""

    @pytest.mark.asyncio
    async def test_invalid_token_returns_401(self):
        """
        测试点1：token 无效（store 中不存在）时返回 HTTP 401。
        """
        from fastapi import HTTPException

        # token_stores 为空，任何 token 都是无效的
        with pytest.raises(HTTPException) as exc_info:
            await zx_config_endpoints.cli_get_config(token="invalid-token-xyz")

        assert exc_info.value.status_code == 401
        assert "Invalid token" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_none_token_returns_401(self):
        """
        token 为 None 时（请求头中没有 token），同样返回 401。
        """
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await zx_config_endpoints.cli_get_config(token=None)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_token_reads_config_file(self):
        """
        token 有效时，端点应读取配置文件并返回内容。
        文件系统操作需要 mock。
        """
        _make_logged_in_store("valid-tok-config")

        fake_config = {"model": "gpt-4", "api_base": "https://api.example.com"}

        with patch("builtins.open", create=True) as mock_open, \
             patch("json.load", return_value=fake_config):
            mock_open.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_open.return_value.__exit__ = MagicMock(return_value=False)

            # 直接 mock open，避免触发真实文件系统
            import io

            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            with patch("litellm.proxy.zx.zx_config_endpoints.json.load", return_value=fake_config):
                result = await zx_config_endpoints.cli_get_config(token="valid-tok-config")

        assert result == fake_config


# ---------------------------------------------------------------------------
# 测试类2：cli_get_config_yaml 端点鉴权
# ---------------------------------------------------------------------------


class TestCliGetConfigYaml:
    """测试 cli_get_config_yaml 端点的 Token 鉴权逻辑。"""

    @pytest.mark.asyncio
    async def test_invalid_token_returns_401(self):
        """
        测试点2：token 无效时返回 HTTP 401。
        """
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await zx_config_endpoints.cli_get_config_yaml(token="not-a-valid-token")

        assert exc_info.value.status_code == 401
        assert "Invalid token" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_none_token_returns_401(self):
        """
        token 为 None 时同样返回 401。
        """
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await zx_config_endpoints.cli_get_config_yaml(token=None)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_token_returns_yaml_with_user_id_substituted(self):
        """
        token 有效时，从 DB 查询 litellm_user_id，并替换 YAML 模板中的 <UTOKEN> 占位符。
        DB 调用需要 mock。

        注意：prisma_client 在 cli_get_config_yaml 函数体内通过
          from litellm.proxy.proxy_server import prisma_client
        延迟导入，因此需要 patch litellm.proxy.proxy_server.prisma_client。
        """
        _make_logged_in_store("valid-tok-yaml")

        # mock prisma_client 数据库查询
        # 注意：find_first 是异步方法，必须用 AsyncMock
        fake_user_info = MagicMock()
        fake_user_info.user_id = "litellm-uid-9999"

        fake_prisma = MagicMock()
        fake_prisma.db.litellm_usertable.find_first = AsyncMock(return_value=fake_user_info)

        yaml_template = "api_key: <UTOKEN>\nmodel: gpt-4\n"

        # cli_get_config_yaml 函数体内通过 "from litellm.proxy.proxy_server import prisma_client"
        # 延迟导入，patch 点是 sys.modules["litellm.proxy.proxy_server"] 里的 prisma_client 属性。
        # 联合测试时两个文件共享同一个 proxy_server mock（sys.modules 全局），
        # 直接设置属性即可，不用 patch.object（避免对象引用差异问题）。
        import sys as _sys
        proxy_mod = _sys.modules["litellm.proxy.proxy_server"]
        original_prisma = getattr(proxy_mod, "prisma_client", None)
        proxy_mod.prisma_client = fake_prisma

        try:
            with patch("builtins.open", create=True) as mock_open:
                mock_file = MagicMock()
                mock_file.read.return_value = yaml_template
                mock_open.return_value.__enter__ = MagicMock(return_value=mock_file)
                mock_open.return_value.__exit__ = MagicMock(return_value=False)

                result = await zx_config_endpoints.cli_get_config_yaml(token="valid-tok-yaml")
        finally:
            # 恢复原始值，避免影响其他测试
            proxy_mod.prisma_client = original_prisma

        # <UTOKEN> 应被替换为实际 user_id
        assert result["data"] == "api_key: litellm-uid-9999\nmodel: gpt-4\n"


# ---------------------------------------------------------------------------
# 测试类3：auth_callback 旧 key 关联逻辑的跳过条件
# ---------------------------------------------------------------------------


class TestAuthCallbackKeyAssociationSkip:
    """
    测试 auth_callback 中关联旧 key 的条件判断：
      - 只有 org_email AND key_hash AND device_id 同时满足，才尝试关联
    """

    def _build_auth_store(
        self,
        auth_key: str,
        key_hash=None,
        device_id=None,
    ):
        """创建等待回调的 TokenStore（登录前状态）。"""
        ts = set_store(
            "cli",
            f"tok-{auth_key}",
            auth_key=auth_key,
            timeout=5,
            data={
                "key_hash": key_hash,
                "key_metadata": {
                    "device_id": device_id,
                    "device_name": "Test Device",
                },
            },
        )
        ts.login = False
        ts.status = "pending"
        return ts

    def _mock_auth(self, user_id="u-001", org_email="test@fzzixun.com"):
        """mock 鉴权服务的 get_access_token 和 get_user_info 方法。"""
        zx_config_endpoints.auth.get_access_token = MagicMock(return_value="at-mock")
        zx_config_endpoints.auth.get_user_info = MagicMock(
            return_value={
                "userId": user_id,
                "orgEmail": org_email,
                "name": "Test",
            }
        )

    @pytest.mark.asyncio
    async def test_key_hash_none_skips_db_query(self):
        """
        测试点3：key_hash 为 None 时不尝试查询数据库，直接重定向。

        条件：if org_email and key_hash and device_id: 中 key_hash=None 导致短路。
        """
        self._build_auth_store("auth-key-no-hash", key_hash=None, device_id="dev-001")
        self._mock_auth()

        from fastapi import Request
        from litellm.proxy.management_endpoints import key_management_endpoints as kme

        # list_keys 不应被调用
        list_keys_mock = AsyncMock()

        with patch.object(kme, "list_keys", list_keys_mock):
            req = Request({"type": "http", "query_string": ""})
            result = await zx_config_endpoints.auth_callback(
                auth_key="auth-key-no-hash", code="code-001", request=req
            )

        # list_keys 不应被调用
        list_keys_mock.assert_not_called()

        # 回调应成功（重定向到 /zx/auth_success）
        assert result.status_code == 302

    @pytest.mark.asyncio
    async def test_device_id_none_skips_db_query_even_with_key_hash(self):
        """
        测试点4：device_id 为 None 时跳过旧 key 关联，即使 key_hash 有值。

        条件：if org_email and key_hash and device_id: 中 device_id=None 导致短路。
        """
        self._build_auth_store(
            "auth-key-no-device",
            key_hash="some-key-hash-value",
            device_id=None,  # device_id 缺失
        )
        self._mock_auth()

        from litellm.proxy.management_endpoints import key_management_endpoints as kme

        list_keys_mock = AsyncMock()

        with patch.object(kme, "list_keys", list_keys_mock):
            from fastapi import Request

            req = Request({"type": "http", "query_string": ""})
            result = await zx_config_endpoints.auth_callback(
                auth_key="auth-key-no-device", code="code-002", request=req
            )

        # list_keys 不应被调用（device_id 为 None，条件不满足）
        list_keys_mock.assert_not_called()
        assert result.status_code == 302

    @pytest.mark.asyncio
    async def test_both_key_hash_and_device_id_present_queries_db(self):
        """
        反向验证：key_hash 和 device_id 都有值时，确实会查询 DB（list_keys 被调用）。
        """
        self._build_auth_store(
            "auth-key-full",
            key_hash="hash-abc123",
            device_id="dev-xyz",
        )
        self._mock_auth(org_email="user@fzzixun.com")

        from litellm.proxy.management_endpoints import key_management_endpoints as kme

        # mock list_keys 返回 0 条记录（不执行 update）
        list_keys_mock = AsyncMock(return_value={"total_count": 0, "keys": []})

        with patch.object(kme, "list_keys", list_keys_mock):
            from fastapi import Request

            req = Request({"type": "http", "query_string": ""})
            result = await zx_config_endpoints.auth_callback(
                auth_key="auth-key-full", code="code-003", request=req
            )

        # list_keys 应该被调用（条件全部满足）
        list_keys_mock.assert_called_once()
        assert result.status_code == 302


# ---------------------------------------------------------------------------
# 测试类4：continue_plugin_dev_data 端点
# ---------------------------------------------------------------------------


class TestContinuePluginDevData:
    """
    测试 continue_plugin_dev_data 端点。

    该端点根据 continue_plugin_dev_data_enabled 开关决定是否处理数据。
    开启时将 utoken 注入到事件 JSON 对象中。
    """

    @pytest.mark.asyncio
    async def test_disabled_returns_success_without_processing(self):
        """
        测试点5：continue_plugin_dev_data_enabled=False 时不处理数据，
        直接返回 {"success": "true"}，不写入文件。
        """
        req = MagicMock()
        req.body = AsyncMock(return_value=b'{"type":"ide","event":"autocomplete"}')

        add_event_mock = MagicMock()

        with patch.object(zx_development_data, "add_continue_plugin_event", add_event_mock), \
             patch.object(zx_config_endpoints, "continue_plugin_dev_data_enabled", False):
            result = await zx_config_endpoints.continue_plugin_dev_data(
                req, utoken="user-token-001"
            )

        # 应直接返回成功，不调用 add_continue_plugin_event
        assert result == {"success": "true"}
        add_event_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_enabled_injects_utoken_into_event(self):
        """
        测试点6：开关开启时，端点正确将 utoken 注入到事件数据的 JSON 对象中。

        源码实现：user_event = f'{event[0]}"litellm_user_id":"{utoken}",{event[1:]}'
        即在 JSON 对象的第一个字符（{）之后插入 litellm_user_id 字段。
        """
        body = b'{"type":"ide","event":"autocomplete","data":"test"}'
        req = MagicMock()
        req.body = AsyncMock(return_value=body)

        captured_events = []

        def capture_event(event_str: str):
            captured_events.append(event_str)

        with patch.object(zx_development_data, "add_continue_plugin_event", capture_event), \
             patch.object(zx_config_endpoints, "continue_plugin_dev_data_enabled", True):
            result = await zx_config_endpoints.continue_plugin_dev_data(
                req, utoken="user-abc-999"
            )

        # 仍返回成功
        assert result == {"success": "true"}

        # add_continue_plugin_event 应被调用一次
        assert len(captured_events) == 1

        event_str = captured_events[0]

        # utoken 应被注入到事件中
        assert "user-abc-999" in event_str
        assert "litellm_user_id" in event_str

        # 原始字段应仍在事件中（注入是在 { 之后插入，不是替换）
        assert '"type"' in event_str
        assert '"event"' in event_str

    @pytest.mark.asyncio
    async def test_enabled_utoken_is_first_field_in_json(self):
        """
        额外测试：开关开启时，litellm_user_id 应作为第一个字段被注入
        （紧跟在 '{' 之后）。
        """
        body = b'{"original_key":"original_value"}'
        req = MagicMock()
        req.body = AsyncMock(return_value=body)

        captured_events = []

        with patch.object(zx_development_data, "add_continue_plugin_event", captured_events.append), \
             patch.object(zx_config_endpoints, "continue_plugin_dev_data_enabled", True):
            await zx_config_endpoints.continue_plugin_dev_data(
                req, utoken="my-user-token"
            )

        assert len(captured_events) == 1
        event_str = captured_events[0]

        # 验证注入格式：{"litellm_user_id":"...", ...原始内容...}
        assert event_str.startswith('{"litellm_user_id":"my-user-token"')

    @pytest.mark.asyncio
    async def test_enabled_none_utoken_injects_none_string(self):
        """
        额外测试：utoken 为 None 时，注入的是字符串 "None"（Python str(None)）。
        此测试覆盖边界情况，确保不会崩溃。
        """
        body = b'{"event":"test"}'
        req = MagicMock()
        req.body = AsyncMock(return_value=body)

        captured_events = []

        with patch.object(zx_development_data, "add_continue_plugin_event", captured_events.append), \
             patch.object(zx_config_endpoints, "continue_plugin_dev_data_enabled", True):
            result = await zx_config_endpoints.continue_plugin_dev_data(
                req, utoken=None  # utoken 为 None
            )

        # 不应抛出异常，正常返回
        assert result == {"success": "true"}
        assert len(captured_events) == 1
        # utoken=None 时，f-string 注入 "None"
        assert '"litellm_user_id":"None"' in captured_events[0]
