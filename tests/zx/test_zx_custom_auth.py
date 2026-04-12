"""
zx_custom_auth_app（user_api_key_auth）单元测试

覆盖：
1. api_key 不以 `sk-zx-u-` 开头 → 直接返回原始 api_key
2. 缺少 `zx-client-id` header → 抛出 ProxyException(401)
3. 缺少 `zx-client-app-id` 和 `zx-user-email` → 抛出 ProxyException(401)
4. 缺少 `zx-timestamp` → 抛出 ProxyException(401)
5. timestamp 不是数字 → 抛出 ProxyException(401)
6. timestamp 已过期（超过 ttl=86400秒）→ 抛出 ProxyException(401)
7. timestamp 在有效期内但 signature 错误 → 抛出 ProxyException(401)

注：mock 掉 security_validator、prisma_client、get_key_object 等外部依赖
"""

import os
import sys
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def _make_request(headers: dict) -> MagicMock:
    """辅助函数：创建带指定 headers 的模拟 Request 对象"""
    mock_request = MagicMock()
    mock_request.headers = headers
    return mock_request


@pytest.fixture(autouse=True)
def mock_module_dependencies():
    """
    在每个测试前 mock 掉模块级别的外部依赖，防止实际导入和调用：
    - proxy_server 模块（proxy_logging_obj、user_api_key_cache）
    - get_key_object（认证检查函数）
    - security_validator（签名校验器）
    - DualCache（防止缓存副作用）
    """
    # 构造假的 proxy_server 模块对象
    mock_proxy_server = MagicMock()
    mock_proxy_server.proxy_logging_obj = MagicMock()
    mock_proxy_server.user_api_key_cache = MagicMock()
    mock_proxy_server.prisma_client = None

    with patch.dict(
        "sys.modules",
        {
            "litellm.proxy.proxy_server": mock_proxy_server,
        },
    ):
        yield


@pytest.fixture()
def auth_module():
    """
    每个测试都重新导入 zx_custom_auth_app 模块，确保 mock 生效。
    使用 importlib 强制重新加载以避免模块缓存问题。
    """
    import importlib

    # 若已存在旧模块，先移除
    module_name = "litellm.proxy.zx.plugins.zx_custom_auth_app"
    if module_name in sys.modules:
        del sys.modules[module_name]

    module = importlib.import_module(module_name)
    return module


class TestUserApiKeyAuth:
    """测试 user_api_key_auth 异步函数"""

    @pytest.mark.asyncio
    async def test_非zx用户key直接返回原始api_key(self, auth_module):
        """api_key 不以 sk-zx-u- 开头时，应直接返回原始 api_key，不做任何校验"""
        regular_key = "sk-regular-key-12345"
        mock_request = _make_request({})

        result = await auth_module.user_api_key_auth(mock_request, regular_key)

        # 应原样返回
        assert result == regular_key

    @pytest.mark.asyncio
    async def test_缺少zx_client_id抛出401(self, auth_module):
        """缺少 zx-client-id header 时应抛出 ProxyException，code=401"""
        from litellm.proxy._types import ProxyException

        api_key = "sk-zx-u-some_signature"
        # 故意不提供 zx-client-id
        mock_request = _make_request(
            {
                "zx-client-app-id": "app_123",
                "zx-timestamp": str(int(time.time())),
            }
        )

        # mock 掉缓存，使其返回 None（触发 header 验证逻辑）
        auth_module.user_key_hashed_token_cache.async_get_cache = AsyncMock(return_value=None)

        with pytest.raises(ProxyException) as exc_info:
            await auth_module.user_api_key_auth(mock_request, api_key)

        assert exc_info.value.code == "401"
        assert "zx-client-id" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_缺少client_app_id和user_email抛出401(self, auth_module):
        """同时缺少 zx-client-app-id 和 zx-user-email 时应抛出 ProxyException，code=401"""
        from litellm.proxy._types import ProxyException

        api_key = "sk-zx-u-some_signature"
        # 提供了 client-id 但没有 app-id 和 email
        mock_request = _make_request(
            {
                "zx-client-id": "client_xyz",
                "zx-timestamp": str(int(time.time())),
            }
        )

        auth_module.user_key_hashed_token_cache.async_get_cache = AsyncMock(return_value=None)

        with pytest.raises(ProxyException) as exc_info:
            await auth_module.user_api_key_auth(mock_request, api_key)

        assert exc_info.value.code == "401"
        # 错误消息应提及 app-id 或 email
        msg = exc_info.value.message
        assert "zx-client-app-id" in msg or "zx_user_email" in msg

    @pytest.mark.asyncio
    async def test_缺少timestamp抛出401(self, auth_module):
        """缺少 zx-timestamp header 时应抛出 ProxyException，code=401"""
        from litellm.proxy._types import ProxyException

        api_key = "sk-zx-u-some_signature"
        # 提供了 client-id 和 app-id，但没有 timestamp
        mock_request = _make_request(
            {
                "zx-client-id": "client_xyz",
                "zx-client-app-id": "app_123",
            }
        )

        auth_module.user_key_hashed_token_cache.async_get_cache = AsyncMock(return_value=None)

        with pytest.raises(ProxyException) as exc_info:
            await auth_module.user_api_key_auth(mock_request, api_key)

        assert exc_info.value.code == "401"
        assert "zx-timestamp" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_timestamp不是数字抛出401(self, auth_module):
        """timestamp 无法转为整数时应抛出 ProxyException，code=401"""
        from litellm.proxy._types import ProxyException

        api_key = "sk-zx-u-some_signature"
        mock_request = _make_request(
            {
                "zx-client-id": "client_xyz",
                "zx-client-app-id": "app_123",
                "zx-timestamp": "not_a_number",  # 非法 timestamp
            }
        )

        auth_module.user_key_hashed_token_cache.async_get_cache = AsyncMock(return_value=None)

        with pytest.raises(ProxyException) as exc_info:
            await auth_module.user_api_key_auth(mock_request, api_key)

        assert exc_info.value.code == "401"
        assert "timestamp" in exc_info.value.message.lower()

    @pytest.mark.asyncio
    async def test_timestamp已过期抛出401(self, auth_module):
        """timestamp 超过 ttl（86400秒=24小时）时应抛出 ProxyException，code=401"""
        from litellm.proxy._types import ProxyException

        api_key = "sk-zx-u-some_signature"
        # 使用 25 小时前的时间戳，肯定过期
        expired_timestamp = int(time.time()) - (25 * 60 * 60)
        mock_request = _make_request(
            {
                "zx-client-id": "client_xyz",
                "zx-client-app-id": "app_123",
                "zx-timestamp": str(expired_timestamp),
            }
        )

        auth_module.user_key_hashed_token_cache.async_get_cache = AsyncMock(return_value=None)

        with pytest.raises(ProxyException) as exc_info:
            await auth_module.user_api_key_auth(mock_request, api_key)

        assert exc_info.value.code == "401"
        assert "expired" in exc_info.value.message.lower() or "timestamp" in exc_info.value.message.lower()

    @pytest.mark.asyncio
    async def test_有效timestamp但签名错误抛出401(self, auth_module):
        """timestamp 在有效期内但签名验证失败时应抛出 ProxyException，code=401"""
        from litellm.proxy._types import ProxyException

        api_key = "sk-zx-u-wrong_signature_here"
        valid_timestamp = str(int(time.time()) - 60)  # 1 分钟前，仍在 24 小时有效期内
        mock_request = _make_request(
            {
                "zx-client-id": "client_xyz",
                "zx-client-app-id": "app_123",
                "zx-timestamp": valid_timestamp,
            }
        )

        auth_module.user_key_hashed_token_cache.async_get_cache = AsyncMock(return_value=None)

        # mock security_validator.validate 返回 False（签名验证失败）
        auth_module.security_validator.validate = MagicMock(return_value=False)

        with pytest.raises(ProxyException) as exc_info:
            await auth_module.user_api_key_auth(mock_request, api_key)

        assert exc_info.value.code == "401"
        assert "signature" in exc_info.value.message.lower()

    @pytest.mark.asyncio
    async def test_缓存命中时直接调用get_key_object(self, auth_module):
        """当缓存中已有 hashed_token 时，应跳过所有 header 验证，直接调用 get_key_object"""
        from unittest.mock import AsyncMock as AM, MagicMock as MM

        api_key = "sk-zx-u-cached_signature"
        cached_token = "hashed_token_from_cache"

        mock_request = _make_request({})  # 没有任何 header 也应能通过（因为缓存命中）

        # 模拟缓存命中
        auth_module.user_key_hashed_token_cache.async_get_cache = AM(return_value=cached_token)

        # mock get_key_object 返回一个假的 UserAPIKeyAuth 对象
        mock_key_object = MM()

        with patch(
            "litellm.proxy.zx.plugins.zx_custom_auth_app.get_key_object",
            new=AM(return_value=mock_key_object),
        ) as mock_get_key:
            result = await auth_module.user_api_key_auth(mock_request, api_key)

        # 应调用了 get_key_object，返回了 mock 对象
        assert result == mock_key_object
        mock_get_key.assert_called_once()
