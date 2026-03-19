"""
SecurityValidator 单元测试

覆盖：
1. _load_credentials - 正确解析环境变量（格式 client_id:client_secret）
2. _load_credentials - 前缀不匹配的环境变量不加载
3. _load_credentials - 格式错误抛出 ValueError
4. _load_credentials - 无匹配环境变量时返回空 dict（不抛出，只警告）
5. _generate_signature - 相同输入产生相同签名
6. _generate_signature - 不同 client_secret 产生不同签名
7. validate - 正确签名验证通过
8. validate - 错误签名验证失败
9. validate - client_id 不存在返回 False
10. validate - client_secret 为 None 返回 False（空值容忍）
"""

import os
import sys
import pytest
from unittest.mock import patch

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from litellm.proxy.zx.zx_security_validator import SecurityValidator


class TestLoadCredentials:
    """测试 _load_credentials 静态方法"""

    def test_正确解析环境变量(self):
        """格式 client_id:client_secret 应被正确解析"""
        fake_env = {
            "MY_PREFIX_1": "app_client_id:my_secret_123",
            "MY_PREFIX_2": "another_id:another_secret",
        }
        with patch.dict(os.environ, fake_env, clear=True):
            result = SecurityValidator._load_credentials("MY_PREFIX_")

        assert result == {
            "app_client_id": "my_secret_123",
            "another_id": "another_secret",
        }

    def test_前缀不匹配的环境变量不加载(self):
        """不符合前缀的环境变量不应被加载"""
        fake_env = {
            "MY_PREFIX_1": "valid_id:valid_secret",
            "OTHER_PREFIX_1": "should_not_load_id:secret",
            "UNRELATED": "foo:bar",
        }
        with patch.dict(os.environ, fake_env, clear=True):
            result = SecurityValidator._load_credentials("MY_PREFIX_")

        # 只有 MY_PREFIX_ 开头的才被加载
        assert "valid_id" in result
        assert "should_not_load_id" not in result
        assert len(result) == 1

    def test_格式错误抛出ValueError(self):
        """不含冒号分隔符的环境变量值应抛出 ValueError"""
        fake_env = {
            "MY_PREFIX_BAD": "no_colon_here",
        }
        with patch.dict(os.environ, fake_env, clear=True):
            with pytest.raises(ValueError) as exc_info:
                SecurityValidator._load_credentials("MY_PREFIX_")

        # 错误消息应包含环境变量名称
        assert "MY_PREFIX_BAD" in str(exc_info.value)

    def test_无匹配环境变量时返回空dict不抛出(self):
        """没有匹配的环境变量时不应抛出异常，只返回空字典（并发出警告）"""
        # 清空所有环境变量，确保没有匹配前缀
        with patch.dict(os.environ, {}, clear=True):
            # 不应抛出异常
            result = SecurityValidator._load_credentials("NONEXISTENT_PREFIX_")

        assert result == {}

    def test_值含冒号时只按第一个冒号分割(self):
        """client_secret 本身含有冒号时，应只按第一个冒号分割"""
        fake_env = {
            "MY_PREFIX_COMPLEX": "my_id:secret:with:colons",
        }
        with patch.dict(os.environ, fake_env, clear=True):
            result = SecurityValidator._load_credentials("MY_PREFIX_")

        # split(":", 1) 只分割第一个冒号
        assert result == {"my_id": "secret:with:colons"}

    def test_空值的环境变量被跳过(self):
        """环境变量值为空字符串时应被跳过"""
        fake_env = {
            "MY_PREFIX_EMPTY": "",
            "MY_PREFIX_VALID": "valid_id:valid_secret",
        }
        with patch.dict(os.environ, fake_env, clear=True):
            result = SecurityValidator._load_credentials("MY_PREFIX_")

        # 空值环境变量不加载，有效的应该加载
        assert result == {"valid_id": "valid_secret"}


class TestGenerateSignature:
    """测试 _generate_signature 静态方法"""

    def test_相同输入产生相同签名(self):
        """使用相同的参数多次调用应产生完全相同的签名"""
        sig1 = SecurityValidator._generate_signature("client_1", "secret_abc", "payload_xyz")
        sig2 = SecurityValidator._generate_signature("client_1", "secret_abc", "payload_xyz")

        assert sig1 == sig2

    def test_不同client_secret产生不同签名(self):
        """相同 client_id 和 payload，但不同 client_secret，应产生不同签名"""
        sig1 = SecurityValidator._generate_signature("client_1", "secret_A", "payload_xyz")
        sig2 = SecurityValidator._generate_signature("client_1", "secret_B", "payload_xyz")

        assert sig1 != sig2

    def test_不同client_id产生不同签名(self):
        """相同 client_secret 和 payload，但不同 client_id，应产生不同签名"""
        sig1 = SecurityValidator._generate_signature("client_1", "same_secret", "payload_xyz")
        sig2 = SecurityValidator._generate_signature("client_2", "same_secret", "payload_xyz")

        assert sig1 != sig2

    def test_不同payload产生不同签名(self):
        """相同 client_id 和 client_secret，但不同 payload，应产生不同签名"""
        sig1 = SecurityValidator._generate_signature("client_1", "secret_abc", "payload_A")
        sig2 = SecurityValidator._generate_signature("client_1", "secret_abc", "payload_B")

        assert sig1 != sig2

    def test_签名是十六进制字符串(self):
        """生成的签名应为十六进制字符串（SHA256 HMAC 输出为 64 字符十六进制）"""
        sig = SecurityValidator._generate_signature("client_1", "secret_abc", "payload_xyz")

        # SHA256 hexdigest 固定为 64 字符
        assert len(sig) == 64
        # 应为十六进制字符
        assert all(c in "0123456789abcdef" for c in sig)


class TestValidate:
    """测试 validate 实例方法"""

    def _make_validator(self, credentials: dict) -> SecurityValidator:
        """辅助方法：直接注入 credentials 创建 SecurityValidator，绕过环境变量加载"""
        # 使用空前缀，使 _load_credentials 返回空字典，然后手动赋值
        with patch.dict(os.environ, {}, clear=True):
            validator = SecurityValidator("NONEXISTENT_PREFIX_")
        # 直接替换内部 credentials
        validator.valid_credentials = credentials
        return validator

    def test_正确签名验证通过(self):
        """当提供正确的 client_id 和对应签名时，validate 应返回 True"""
        client_id = "test_client"
        client_secret = "test_secret_key"
        payload = "user@example.com:1700000000"

        # 预先生成正确签名
        correct_signature = SecurityValidator._generate_signature(client_id, client_secret, payload)

        validator = self._make_validator({client_id: client_secret})
        result = validator.validate(client_id, correct_signature, payload)

        assert result is True

    def test_错误签名验证失败(self):
        """当签名错误时，validate 应返回 False"""
        client_id = "test_client"
        client_secret = "test_secret_key"
        payload = "user@example.com:1700000000"
        wrong_signature = "0" * 64  # 明显错误的签名

        validator = self._make_validator({client_id: client_secret})
        result = validator.validate(client_id, wrong_signature, payload)

        assert result is False

    def test_client_id不存在返回False(self):
        """当 client_id 不在已加载的凭证中时，validate 应返回 False"""
        validator = self._make_validator({"known_client": "some_secret"})
        result = validator.validate("unknown_client", "any_signature", "any_payload")

        assert result is False

    def test_client_secret为None返回False(self):
        """当 client_id 存在但对应的 client_secret 为 None 时，validate 应返回 False（空值容忍）"""
        client_id = "test_client"

        # 手动将 client_secret 设置为 None，模拟空值场景
        validator = self._make_validator({client_id: None})
        result = validator.validate(client_id, "any_signature", "any_payload")

        assert result is False

    def test_payload不同时验证失败(self):
        """当 payload 与签名时使用的 payload 不匹配时，validate 应返回 False"""
        client_id = "test_client"
        client_secret = "test_secret_key"
        original_payload = "user@example.com:1700000000"
        different_payload = "other@example.com:1700000000"

        # 用 original_payload 生成签名
        signature = SecurityValidator._generate_signature(client_id, client_secret, original_payload)

        validator = self._make_validator({client_id: client_secret})
        # 但用 different_payload 验证，应失败
        result = validator.validate(client_id, signature, different_payload)

        assert result is False
