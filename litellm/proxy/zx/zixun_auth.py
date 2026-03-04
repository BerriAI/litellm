import os
import json
import time
import urllib.request
import urllib.parse
import urllib.error
from typing import Optional, Dict, Any


class ZixunAuth:
    def __init__(
        self,
        auth_host: Optional[str] = None,
        auth_api_host: Optional[str] = None,
        auth_app_key: Optional[str] = None,
        auth_app_secret: Optional[str] = None,
    ):
        """
        初始化 ZixunAuth 类

        :param auth_host: 认证服务器主机
        :param auth_api_host: 认证服务器 API 主机
        :param auth_app_key: 应用 AppKey
        :param auth_app_secret: 应用 AppSecret
        """
        # 优先使用传入的参数，如果没有则从环境变量获取
        self.auth_server_host = self._clean_url(
            auth_host or os.environ.get("ZX_AUTH_HOST")
        )
        self.auth_server_api_host = self._clean_url(
            auth_api_host or os.environ.get("ZX_AUTH_API_HOST")
        )
        self.auth_app_key = auth_app_key or os.environ.get("ZX_AUTH_APP_KEY")
        self.auth_app_secret = auth_app_secret or os.environ.get("ZX_AUTH_APP_SECRET")

        # 验证必要参数
        self._validate_config()

    def _validate_config(self):
        """验证配置参数是否齐全"""
        required_params = [
            ("auth_server_host", self.auth_server_host),
            ("auth_server_api_host", self.auth_server_api_host),
            ("auth_app_key", self.auth_app_key),
            ("auth_app_secret", self.auth_app_secret),
        ]

        for param_name, param_value in required_params:
            if not param_value:
                raise ValueError(f"缺少必要参数：{param_name}")

    @staticmethod
    def _clean_url(url: Optional[str]) -> str:
        """清理 URL，移除末尾的斜杠"""
        if url and url.endswith("/"):
            return url[:-1]
        return url or ""

    def generate_oauth_url(self, redirect_url: str) -> str:
        """
        生成 OAuth 认证 URL

        :param redirect_url: 回调 URL
        :return: OAuth 认证 URL
        """
        return (
            f"{self.auth_server_host}/oauth/authorize"
            f"?redirect_uri={urllib.parse.quote(redirect_url)}"
            f"&response_type=code"
            f"&client_id={self.auth_app_key}"
            f"&scope=openid"
            f"&state={int(time.time() * 1000)}"
            f"&prompt=login%20consent"
        )

    def get_access_token(self, auth_code: str) -> str:
        """
        获取访问令牌

        :param auth_code: 授权码
        :return: 访问令牌
        """

        # 准备请求参数
        url = f"{self.auth_server_api_host}/v1.0/oauth/code/token"
        headers = {"Content-Type": "application/json"}
        data = json.dumps(
            {
                "clientId": self.auth_app_key,
                "clientSecret": self.auth_app_secret,
                "code": auth_code,
            }
        ).encode("utf-8")

        body = None
        try:
            # 发送请求
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req) as response:
                body = json.loads(response.read().decode("utf-8"))

            # 处理响应
            if body.get("code") != 0:
                raise ValueError(f"获取 Access Token 失败: {body}")

            data = body["data"]

            return data["accessToken"]

        except Exception as e:
            raise RuntimeError(f"获取 Access Token body[{body}] 时发生错误: {e}")

    def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """
        获取用户信息

        :param access_token: 访问令牌
        :return: 用户信息字典
        """
        url = f"{self.auth_server_api_host}/v1.0/oauth/getAuthUser"
        headers = {"Content-Type": "application/json", "zx-access-token": access_token}
        data = json.dumps(
            {"clientId": self.auth_app_key, "clientSecret": self.auth_app_secret}
        ).encode("utf-8")

        try:
            # 发送请求
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req) as response:
                # with urllib.request.urlopen(req, context=disable_ssl_context) as response:
                body = json.loads(response.read().decode("utf-8"))

            # 处理响应
            if body.get("code") != 0:
                raise ValueError(f"获取用户信息失败: {access_token}")

            return body["data"]["auth_user"]

        except (urllib.error.URLError, ValueError) as e:
            raise RuntimeError(f"获取用户信息时发生错误: {e}")

    def __repr__(self):
        return f"ZixunAuth(host={self.auth_server_host})"


# 使用示例
def main():
    try:
        # 方法1：使用环境变量
        auth = ZixunAuth()

        # 方法2：显式传入参数
        # auth = ZixunAuth(
        #     auth_host='https://auth.example.com',
        #     auth_api_host='https://api.example.com',
        #     auth_app_key='your_app_key',
        #     auth_app_secret='your_app_secret'
        # )

        # 生成 OAuth URL
        redirect_url = "http://127.0.0.1"
        oauth_url = auth.generate_oauth_url(redirect_url)
        print(f"请复制到浏览器执行: {oauth_url}")

        # 获取 Access Token（通过命令行获取）
        auth_code = input("请输入浏览器重定向地址中的authCode参数：")
        access_token = auth.get_access_token(auth_code)
        print(f"Access Token: {access_token}")

        # 获取用户信息
        user_info = auth.get_user_info(access_token)
        print(f"User Info: {user_info}")

    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()
