import os
import hmac
import hashlib
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Header, status
from pydantic import BaseModel, EmailStr, Field

# 安全校验类
class SecurityValidator:
    """处理API安全认证"""
    
    def __init__(self, prefix: str):
        self.valid_credentials = self._load_credentials(prefix)
    
    @staticmethod
    def _load_credentials(prefix) -> Dict[str, str]:
        """
        从环境变量加载多个client_id和client_secret
        环境变量格式: ZX_APP_CLIENT_CREDENTIALS_1, ZX_APP_CLIENT_CREDENTIALS_2, etc.
        格式: ZX_APP_CLIENT_CREDENTIALS_N=client_id:client_secret
        """
        credentials = {}
        
        for env_key, env_value in os.environ.items():
            if not env_value:
                continue
            if not env_key.startswith(prefix):
                continue
            
            try:
                client_id, client_secret = env_value.split(":", 1)
                credentials[client_id.strip()] = client_secret.strip()
            except ValueError:
                raise ValueError(f"Invalid format for {env_key}. Expected 'client_id:client_secret'")
            
        if not credentials:
            logging.warning(f"No valid client credentials found in environment variables prefix: {prefix}")
            # raise ValueError("No valid client credentials found in environment variables")
        
        return credentials
    
    @staticmethod
    def _generate_signature(client_id: str, client_secret: str, payload: str) -> str:
        """
        使用HMAC-SHA256生成签名
        """
        message = f"{client_id}:{payload}".encode()
        signature = hmac.new(
            client_secret.encode(),
            message,
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def validate(self, client_id: str, signature: str, payload: str) -> bool:
        """
        验证请求的签名
        """
        if client_id not in self.valid_credentials:
            return False
        
        client_secret = self.valid_credentials[client_id]
        if client_secret is None:
            return False

        expected_signature = self._generate_signature(client_id, client_secret, payload)
        
        # 使用常时间比较防止时序攻击
        return hmac.compare_digest(signature, expected_signature)
