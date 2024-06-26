# +-------------------------------------------------------------+
#
#           Use SecretDetection /moderations for your LLM calls
#
# +-------------------------------------------------------------+
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import sys, os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from typing import Optional, Literal, Union
import litellm, traceback, sys, uuid
from litellm.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.integrations.custom_logger import CustomLogger
from fastapi import HTTPException
from litellm._logging import verbose_proxy_logger
from litellm.utils import (
    ModelResponse,
    EmbeddingResponse,
    ImageResponse,
    StreamingChoices,
)
from datetime import datetime
import aiohttp, asyncio
from litellm._logging import verbose_proxy_logger
import tempfile
from litellm._logging import verbose_proxy_logger


litellm.set_verbose = True


class _ENTERPRISE_SecretDetection(CustomLogger):
    def __init__(self):
        pass

    def scan_message_for_secrets(self, message_content: str):
        from detect_secrets import SecretsCollection
        from detect_secrets.settings import default_settings

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(message_content.encode("utf-8"))
        temp_file.close()

        secrets = SecretsCollection()
        with default_settings():
            secrets.scan_file(temp_file.name)

        os.remove(temp_file.name)

        detected_secrets = []
        for file in secrets.files:
            for found_secret in secrets[file]:
                if found_secret.secret_value is None:
                    continue
                detected_secrets.append(
                    {"type": found_secret.type, "value": found_secret.secret_value}
                )

        return detected_secrets

    #### CALL HOOKS - proxy only ####
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,  # "completion", "embeddings", "image_generation", "moderation"
    ):
        from detect_secrets import SecretsCollection
        from detect_secrets.settings import default_settings

        if "messages" in data and isinstance(data["messages"], list):
            for message in data["messages"]:
                if "content" in message and isinstance(message["content"], str):
                    detected_secrets = self.scan_message_for_secrets(message["content"])

                    for secret in detected_secrets:
                        message["content"] = message["content"].replace(
                            secret["value"], "[REDACTED]"
                        )

                    if len(detected_secrets) > 0:
                        secret_types = [secret["type"] for secret in detected_secrets]
                        verbose_proxy_logger.warning(
                            f"Detected and redacted secrets in message: {secret_types}"
                        )

        if "prompt" in data:
            if isinstance(data["prompt"], str):
                detected_secrets = self.scan_message_for_secrets(data["prompt"])
                for secret in detected_secrets:
                    data["prompt"] = data["prompt"].replace(
                        secret["value"], "[REDACTED]"
                    )
                if len(detected_secrets) > 0:
                    secret_types = [secret["type"] for secret in detected_secrets]
                    verbose_proxy_logger.warning(
                        f"Detected and redacted secrets in prompt: {secret_types}"
                    )
            elif isinstance(data["prompt"], list):
                for item in data["prompt"]:
                    if isinstance(item, str):
                        detected_secrets = self.scan_message_for_secrets(item)
                        for secret in detected_secrets:
                            item = item.replace(secret["value"], "[REDACTED]")
                        if len(detected_secrets) > 0:
                            secret_types = [
                                secret["type"] for secret in detected_secrets
                            ]
                            verbose_proxy_logger.warning(
                                f"Detected and redacted secrets in prompt: {secret_types}"
                            )

        if "input" in data:
            if isinstance(data["input"], str):
                detected_secrets = self.scan_message_for_secrets(data["input"])
                for secret in detected_secrets:
                    data["input"] = data["input"].replace(secret["value"], "[REDACTED]")
                if len(detected_secrets) > 0:
                    secret_types = [secret["type"] for secret in detected_secrets]
                    verbose_proxy_logger.warning(
                        f"Detected and redacted secrets in input: {secret_types}"
                    )
            elif isinstance(data["input"], list):
                _input_in_request = data["input"]
                for idx, item in enumerate(_input_in_request):
                    if isinstance(item, str):
                        detected_secrets = self.scan_message_for_secrets(item)
                        for secret in detected_secrets:
                            _input_in_request[idx] = item.replace(
                                secret["value"], "[REDACTED]"
                            )
                        if len(detected_secrets) > 0:
                            secret_types = [
                                secret["type"] for secret in detected_secrets
                            ]
                            verbose_proxy_logger.warning(
                                f"Detected and redacted secrets in input: {secret_types}"
                            )
                verbose_proxy_logger.debug("Data after redacting input %s", data)
        return
