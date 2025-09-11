from __future__ import (
    annotations as _annotations,
)  # important for BaseSettings import to work

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Optional

from ..utils import parse_env_vars
from .env import EnvSettingsSource

if TYPE_CHECKING:
    from pydantic_settings.main import BaseSettings


boto3_client = None
SecretsManagerClient = None


def import_aws_secrets_manager() -> None:
    global boto3_client
    global SecretsManagerClient

    try:
        from boto3 import client as boto3_client
        from mypy_boto3_secretsmanager.client import SecretsManagerClient
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "AWS Secrets Manager dependencies are not installed, run `pip install pydantic-settings[aws-secrets-manager]`"
        ) from e


class AWSSecretsManagerSettingsSource(EnvSettingsSource):
    _secret_id: str
    _secretsmanager_client: SecretsManagerClient  # type: ignore

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        secret_id: str,
        region_name: str | None = None,
        case_sensitive: bool | None = True,
        env_prefix: str | None = None,
        env_parse_none_str: str | None = None,
        env_parse_enums: bool | None = None,
    ) -> None:
        import_aws_secrets_manager()
        self._secretsmanager_client = boto3_client("secretsmanager", region_name=region_name)  # type: ignore
        self._secret_id = secret_id
        super().__init__(
            settings_cls,
            case_sensitive=case_sensitive,
            env_prefix=env_prefix,
            env_nested_delimiter="--",
            env_ignore_empty=False,
            env_parse_none_str=env_parse_none_str,
            env_parse_enums=env_parse_enums,
        )

    def _load_env_vars(self) -> Mapping[str, Optional[str]]:
        response = self._secretsmanager_client.get_secret_value(SecretId=self._secret_id)  # type: ignore

        return parse_env_vars(
            json.loads(response["SecretString"]),
            self.case_sensitive,
            self.env_ignore_empty,
            self.env_parse_none_str,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(secret_id={self._secret_id!r}, "
            f"env_nested_delimiter={self.env_nested_delimiter!r})"
        )


__all__ = [
    "AWSSecretsManagerSettingsSource",
]
