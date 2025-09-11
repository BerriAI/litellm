from __future__ import annotations as _annotations

from collections.abc import Iterator, Mapping
from functools import cached_property
from typing import TYPE_CHECKING, Optional

from .env import EnvSettingsSource

if TYPE_CHECKING:
    from google.auth import default as google_auth_default
    from google.auth.credentials import Credentials
    from google.cloud.secretmanager import SecretManagerServiceClient

    from pydantic_settings.main import BaseSettings
else:
    Credentials = None
    SecretManagerServiceClient = None
    google_auth_default = None


def import_gcp_secret_manager() -> None:
    global Credentials
    global SecretManagerServiceClient
    global google_auth_default

    try:
        from google.auth import default as google_auth_default
        from google.auth.credentials import Credentials
        from google.cloud.secretmanager import SecretManagerServiceClient
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "GCP Secret Manager dependencies are not installed, run `pip install pydantic-settings[gcp-secret-manager]`"
        ) from e


class GoogleSecretManagerMapping(Mapping[str, Optional[str]]):
    _loaded_secrets: dict[str, str | None]
    _secret_client: SecretManagerServiceClient

    def __init__(
        self,
        secret_client: SecretManagerServiceClient,
        project_id: str,
        case_sensitive: bool,
    ) -> None:
        self._loaded_secrets = {}
        self._secret_client = secret_client
        self._project_id = project_id
        self._case_sensitive = case_sensitive

    @property
    def _gcp_project_path(self) -> str:
        return self._secret_client.common_project_path(self._project_id)

    @cached_property
    def _secret_names(self) -> list[str]:
        rv: list[str] = []

        secrets = self._secret_client.list_secrets(parent=self._gcp_project_path)
        for secret in secrets:
            name = self._secret_client.parse_secret_path(secret.name).get("secret", "")
            if not self._case_sensitive:
                name = name.lower()
            rv.append(name)
        return rv

    def _secret_version_path(self, key: str, version: str = "latest") -> str:
        return self._secret_client.secret_version_path(self._project_id, key, version)

    def __getitem__(self, key: str) -> str | None:
        if not self._case_sensitive:
            key = key.lower()
        if key not in self._loaded_secrets:
            # If we know the key isn't available in secret manager, raise a key error
            if key not in self._secret_names:
                raise KeyError(key)

            try:
                self._loaded_secrets[key] = self._secret_client.access_secret_version(
                    name=self._secret_version_path(key)
                ).payload.data.decode("UTF-8")
            except Exception:
                raise KeyError(key)

        return self._loaded_secrets[key]

    def __len__(self) -> int:
        return len(self._secret_names)

    def __iter__(self) -> Iterator[str]:
        return iter(self._secret_names)


class GoogleSecretManagerSettingsSource(EnvSettingsSource):
    _credentials: Credentials
    _secret_client: SecretManagerServiceClient
    _project_id: str

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        credentials: Credentials | None = None,
        project_id: str | None = None,
        env_prefix: str | None = None,
        env_parse_none_str: str | None = None,
        env_parse_enums: bool | None = None,
        secret_client: SecretManagerServiceClient | None = None,
        case_sensitive: bool | None = True,
    ) -> None:
        # Import Google Packages if they haven't already been imported
        if (
            SecretManagerServiceClient is None
            or Credentials is None
            or google_auth_default is None
        ):
            import_gcp_secret_manager()

        # If credentials or project_id are not passed, then
        # try to get them from the default function
        if not credentials or not project_id:
            _creds, _project_id = google_auth_default()  # type: ignore[no-untyped-call]

        # Set the credentials and/or project id if they weren't specified
        if credentials is None:
            credentials = _creds

        if project_id is None:
            if isinstance(_project_id, str):
                project_id = _project_id
            else:
                raise AttributeError(
                    "project_id is required to be specified either as an argument or from the google.auth.default. See https://google-auth.readthedocs.io/en/master/reference/google.auth.html#google.auth.default"
                )

        self._credentials: Credentials = credentials
        self._project_id: str = project_id

        if secret_client:
            self._secret_client = secret_client
        else:
            self._secret_client = SecretManagerServiceClient(
                credentials=self._credentials
            )

        super().__init__(
            settings_cls,
            case_sensitive=case_sensitive,
            env_prefix=env_prefix,
            env_ignore_empty=False,
            env_parse_none_str=env_parse_none_str,
            env_parse_enums=env_parse_enums,
        )

    def _load_env_vars(self) -> Mapping[str, Optional[str]]:
        return GoogleSecretManagerMapping(
            self._secret_client,
            project_id=self._project_id,
            case_sensitive=self.case_sensitive,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(project_id={self._project_id!r}, env_nested_delimiter={self.env_nested_delimiter!r})"


__all__ = ["GoogleSecretManagerSettingsSource", "GoogleSecretManagerMapping"]
