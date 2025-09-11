from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Literal

import polars._utils.logging
from polars._utils.logging import eprint, verbose
from polars._utils.unstable import issue_unstable_warning
from polars.io.cloud.credential_provider._providers import (
    CredentialProvider,
    CredentialProviderAWS,
    CredentialProviderAzure,
    CredentialProviderGCP,
)

if TYPE_CHECKING:
    from polars.io.cloud.credential_provider._providers import (
        CredentialProviderFunction,
        CredentialProviderFunctionReturn,
    )

# https://docs.rs/object_store/latest/object_store/enum.ClientConfigKey.html
OBJECT_STORE_CLIENT_OPTIONS: frozenset[str] = frozenset(
    [
        "allow_http",
        "allow_invalid_certificates",
        "connect_timeout",
        "default_content_type",
        "http1_only",
        "http2_only",
        "http2_keep_alive_interval",
        "http2_keep_alive_timeout",
        "http2_keep_alive_while_idle",
        "http2_max_frame_size",
        "pool_idle_timeout",
        "pool_max_idle_per_host",
        "proxy_url",
        "proxy_ca_certificate",
        "proxy_excludes",
        "timeout",
        "user_agent",
    ]
)


class CredentialProviderBuilder:
    """
    Builds credential providers.

    This is used to defer credential provider initialization to happen at
    `collect()` rather than immediately during query construction. This makes
    the behavior predictable when queries are sent to another environment for
    execution.
    """

    def __init__(
        self,
        credential_provider_init: CredentialProviderBuilderImpl,
    ) -> None:
        """
        Initialize configuration for building a credential provider.

        Parameters
        ----------
        credential_provider_init
            Initializer function that returns a credential provider.
        """
        self.credential_provider_init = credential_provider_init

    # Note: The rust-side expects this exact function name.
    def build_credential_provider(
        self,
    ) -> CredentialProvider | CredentialProviderFunction | None:
        """Instantiate a credential provider from configuration."""
        verbose = polars._utils.logging.verbose()

        if verbose:
            eprint(
                "[CredentialProviderBuilder]: Begin initialize "
                f"{self.credential_provider_init!r}"
            )

        v = self.credential_provider_init()

        if verbose:
            if v is not None:
                eprint(
                    f"[CredentialProviderBuilder]: Initialized {v!r} "
                    f"from {self.credential_provider_init!r}"
                )
            else:
                eprint(
                    f"[CredentialProviderBuilder]: No provider initialized "
                    f"from {self.credential_provider_init!r}"
                )

        return v

    @classmethod
    def from_initialized_provider(
        cls, credential_provider: CredentialProviderFunction
    ) -> CredentialProviderBuilder:
        """Initialize with an already constructed provider."""
        return cls(InitializedCredentialProvider(credential_provider))

    def __getstate__(self) -> Any:
        state = self.credential_provider_init

        if verbose():
            eprint(f"[CredentialProviderBuilder]: __getstate__(): {state = !r} ")

        return state

    def __setstate__(self, state: Any) -> None:
        self.credential_provider_init = state

        if verbose():
            eprint(f"[CredentialProviderBuilder]: __setstate__(): {self = !r}")

    def __repr__(self) -> str:
        return f"CredentialProviderBuilder({self.credential_provider_init!r})"


class CredentialProviderBuilderImpl(abc.ABC):
    @abc.abstractmethod
    def __call__(self) -> CredentialProviderFunction | None:
        pass

    @property
    @abc.abstractmethod
    def provider_repr(self) -> str:
        """Used for logging."""

    def __repr__(self) -> str:
        provider_repr = self.provider_repr
        builder_name = type(self).__name__

        return f"{provider_repr} @ {builder_name}"


# Wraps an already ininitialized credential provider into the builder interface.
# Used for e.g. user-provided credential providers.
class InitializedCredentialProvider(CredentialProviderBuilderImpl):
    """Wraps an already initialized credential provider."""

    def __init__(self, credential_provider: CredentialProviderFunction | None) -> None:
        self.credential_provider = credential_provider

    def __call__(self) -> CredentialProviderFunction | None:
        return self.credential_provider

    @property
    def provider_repr(self) -> str:
        return repr(self.credential_provider)


# Represents an automatic initialization configuration. This is created for
# credential_provider="auto".
class AutoInit(CredentialProviderBuilderImpl):
    def __init__(self, cls: Any, **kw: Any) -> None:
        self.cls = cls
        self.kw = kw

    def __call__(self) -> Any:
        # This is used for credential_provider="auto", which allows for
        # ImportErrors.
        try:
            return self.cls(**self.kw)
        except ImportError as e:
            if verbose():
                eprint(f"failed to auto-initialize {self.provider_repr}: {e!r}")

        return None

    @property
    def provider_repr(self) -> str:
        return self.cls.__name__


class UserProvidedGCPToken(CredentialProvider):
    """User-provided GCP token in storage_options."""

    def __init__(self, token: str) -> None:
        self.token = token

    def __call__(self) -> CredentialProviderFunctionReturn:
        """Fetches the credentials."""
        return {"bearer_token": self.token}, None


def _init_credential_provider_builder(
    credential_provider: CredentialProviderFunction
    | CredentialProviderBuilder
    | Literal["auto"]
    | None,
    source: Any,
    storage_options: dict[str, Any] | None,
    caller_name: str,
) -> CredentialProviderBuilder | None:
    def f() -> CredentialProviderBuilder | None:
        # Note: The behavior of this function should depend only on the function
        # parameters. Any environment-specific behavior should take place inside
        # instantiated credential providers.

        from polars.io.cloud._utils import (
            _first_scan_path,
            _get_path_scheme,
            _is_aws_cloud,
            _is_azure_cloud,
            _is_gcp_cloud,
        )

        if credential_provider is None:
            return None

        if isinstance(credential_provider, CredentialProviderBuilder):
            # This happens when the catalog client auto-inits and passes it to
            # scan/write_delta, which calls us again.
            return credential_provider

        if credential_provider != "auto":
            msg = f"the `credential_provider` parameter of `{caller_name}` is considered unstable."
            issue_unstable_warning(msg)

            return CredentialProviderBuilder.from_initialized_provider(
                credential_provider
            )

        if (path := _first_scan_path(source)) is None:
            return None

        if (scheme := _get_path_scheme(path)) is None:
            return None

        if _is_azure_cloud(scheme):
            tenant_id = None
            storage_account = None

            if storage_options is not None:
                for k, v in storage_options.items():
                    k = k.lower()

                    # https://docs.rs/object_store/latest/object_store/azure/enum.AzureConfigKey.html
                    if k in {
                        "azure_storage_tenant_id",
                        "azure_storage_authority_id",
                        "azure_tenant_id",
                        "azure_authority_id",
                        "tenant_id",
                        "authority_id",
                    }:
                        tenant_id = v
                    elif k in {"azure_storage_account_name", "account_name"}:
                        storage_account = v
                    elif k in {"azure_use_azure_cli", "use_azure_cli"}:
                        continue
                    elif k in OBJECT_STORE_CLIENT_OPTIONS:
                        continue
                    else:
                        # We assume some sort of access key was given, so we
                        # just dispatch to the rust side.
                        return None

            storage_account = (
                # Prefer the one embedded in the path
                CredentialProviderAzure._extract_adls_uri_storage_account(str(path))
                or storage_account
            )

            return CredentialProviderBuilder(
                AutoInit(
                    CredentialProviderAzure,
                    tenant_id=tenant_id,
                    _storage_account=storage_account,
                )
            )

        elif _is_aws_cloud(scheme):
            region = None
            profile = None
            default_region = None
            unhandled_key = None
            has_endpoint_url = False

            if storage_options is not None:
                for k, v in storage_options.items():
                    k = k.lower()

                    # https://docs.rs/object_store/latest/object_store/aws/enum.AmazonS3ConfigKey.html
                    if k in {"aws_region", "region"}:
                        region = v
                    elif k in {"aws_default_region", "default_region"}:
                        default_region = v
                    elif k in {"aws_profile", "profile"}:
                        profile = v
                    elif k in {
                        "aws_endpoint",
                        "aws_endpoint_url",
                        "endpoint",
                        "endpoint_url",
                    }:
                        has_endpoint_url = True
                    elif k in OBJECT_STORE_CLIENT_OPTIONS:
                        continue
                    else:
                        # We assume this is some sort of access key
                        unhandled_key = k

            if unhandled_key is not None:
                if profile is not None:
                    msg = (
                        "unsupported: cannot combine aws_profile with "
                        f"{unhandled_key} in storage_options"
                    )
                    raise ValueError(msg)

            return CredentialProviderBuilder(
                AutoInit(
                    CredentialProviderAWS,
                    profile_name=profile,
                    region_name=region or default_region,
                    _auto_init_unhandled_key=unhandled_key,
                    _storage_options_has_endpoint_url=has_endpoint_url,
                )
            )

        elif _is_gcp_cloud(scheme):
            token = None
            unhandled_key = None

            if storage_options is not None:
                for k, v in storage_options.items():
                    k = k.lower()

                    # https://docs.rs/object_store/latest/object_store/gcp/enum.GoogleConfigKey.html
                    if k in {"token", "bearer_token"}:
                        token = v
                    elif k in {
                        "google_bucket",
                        "google_bucket_name",
                        "bucket",
                        "bucket_name",
                    }:
                        continue
                    elif k in OBJECT_STORE_CLIENT_OPTIONS:
                        continue
                    else:
                        # We assume some sort of access key was given, so we
                        # just dispatch to the rust side.
                        unhandled_key = k

            if unhandled_key is not None:
                if token is not None:
                    msg = (
                        "unsupported: cannot combine token with "
                        f"{unhandled_key} in storage_options"
                    )
                    raise ValueError(msg)

                return None

            if token is not None:
                return CredentialProviderBuilder(
                    InitializedCredentialProvider(UserProvidedGCPToken(token))
                )

            return CredentialProviderBuilder(AutoInit(CredentialProviderGCP))

        return None

    credential_provider_init = f()

    if verbose():
        eprint(f"_init_credential_provider_builder(): {credential_provider_init = !r}")

    return credential_provider_init
