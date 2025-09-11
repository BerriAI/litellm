# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from collections import OrderedDict
from http import HTTPStatus
import json
import logging as std_logging
import os
import re
from typing import (
    Callable,
    Dict,
    Mapping,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)
import warnings

from google.api_core import client_options as client_options_lib
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.exceptions import MutualTLSChannelError  # type: ignore
from google.auth.transport import mtls  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.oauth2 import service_account  # type: ignore
import google.protobuf

from google.cloud.iam_admin_v1 import gapic_version as package_version

try:
    OptionalRetry = Union[retries.Retry, gapic_v1.method._MethodDefault, None]
except AttributeError:  # pragma: NO COVER
    OptionalRetry = Union[retries.Retry, object, None]  # type: ignore

try:
    from google.api_core import client_logging  # type: ignore

    CLIENT_LOGGING_SUPPORTED = True  # pragma: NO COVER
except ImportError:  # pragma: NO COVER
    CLIENT_LOGGING_SUPPORTED = False

_LOGGER = std_logging.getLogger(__name__)

from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore

from google.cloud.iam_admin_v1.services.iam import pagers
from google.cloud.iam_admin_v1.types import iam

from .transports.base import DEFAULT_CLIENT_INFO, IAMTransport
from .transports.grpc import IAMGrpcTransport
from .transports.grpc_asyncio import IAMGrpcAsyncIOTransport


class IAMClientMeta(type):
    """Metaclass for the IAM client.

    This provides class-level methods for building and retrieving
    support objects (e.g. transport) without polluting the client instance
    objects.
    """

    _transport_registry = OrderedDict()  # type: Dict[str, Type[IAMTransport]]
    _transport_registry["grpc"] = IAMGrpcTransport
    _transport_registry["grpc_asyncio"] = IAMGrpcAsyncIOTransport

    def get_transport_class(
        cls,
        label: Optional[str] = None,
    ) -> Type[IAMTransport]:
        """Returns an appropriate transport class.

        Args:
            label: The name of the desired transport. If none is
                provided, then the first transport in the registry is used.

        Returns:
            The transport class to use.
        """
        # If a specific transport is requested, return that one.
        if label:
            return cls._transport_registry[label]

        # No transport is requested; return the default (that is, the first one
        # in the dictionary).
        return next(iter(cls._transport_registry.values()))


class IAMClient(metaclass=IAMClientMeta):
    """Creates and manages Identity and Access Management (IAM) resources.

    You can use this service to work with all of the following
    resources:

    -  **Service accounts**, which identify an application or a virtual
       machine (VM) instance rather than a person
    -  **Service account keys**, which service accounts use to
       authenticate with Google APIs
    -  **IAM policies for service accounts**, which specify the roles
       that a principal has for the service account
    -  **IAM custom roles**, which help you limit the number of
       permissions that you grant to principals

    In addition, you can use this service to complete the following
    tasks, among others:

    -  Test whether a service account can use specific permissions
    -  Check which roles you can grant for a specific resource
    -  Lint, or validate, condition expressions in an IAM policy

    When you read data from the IAM API, each read is eventually
    consistent. In other words, if you write data with the IAM API, then
    immediately read that data, the read operation might return an older
    version of the data. To deal with this behavior, your application
    can retry the request with truncated exponential backoff.

    In contrast, writing data to the IAM API is sequentially consistent.
    In other words, write operations are always processed in the order
    in which they were received.
    """

    @staticmethod
    def _get_default_mtls_endpoint(api_endpoint):
        """Converts api endpoint to mTLS endpoint.

        Convert "*.sandbox.googleapis.com" and "*.googleapis.com" to
        "*.mtls.sandbox.googleapis.com" and "*.mtls.googleapis.com" respectively.
        Args:
            api_endpoint (Optional[str]): the api endpoint to convert.
        Returns:
            str: converted mTLS api endpoint.
        """
        if not api_endpoint:
            return api_endpoint

        mtls_endpoint_re = re.compile(
            r"(?P<name>[^.]+)(?P<mtls>\.mtls)?(?P<sandbox>\.sandbox)?(?P<googledomain>\.googleapis\.com)?"
        )

        m = mtls_endpoint_re.match(api_endpoint)
        name, mtls, sandbox, googledomain = m.groups()
        if mtls or not googledomain:
            return api_endpoint

        if sandbox:
            return api_endpoint.replace(
                "sandbox.googleapis.com", "mtls.sandbox.googleapis.com"
            )

        return api_endpoint.replace(".googleapis.com", ".mtls.googleapis.com")

    # Note: DEFAULT_ENDPOINT is deprecated. Use _DEFAULT_ENDPOINT_TEMPLATE instead.
    DEFAULT_ENDPOINT = "iam.googleapis.com"
    DEFAULT_MTLS_ENDPOINT = _get_default_mtls_endpoint.__func__(  # type: ignore
        DEFAULT_ENDPOINT
    )

    _DEFAULT_ENDPOINT_TEMPLATE = "iam.{UNIVERSE_DOMAIN}"
    _DEFAULT_UNIVERSE = "googleapis.com"

    @classmethod
    def from_service_account_info(cls, info: dict, *args, **kwargs):
        """Creates an instance of this client using the provided credentials
            info.

        Args:
            info (dict): The service account private key info.
            args: Additional arguments to pass to the constructor.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            IAMClient: The constructed client.
        """
        credentials = service_account.Credentials.from_service_account_info(info)
        kwargs["credentials"] = credentials
        return cls(*args, **kwargs)

    @classmethod
    def from_service_account_file(cls, filename: str, *args, **kwargs):
        """Creates an instance of this client using the provided credentials
            file.

        Args:
            filename (str): The path to the service account private key json
                file.
            args: Additional arguments to pass to the constructor.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            IAMClient: The constructed client.
        """
        credentials = service_account.Credentials.from_service_account_file(filename)
        kwargs["credentials"] = credentials
        return cls(*args, **kwargs)

    from_service_account_json = from_service_account_file

    @property
    def transport(self) -> IAMTransport:
        """Returns the transport used by the client instance.

        Returns:
            IAMTransport: The transport used by the client
                instance.
        """
        return self._transport

    @staticmethod
    def key_path(
        project: str,
        service_account: str,
        key: str,
    ) -> str:
        """Returns a fully-qualified key string."""
        return "projects/{project}/serviceAccounts/{service_account}/keys/{key}".format(
            project=project,
            service_account=service_account,
            key=key,
        )

    @staticmethod
    def parse_key_path(path: str) -> Dict[str, str]:
        """Parses a key path into its component segments."""
        m = re.match(
            r"^projects/(?P<project>.+?)/serviceAccounts/(?P<service_account>.+?)/keys/(?P<key>.+?)$",
            path,
        )
        return m.groupdict() if m else {}

    @staticmethod
    def service_account_path(
        project: str,
        service_account: str,
    ) -> str:
        """Returns a fully-qualified service_account string."""
        return "projects/{project}/serviceAccounts/{service_account}".format(
            project=project,
            service_account=service_account,
        )

    @staticmethod
    def parse_service_account_path(path: str) -> Dict[str, str]:
        """Parses a service_account path into its component segments."""
        m = re.match(
            r"^projects/(?P<project>.+?)/serviceAccounts/(?P<service_account>.+?)$",
            path,
        )
        return m.groupdict() if m else {}

    @staticmethod
    def common_billing_account_path(
        billing_account: str,
    ) -> str:
        """Returns a fully-qualified billing_account string."""
        return "billingAccounts/{billing_account}".format(
            billing_account=billing_account,
        )

    @staticmethod
    def parse_common_billing_account_path(path: str) -> Dict[str, str]:
        """Parse a billing_account path into its component segments."""
        m = re.match(r"^billingAccounts/(?P<billing_account>.+?)$", path)
        return m.groupdict() if m else {}

    @staticmethod
    def common_folder_path(
        folder: str,
    ) -> str:
        """Returns a fully-qualified folder string."""
        return "folders/{folder}".format(
            folder=folder,
        )

    @staticmethod
    def parse_common_folder_path(path: str) -> Dict[str, str]:
        """Parse a folder path into its component segments."""
        m = re.match(r"^folders/(?P<folder>.+?)$", path)
        return m.groupdict() if m else {}

    @staticmethod
    def common_organization_path(
        organization: str,
    ) -> str:
        """Returns a fully-qualified organization string."""
        return "organizations/{organization}".format(
            organization=organization,
        )

    @staticmethod
    def parse_common_organization_path(path: str) -> Dict[str, str]:
        """Parse a organization path into its component segments."""
        m = re.match(r"^organizations/(?P<organization>.+?)$", path)
        return m.groupdict() if m else {}

    @staticmethod
    def common_project_path(
        project: str,
    ) -> str:
        """Returns a fully-qualified project string."""
        return "projects/{project}".format(
            project=project,
        )

    @staticmethod
    def parse_common_project_path(path: str) -> Dict[str, str]:
        """Parse a project path into its component segments."""
        m = re.match(r"^projects/(?P<project>.+?)$", path)
        return m.groupdict() if m else {}

    @staticmethod
    def common_location_path(
        project: str,
        location: str,
    ) -> str:
        """Returns a fully-qualified location string."""
        return "projects/{project}/locations/{location}".format(
            project=project,
            location=location,
        )

    @staticmethod
    def parse_common_location_path(path: str) -> Dict[str, str]:
        """Parse a location path into its component segments."""
        m = re.match(r"^projects/(?P<project>.+?)/locations/(?P<location>.+?)$", path)
        return m.groupdict() if m else {}

    @classmethod
    def get_mtls_endpoint_and_cert_source(
        cls, client_options: Optional[client_options_lib.ClientOptions] = None
    ):
        """Deprecated. Return the API endpoint and client cert source for mutual TLS.

        The client cert source is determined in the following order:
        (1) if `GOOGLE_API_USE_CLIENT_CERTIFICATE` environment variable is not "true", the
        client cert source is None.
        (2) if `client_options.client_cert_source` is provided, use the provided one; if the
        default client cert source exists, use the default one; otherwise the client cert
        source is None.

        The API endpoint is determined in the following order:
        (1) if `client_options.api_endpoint` if provided, use the provided one.
        (2) if `GOOGLE_API_USE_CLIENT_CERTIFICATE` environment variable is "always", use the
        default mTLS endpoint; if the environment variable is "never", use the default API
        endpoint; otherwise if client cert source exists, use the default mTLS endpoint, otherwise
        use the default API endpoint.

        More details can be found at https://google.aip.dev/auth/4114.

        Args:
            client_options (google.api_core.client_options.ClientOptions): Custom options for the
                client. Only the `api_endpoint` and `client_cert_source` properties may be used
                in this method.

        Returns:
            Tuple[str, Callable[[], Tuple[bytes, bytes]]]: returns the API endpoint and the
                client cert source to use.

        Raises:
            google.auth.exceptions.MutualTLSChannelError: If any errors happen.
        """

        warnings.warn(
            "get_mtls_endpoint_and_cert_source is deprecated. Use the api_endpoint property instead.",
            DeprecationWarning,
        )
        if client_options is None:
            client_options = client_options_lib.ClientOptions()
        use_client_cert = os.getenv("GOOGLE_API_USE_CLIENT_CERTIFICATE", "false")
        use_mtls_endpoint = os.getenv("GOOGLE_API_USE_MTLS_ENDPOINT", "auto")
        if use_client_cert not in ("true", "false"):
            raise ValueError(
                "Environment variable `GOOGLE_API_USE_CLIENT_CERTIFICATE` must be either `true` or `false`"
            )
        if use_mtls_endpoint not in ("auto", "never", "always"):
            raise MutualTLSChannelError(
                "Environment variable `GOOGLE_API_USE_MTLS_ENDPOINT` must be `never`, `auto` or `always`"
            )

        # Figure out the client cert source to use.
        client_cert_source = None
        if use_client_cert == "true":
            if client_options.client_cert_source:
                client_cert_source = client_options.client_cert_source
            elif mtls.has_default_client_cert_source():
                client_cert_source = mtls.default_client_cert_source()

        # Figure out which api endpoint to use.
        if client_options.api_endpoint is not None:
            api_endpoint = client_options.api_endpoint
        elif use_mtls_endpoint == "always" or (
            use_mtls_endpoint == "auto" and client_cert_source
        ):
            api_endpoint = cls.DEFAULT_MTLS_ENDPOINT
        else:
            api_endpoint = cls.DEFAULT_ENDPOINT

        return api_endpoint, client_cert_source

    @staticmethod
    def _read_environment_variables():
        """Returns the environment variables used by the client.

        Returns:
            Tuple[bool, str, str]: returns the GOOGLE_API_USE_CLIENT_CERTIFICATE,
            GOOGLE_API_USE_MTLS_ENDPOINT, and GOOGLE_CLOUD_UNIVERSE_DOMAIN environment variables.

        Raises:
            ValueError: If GOOGLE_API_USE_CLIENT_CERTIFICATE is not
                any of ["true", "false"].
            google.auth.exceptions.MutualTLSChannelError: If GOOGLE_API_USE_MTLS_ENDPOINT
                is not any of ["auto", "never", "always"].
        """
        use_client_cert = os.getenv(
            "GOOGLE_API_USE_CLIENT_CERTIFICATE", "false"
        ).lower()
        use_mtls_endpoint = os.getenv("GOOGLE_API_USE_MTLS_ENDPOINT", "auto").lower()
        universe_domain_env = os.getenv("GOOGLE_CLOUD_UNIVERSE_DOMAIN")
        if use_client_cert not in ("true", "false"):
            raise ValueError(
                "Environment variable `GOOGLE_API_USE_CLIENT_CERTIFICATE` must be either `true` or `false`"
            )
        if use_mtls_endpoint not in ("auto", "never", "always"):
            raise MutualTLSChannelError(
                "Environment variable `GOOGLE_API_USE_MTLS_ENDPOINT` must be `never`, `auto` or `always`"
            )
        return use_client_cert == "true", use_mtls_endpoint, universe_domain_env

    @staticmethod
    def _get_client_cert_source(provided_cert_source, use_cert_flag):
        """Return the client cert source to be used by the client.

        Args:
            provided_cert_source (bytes): The client certificate source provided.
            use_cert_flag (bool): A flag indicating whether to use the client certificate.

        Returns:
            bytes or None: The client cert source to be used by the client.
        """
        client_cert_source = None
        if use_cert_flag:
            if provided_cert_source:
                client_cert_source = provided_cert_source
            elif mtls.has_default_client_cert_source():
                client_cert_source = mtls.default_client_cert_source()
        return client_cert_source

    @staticmethod
    def _get_api_endpoint(
        api_override, client_cert_source, universe_domain, use_mtls_endpoint
    ):
        """Return the API endpoint used by the client.

        Args:
            api_override (str): The API endpoint override. If specified, this is always
                the return value of this function and the other arguments are not used.
            client_cert_source (bytes): The client certificate source used by the client.
            universe_domain (str): The universe domain used by the client.
            use_mtls_endpoint (str): How to use the mTLS endpoint, which depends also on the other parameters.
                Possible values are "always", "auto", or "never".

        Returns:
            str: The API endpoint to be used by the client.
        """
        if api_override is not None:
            api_endpoint = api_override
        elif use_mtls_endpoint == "always" or (
            use_mtls_endpoint == "auto" and client_cert_source
        ):
            _default_universe = IAMClient._DEFAULT_UNIVERSE
            if universe_domain != _default_universe:
                raise MutualTLSChannelError(
                    f"mTLS is not supported in any universe other than {_default_universe}."
                )
            api_endpoint = IAMClient.DEFAULT_MTLS_ENDPOINT
        else:
            api_endpoint = IAMClient._DEFAULT_ENDPOINT_TEMPLATE.format(
                UNIVERSE_DOMAIN=universe_domain
            )
        return api_endpoint

    @staticmethod
    def _get_universe_domain(
        client_universe_domain: Optional[str], universe_domain_env: Optional[str]
    ) -> str:
        """Return the universe domain used by the client.

        Args:
            client_universe_domain (Optional[str]): The universe domain configured via the client options.
            universe_domain_env (Optional[str]): The universe domain configured via the "GOOGLE_CLOUD_UNIVERSE_DOMAIN" environment variable.

        Returns:
            str: The universe domain to be used by the client.

        Raises:
            ValueError: If the universe domain is an empty string.
        """
        universe_domain = IAMClient._DEFAULT_UNIVERSE
        if client_universe_domain is not None:
            universe_domain = client_universe_domain
        elif universe_domain_env is not None:
            universe_domain = universe_domain_env
        if len(universe_domain.strip()) == 0:
            raise ValueError("Universe Domain cannot be an empty string.")
        return universe_domain

    def _validate_universe_domain(self):
        """Validates client's and credentials' universe domains are consistent.

        Returns:
            bool: True iff the configured universe domain is valid.

        Raises:
            ValueError: If the configured universe domain is not valid.
        """

        # NOTE (b/349488459): universe validation is disabled until further notice.
        return True

    def _add_cred_info_for_auth_errors(
        self, error: core_exceptions.GoogleAPICallError
    ) -> None:
        """Adds credential info string to error details for 401/403/404 errors.

        Args:
            error (google.api_core.exceptions.GoogleAPICallError): The error to add the cred info.
        """
        if error.code not in [
            HTTPStatus.UNAUTHORIZED,
            HTTPStatus.FORBIDDEN,
            HTTPStatus.NOT_FOUND,
        ]:
            return

        cred = self._transport._credentials

        # get_cred_info is only available in google-auth>=2.35.0
        if not hasattr(cred, "get_cred_info"):
            return

        # ignore the type check since pypy test fails when get_cred_info
        # is not available
        cred_info = cred.get_cred_info()  # type: ignore
        if cred_info and hasattr(error._details, "append"):
            error._details.append(json.dumps(cred_info))

    @property
    def api_endpoint(self):
        """Return the API endpoint used by the client instance.

        Returns:
            str: The API endpoint used by the client instance.
        """
        return self._api_endpoint

    @property
    def universe_domain(self) -> str:
        """Return the universe domain used by the client instance.

        Returns:
            str: The universe domain used by the client instance.
        """
        return self._universe_domain

    def __init__(
        self,
        *,
        credentials: Optional[ga_credentials.Credentials] = None,
        transport: Optional[
            Union[str, IAMTransport, Callable[..., IAMTransport]]
        ] = None,
        client_options: Optional[Union[client_options_lib.ClientOptions, dict]] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
    ) -> None:
        """Instantiates the iam client.

        Args:
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            transport (Optional[Union[str,IAMTransport,Callable[..., IAMTransport]]]):
                The transport to use, or a Callable that constructs and returns a new transport.
                If a Callable is given, it will be called with the same set of initialization
                arguments as used in the IAMTransport constructor.
                If set to None, a transport is chosen automatically.
            client_options (Optional[Union[google.api_core.client_options.ClientOptions, dict]]):
                Custom options for the client.

                1. The ``api_endpoint`` property can be used to override the
                default endpoint provided by the client when ``transport`` is
                not explicitly provided. Only if this property is not set and
                ``transport`` was not explicitly provided, the endpoint is
                determined by the GOOGLE_API_USE_MTLS_ENDPOINT environment
                variable, which have one of the following values:
                "always" (always use the default mTLS endpoint), "never" (always
                use the default regular endpoint) and "auto" (auto-switch to the
                default mTLS endpoint if client certificate is present; this is
                the default value).

                2. If the GOOGLE_API_USE_CLIENT_CERTIFICATE environment variable
                is "true", then the ``client_cert_source`` property can be used
                to provide a client certificate for mTLS transport. If
                not provided, the default SSL client certificate will be used if
                present. If GOOGLE_API_USE_CLIENT_CERTIFICATE is "false" or not
                set, no client certificate will be used.

                3. The ``universe_domain`` property can be used to override the
                default "googleapis.com" universe. Note that the ``api_endpoint``
                property still takes precedence; and ``universe_domain`` is
                currently not supported for mTLS.

            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you're developing
                your own client library.

        Raises:
            google.auth.exceptions.MutualTLSChannelError: If mutual TLS transport
                creation failed for any reason.
        """
        self._client_options = client_options
        if isinstance(self._client_options, dict):
            self._client_options = client_options_lib.from_dict(self._client_options)
        if self._client_options is None:
            self._client_options = client_options_lib.ClientOptions()
        self._client_options = cast(
            client_options_lib.ClientOptions, self._client_options
        )

        universe_domain_opt = getattr(self._client_options, "universe_domain", None)

        (
            self._use_client_cert,
            self._use_mtls_endpoint,
            self._universe_domain_env,
        ) = IAMClient._read_environment_variables()
        self._client_cert_source = IAMClient._get_client_cert_source(
            self._client_options.client_cert_source, self._use_client_cert
        )
        self._universe_domain = IAMClient._get_universe_domain(
            universe_domain_opt, self._universe_domain_env
        )
        self._api_endpoint = None  # updated below, depending on `transport`

        # Initialize the universe domain validation.
        self._is_universe_domain_valid = False

        if CLIENT_LOGGING_SUPPORTED:  # pragma: NO COVER
            # Setup logging.
            client_logging.initialize_logging()

        api_key_value = getattr(self._client_options, "api_key", None)
        if api_key_value and credentials:
            raise ValueError(
                "client_options.api_key and credentials are mutually exclusive"
            )

        # Save or instantiate the transport.
        # Ordinarily, we provide the transport, but allowing a custom transport
        # instance provides an extensibility point for unusual situations.
        transport_provided = isinstance(transport, IAMTransport)
        if transport_provided:
            # transport is a IAMTransport instance.
            if credentials or self._client_options.credentials_file or api_key_value:
                raise ValueError(
                    "When providing a transport instance, "
                    "provide its credentials directly."
                )
            if self._client_options.scopes:
                raise ValueError(
                    "When providing a transport instance, provide its scopes "
                    "directly."
                )
            self._transport = cast(IAMTransport, transport)
            self._api_endpoint = self._transport.host

        self._api_endpoint = self._api_endpoint or IAMClient._get_api_endpoint(
            self._client_options.api_endpoint,
            self._client_cert_source,
            self._universe_domain,
            self._use_mtls_endpoint,
        )

        if not transport_provided:
            import google.auth._default  # type: ignore

            if api_key_value and hasattr(
                google.auth._default, "get_api_key_credentials"
            ):
                credentials = google.auth._default.get_api_key_credentials(
                    api_key_value
                )

            transport_init: Union[Type[IAMTransport], Callable[..., IAMTransport]] = (
                IAMClient.get_transport_class(transport)
                if isinstance(transport, str) or transport is None
                else cast(Callable[..., IAMTransport], transport)
            )
            # initialize with the provided callable or the passed in class
            self._transport = transport_init(
                credentials=credentials,
                credentials_file=self._client_options.credentials_file,
                host=self._api_endpoint,
                scopes=self._client_options.scopes,
                client_cert_source_for_mtls=self._client_cert_source,
                quota_project_id=self._client_options.quota_project_id,
                client_info=client_info,
                always_use_jwt_access=True,
                api_audience=self._client_options.api_audience,
            )

        if "async" not in str(self._transport):
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                std_logging.DEBUG
            ):  # pragma: NO COVER
                _LOGGER.debug(
                    "Created client `google.iam.admin_v1.IAMClient`.",
                    extra={
                        "serviceName": "google.iam.admin.v1.IAM",
                        "universeDomain": getattr(
                            self._transport._credentials, "universe_domain", ""
                        ),
                        "credentialsType": f"{type(self._transport._credentials).__module__}.{type(self._transport._credentials).__qualname__}",
                        "credentialsInfo": getattr(
                            self.transport._credentials, "get_cred_info", lambda: None
                        )(),
                    }
                    if hasattr(self._transport, "_credentials")
                    else {
                        "serviceName": "google.iam.admin.v1.IAM",
                        "credentialsType": None,
                    },
                )

    def list_service_accounts(
        self,
        request: Optional[Union[iam.ListServiceAccountsRequest, dict]] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListServiceAccountsPager:
        r"""Lists every [ServiceAccount][google.iam.admin.v1.ServiceAccount]
        that belongs to a specific project.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_list_service_accounts():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.ListServiceAccountsRequest(
                    name="name_value",
                )

                # Make the request
                page_result = client.list_service_accounts(request=request)

                # Handle the response
                for response in page_result:
                    print(response)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.ListServiceAccountsRequest, dict]):
                The request object. The service account list request.
            name (str):
                Required. The resource name of the project associated
                with the service accounts, such as
                ``projects/my-project-123``.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.iam_admin_v1.services.iam.pagers.ListServiceAccountsPager:
                The service account list response.

                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.ListServiceAccountsRequest):
            request = iam.ListServiceAccountsRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if name is not None:
                request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.list_service_accounts]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.ListServiceAccountsPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def get_service_account(
        self,
        request: Optional[Union[iam.GetServiceAccountRequest, dict]] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> iam.ServiceAccount:
        r"""Gets a [ServiceAccount][google.iam.admin.v1.ServiceAccount].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_get_service_account():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.GetServiceAccountRequest(
                    name="name_value",
                )

                # Make the request
                response = client.get_service_account(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.GetServiceAccountRequest, dict]):
                The request object. The service account get request.
            name (str):
                Required. The resource name of the service account in
                the following format:
                ``projects/{PROJECT_ID}/serviceAccounts/{ACCOUNT}``.
                Using ``-`` as a wildcard for the ``PROJECT_ID`` will
                infer the project from the account. The ``ACCOUNT``
                value can be the ``email`` address or the ``unique_id``
                of the service account.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.iam_admin_v1.types.ServiceAccount:
                An IAM service account.

                   A service account is an account for an application or
                   a virtual machine (VM) instance, not a person. You
                   can use a service account to call Google APIs. To
                   learn more, read the [overview of service
                   accounts](\ https://cloud.google.com/iam/help/service-accounts/overview).

                   When you create a service account, you specify the
                   project ID that owns the service account, as well as
                   a name that must be unique within the project. IAM
                   uses these values to create an email address that
                   identifies the service account.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.GetServiceAccountRequest):
            request = iam.GetServiceAccountRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if name is not None:
                request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.get_service_account]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def create_service_account(
        self,
        request: Optional[Union[iam.CreateServiceAccountRequest, dict]] = None,
        *,
        name: Optional[str] = None,
        account_id: Optional[str] = None,
        service_account: Optional[iam.ServiceAccount] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> iam.ServiceAccount:
        r"""Creates a [ServiceAccount][google.iam.admin.v1.ServiceAccount].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_create_service_account():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.CreateServiceAccountRequest(
                    name="name_value",
                    account_id="account_id_value",
                )

                # Make the request
                response = client.create_service_account(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.CreateServiceAccountRequest, dict]):
                The request object. The service account create request.
            name (str):
                Required. The resource name of the project associated
                with the service accounts, such as
                ``projects/my-project-123``.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            account_id (str):
                Required. The account id that is used to generate the
                service account email address and a stable unique id. It
                is unique within a project, must be 6-30 characters
                long, and match the regular expression
                ``[a-z]([-a-z0-9]*[a-z0-9])`` to comply with RFC1035.

                This corresponds to the ``account_id`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            service_account (google.cloud.iam_admin_v1.types.ServiceAccount):
                The [ServiceAccount][google.iam.admin.v1.ServiceAccount]
                resource to create. Currently, only the following values
                are user assignable: ``display_name`` and
                ``description``.

                This corresponds to the ``service_account`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.iam_admin_v1.types.ServiceAccount:
                An IAM service account.

                   A service account is an account for an application or
                   a virtual machine (VM) instance, not a person. You
                   can use a service account to call Google APIs. To
                   learn more, read the [overview of service
                   accounts](\ https://cloud.google.com/iam/help/service-accounts/overview).

                   When you create a service account, you specify the
                   project ID that owns the service account, as well as
                   a name that must be unique within the project. IAM
                   uses these values to create an email address that
                   identifies the service account.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name, account_id, service_account]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.CreateServiceAccountRequest):
            request = iam.CreateServiceAccountRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if name is not None:
                request.name = name
            if account_id is not None:
                request.account_id = account_id
            if service_account is not None:
                request.service_account = service_account

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.create_service_account]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def update_service_account(
        self,
        request: Optional[Union[iam.ServiceAccount, dict]] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> iam.ServiceAccount:
        r"""**Note:** We are in the process of deprecating this method. Use
        [PatchServiceAccount][google.iam.admin.v1.IAM.PatchServiceAccount]
        instead.

        Updates a [ServiceAccount][google.iam.admin.v1.ServiceAccount].

        You can update only the ``display_name`` field.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_update_service_account():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.ServiceAccount(
                )

                # Make the request
                response = client.update_service_account(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.ServiceAccount, dict]):
                The request object. An IAM service account.

                A service account is an account for an application or a
                virtual machine (VM) instance, not a person. You can use
                a service account to call Google APIs. To learn more,
                read the `overview of service
                accounts <https://cloud.google.com/iam/help/service-accounts/overview>`__.

                When you create a service account, you specify the
                project ID that owns the service account, as well as a
                name that must be unique within the project. IAM uses
                these values to create an email address that identifies
                the service account.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.iam_admin_v1.types.ServiceAccount:
                An IAM service account.

                   A service account is an account for an application or
                   a virtual machine (VM) instance, not a person. You
                   can use a service account to call Google APIs. To
                   learn more, read the [overview of service
                   accounts](\ https://cloud.google.com/iam/help/service-accounts/overview).

                   When you create a service account, you specify the
                   project ID that owns the service account, as well as
                   a name that must be unique within the project. IAM
                   uses these values to create an email address that
                   identifies the service account.

        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.ServiceAccount):
            request = iam.ServiceAccount(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.update_service_account]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def patch_service_account(
        self,
        request: Optional[Union[iam.PatchServiceAccountRequest, dict]] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> iam.ServiceAccount:
        r"""Patches a [ServiceAccount][google.iam.admin.v1.ServiceAccount].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_patch_service_account():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.PatchServiceAccountRequest(
                )

                # Make the request
                response = client.patch_service_account(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.PatchServiceAccountRequest, dict]):
                The request object. The service account patch request.

                You can patch only the ``display_name`` and
                ``description`` fields. You must use the ``update_mask``
                field to specify which of these fields you want to
                patch.

                Only the fields specified in the request are guaranteed
                to be returned in the response. Other fields may be
                empty in the response.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.iam_admin_v1.types.ServiceAccount:
                An IAM service account.

                   A service account is an account for an application or
                   a virtual machine (VM) instance, not a person. You
                   can use a service account to call Google APIs. To
                   learn more, read the [overview of service
                   accounts](\ https://cloud.google.com/iam/help/service-accounts/overview).

                   When you create a service account, you specify the
                   project ID that owns the service account, as well as
                   a name that must be unique within the project. IAM
                   uses these values to create an email address that
                   identifies the service account.

        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.PatchServiceAccountRequest):
            request = iam.PatchServiceAccountRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.patch_service_account]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (("service_account.name", request.service_account.name),)
            ),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def delete_service_account(
        self,
        request: Optional[Union[iam.DeleteServiceAccountRequest, dict]] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Deletes a [ServiceAccount][google.iam.admin.v1.ServiceAccount].

        **Warning:** After you delete a service account, you might not
        be able to undelete it. If you know that you need to re-enable
        the service account in the future, use
        [DisableServiceAccount][google.iam.admin.v1.IAM.DisableServiceAccount]
        instead.

        If you delete a service account, IAM permanently removes the
        service account 30 days later. Google Cloud cannot recover the
        service account after it is permanently removed, even if you
        file a support request.

        To help avoid unplanned outages, we recommend that you disable
        the service account before you delete it. Use
        [DisableServiceAccount][google.iam.admin.v1.IAM.DisableServiceAccount]
        to disable the service account, then wait at least 24 hours and
        watch for unintended consequences. If there are no unintended
        consequences, you can delete the service account.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_delete_service_account():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.DeleteServiceAccountRequest(
                    name="name_value",
                )

                # Make the request
                client.delete_service_account(request=request)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.DeleteServiceAccountRequest, dict]):
                The request object. The service account delete request.
            name (str):
                Required. The resource name of the service account in
                the following format:
                ``projects/{PROJECT_ID}/serviceAccounts/{ACCOUNT}``.
                Using ``-`` as a wildcard for the ``PROJECT_ID`` will
                infer the project from the account. The ``ACCOUNT``
                value can be the ``email`` address or the ``unique_id``
                of the service account.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.DeleteServiceAccountRequest):
            request = iam.DeleteServiceAccountRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if name is not None:
                request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.delete_service_account]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

    def undelete_service_account(
        self,
        request: Optional[Union[iam.UndeleteServiceAccountRequest, dict]] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> iam.UndeleteServiceAccountResponse:
        r"""Restores a deleted
        [ServiceAccount][google.iam.admin.v1.ServiceAccount].

        **Important:** It is not always possible to restore a deleted
        service account. Use this method only as a last resort.

        After you delete a service account, IAM permanently removes the
        service account 30 days later. There is no way to restore a
        deleted service account that has been permanently removed.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_undelete_service_account():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.UndeleteServiceAccountRequest(
                )

                # Make the request
                response = client.undelete_service_account(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.UndeleteServiceAccountRequest, dict]):
                The request object. The service account undelete request.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.iam_admin_v1.types.UndeleteServiceAccountResponse:

        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.UndeleteServiceAccountRequest):
            request = iam.UndeleteServiceAccountRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.undelete_service_account]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def enable_service_account(
        self,
        request: Optional[Union[iam.EnableServiceAccountRequest, dict]] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Enables a [ServiceAccount][google.iam.admin.v1.ServiceAccount]
        that was disabled by
        [DisableServiceAccount][google.iam.admin.v1.IAM.DisableServiceAccount].

        If the service account is already enabled, then this method has
        no effect.

        If the service account was disabled by other means—for example,
        if Google disabled the service account because it was
        compromised—you cannot use this method to enable the service
        account.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_enable_service_account():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.EnableServiceAccountRequest(
                )

                # Make the request
                client.enable_service_account(request=request)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.EnableServiceAccountRequest, dict]):
                The request object. The service account enable request.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.EnableServiceAccountRequest):
            request = iam.EnableServiceAccountRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.enable_service_account]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

    def disable_service_account(
        self,
        request: Optional[Union[iam.DisableServiceAccountRequest, dict]] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Disables a [ServiceAccount][google.iam.admin.v1.ServiceAccount]
        immediately.

        If an application uses the service account to authenticate, that
        application can no longer call Google APIs or access Google
        Cloud resources. Existing access tokens for the service account
        are rejected, and requests for new access tokens will fail.

        To re-enable the service account, use
        [EnableServiceAccount][google.iam.admin.v1.IAM.EnableServiceAccount].
        After you re-enable the service account, its existing access
        tokens will be accepted, and you can request new access tokens.

        To help avoid unplanned outages, we recommend that you disable
        the service account before you delete it. Use this method to
        disable the service account, then wait at least 24 hours and
        watch for unintended consequences. If there are no unintended
        consequences, you can delete the service account with
        [DeleteServiceAccount][google.iam.admin.v1.IAM.DeleteServiceAccount].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_disable_service_account():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.DisableServiceAccountRequest(
                )

                # Make the request
                client.disable_service_account(request=request)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.DisableServiceAccountRequest, dict]):
                The request object. The service account disable request.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.DisableServiceAccountRequest):
            request = iam.DisableServiceAccountRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.disable_service_account]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

    def list_service_account_keys(
        self,
        request: Optional[Union[iam.ListServiceAccountKeysRequest, dict]] = None,
        *,
        name: Optional[str] = None,
        key_types: Optional[
            MutableSequence[iam.ListServiceAccountKeysRequest.KeyType]
        ] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> iam.ListServiceAccountKeysResponse:
        r"""Lists every
        [ServiceAccountKey][google.iam.admin.v1.ServiceAccountKey] for a
        service account.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_list_service_account_keys():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.ListServiceAccountKeysRequest(
                    name="name_value",
                )

                # Make the request
                response = client.list_service_account_keys(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.ListServiceAccountKeysRequest, dict]):
                The request object. The service account keys list
                request.
            name (str):
                Required. The resource name of the service account in
                the following format:
                ``projects/{PROJECT_ID}/serviceAccounts/{ACCOUNT}``.

                Using ``-`` as a wildcard for the ``PROJECT_ID``, will
                infer the project from the account. The ``ACCOUNT``
                value can be the ``email`` address or the ``unique_id``
                of the service account.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            key_types (MutableSequence[google.cloud.iam_admin_v1.types.ListServiceAccountKeysRequest.KeyType]):
                Filters the types of keys the user
                wants to include in the list response.
                Duplicate key types are not allowed. If
                no key type is provided, all keys are
                returned.

                This corresponds to the ``key_types`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.iam_admin_v1.types.ListServiceAccountKeysResponse:
                The service account keys list
                response.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name, key_types]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.ListServiceAccountKeysRequest):
            request = iam.ListServiceAccountKeysRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if name is not None:
                request.name = name
            if key_types is not None:
                request.key_types = key_types

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[
            self._transport.list_service_account_keys
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def get_service_account_key(
        self,
        request: Optional[Union[iam.GetServiceAccountKeyRequest, dict]] = None,
        *,
        name: Optional[str] = None,
        public_key_type: Optional[iam.ServiceAccountPublicKeyType] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> iam.ServiceAccountKey:
        r"""Gets a
        [ServiceAccountKey][google.iam.admin.v1.ServiceAccountKey].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_get_service_account_key():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.GetServiceAccountKeyRequest(
                    name="name_value",
                )

                # Make the request
                response = client.get_service_account_key(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.GetServiceAccountKeyRequest, dict]):
                The request object. The service account key get by id
                request.
            name (str):
                Required. The resource name of the service account key
                in the following format:
                ``projects/{PROJECT_ID}/serviceAccounts/{ACCOUNT}/keys/{key}``.

                Using ``-`` as a wildcard for the ``PROJECT_ID`` will
                infer the project from the account. The ``ACCOUNT``
                value can be the ``email`` address or the ``unique_id``
                of the service account.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            public_key_type (google.cloud.iam_admin_v1.types.ServiceAccountPublicKeyType):
                Optional. The output format of the public key. The
                default is ``TYPE_NONE``, which means that the public
                key is not returned.

                This corresponds to the ``public_key_type`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.iam_admin_v1.types.ServiceAccountKey:
                Represents a service account key.

                A service account has two sets of
                key-pairs: user-managed, and
                system-managed.

                User-managed key-pairs can be created
                and deleted by users.  Users are
                responsible for rotating these keys
                periodically to ensure security of their
                service accounts.  Users retain the
                private key of these key-pairs, and
                Google retains ONLY the public key.

                System-managed keys are automatically
                rotated by Google, and are used for
                signing for a maximum of two weeks. The
                rotation process is probabilistic, and
                usage of the new key will gradually ramp
                up and down over the key's lifetime.

                If you cache the public key set for a
                service account, we recommend that you
                update the cache every 15 minutes.
                User-managed keys can be added and
                removed at any time, so it is important
                to update the cache frequently. For
                Google-managed keys, Google will publish
                a key at least 6 hours before it is
                first used for signing and will keep
                publishing it for at least 6 hours after
                it was last used for signing.

                Public keys for all service accounts are
                also published at the OAuth2 Service
                Account API.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name, public_key_type]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.GetServiceAccountKeyRequest):
            request = iam.GetServiceAccountKeyRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if name is not None:
                request.name = name
            if public_key_type is not None:
                request.public_key_type = public_key_type

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.get_service_account_key]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def create_service_account_key(
        self,
        request: Optional[Union[iam.CreateServiceAccountKeyRequest, dict]] = None,
        *,
        name: Optional[str] = None,
        private_key_type: Optional[iam.ServiceAccountPrivateKeyType] = None,
        key_algorithm: Optional[iam.ServiceAccountKeyAlgorithm] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> iam.ServiceAccountKey:
        r"""Creates a
        [ServiceAccountKey][google.iam.admin.v1.ServiceAccountKey].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_create_service_account_key():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.CreateServiceAccountKeyRequest(
                    name="name_value",
                )

                # Make the request
                response = client.create_service_account_key(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.CreateServiceAccountKeyRequest, dict]):
                The request object. The service account key create
                request.
            name (str):
                Required. The resource name of the service account in
                the following format:
                ``projects/{PROJECT_ID}/serviceAccounts/{ACCOUNT}``.
                Using ``-`` as a wildcard for the ``PROJECT_ID`` will
                infer the project from the account. The ``ACCOUNT``
                value can be the ``email`` address or the ``unique_id``
                of the service account.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            private_key_type (google.cloud.iam_admin_v1.types.ServiceAccountPrivateKeyType):
                The output format of the private key. The default value
                is ``TYPE_GOOGLE_CREDENTIALS_FILE``, which is the Google
                Credentials File format.

                This corresponds to the ``private_key_type`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            key_algorithm (google.cloud.iam_admin_v1.types.ServiceAccountKeyAlgorithm):
                Which type of key and algorithm to
                use for the key. The default is
                currently a 2K RSA key.  However this
                may change in the future.

                This corresponds to the ``key_algorithm`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.iam_admin_v1.types.ServiceAccountKey:
                Represents a service account key.

                A service account has two sets of
                key-pairs: user-managed, and
                system-managed.

                User-managed key-pairs can be created
                and deleted by users.  Users are
                responsible for rotating these keys
                periodically to ensure security of their
                service accounts.  Users retain the
                private key of these key-pairs, and
                Google retains ONLY the public key.

                System-managed keys are automatically
                rotated by Google, and are used for
                signing for a maximum of two weeks. The
                rotation process is probabilistic, and
                usage of the new key will gradually ramp
                up and down over the key's lifetime.

                If you cache the public key set for a
                service account, we recommend that you
                update the cache every 15 minutes.
                User-managed keys can be added and
                removed at any time, so it is important
                to update the cache frequently. For
                Google-managed keys, Google will publish
                a key at least 6 hours before it is
                first used for signing and will keep
                publishing it for at least 6 hours after
                it was last used for signing.

                Public keys for all service accounts are
                also published at the OAuth2 Service
                Account API.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name, private_key_type, key_algorithm]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.CreateServiceAccountKeyRequest):
            request = iam.CreateServiceAccountKeyRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if name is not None:
                request.name = name
            if private_key_type is not None:
                request.private_key_type = private_key_type
            if key_algorithm is not None:
                request.key_algorithm = key_algorithm

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[
            self._transport.create_service_account_key
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def upload_service_account_key(
        self,
        request: Optional[Union[iam.UploadServiceAccountKeyRequest, dict]] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> iam.ServiceAccountKey:
        r"""Uploads the public key portion of a key pair that you manage,
        and associates the public key with a
        [ServiceAccount][google.iam.admin.v1.ServiceAccount].

        After you upload the public key, you can use the private key
        from the key pair as a service account key.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_upload_service_account_key():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.UploadServiceAccountKeyRequest(
                )

                # Make the request
                response = client.upload_service_account_key(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.UploadServiceAccountKeyRequest, dict]):
                The request object. The service account key upload
                request.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.iam_admin_v1.types.ServiceAccountKey:
                Represents a service account key.

                A service account has two sets of
                key-pairs: user-managed, and
                system-managed.

                User-managed key-pairs can be created
                and deleted by users.  Users are
                responsible for rotating these keys
                periodically to ensure security of their
                service accounts.  Users retain the
                private key of these key-pairs, and
                Google retains ONLY the public key.

                System-managed keys are automatically
                rotated by Google, and are used for
                signing for a maximum of two weeks. The
                rotation process is probabilistic, and
                usage of the new key will gradually ramp
                up and down over the key's lifetime.

                If you cache the public key set for a
                service account, we recommend that you
                update the cache every 15 minutes.
                User-managed keys can be added and
                removed at any time, so it is important
                to update the cache frequently. For
                Google-managed keys, Google will publish
                a key at least 6 hours before it is
                first used for signing and will keep
                publishing it for at least 6 hours after
                it was last used for signing.

                Public keys for all service accounts are
                also published at the OAuth2 Service
                Account API.

        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.UploadServiceAccountKeyRequest):
            request = iam.UploadServiceAccountKeyRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[
            self._transport.upload_service_account_key
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def delete_service_account_key(
        self,
        request: Optional[Union[iam.DeleteServiceAccountKeyRequest, dict]] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Deletes a
        [ServiceAccountKey][google.iam.admin.v1.ServiceAccountKey].
        Deleting a service account key does not revoke short-lived
        credentials that have been issued based on the service account
        key.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_delete_service_account_key():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.DeleteServiceAccountKeyRequest(
                    name="name_value",
                )

                # Make the request
                client.delete_service_account_key(request=request)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.DeleteServiceAccountKeyRequest, dict]):
                The request object. The service account key delete
                request.
            name (str):
                Required. The resource name of the service account key
                in the following format:
                ``projects/{PROJECT_ID}/serviceAccounts/{ACCOUNT}/keys/{key}``.
                Using ``-`` as a wildcard for the ``PROJECT_ID`` will
                infer the project from the account. The ``ACCOUNT``
                value can be the ``email`` address or the ``unique_id``
                of the service account.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.DeleteServiceAccountKeyRequest):
            request = iam.DeleteServiceAccountKeyRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if name is not None:
                request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[
            self._transport.delete_service_account_key
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

    def disable_service_account_key(
        self,
        request: Optional[Union[iam.DisableServiceAccountKeyRequest, dict]] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Disable a
        [ServiceAccountKey][google.iam.admin.v1.ServiceAccountKey]. A
        disabled service account key can be re-enabled with
        [EnableServiceAccountKey][google.iam.admin.v1.IAM.EnableServiceAccountKey].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_disable_service_account_key():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.DisableServiceAccountKeyRequest(
                    name="name_value",
                )

                # Make the request
                client.disable_service_account_key(request=request)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.DisableServiceAccountKeyRequest, dict]):
                The request object. The service account key disable
                request.
            name (str):
                Required. The resource name of the service account key
                in the following format:
                ``projects/{PROJECT_ID}/serviceAccounts/{ACCOUNT}/keys/{key}``.

                Using ``-`` as a wildcard for the ``PROJECT_ID`` will
                infer the project from the account. The ``ACCOUNT``
                value can be the ``email`` address or the ``unique_id``
                of the service account.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.DisableServiceAccountKeyRequest):
            request = iam.DisableServiceAccountKeyRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if name is not None:
                request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[
            self._transport.disable_service_account_key
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

    def enable_service_account_key(
        self,
        request: Optional[Union[iam.EnableServiceAccountKeyRequest, dict]] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Enable a
        [ServiceAccountKey][google.iam.admin.v1.ServiceAccountKey].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_enable_service_account_key():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.EnableServiceAccountKeyRequest(
                    name="name_value",
                )

                # Make the request
                client.enable_service_account_key(request=request)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.EnableServiceAccountKeyRequest, dict]):
                The request object. The service account key enable
                request.
            name (str):
                Required. The resource name of the service account key
                in the following format:
                ``projects/{PROJECT_ID}/serviceAccounts/{ACCOUNT}/keys/{key}``.

                Using ``-`` as a wildcard for the ``PROJECT_ID`` will
                infer the project from the account. The ``ACCOUNT``
                value can be the ``email`` address or the ``unique_id``
                of the service account.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.EnableServiceAccountKeyRequest):
            request = iam.EnableServiceAccountKeyRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if name is not None:
                request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[
            self._transport.enable_service_account_key
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

    def sign_blob(
        self,
        request: Optional[Union[iam.SignBlobRequest, dict]] = None,
        *,
        name: Optional[str] = None,
        bytes_to_sign: Optional[bytes] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> iam.SignBlobResponse:
        r"""**Note:** This method is deprecated. Use the
        ```signBlob`` <https://cloud.google.com/iam/help/rest-credentials/v1/projects.serviceAccounts/signBlob>`__
        method in the IAM Service Account Credentials API instead. If
        you currently use this method, see the `migration
        guide <https://cloud.google.com/iam/help/credentials/migrate-api>`__
        for instructions.

        Signs a blob using the system-managed private key for a
        [ServiceAccount][google.iam.admin.v1.ServiceAccount].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_sign_blob():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.SignBlobRequest(
                    name="name_value",
                    bytes_to_sign=b'bytes_to_sign_blob',
                )

                # Make the request
                response = client.sign_blob(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.SignBlobRequest, dict]):
                The request object. Deprecated. `Migrate to Service Account Credentials
                API <https://cloud.google.com/iam/help/credentials/migrate-api>`__.

                The service account sign blob request.
            name (str):
                Required. Deprecated. `Migrate to Service Account
                Credentials
                API <https://cloud.google.com/iam/help/credentials/migrate-api>`__.

                The resource name of the service account in the
                following format:
                ``projects/{PROJECT_ID}/serviceAccounts/{ACCOUNT}``.
                Using ``-`` as a wildcard for the ``PROJECT_ID`` will
                infer the project from the account. The ``ACCOUNT``
                value can be the ``email`` address or the ``unique_id``
                of the service account.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            bytes_to_sign (bytes):
                Required. Deprecated. `Migrate to Service Account
                Credentials
                API <https://cloud.google.com/iam/help/credentials/migrate-api>`__.

                The bytes to sign.

                This corresponds to the ``bytes_to_sign`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.iam_admin_v1.types.SignBlobResponse:
                Deprecated. [Migrate to Service Account Credentials
                   API](\ https://cloud.google.com/iam/help/credentials/migrate-api).

                   The service account sign blob response.

        """
        warnings.warn("IAMClient.sign_blob is deprecated", DeprecationWarning)

        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name, bytes_to_sign]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.SignBlobRequest):
            request = iam.SignBlobRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if name is not None:
                request.name = name
            if bytes_to_sign is not None:
                request.bytes_to_sign = bytes_to_sign

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.sign_blob]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def sign_jwt(
        self,
        request: Optional[Union[iam.SignJwtRequest, dict]] = None,
        *,
        name: Optional[str] = None,
        payload: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> iam.SignJwtResponse:
        r"""**Note:** This method is deprecated. Use the
        ```signJwt`` <https://cloud.google.com/iam/help/rest-credentials/v1/projects.serviceAccounts/signJwt>`__
        method in the IAM Service Account Credentials API instead. If
        you currently use this method, see the `migration
        guide <https://cloud.google.com/iam/help/credentials/migrate-api>`__
        for instructions.

        Signs a JSON Web Token (JWT) using the system-managed private
        key for a [ServiceAccount][google.iam.admin.v1.ServiceAccount].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_sign_jwt():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.SignJwtRequest(
                    name="name_value",
                    payload="payload_value",
                )

                # Make the request
                response = client.sign_jwt(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.SignJwtRequest, dict]):
                The request object. Deprecated. `Migrate to Service Account Credentials
                API <https://cloud.google.com/iam/help/credentials/migrate-api>`__.

                The service account sign JWT request.
            name (str):
                Required. Deprecated. `Migrate to Service Account
                Credentials
                API <https://cloud.google.com/iam/help/credentials/migrate-api>`__.

                The resource name of the service account in the
                following format:
                ``projects/{PROJECT_ID}/serviceAccounts/{ACCOUNT}``.
                Using ``-`` as a wildcard for the ``PROJECT_ID`` will
                infer the project from the account. The ``ACCOUNT``
                value can be the ``email`` address or the ``unique_id``
                of the service account.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            payload (str):
                Required. Deprecated. `Migrate to Service Account
                Credentials
                API <https://cloud.google.com/iam/help/credentials/migrate-api>`__.

                The JWT payload to sign. Must be a serialized JSON
                object that contains a JWT Claims Set. For example:
                ``{"sub": "user@example.com", "iat": 313435}``

                If the JWT Claims Set contains an expiration time
                (``exp``) claim, it must be an integer timestamp that is
                not in the past and no more than 12 hours in the future.

                If the JWT Claims Set does not contain an expiration
                time (``exp``) claim, this claim is added automatically,
                with a timestamp that is 1 hour in the future.

                This corresponds to the ``payload`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.iam_admin_v1.types.SignJwtResponse:
                Deprecated. [Migrate to Service Account Credentials
                   API](\ https://cloud.google.com/iam/help/credentials/migrate-api).

                   The service account sign JWT response.

        """
        warnings.warn("IAMClient.sign_jwt is deprecated", DeprecationWarning)

        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name, payload]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.SignJwtRequest):
            request = iam.SignJwtRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if name is not None:
                request.name = name
            if payload is not None:
                request.payload = payload

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.sign_jwt]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def get_iam_policy(
        self,
        request: Optional[Union[iam_policy_pb2.GetIamPolicyRequest, dict]] = None,
        *,
        resource: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> policy_pb2.Policy:
        r"""Gets the IAM policy that is attached to a
        [ServiceAccount][google.iam.admin.v1.ServiceAccount]. This IAM
        policy specifies which principals have access to the service
        account.

        This method does not tell you whether the service account has
        been granted any roles on other resources. To check whether a
        service account has role grants on a resource, use the
        ``getIamPolicy`` method for that resource. For example, to view
        the role grants for a project, call the Resource Manager API's
        ```projects.getIamPolicy`` <https://cloud.google.com/resource-manager/reference/rest/v1/projects/getIamPolicy>`__
        method.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1
            from google.iam.v1 import iam_policy_pb2  # type: ignore

            def sample_get_iam_policy():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_policy_pb2.GetIamPolicyRequest(
                    resource="resource_value",
                )

                # Make the request
                response = client.get_iam_policy(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.iam.v1.iam_policy_pb2.GetIamPolicyRequest, dict]):
                The request object. Request message for ``GetIamPolicy`` method.
            resource (str):
                REQUIRED: The resource for which the
                policy is being requested. See the
                operation documentation for the
                appropriate value for this field.

                This corresponds to the ``resource`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.iam.v1.policy_pb2.Policy:
                An Identity and Access Management (IAM) policy, which specifies access
                   controls for Google Cloud resources.

                   A Policy is a collection of bindings. A binding binds
                   one or more members, or principals, to a single role.
                   Principals can be user accounts, service accounts,
                   Google groups, and domains (such as G Suite). A role
                   is a named list of permissions; each role can be an
                   IAM predefined role or a user-created custom role.

                   For some types of Google Cloud resources, a binding
                   can also specify a condition, which is a logical
                   expression that allows access to a resource only if
                   the expression evaluates to true. A condition can add
                   constraints based on attributes of the request, the
                   resource, or both. To learn which resources support
                   conditions in their IAM policies, see the [IAM
                   documentation](\ https://cloud.google.com/iam/help/conditions/resource-policies).

                   **JSON example:**

                   :literal:`\`     {       "bindings": [         {           "role": "roles/resourcemanager.organizationAdmin",           "members": [             "user:mike@example.com",             "group:admins@example.com",             "domain:google.com",             "serviceAccount:my-project-id@appspot.gserviceaccount.com"           ]         },         {           "role": "roles/resourcemanager.organizationViewer",           "members": [             "user:eve@example.com"           ],           "condition": {             "title": "expirable access",             "description": "Does not grant access after Sep 2020",             "expression": "request.time <             timestamp('2020-10-01T00:00:00.000Z')",           }         }       ],       "etag": "BwWWja0YfJA=",       "version": 3     }`\ \`

                   **YAML example:**

                   :literal:`\`     bindings:     - members:       - user:mike@example.com       - group:admins@example.com       - domain:google.com       - serviceAccount:my-project-id@appspot.gserviceaccount.com       role: roles/resourcemanager.organizationAdmin     - members:       - user:eve@example.com       role: roles/resourcemanager.organizationViewer       condition:         title: expirable access         description: Does not grant access after Sep 2020         expression: request.time < timestamp('2020-10-01T00:00:00.000Z')     etag: BwWWja0YfJA=     version: 3`\ \`

                   For a description of IAM and its features, see the
                   [IAM
                   documentation](\ https://cloud.google.com/iam/docs/).

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [resource]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        if isinstance(request, dict):
            # - The request isn't a proto-plus wrapped type,
            #   so it must be constructed via keyword expansion.
            request = iam_policy_pb2.GetIamPolicyRequest(**request)
        elif not request:
            # Null request, just make one.
            request = iam_policy_pb2.GetIamPolicyRequest()
            if resource is not None:
                request.resource = resource

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.get_iam_policy]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("resource", request.resource),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def set_iam_policy(
        self,
        request: Optional[Union[iam_policy_pb2.SetIamPolicyRequest, dict]] = None,
        *,
        resource: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> policy_pb2.Policy:
        r"""Sets the IAM policy that is attached to a
        [ServiceAccount][google.iam.admin.v1.ServiceAccount].

        Use this method to grant or revoke access to the service
        account. For example, you could grant a principal the ability to
        impersonate the service account.

        This method does not enable the service account to access other
        resources. To grant roles to a service account on a resource,
        follow these steps:

        1. Call the resource's ``getIamPolicy`` method to get its
           current IAM policy.
        2. Edit the policy so that it binds the service account to an
           IAM role for the resource.
        3. Call the resource's ``setIamPolicy`` method to update its IAM
           policy.

        For detailed instructions, see `Manage access to project,
        folders, and
        organizations <https://cloud.google.com/iam/help/service-accounts/granting-access-to-service-accounts>`__
        or `Manage access to other
        resources <https://cloud.google.com/iam/help/access/manage-other-resources>`__.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1
            from google.iam.v1 import iam_policy_pb2  # type: ignore

            def sample_set_iam_policy():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_policy_pb2.SetIamPolicyRequest(
                    resource="resource_value",
                )

                # Make the request
                response = client.set_iam_policy(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.iam.v1.iam_policy_pb2.SetIamPolicyRequest, dict]):
                The request object. Request message for ``SetIamPolicy`` method.
            resource (str):
                REQUIRED: The resource for which the
                policy is being specified. See the
                operation documentation for the
                appropriate value for this field.

                This corresponds to the ``resource`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.iam.v1.policy_pb2.Policy:
                An Identity and Access Management (IAM) policy, which specifies access
                   controls for Google Cloud resources.

                   A Policy is a collection of bindings. A binding binds
                   one or more members, or principals, to a single role.
                   Principals can be user accounts, service accounts,
                   Google groups, and domains (such as G Suite). A role
                   is a named list of permissions; each role can be an
                   IAM predefined role or a user-created custom role.

                   For some types of Google Cloud resources, a binding
                   can also specify a condition, which is a logical
                   expression that allows access to a resource only if
                   the expression evaluates to true. A condition can add
                   constraints based on attributes of the request, the
                   resource, or both. To learn which resources support
                   conditions in their IAM policies, see the [IAM
                   documentation](\ https://cloud.google.com/iam/help/conditions/resource-policies).

                   **JSON example:**

                   :literal:`\`     {       "bindings": [         {           "role": "roles/resourcemanager.organizationAdmin",           "members": [             "user:mike@example.com",             "group:admins@example.com",             "domain:google.com",             "serviceAccount:my-project-id@appspot.gserviceaccount.com"           ]         },         {           "role": "roles/resourcemanager.organizationViewer",           "members": [             "user:eve@example.com"           ],           "condition": {             "title": "expirable access",             "description": "Does not grant access after Sep 2020",             "expression": "request.time <             timestamp('2020-10-01T00:00:00.000Z')",           }         }       ],       "etag": "BwWWja0YfJA=",       "version": 3     }`\ \`

                   **YAML example:**

                   :literal:`\`     bindings:     - members:       - user:mike@example.com       - group:admins@example.com       - domain:google.com       - serviceAccount:my-project-id@appspot.gserviceaccount.com       role: roles/resourcemanager.organizationAdmin     - members:       - user:eve@example.com       role: roles/resourcemanager.organizationViewer       condition:         title: expirable access         description: Does not grant access after Sep 2020         expression: request.time < timestamp('2020-10-01T00:00:00.000Z')     etag: BwWWja0YfJA=     version: 3`\ \`

                   For a description of IAM and its features, see the
                   [IAM
                   documentation](\ https://cloud.google.com/iam/docs/).

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [resource]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        if isinstance(request, dict):
            # - The request isn't a proto-plus wrapped type,
            #   so it must be constructed via keyword expansion.
            request = iam_policy_pb2.SetIamPolicyRequest(**request)
        elif not request:
            # Null request, just make one.
            request = iam_policy_pb2.SetIamPolicyRequest()
            if resource is not None:
                request.resource = resource

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.set_iam_policy]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("resource", request.resource),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def test_iam_permissions(
        self,
        request: Optional[Union[iam_policy_pb2.TestIamPermissionsRequest, dict]] = None,
        *,
        resource: Optional[str] = None,
        permissions: Optional[MutableSequence[str]] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> iam_policy_pb2.TestIamPermissionsResponse:
        r"""Tests whether the caller has the specified permissions on a
        [ServiceAccount][google.iam.admin.v1.ServiceAccount].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1
            from google.iam.v1 import iam_policy_pb2  # type: ignore

            def sample_test_iam_permissions():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_policy_pb2.TestIamPermissionsRequest(
                    resource="resource_value",
                    permissions=['permissions_value1', 'permissions_value2'],
                )

                # Make the request
                response = client.test_iam_permissions(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.iam.v1.iam_policy_pb2.TestIamPermissionsRequest, dict]):
                The request object. Request message for ``TestIamPermissions`` method.
            resource (str):
                REQUIRED: The resource for which the
                policy detail is being requested. See
                the operation documentation for the
                appropriate value for this field.

                This corresponds to the ``resource`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            permissions (MutableSequence[str]):
                The set of permissions to check for the ``resource``.
                Permissions with wildcards (such as '*' or 'storage.*')
                are not allowed. For more information see `IAM
                Overview <https://cloud.google.com/iam/docs/overview#permissions>`__.

                This corresponds to the ``permissions`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.iam.v1.iam_policy_pb2.TestIamPermissionsResponse:
                Response message for TestIamPermissions method.
        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [resource, permissions]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        if isinstance(request, dict):
            # - The request isn't a proto-plus wrapped type,
            #   so it must be constructed via keyword expansion.
            request = iam_policy_pb2.TestIamPermissionsRequest(**request)
        elif not request:
            # Null request, just make one.
            request = iam_policy_pb2.TestIamPermissionsRequest()
            if resource is not None:
                request.resource = resource
            if permissions:
                request.permissions.extend(permissions)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.test_iam_permissions]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("resource", request.resource),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def query_grantable_roles(
        self,
        request: Optional[Union[iam.QueryGrantableRolesRequest, dict]] = None,
        *,
        full_resource_name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.QueryGrantableRolesPager:
        r"""Lists roles that can be granted on a Google Cloud
        resource. A role is grantable if the IAM policy for the
        resource can contain bindings to the role.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_query_grantable_roles():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.QueryGrantableRolesRequest(
                    full_resource_name="full_resource_name_value",
                )

                # Make the request
                page_result = client.query_grantable_roles(request=request)

                # Handle the response
                for response in page_result:
                    print(response)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.QueryGrantableRolesRequest, dict]):
                The request object. The grantable role query request.
            full_resource_name (str):
                Required. The full resource name to query from the list
                of grantable roles.

                The name follows the Google Cloud Platform resource
                format. For example, a Cloud Platform project with id
                ``my-project`` will be named
                ``//cloudresourcemanager.googleapis.com/projects/my-project``.

                This corresponds to the ``full_resource_name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.iam_admin_v1.services.iam.pagers.QueryGrantableRolesPager:
                The grantable role query response.

                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [full_resource_name]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.QueryGrantableRolesRequest):
            request = iam.QueryGrantableRolesRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if full_resource_name is not None:
                request.full_resource_name = full_resource_name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.query_grantable_roles]

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.QueryGrantableRolesPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def list_roles(
        self,
        request: Optional[Union[iam.ListRolesRequest, dict]] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListRolesPager:
        r"""Lists every predefined [Role][google.iam.admin.v1.Role] that IAM
        supports, or every custom role that is defined for an
        organization or project.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_list_roles():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.ListRolesRequest(
                )

                # Make the request
                page_result = client.list_roles(request=request)

                # Handle the response
                for response in page_result:
                    print(response)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.ListRolesRequest, dict]):
                The request object. The request to get all roles defined
                under a resource.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.iam_admin_v1.services.iam.pagers.ListRolesPager:
                The response containing the roles
                defined under a resource.
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.ListRolesRequest):
            request = iam.ListRolesRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.list_roles]

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.ListRolesPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def get_role(
        self,
        request: Optional[Union[iam.GetRoleRequest, dict]] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> iam.Role:
        r"""Gets the definition of a [Role][google.iam.admin.v1.Role].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_get_role():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.GetRoleRequest(
                )

                # Make the request
                response = client.get_role(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.GetRoleRequest, dict]):
                The request object. The request to get the definition of
                an existing role.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.iam_admin_v1.types.Role:
                A role in the Identity and Access
                Management API.

        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.GetRoleRequest):
            request = iam.GetRoleRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.get_role]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def create_role(
        self,
        request: Optional[Union[iam.CreateRoleRequest, dict]] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> iam.Role:
        r"""Creates a new custom [Role][google.iam.admin.v1.Role].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_create_role():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.CreateRoleRequest(
                )

                # Make the request
                response = client.create_role(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.CreateRoleRequest, dict]):
                The request object. The request to create a new role.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.iam_admin_v1.types.Role:
                A role in the Identity and Access
                Management API.

        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.CreateRoleRequest):
            request = iam.CreateRoleRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.create_role]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def update_role(
        self,
        request: Optional[Union[iam.UpdateRoleRequest, dict]] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> iam.Role:
        r"""Updates the definition of a custom
        [Role][google.iam.admin.v1.Role].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_update_role():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.UpdateRoleRequest(
                )

                # Make the request
                response = client.update_role(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.UpdateRoleRequest, dict]):
                The request object. The request to update a role.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.iam_admin_v1.types.Role:
                A role in the Identity and Access
                Management API.

        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.UpdateRoleRequest):
            request = iam.UpdateRoleRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.update_role]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def delete_role(
        self,
        request: Optional[Union[iam.DeleteRoleRequest, dict]] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> iam.Role:
        r"""Deletes a custom [Role][google.iam.admin.v1.Role].

        When you delete a custom role, the following changes occur
        immediately:

        -  You cannot bind a principal to the custom role in an IAM
           [Policy][google.iam.v1.Policy].
        -  Existing bindings to the custom role are not changed, but
           they have no effect.
        -  By default, the response from
           [ListRoles][google.iam.admin.v1.IAM.ListRoles] does not
           include the custom role.

        You have 7 days to undelete the custom role. After 7 days, the
        following changes occur:

        -  The custom role is permanently deleted and cannot be
           recovered.
        -  If an IAM policy contains a binding to the custom role, the
           binding is permanently removed.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_delete_role():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.DeleteRoleRequest(
                )

                # Make the request
                response = client.delete_role(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.DeleteRoleRequest, dict]):
                The request object. The request to delete an existing
                role.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.iam_admin_v1.types.Role:
                A role in the Identity and Access
                Management API.

        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.DeleteRoleRequest):
            request = iam.DeleteRoleRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.delete_role]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def undelete_role(
        self,
        request: Optional[Union[iam.UndeleteRoleRequest, dict]] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> iam.Role:
        r"""Undeletes a custom [Role][google.iam.admin.v1.Role].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_undelete_role():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.UndeleteRoleRequest(
                )

                # Make the request
                response = client.undelete_role(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.UndeleteRoleRequest, dict]):
                The request object. The request to undelete an existing
                role.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.iam_admin_v1.types.Role:
                A role in the Identity and Access
                Management API.

        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.UndeleteRoleRequest):
            request = iam.UndeleteRoleRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.undelete_role]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def query_testable_permissions(
        self,
        request: Optional[Union[iam.QueryTestablePermissionsRequest, dict]] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.QueryTestablePermissionsPager:
        r"""Lists every permission that you can test on a
        resource. A permission is testable if you can check
        whether a principal has that permission on the resource.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_query_testable_permissions():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.QueryTestablePermissionsRequest(
                )

                # Make the request
                page_result = client.query_testable_permissions(request=request)

                # Handle the response
                for response in page_result:
                    print(response)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.QueryTestablePermissionsRequest, dict]):
                The request object. A request to get permissions which
                can be tested on a resource.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.iam_admin_v1.services.iam.pagers.QueryTestablePermissionsPager:
                The response containing permissions
                which can be tested on a resource.
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.QueryTestablePermissionsRequest):
            request = iam.QueryTestablePermissionsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[
            self._transport.query_testable_permissions
        ]

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.QueryTestablePermissionsPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def query_auditable_services(
        self,
        request: Optional[Union[iam.QueryAuditableServicesRequest, dict]] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> iam.QueryAuditableServicesResponse:
        r"""Returns a list of services that allow you to opt into audit logs
        that are not generated by default.

        To learn more about audit logs, see the `Logging
        documentation <https://cloud.google.com/logging/docs/audit>`__.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_query_auditable_services():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.QueryAuditableServicesRequest(
                )

                # Make the request
                response = client.query_auditable_services(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.QueryAuditableServicesRequest, dict]):
                The request object. A request to get the list of
                auditable services for a resource.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.iam_admin_v1.types.QueryAuditableServicesResponse:
                A response containing a list of
                auditable services for a resource.

        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.QueryAuditableServicesRequest):
            request = iam.QueryAuditableServicesRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.query_auditable_services]

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def lint_policy(
        self,
        request: Optional[Union[iam.LintPolicyRequest, dict]] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> iam.LintPolicyResponse:
        r"""Lints, or validates, an IAM policy. Currently checks the
        [google.iam.v1.Binding.condition][google.iam.v1.Binding.condition]
        field, which contains a condition expression for a role binding.

        Successful calls to this method always return an HTTP ``200 OK``
        status code, even if the linter detects an issue in the IAM
        policy.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_admin_v1

            def sample_lint_policy():
                # Create a client
                client = iam_admin_v1.IAMClient()

                # Initialize request argument(s)
                request = iam_admin_v1.LintPolicyRequest(
                )

                # Make the request
                response = client.lint_policy(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.iam_admin_v1.types.LintPolicyRequest, dict]):
                The request object. The request to lint a Cloud IAM
                policy object.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.iam_admin_v1.types.LintPolicyResponse:
                The response of a lint operation. An
                empty response indicates the operation
                was able to fully execute and no lint
                issue was found.

        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, iam.LintPolicyRequest):
            request = iam.LintPolicyRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.lint_policy]

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def __enter__(self) -> "IAMClient":
        return self

    def __exit__(self, type, value, traceback):
        """Releases underlying transport's resources.

        .. warning::
            ONLY use as a context manager if the transport is NOT shared
            with other clients! Exiting the with block will CLOSE the transport
            and may cause errors in other clients!
        """
        self.transport.close()


DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=package_version.__version__
)

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__

__all__ = ("IAMClient",)
