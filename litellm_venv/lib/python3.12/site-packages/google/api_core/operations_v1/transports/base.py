# -*- coding: utf-8 -*-
# Copyright 2020 Google LLC
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
import abc
import re
from typing import Awaitable, Callable, Optional, Sequence, Union

import google.api_core  # type: ignore
from google.api_core import exceptions as core_exceptions  # type: ignore
from google.api_core import gapic_v1  # type: ignore
from google.api_core import retry as retries  # type: ignore
from google.api_core import version
import google.auth  # type: ignore
from google.auth import credentials as ga_credentials  # type: ignore
from google.longrunning import operations_pb2
from google.oauth2 import service_account  # type: ignore
import google.protobuf
from google.protobuf import empty_pb2, json_format  # type: ignore
from grpc import Compression


PROTOBUF_VERSION = google.protobuf.__version__

DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=version.__version__,
)


class OperationsTransport(abc.ABC):
    """Abstract transport class for Operations."""

    AUTH_SCOPES = ()

    DEFAULT_HOST: str = "longrunning.googleapis.com"

    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        # TODO(https://github.com/googleapis/python-api-core/issues/709): update type hint for credentials to include `google.auth.aio.Credentials`.
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        quota_project_id: Optional[str] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
        always_use_jwt_access: Optional[bool] = False,
        url_scheme="https",
        **kwargs,
    ) -> None:
        """Instantiate the transport.

        Args:
            host (Optional[str]):
                 The hostname to connect to.
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is mutually exclusive with credentials.

                .. warning::
                    Important: If you accept a credential configuration (credential JSON/File/Stream)
                    from an external source for authentication to Google Cloud Platform, you must
                    validate it before providing it to any Google API or client library. Providing an
                    unvalidated credential configuration to Google APIs or libraries can compromise
                    the security of your systems and data. For more information, refer to
                    `Validate credential configurations from external sources`_.

                .. _Validate credential configurations from external sources:

                https://cloud.google.com/docs/authentication/external/externally-sourced-credentials
            scopes (Optional[Sequence[str]]): A list of scopes.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you're developing
                your own client library.
            always_use_jwt_access (Optional[bool]): Whether self signed JWT should
                be used for service account credentials.
            url_scheme: the protocol scheme for the API endpoint.  Normally
                "https", but for testing or local servers,
                "http" can be specified.
        """
        maybe_url_match = re.match("^(?P<scheme>http(?:s)?://)?(?P<host>.*)$", host)
        if maybe_url_match is None:
            raise ValueError(
                f"Unexpected hostname structure: {host}"
            )  # pragma: NO COVER

        url_match_items = maybe_url_match.groupdict()

        host = f"{url_scheme}://{host}" if not url_match_items["scheme"] else host

        # Save the hostname. Default to port 443 (HTTPS) if none is specified.
        if ":" not in host:
            host += ":443"  # pragma: NO COVER
        self._host = host

        scopes_kwargs = {"scopes": scopes, "default_scopes": self.AUTH_SCOPES}

        # Save the scopes.
        self._scopes = scopes

        # If no credentials are provided, then determine the appropriate
        # defaults.
        if credentials and credentials_file:
            raise core_exceptions.DuplicateCredentialArgs(
                "'credentials_file' and 'credentials' are mutually exclusive"
            )

        if credentials_file is not None:
            credentials, _ = google.auth.load_credentials_from_file(
                credentials_file, **scopes_kwargs, quota_project_id=quota_project_id
            )

        elif credentials is None:
            credentials, _ = google.auth.default(
                **scopes_kwargs, quota_project_id=quota_project_id
            )

        # If the credentials are service account credentials, then always try to use self signed JWT.
        if (
            always_use_jwt_access
            and isinstance(credentials, service_account.Credentials)
            and hasattr(service_account.Credentials, "with_always_use_jwt_access")
        ):
            credentials = credentials.with_always_use_jwt_access(True)

        # Save the credentials.
        self._credentials = credentials

    def _prep_wrapped_messages(self, client_info):
        # Precompute the wrapped methods.
        self._wrapped_methods = {
            self.list_operations: gapic_v1.method.wrap_method(
                self.list_operations,
                default_retry=retries.Retry(
                    initial=0.5,
                    maximum=10.0,
                    multiplier=2.0,
                    predicate=retries.if_exception_type(
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=10.0,
                ),
                default_timeout=10.0,
                default_compression=Compression.NoCompression,
                client_info=client_info,
            ),
            self.get_operation: gapic_v1.method.wrap_method(
                self.get_operation,
                default_retry=retries.Retry(
                    initial=0.5,
                    maximum=10.0,
                    multiplier=2.0,
                    predicate=retries.if_exception_type(
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=10.0,
                ),
                default_timeout=10.0,
                default_compression=Compression.NoCompression,
                client_info=client_info,
            ),
            self.delete_operation: gapic_v1.method.wrap_method(
                self.delete_operation,
                default_retry=retries.Retry(
                    initial=0.5,
                    maximum=10.0,
                    multiplier=2.0,
                    predicate=retries.if_exception_type(
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=10.0,
                ),
                default_timeout=10.0,
                default_compression=Compression.NoCompression,
                client_info=client_info,
            ),
            self.cancel_operation: gapic_v1.method.wrap_method(
                self.cancel_operation,
                default_retry=retries.Retry(
                    initial=0.5,
                    maximum=10.0,
                    multiplier=2.0,
                    predicate=retries.if_exception_type(
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=10.0,
                ),
                default_timeout=10.0,
                default_compression=Compression.NoCompression,
                client_info=client_info,
            ),
        }

    def close(self):
        """Closes resources associated with the transport.

        .. warning::
             Only call this method if the transport is NOT shared
             with other clients - this may cause errors in other clients!
        """
        raise NotImplementedError()

    def _convert_protobuf_message_to_dict(
        self, message: google.protobuf.message.Message
    ):
        r"""Converts protobuf message to a dictionary.

        When the dictionary is encoded to JSON, it conforms to proto3 JSON spec.

        Args:
            message(google.protobuf.message.Message): The protocol buffers message
                instance to serialize.

        Returns:
            A dict representation of the protocol buffer message.
        """
        # TODO(https://github.com/googleapis/python-api-core/issues/643): For backwards compatibility
        # with protobuf 3.x 4.x, Remove once support for protobuf 3.x and 4.x is dropped.
        if PROTOBUF_VERSION[0:2] in ["3.", "4."]:
            result = json_format.MessageToDict(
                message,
                preserving_proto_field_name=True,
                including_default_value_fields=True,  # type: ignore # backward compatibility
            )
        else:
            result = json_format.MessageToDict(
                message,
                preserving_proto_field_name=True,
                always_print_fields_with_no_presence=True,
            )

        return result

    @property
    def list_operations(
        self,
    ) -> Callable[
        [operations_pb2.ListOperationsRequest],
        Union[
            operations_pb2.ListOperationsResponse,
            Awaitable[operations_pb2.ListOperationsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def get_operation(
        self,
    ) -> Callable[
        [operations_pb2.GetOperationRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def delete_operation(
        self,
    ) -> Callable[
        [operations_pb2.DeleteOperationRequest],
        Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]],
    ]:
        raise NotImplementedError()

    @property
    def cancel_operation(
        self,
    ) -> Callable[
        [operations_pb2.CancelOperationRequest],
        Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]],
    ]:
        raise NotImplementedError()


__all__ = ("OperationsTransport",)
