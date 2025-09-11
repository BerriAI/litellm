# Copyright 2019 Google LLC
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

"""Client options class.

Client options provide a consistent interface for user options to be defined
across clients.

You can pass a client options object to a client.

.. code-block:: python

    from google.api_core.client_options import ClientOptions
    from google.cloud.vision_v1 import ImageAnnotatorClient

    def get_client_cert():
        # code to load client certificate and private key.
        return client_cert_bytes, client_private_key_bytes

    options = ClientOptions(api_endpoint="foo.googleapis.com",
        client_cert_source=get_client_cert)

    client = ImageAnnotatorClient(client_options=options)

You can also pass a mapping object.

.. code-block:: python

    from google.cloud.vision_v1 import ImageAnnotatorClient

    client = ImageAnnotatorClient(
        client_options={
            "api_endpoint": "foo.googleapis.com",
            "client_cert_source" : get_client_cert
        })


"""

from typing import Callable, Mapping, Optional, Sequence, Tuple


class ClientOptions(object):
    """Client Options used to set options on clients.

    Args:
        api_endpoint (Optional[str]): The desired API endpoint, e.g.,
            compute.googleapis.com
        client_cert_source (Optional[Callable[[], Tuple[bytes, bytes]]]): A callback
            which returns client certificate bytes and private key bytes both in
            PEM format. ``client_cert_source`` and ``client_encrypted_cert_source``
            are mutually exclusive.
        client_encrypted_cert_source (Optional[Callable[[], Tuple[str, str, bytes]]]):
            A callback which returns client certificate file path, encrypted
            private key file path, and the passphrase bytes.``client_cert_source``
            and ``client_encrypted_cert_source`` are mutually exclusive.
        quota_project_id (Optional[str]): A project name that a client's
            quota belongs to.
        credentials_file (Optional[str]): A path to a file storing credentials.
            ``credentials_file` and ``api_key`` are mutually exclusive.

            .. warning::
                Important: If you accept a credential configuration (credential JSON/File/Stream)
                from an external source for authentication to Google Cloud Platform, you must
                validate it before providing it to any Google API or client library. Providing an
                unvalidated credential configuration to Google APIs or libraries can compromise
                the security of your systems and data. For more information, refer to
                `Validate credential configurations from external sources`_.

            .. _Validate credential configurations from external sources:

            https://cloud.google.com/docs/authentication/external/externally-sourced-credentials
        scopes (Optional[Sequence[str]]): OAuth access token override scopes.
        api_key (Optional[str]): Google API key. ``credentials_file`` and
            ``api_key`` are mutually exclusive.
        api_audience (Optional[str]): The intended audience for the API calls
            to the service that will be set when using certain 3rd party
            authentication flows. Audience is typically a resource identifier.
            If not set, the service endpoint value will be used as a default.
            An example of a valid ``api_audience`` is: "https://language.googleapis.com".
        universe_domain (Optional[str]): The desired universe domain. This must match
            the one in credentials. If not set, the default universe domain is
            `googleapis.com`. If both `api_endpoint` and `universe_domain` are set,
            then `api_endpoint` is used as the service endpoint. If `api_endpoint` is
            not specified, the format will be `{service}.{universe_domain}`.

    Raises:
        ValueError: If both ``client_cert_source`` and ``client_encrypted_cert_source``
            are provided, or both ``credentials_file`` and ``api_key`` are provided.
    """

    def __init__(
        self,
        api_endpoint: Optional[str] = None,
        client_cert_source: Optional[Callable[[], Tuple[bytes, bytes]]] = None,
        client_encrypted_cert_source: Optional[
            Callable[[], Tuple[str, str, bytes]]
        ] = None,
        quota_project_id: Optional[str] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        api_key: Optional[str] = None,
        api_audience: Optional[str] = None,
        universe_domain: Optional[str] = None,
    ):
        if client_cert_source and client_encrypted_cert_source:
            raise ValueError(
                "client_cert_source and client_encrypted_cert_source are mutually exclusive"
            )
        if api_key and credentials_file:
            raise ValueError("api_key and credentials_file are mutually exclusive")
        self.api_endpoint = api_endpoint
        self.client_cert_source = client_cert_source
        self.client_encrypted_cert_source = client_encrypted_cert_source
        self.quota_project_id = quota_project_id
        self.credentials_file = credentials_file
        self.scopes = scopes
        self.api_key = api_key
        self.api_audience = api_audience
        self.universe_domain = universe_domain

    def __repr__(self) -> str:
        return "ClientOptions: " + repr(self.__dict__)


def from_dict(options: Mapping[str, object]) -> ClientOptions:
    """Construct a client options object from a mapping object.

    Args:
        options (collections.abc.Mapping): A mapping object with client options.
            See the docstring for ClientOptions for details on valid arguments.
    """

    client_options = ClientOptions()

    for key, value in options.items():
        if hasattr(client_options, key):
            setattr(client_options, key, value)
        else:
            raise ValueError("ClientOptions does not accept an option '" + key + "'")

    return client_options
