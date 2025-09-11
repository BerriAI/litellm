# Copyright 2015 Google LLC
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

"""Base classes for client used to interact with Google Cloud APIs."""

import io
import json
import os
from pickle import PicklingError
from typing import Tuple
from typing import Union

import google.api_core.client_options
import google.api_core.exceptions
import google.auth
from google.auth import environment_vars
import google.auth.credentials
import google.auth.transport.requests
from google.cloud._helpers import _determine_default_project
from google.oauth2 import service_account

try:
    import google.auth.api_key

    HAS_GOOGLE_AUTH_API_KEY = True
except ImportError:
    HAS_GOOGLE_AUTH_API_KEY = False


_GOOGLE_AUTH_CREDENTIALS_HELP = (
    "This library only supports credentials from google-auth-library-python. "
    "See https://google-auth.readthedocs.io/en/latest/ "
    "for help on authentication with this library."
)

# Default timeout for auth requests.
_CREDENTIALS_REFRESH_TIMEOUT = 300


class _ClientFactoryMixin(object):
    """Mixin to allow factories that create credentials.

    .. note::

        This class is virtual.
    """

    _SET_PROJECT = False

    @classmethod
    def from_service_account_info(cls, info, *args, **kwargs):
        """Factory to retrieve JSON credentials while creating client.

        :type info: dict
        :param info:
            The JSON object with a private key and other credentials
            information (downloaded from the Google APIs console).

        :type args: tuple
        :param args: Remaining positional arguments to pass to constructor.

        :param kwargs: Remaining keyword arguments to pass to constructor.

        :rtype: :class:`_ClientFactoryMixin`
        :returns: The client created with the retrieved JSON credentials.
        :raises TypeError: if there is a conflict with the kwargs
                 and the credentials created by the factory.
        """
        if "credentials" in kwargs:
            raise TypeError("credentials must not be in keyword arguments")

        credentials = service_account.Credentials.from_service_account_info(info)
        if cls._SET_PROJECT:
            if "project" not in kwargs:
                kwargs["project"] = info.get("project_id")

        kwargs["credentials"] = credentials
        return cls(*args, **kwargs)

    @classmethod
    def from_service_account_json(cls, json_credentials_path, *args, **kwargs):
        """Factory to retrieve JSON credentials while creating client.

        :type json_credentials_path: str
        :param json_credentials_path: The path to a private key file (this file
                                      was given to you when you created the
                                      service account). This file must contain
                                      a JSON object with a private key and
                                      other credentials information (downloaded
                                      from the Google APIs console).

        :type args: tuple
        :param args: Remaining positional arguments to pass to constructor.

        :param kwargs: Remaining keyword arguments to pass to constructor.

        :rtype: :class:`_ClientFactoryMixin`
        :returns: The client created with the retrieved JSON credentials.
        :raises TypeError: if there is a conflict with the kwargs
                 and the credentials created by the factory.
        """
        with io.open(json_credentials_path, "r", encoding="utf-8") as json_fi:
            credentials_info = json.load(json_fi)

        return cls.from_service_account_info(credentials_info, *args, **kwargs)


class Client(_ClientFactoryMixin):
    """Client to bundle configuration needed for API requests.

    Stores ``credentials`` and an HTTP object so that subclasses
    can pass them along to a connection class.

    If no value is passed in for ``_http``, a :class:`requests.Session` object
    will be created and authorized with the ``credentials``. If not, the
    ``credentials`` and ``_http`` need not be related.

    Callers and subclasses may seek to use the private key from
    ``credentials`` to sign data.

    Args:
        credentials (google.auth.credentials.Credentials):
            (Optional) The OAuth2 Credentials to use for this client. If not
            passed (and if no ``_http`` object is passed), falls back to the
            default inferred from the environment.
        client_options (google.api_core.client_options.ClientOptions):
            (Optional) Custom options for the client.
        _http (requests.Session):
            (Optional) HTTP object to make requests. Can be any object that
            defines ``request()`` with the same interface as
            :meth:`requests.Session.request`. If not passed, an ``_http``
            object is created that is bound to the ``credentials`` for the
            current object.
            This parameter should be considered private, and could change in
            the future.

    Raises:
        google.auth.exceptions.DefaultCredentialsError:
            Raised if ``credentials`` is not specified and the library fails
            to acquire default credentials.
    """

    SCOPE: Union[Tuple[str, ...], None] = None
    """The scopes required for authenticating with a service.

    Needs to be set by subclasses.
    """

    def __init__(self, credentials=None, _http=None, client_options=None):
        if isinstance(client_options, dict):
            client_options = google.api_core.client_options.from_dict(client_options)
        if client_options is None:
            client_options = google.api_core.client_options.ClientOptions()

        if credentials and client_options.credentials_file:
            raise google.api_core.exceptions.DuplicateCredentialArgs(
                "'credentials' and 'client_options.credentials_file' are mutually exclusive."
            )

        if (
            HAS_GOOGLE_AUTH_API_KEY
            and client_options.api_key
            and (credentials or client_options.credentials_file)
        ):
            raise google.api_core.exceptions.DuplicateCredentialArgs(
                "'client_options.api_key' is mutually exclusive with 'credentials' and 'client_options.credentials_file'."
            )

        if credentials and not isinstance(
            credentials, google.auth.credentials.Credentials
        ):
            raise ValueError(_GOOGLE_AUTH_CREDENTIALS_HELP)

        scopes = client_options.scopes or self.SCOPE

        # if no http is provided, credentials must exist
        if not _http and credentials is None:
            if client_options.credentials_file:
                credentials, _ = google.auth.load_credentials_from_file(
                    client_options.credentials_file, scopes=scopes
                )
            elif HAS_GOOGLE_AUTH_API_KEY and client_options.api_key is not None:
                credentials = google.auth.api_key.Credentials(client_options.api_key)
            else:
                credentials, _ = google.auth.default(scopes=scopes)

        self._credentials = google.auth.credentials.with_scopes_if_required(
            credentials, scopes=scopes
        )

        if client_options.quota_project_id:
            self._credentials = self._credentials.with_quota_project(
                client_options.quota_project_id
            )

        self._http_internal = _http
        self._client_cert_source = client_options.client_cert_source

    def __getstate__(self):
        """Explicitly state that clients are not pickleable."""
        raise PicklingError(
            "\n".join(
                [
                    "Pickling client objects is explicitly not supported.",
                    "Clients have non-trivial state that is local and unpickleable.",
                ]
            )
        )

    @property
    def _http(self):
        """Getter for object used for HTTP transport.

        :rtype: :class:`~requests.Session`
        :returns: An HTTP object.
        """
        if self._http_internal is None:
            self._http_internal = google.auth.transport.requests.AuthorizedSession(
                self._credentials,
                refresh_timeout=_CREDENTIALS_REFRESH_TIMEOUT,
            )
            self._http_internal.configure_mtls_channel(self._client_cert_source)
        return self._http_internal

    def close(self):
        """Clean up transport, if set.

        Suggested use:

        .. code-block:: python

           import contextlib

           with contextlib.closing(client):  # closes on exit
               do_something_with(client)
        """
        if self._http_internal is not None:
            self._http_internal.close()


class _ClientProjectMixin(object):
    """Mixin to allow setting the project on the client.

    :type project: str
    :param project:
        (Optional) the project which the client acts on behalf of. If not
        passed, falls back to the default inferred from the environment.

    :type credentials: :class:`google.auth.credentials.Credentials`
    :param credentials:
        (Optional) credentials used to discover a project, if not passed.

    :raises: :class:`EnvironmentError` if the project is neither passed in nor
             set on the credentials or in the environment. :class:`ValueError`
             if the project value is invalid.
    """

    def __init__(self, project=None, credentials=None):
        # This test duplicates the one from `google.auth.default`, but earlier,
        # for backward compatibility:  we want the environment variable to
        # override any project set on the credentials.  See:
        # https://github.com/googleapis/python-cloud-core/issues/27
        if project is None:
            project = os.getenv(
                environment_vars.PROJECT,
                os.getenv(environment_vars.LEGACY_PROJECT),
            )

        # Project set on explicit credentials overrides discovery from
        # SDK / GAE / GCE.
        if project is None and credentials is not None:
            project = getattr(credentials, "project_id", None)

        if project is None:
            project = self._determine_default(project)

        if project is None:
            raise EnvironmentError(
                "Project was not passed and could not be "
                "determined from the environment."
            )

        if isinstance(project, bytes):
            project = project.decode("utf-8")

        if not isinstance(project, str):
            raise ValueError("Project must be a string.")

        self.project = project

    @staticmethod
    def _determine_default(project):
        """Helper:  use default project detection."""
        return _determine_default_project(project)


class ClientWithProject(Client, _ClientProjectMixin):
    """Client that also stores a project.

    :type project: str
    :param project: the project which the client acts on behalf of. If not
                    passed falls back to the default inferred from the
                    environment.

    :type credentials: :class:`~google.auth.credentials.Credentials`
    :param credentials: (Optional) The OAuth2 Credentials to use for this
                        client. If not passed (and if no ``_http`` object is
                        passed), falls back to the default inferred from the
                        environment.

    :type _http: :class:`~requests.Session`
    :param _http: (Optional) HTTP object to make requests. Can be any object
                  that defines ``request()`` with the same interface as
                  :meth:`~requests.Session.request`. If not passed, an
                  ``_http`` object is created that is bound to the
                  ``credentials`` for the current object.
                  This parameter should be considered private, and could
                  change in the future.

    :raises: :class:`ValueError` if the project is neither passed in nor
             set in the environment.
    """

    _SET_PROJECT = True  # Used by from_service_account_json()

    def __init__(self, project=None, credentials=None, client_options=None, _http=None):
        _ClientProjectMixin.__init__(self, project=project, credentials=credentials)
        Client.__init__(
            self, credentials=credentials, client_options=client_options, _http=_http
        )
