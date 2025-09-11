# Copyright 2014 Google LLC
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

"""Shared implementation of connections to API servers."""

import collections
import collections.abc
import json
import os
import platform
from typing import Optional
from urllib.parse import urlencode
import warnings

from google.api_core.client_info import ClientInfo
from google.cloud import exceptions
from google.cloud import version


API_BASE_URL = "https://www.googleapis.com"
"""The base of the API call URL."""

DEFAULT_USER_AGENT = "gcloud-python/{0}".format(version.__version__)
"""The user agent for google-cloud-python requests."""

CLIENT_INFO_HEADER = "X-Goog-API-Client"
CLIENT_INFO_TEMPLATE = "gl-python/" + platform.python_version() + " gccl/{}"

_USER_AGENT_ALL_CAPS_DEPRECATED = """\
The 'USER_AGENT' class-level attribute is deprecated.  Please use
'user_agent' instead.
"""

_EXTRA_HEADERS_ALL_CAPS_DEPRECATED = """\
The '_EXTRA_HEADERS' class-level attribute is deprecated.  Please use
'extra_headers' instead.
"""

_DEFAULT_TIMEOUT = 60  # in seconds


class Connection(object):
    """A generic connection to Google Cloud Platform.

    :type client: :class:`~google.cloud.client.Client`
    :param client: The client that owns the current connection.

    :type client_info: :class:`~google.api_core.client_info.ClientInfo`
    :param client_info: (Optional) instance used to generate user agent.
    """

    _user_agent = DEFAULT_USER_AGENT

    def __init__(self, client, client_info=None):
        self._client = client

        if client_info is None:
            client_info = ClientInfo()

        self._client_info = client_info
        self._extra_headers = {}

    @property
    def USER_AGENT(self):
        """Deprecated:  get / set user agent sent by connection.

        :rtype: str
        :returns: user agent
        """
        warnings.warn(_USER_AGENT_ALL_CAPS_DEPRECATED, DeprecationWarning, stacklevel=2)
        return self.user_agent

    @USER_AGENT.setter
    def USER_AGENT(self, value):
        warnings.warn(_USER_AGENT_ALL_CAPS_DEPRECATED, DeprecationWarning, stacklevel=2)
        self.user_agent = value

    @property
    def user_agent(self):
        """Get / set user agent sent by connection.

        :rtype: str
        :returns: user agent
        """
        return self._client_info.to_user_agent()

    @user_agent.setter
    def user_agent(self, value):
        self._client_info.user_agent = value

    @property
    def _EXTRA_HEADERS(self):
        """Deprecated:  get / set extra headers sent by connection.

        :rtype: dict
        :returns: header keys / values
        """
        warnings.warn(
            _EXTRA_HEADERS_ALL_CAPS_DEPRECATED, DeprecationWarning, stacklevel=2
        )
        return self.extra_headers

    @_EXTRA_HEADERS.setter
    def _EXTRA_HEADERS(self, value):
        warnings.warn(
            _EXTRA_HEADERS_ALL_CAPS_DEPRECATED, DeprecationWarning, stacklevel=2
        )
        self.extra_headers = value

    @property
    def extra_headers(self):
        """Get / set extra headers sent by connection.

        :rtype: dict
        :returns: header keys / values
        """
        return self._extra_headers

    @extra_headers.setter
    def extra_headers(self, value):
        self._extra_headers = value

    @property
    def credentials(self):
        """Getter for current credentials.

        :rtype: :class:`google.auth.credentials.Credentials` or
                :class:`NoneType`
        :returns: The credentials object associated with this connection.
        """
        return self._client._credentials

    @property
    def http(self):
        """A getter for the HTTP transport used in talking to the API.

        Returns:
            google.auth.transport.requests.AuthorizedSession:
                A :class:`requests.Session` instance.
        """
        return self._client._http


class JSONConnection(Connection):
    """A connection to a Google JSON-based API.

    These APIs are discovery based. For reference:

        https://developers.google.com/discovery/

    This defines :meth:`api_request` for making a generic JSON
    API request and API requests are created elsewhere.

    * :attr:`API_BASE_URL`
    * :attr:`API_VERSION`
    * :attr:`API_URL_TEMPLATE`

    must be updated by subclasses.
    """

    API_BASE_URL: Optional[str] = None
    """The base of the API call URL."""

    API_BASE_MTLS_URL: Optional[str] = None
    """The base of the API call URL for mutual TLS."""

    ALLOW_AUTO_SWITCH_TO_MTLS_URL = False
    """Indicates if auto switch to mTLS url is allowed."""

    API_VERSION: Optional[str] = None
    """The version of the API, used in building the API call's URL."""

    API_URL_TEMPLATE: Optional[str] = None
    """A template for the URL of a particular API call."""

    def get_api_base_url_for_mtls(self, api_base_url=None):
        """Return the api base url for mutual TLS.

        Typically, you shouldn't need to use this method.

        The logic is as follows:

        If `api_base_url` is provided, just return this value; otherwise, the
        return value depends `GOOGLE_API_USE_MTLS_ENDPOINT` environment variable
        value.

        If the environment variable value is "always", return `API_BASE_MTLS_URL`.
        If the environment variable value is "never", return `API_BASE_URL`.
        Otherwise, if `ALLOW_AUTO_SWITCH_TO_MTLS_URL` is True and the underlying
        http is mTLS, then return `API_BASE_MTLS_URL`; otherwise return `API_BASE_URL`.

        :type api_base_url: str
        :param api_base_url: User provided api base url. It takes precedence over
                             `API_BASE_URL` and `API_BASE_MTLS_URL`.

        :rtype: str
        :returns: The api base url used for mTLS.
        """
        if api_base_url:
            return api_base_url

        env = os.getenv("GOOGLE_API_USE_MTLS_ENDPOINT", "auto")
        if env == "always":
            url_to_use = self.API_BASE_MTLS_URL
        elif env == "never":
            url_to_use = self.API_BASE_URL
        else:
            if self.ALLOW_AUTO_SWITCH_TO_MTLS_URL:
                url_to_use = (
                    self.API_BASE_MTLS_URL if self.http.is_mtls else self.API_BASE_URL
                )
            else:
                url_to_use = self.API_BASE_URL
        return url_to_use

    def build_api_url(
        self, path, query_params=None, api_base_url=None, api_version=None
    ):
        """Construct an API url given a few components, some optional.

        Typically, you shouldn't need to use this method.

        :type path: str
        :param path: The path to the resource (ie, ``'/b/bucket-name'``).

        :type query_params: dict or list
        :param query_params: A dictionary of keys and values (or list of
                             key-value pairs) to insert into the query
                             string of the URL.

        :type api_base_url: str
        :param api_base_url: The base URL for the API endpoint.
                             Typically you won't have to provide this.

        :type api_version: str
        :param api_version: The version of the API to call.
                            Typically you shouldn't provide this and instead
                            use the default for the library.

        :rtype: str
        :returns: The URL assembled from the pieces provided.
        """
        url = self.API_URL_TEMPLATE.format(
            api_base_url=self.get_api_base_url_for_mtls(api_base_url),
            api_version=(api_version or self.API_VERSION),
            path=path,
        )

        query_params = query_params or {}

        if isinstance(query_params, collections.abc.Mapping):
            query_params = query_params.copy()
        else:
            query_params_dict = collections.defaultdict(list)
            for key, value in query_params:
                query_params_dict[key].append(value)
            query_params = query_params_dict

        query_params.setdefault("prettyPrint", "false")

        url += "?" + urlencode(query_params, doseq=True)

        return url

    def _make_request(
        self,
        method,
        url,
        data=None,
        content_type=None,
        headers=None,
        target_object=None,
        timeout=_DEFAULT_TIMEOUT,
        extra_api_info=None,
    ):
        """A low level method to send a request to the API.

        Typically, you shouldn't need to use this method.

        :type method: str
        :param method: The HTTP method to use in the request.

        :type url: str
        :param url: The URL to send the request to.

        :type data: str
        :param data: The data to send as the body of the request.

        :type content_type: str
        :param content_type: The proper MIME type of the data provided.

        :type headers: dict
        :param headers: (Optional) A dictionary of HTTP headers to send with
                        the request. If passed, will be modified directly
                        here with added headers.

        :type target_object: object
        :param target_object:
            (Optional) Argument to be used by library callers.  This can allow
            custom behavior, for example, to defer an HTTP request and complete
            initialization of the object at a later time.

        :type timeout: float or tuple
        :param timeout: (optional) The amount of time, in seconds, to wait
            for the server response.

            Can also be passed as a tuple (connect_timeout, read_timeout).
            See :meth:`requests.Session.request` documentation for details.

        :type extra_api_info: string
        :param extra_api_info: (optional) Extra api info to be appended to
            the X-Goog-API-Client header

        :rtype: :class:`requests.Response`
        :returns: The HTTP response.
        """
        headers = headers or {}
        headers.update(self.extra_headers)
        headers["Accept-Encoding"] = "gzip"

        if content_type:
            headers["Content-Type"] = content_type

        if extra_api_info:
            headers[CLIENT_INFO_HEADER] = f"{self.user_agent} {extra_api_info}"
        else:
            headers[CLIENT_INFO_HEADER] = self.user_agent
        headers["User-Agent"] = self.user_agent

        return self._do_request(
            method, url, headers, data, target_object, timeout=timeout
        )

    def _do_request(
        self, method, url, headers, data, target_object, timeout=_DEFAULT_TIMEOUT
    ):  # pylint: disable=unused-argument
        """Low-level helper:  perform the actual API request over HTTP.

        Allows batch context managers to override and defer a request.

        :type method: str
        :param method: The HTTP method to use in the request.

        :type url: str
        :param url: The URL to send the request to.

        :type headers: dict
        :param headers: A dictionary of HTTP headers to send with the request.

        :type data: str
        :param data: The data to send as the body of the request.

        :type target_object: object
        :param target_object:
            (Optional) Unused ``target_object`` here but may be used by a
            superclass.

        :type timeout: float or tuple
        :param timeout: (optional) The amount of time, in seconds, to wait
            for the server response.

            Can also be passed as a tuple (connect_timeout, read_timeout).
            See :meth:`requests.Session.request` documentation for details.

        :rtype: :class:`requests.Response`
        :returns: The HTTP response.
        """
        return self.http.request(
            url=url, method=method, headers=headers, data=data, timeout=timeout
        )

    def api_request(
        self,
        method,
        path,
        query_params=None,
        data=None,
        content_type=None,
        headers=None,
        api_base_url=None,
        api_version=None,
        expect_json=True,
        _target_object=None,
        timeout=_DEFAULT_TIMEOUT,
        extra_api_info=None,
    ):
        """Make a request over the HTTP transport to the API.

        You shouldn't need to use this method, but if you plan to
        interact with the API using these primitives, this is the
        correct one to use.

        :type method: str
        :param method: The HTTP method name (ie, ``GET``, ``POST``, etc).
                       Required.

        :type path: str
        :param path: The path to the resource (ie, ``'/b/bucket-name'``).
                     Required.

        :type query_params: dict or list
        :param query_params: A dictionary of keys and values (or list of
                             key-value pairs) to insert into the query
                             string of the URL.

        :type data: str
        :param data: The data to send as the body of the request. Default is
                     the empty string.

        :type content_type: str
        :param content_type: The proper MIME type of the data provided. Default
                             is None.

        :type headers: dict
        :param headers: extra HTTP headers to be sent with the request.

        :type api_base_url: str
        :param api_base_url: The base URL for the API endpoint.
                             Typically you won't have to provide this.
                             Default is the standard API base URL.

        :type api_version: str
        :param api_version: The version of the API to call.  Typically
                            you shouldn't provide this and instead use
                            the default for the library.  Default is the
                            latest API version supported by
                            google-cloud-python.

        :type expect_json: bool
        :param expect_json: If True, this method will try to parse the
                            response as JSON and raise an exception if
                            that cannot be done.  Default is True.

        :type _target_object: :class:`object`
        :param _target_object:
            (Optional) Protected argument to be used by library callers. This
            can allow custom behavior, for example, to defer an HTTP request
            and complete initialization of the object at a later time.

        :type timeout: float or tuple
        :param timeout: (optional) The amount of time, in seconds, to wait
            for the server response.

            Can also be passed as a tuple (connect_timeout, read_timeout).
            See :meth:`requests.Session.request` documentation for details.

        :type extra_api_info: string
        :param extra_api_info: (optional) Extra api info to be appended to
            the X-Goog-API-Client header

        :raises ~google.cloud.exceptions.GoogleCloudError: if the response code
            is not 200 OK.
        :raises ValueError: if the response content type is not JSON.
        :rtype: dict or str
        :returns: The API response payload, either as a raw string or
                  a dictionary if the response is valid JSON.
        """
        url = self.build_api_url(
            path=path,
            query_params=query_params,
            api_base_url=api_base_url,
            api_version=api_version,
        )

        # Making the executive decision that any dictionary
        # data will be sent properly as JSON.
        if data and isinstance(data, dict):
            data = json.dumps(data)
            content_type = "application/json"

        response = self._make_request(
            method=method,
            url=url,
            data=data,
            content_type=content_type,
            headers=headers,
            target_object=_target_object,
            timeout=timeout,
            extra_api_info=extra_api_info,
        )

        if not 200 <= response.status_code < 300:
            raise exceptions.from_http_response(response)

        if expect_json and response.content:
            return response.json()
        else:
            return response.content
