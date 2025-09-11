# Copyright 2017 Google LLC
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


import base64
import binascii
import collections
import datetime
import hashlib
import json

import http
import urllib

import google.auth.credentials

from google.auth import exceptions
from google.auth.transport import requests
from google.cloud import _helpers
from google.cloud.storage._helpers import _DEFAULT_UNIVERSE_DOMAIN
from google.cloud.storage._helpers import _NOW
from google.cloud.storage._helpers import _UTC
from google.cloud.storage.retry import DEFAULT_RETRY


# `google.cloud.storage._signing.NOW` is deprecated.
# Use `_NOW(_UTC)` instead.
NOW = datetime.datetime.utcnow

SERVICE_ACCOUNT_URL = (
    "https://googleapis.dev/python/google-api-core/latest/"
    "auth.html#setting-up-a-service-account"
)


def ensure_signed_credentials(credentials):
    """Raise AttributeError if the credentials are unsigned.

    :type credentials: :class:`google.auth.credentials.Signing`
    :param credentials: The credentials used to create a private key
                        for signing text.

    :raises: :exc:`AttributeError` if credentials is not an instance
            of :class:`google.auth.credentials.Signing`.
    """
    if not isinstance(credentials, google.auth.credentials.Signing):
        raise AttributeError(
            "you need a private key to sign credentials."
            "the credentials you are currently using {} "
            "just contains a token. see {} for more "
            "details.".format(type(credentials), SERVICE_ACCOUNT_URL)
        )


def get_signed_query_params_v2(credentials, expiration, string_to_sign):
    """Gets query parameters for creating a signed URL.

    :type credentials: :class:`google.auth.credentials.Signing`
    :param credentials: The credentials used to create a private key
                        for signing text.

    :type expiration: int or long
    :param expiration: When the signed URL should expire.

    :type string_to_sign: str
    :param string_to_sign: The string to be signed by the credentials.

    :raises: :exc:`AttributeError` if credentials is not an instance
            of :class:`google.auth.credentials.Signing`.

    :rtype: dict
    :returns: Query parameters matching the signing credentials with a
              signed payload.
    """
    ensure_signed_credentials(credentials)
    signature_bytes = credentials.sign_bytes(string_to_sign.encode("ascii"))
    signature = base64.b64encode(signature_bytes)
    service_account_name = credentials.signer_email
    return {
        "GoogleAccessId": service_account_name,
        "Expires": expiration,
        "Signature": signature,
    }


def get_expiration_seconds_v2(expiration):
    """Convert 'expiration' to a number of seconds in the future.

    :type expiration: Union[Integer, datetime.datetime, datetime.timedelta]
    :param expiration: Point in time when the signed URL should expire. If
                       a ``datetime`` instance is passed without an explicit
                       ``tzinfo`` set,  it will be assumed to be ``UTC``.

    :raises: :exc:`TypeError` when expiration is not a valid type.

    :rtype: int
    :returns: a timestamp as an absolute number of seconds since epoch.
    """
    # If it's a timedelta, add it to `now` in UTC.
    if isinstance(expiration, datetime.timedelta):
        now = _NOW(_UTC)
        expiration = now + expiration

    # If it's a datetime, convert to a timestamp.
    if isinstance(expiration, datetime.datetime):
        micros = _helpers._microseconds_from_datetime(expiration)
        expiration = micros // 10**6

    if not isinstance(expiration, int):
        raise TypeError(
            "Expected an integer timestamp, datetime, or "
            "timedelta. Got %s" % type(expiration)
        )
    return expiration


_EXPIRATION_TYPES = (int, datetime.datetime, datetime.timedelta)


def get_expiration_seconds_v4(expiration):
    """Convert 'expiration' to a number of seconds offset from the current time.

    :type expiration: Union[Integer, datetime.datetime, datetime.timedelta]
    :param expiration: Point in time when the signed URL should expire. If
                       a ``datetime`` instance is passed without an explicit
                       ``tzinfo`` set,  it will be assumed to be ``UTC``.

    :raises: :exc:`TypeError` when expiration is not a valid type.
    :raises: :exc:`ValueError` when expiration is too large.
    :rtype: Integer
    :returns: seconds in the future when the signed URL will expire
    """
    if not isinstance(expiration, _EXPIRATION_TYPES):
        raise TypeError(
            "Expected an integer timestamp, datetime, or "
            "timedelta. Got %s" % type(expiration)
        )

    now = _NOW(_UTC)

    if isinstance(expiration, int):
        seconds = expiration

    if isinstance(expiration, datetime.datetime):
        if expiration.tzinfo is None:
            expiration = expiration.replace(tzinfo=_helpers.UTC)
        expiration = expiration - now

    if isinstance(expiration, datetime.timedelta):
        seconds = int(expiration.total_seconds())

    if seconds > SEVEN_DAYS:
        raise ValueError(f"Max allowed expiration interval is seven days {SEVEN_DAYS}")

    return seconds


def get_canonical_headers(headers):
    """Canonicalize headers for signing.

    See:
    https://cloud.google.com/storage/docs/access-control/signed-urls#about-canonical-extension-headers

    :type headers: Union[dict|List(Tuple(str,str))]
    :param headers:
        (Optional) Additional HTTP headers to be included as part of the
        signed URLs.  See:
        https://cloud.google.com/storage/docs/xml-api/reference-headers
        Requests using the signed URL *must* pass the specified header
        (name and value) with each request for the URL.

    :rtype: str
    :returns: List of headers, normalized / sortted per the URL refernced above.
    """
    if headers is None:
        headers = []
    elif isinstance(headers, dict):
        headers = list(headers.items())

    if not headers:
        return [], []

    normalized = collections.defaultdict(list)
    for key, val in headers:
        key = key.lower().strip()
        val = " ".join(val.split())
        normalized[key].append(val)

    ordered_headers = sorted((key, ",".join(val)) for key, val in normalized.items())

    canonical_headers = ["{}:{}".format(*item) for item in ordered_headers]
    return canonical_headers, ordered_headers


_Canonical = collections.namedtuple(
    "_Canonical", ["method", "resource", "query_parameters", "headers"]
)


def canonicalize_v2(method, resource, query_parameters, headers):
    """Canonicalize method, resource per the V2 spec.

    :type method: str
    :param method: The HTTP verb that will be used when requesting the URL.
                   Defaults to ``'GET'``. If method is ``'RESUMABLE'`` then the
                   signature will additionally contain the `x-goog-resumable`
                   header, and the method changed to POST. See the signed URL
                   docs regarding this flow:
                   https://cloud.google.com/storage/docs/access-control/signed-urls

    :type resource: str
    :param resource: A pointer to a specific resource
                     (typically, ``/bucket-name/path/to/blob.txt``).

    :type query_parameters: dict
    :param query_parameters:
        (Optional) Additional query parameters to be included as part of the
        signed URLs.  See:
        https://cloud.google.com/storage/docs/xml-api/reference-headers#query

    :type headers: Union[dict|List(Tuple(str,str))]
    :param headers:
        (Optional) Additional HTTP headers to be included as part of the
        signed URLs.  See:
        https://cloud.google.com/storage/docs/xml-api/reference-headers
        Requests using the signed URL *must* pass the specified header
        (name and value) with each request for the URL.

    :rtype: :class:_Canonical
    :returns: Canonical method, resource, query_parameters, and headers.
    """
    headers, _ = get_canonical_headers(headers)

    if method == "RESUMABLE":
        method = "POST"
        headers.append("x-goog-resumable:start")

    if query_parameters is None:
        return _Canonical(method, resource, [], headers)

    normalized_qp = sorted(
        (key.lower(), value and value.strip() or "")
        for key, value in query_parameters.items()
    )
    encoded_qp = urllib.parse.urlencode(normalized_qp)
    canonical_resource = f"{resource}?{encoded_qp}"
    return _Canonical(method, canonical_resource, normalized_qp, headers)


def generate_signed_url_v2(
    credentials,
    resource,
    expiration,
    api_access_endpoint="",
    method="GET",
    content_md5=None,
    content_type=None,
    response_type=None,
    response_disposition=None,
    generation=None,
    headers=None,
    query_parameters=None,
    service_account_email=None,
    access_token=None,
    universe_domain=None,
):
    """Generate a V2 signed URL to provide query-string auth'n to a resource.

    .. note::

        Assumes ``credentials`` implements the
        :class:`google.auth.credentials.Signing` interface. Also assumes
        ``credentials`` has a ``signer_email`` property which
        identifies the credentials.

    .. note::

        If you are on Google Compute Engine, you can't generate a signed URL.
        If you'd like to be able to generate a signed URL from GCE, you can use a
        standard service account from a JSON file rather than a GCE service account.

    See headers [reference](https://cloud.google.com/storage/docs/reference-headers)
    for more details on optional arguments.

    :type credentials: :class:`google.auth.credentials.Signing`
    :param credentials: Credentials object with an associated private key to
                        sign text.

    :type resource: str
    :param resource: A pointer to a specific resource
                     (typically, ``/bucket-name/path/to/blob.txt``).
                     Caller should have already URL-encoded the value.

    :type expiration: Union[Integer, datetime.datetime, datetime.timedelta]
    :param expiration: Point in time when the signed URL should expire. If
                       a ``datetime`` instance is passed without an explicit
                       ``tzinfo`` set,  it will be assumed to be ``UTC``.

    :type api_access_endpoint: str
    :param api_access_endpoint: (Optional) URI base. Defaults to empty string.

    :type method: str
    :param method: The HTTP verb that will be used when requesting the URL.
                   Defaults to ``'GET'``. If method is ``'RESUMABLE'`` then the
                   signature will additionally contain the `x-goog-resumable`
                   header, and the method changed to POST. See the signed URL
                   docs regarding this flow:
                   https://cloud.google.com/storage/docs/access-control/signed-urls


    :type content_md5: str
    :param content_md5: (Optional) The MD5 hash of the object referenced by
                        ``resource``.

    :type content_type: str
    :param content_type: (Optional) The content type of the object referenced
                         by ``resource``.

    :type response_type: str
    :param response_type: (Optional) Content type of responses to requests for
                          the signed URL. Ignored if content_type is set on
                          object/blob metadata.

    :type response_disposition: str
    :param response_disposition: (Optional) Content disposition of responses to
                                 requests for the signed URL.

    :type generation: str
    :param generation: (Optional) A value that indicates which generation of
                       the resource to fetch.

    :type headers: Union[dict|List(Tuple(str,str))]
    :param headers:
        (Optional) Additional HTTP headers to be included as part of the
        signed URLs.  See:
        https://cloud.google.com/storage/docs/xml-api/reference-headers
        Requests using the signed URL *must* pass the specified header
        (name and value) with each request for the URL.

    :type service_account_email: str
    :param service_account_email: (Optional) E-mail address of the service account.

    :type access_token: str
    :param access_token: (Optional) Access token for a service account.

    :type query_parameters: dict
    :param query_parameters:
        (Optional) Additional query parameters to be included as part of the
        signed URLs.  See:
        https://cloud.google.com/storage/docs/xml-api/reference-headers#query

    :raises: :exc:`TypeError` when expiration is not a valid type.
    :raises: :exc:`AttributeError` if credentials is not an instance
            of :class:`google.auth.credentials.Signing`.

    :rtype: str
    :returns: A signed URL you can use to access the resource
              until expiration.
    """
    expiration_stamp = get_expiration_seconds_v2(expiration)

    canonical = canonicalize_v2(method, resource, query_parameters, headers)

    # Generate the string to sign.
    elements_to_sign = [
        canonical.method,
        content_md5 or "",
        content_type or "",
        str(expiration_stamp),
    ]
    elements_to_sign.extend(canonical.headers)
    elements_to_sign.append(canonical.resource)
    string_to_sign = "\n".join(elements_to_sign)

    # If you are on Google Compute Engine, you can't generate a signed URL.
    # See https://github.com/googleapis/google-cloud-python/issues/922
    # Set the right query parameters.
    if access_token and service_account_email:
        signature = _sign_message(
            string_to_sign, access_token, service_account_email, universe_domain
        )
        signed_query_params = {
            "GoogleAccessId": service_account_email,
            "Expires": expiration_stamp,
            "Signature": signature,
        }
    else:
        signed_query_params = get_signed_query_params_v2(
            credentials, expiration_stamp, string_to_sign
        )

    if response_type is not None:
        signed_query_params["response-content-type"] = response_type
    if response_disposition is not None:
        signed_query_params["response-content-disposition"] = response_disposition
    if generation is not None:
        signed_query_params["generation"] = generation

    signed_query_params.update(canonical.query_parameters)
    sorted_signed_query_params = sorted(signed_query_params.items())

    # Return the built URL.
    return "{endpoint}{resource}?{querystring}".format(
        endpoint=api_access_endpoint,
        resource=resource,
        querystring=urllib.parse.urlencode(sorted_signed_query_params),
    )


SEVEN_DAYS = 7 * 24 * 60 * 60  # max age for V4 signed URLs.
DEFAULT_ENDPOINT = "https://storage.googleapis.com"


def generate_signed_url_v4(
    credentials,
    resource,
    expiration,
    api_access_endpoint=DEFAULT_ENDPOINT,
    method="GET",
    content_md5=None,
    content_type=None,
    response_type=None,
    response_disposition=None,
    generation=None,
    headers=None,
    query_parameters=None,
    service_account_email=None,
    access_token=None,
    universe_domain=None,
    _request_timestamp=None,  # for testing only
):
    """Generate a V4 signed URL to provide query-string auth'n to a resource.

    .. note::

        Assumes ``credentials`` implements the
        :class:`google.auth.credentials.Signing` interface. Also assumes
        ``credentials`` has a ``signer_email`` property which
        identifies the credentials.

    .. note::

        If you are on Google Compute Engine, you can't generate a signed URL.
        If you'd like to be able to generate a signed URL from GCE,you can use a
        standard service account from a JSON file rather than a GCE service account.

    See headers [reference](https://cloud.google.com/storage/docs/reference-headers)
    for more details on optional arguments.

    :type credentials: :class:`google.auth.credentials.Signing`
    :param credentials: Credentials object with an associated private key to
                        sign text. That credentials must provide signer_email
                        only if service_account_email and access_token are not
                        passed.

    :type resource: str
    :param resource: A pointer to a specific resource
                     (typically, ``/bucket-name/path/to/blob.txt``).
                     Caller should have already URL-encoded the value.

    :type expiration: Union[Integer, datetime.datetime, datetime.timedelta]
    :param expiration: Point in time when the signed URL should expire. If
                       a ``datetime`` instance is passed without an explicit
                       ``tzinfo`` set,  it will be assumed to be ``UTC``.

    :type api_access_endpoint: str
    :param api_access_endpoint: URI base. Defaults to
                                "https://storage.googleapis.com/"

    :type method: str
    :param method: The HTTP verb that will be used when requesting the URL.
                   Defaults to ``'GET'``. If method is ``'RESUMABLE'`` then the
                   signature will additionally contain the `x-goog-resumable`
                   header, and the method changed to POST. See the signed URL
                   docs regarding this flow:
                   https://cloud.google.com/storage/docs/access-control/signed-urls


    :type content_md5: str
    :param content_md5: (Optional) The MD5 hash of the object referenced by
                        ``resource``.

    :type content_type: str
    :param content_type: (Optional) The content type of the object referenced
                         by ``resource``.

    :type response_type: str
    :param response_type: (Optional) Content type of responses to requests for
                          the signed URL. Ignored if content_type is set on
                          object/blob metadata.

    :type response_disposition: str
    :param response_disposition: (Optional) Content disposition of responses to
                                 requests for the signed URL.

    :type generation: str
    :param generation: (Optional) A value that indicates which generation of
                       the resource to fetch.

    :type headers: dict
    :param headers:
        (Optional) Additional HTTP headers to be included as part of the
        signed URLs.  See:
        https://cloud.google.com/storage/docs/xml-api/reference-headers
        Requests using the signed URL *must* pass the specified header
        (name and value) with each request for the URL.

    :type query_parameters: dict
    :param query_parameters:
        (Optional) Additional query parameters to be included as part of the
        signed URLs.  See:
        https://cloud.google.com/storage/docs/xml-api/reference-headers#query

    :type service_account_email: str
    :param service_account_email: (Optional) E-mail address of the service account.

    :type access_token: str
    :param access_token: (Optional) Access token for a service account.

    :raises: :exc:`TypeError` when expiration is not a valid type.
    :raises: :exc:`AttributeError` if credentials is not an instance
            of :class:`google.auth.credentials.Signing`.

    :rtype: str
    :returns: A signed URL you can use to access the resource
              until expiration.
    """
    expiration_seconds = get_expiration_seconds_v4(expiration)

    if _request_timestamp is None:
        request_timestamp, datestamp = get_v4_now_dtstamps()
    else:
        request_timestamp = _request_timestamp
        datestamp = _request_timestamp[:8]

    # If you are on Google Compute Engine, you can't generate a signed URL.
    # See https://github.com/googleapis/google-cloud-python/issues/922
    client_email = service_account_email
    if not access_token or not service_account_email:
        ensure_signed_credentials(credentials)
        client_email = credentials.signer_email

    credential_scope = f"{datestamp}/auto/storage/goog4_request"
    credential = f"{client_email}/{credential_scope}"

    if headers is None:
        headers = {}

    if content_type is not None:
        headers["Content-Type"] = content_type

    if content_md5 is not None:
        headers["Content-MD5"] = content_md5

    header_names = [key.lower() for key in headers]
    if "host" not in header_names:
        headers["Host"] = urllib.parse.urlparse(api_access_endpoint).netloc

    if method.upper() == "RESUMABLE":
        method = "POST"
        headers["x-goog-resumable"] = "start"

    canonical_headers, ordered_headers = get_canonical_headers(headers)
    canonical_header_string = (
        "\n".join(canonical_headers) + "\n"
    )  # Yes, Virginia, the extra newline is part of the spec.
    signed_headers = ";".join([key for key, _ in ordered_headers])

    if query_parameters is None:
        query_parameters = {}
    else:
        query_parameters = {key: value or "" for key, value in query_parameters.items()}

    query_parameters["X-Goog-Algorithm"] = "GOOG4-RSA-SHA256"
    query_parameters["X-Goog-Credential"] = credential
    query_parameters["X-Goog-Date"] = request_timestamp
    query_parameters["X-Goog-Expires"] = expiration_seconds
    query_parameters["X-Goog-SignedHeaders"] = signed_headers

    if response_type is not None:
        query_parameters["response-content-type"] = response_type

    if response_disposition is not None:
        query_parameters["response-content-disposition"] = response_disposition

    if generation is not None:
        query_parameters["generation"] = generation

    canonical_query_string = _url_encode(query_parameters)

    lowercased_headers = dict(ordered_headers)

    if "x-goog-content-sha256" in lowercased_headers:
        payload = lowercased_headers["x-goog-content-sha256"]
    else:
        payload = "UNSIGNED-PAYLOAD"

    canonical_elements = [
        method,
        resource,
        canonical_query_string,
        canonical_header_string,
        signed_headers,
        payload,
    ]
    canonical_request = "\n".join(canonical_elements)

    canonical_request_hash = hashlib.sha256(
        canonical_request.encode("ascii")
    ).hexdigest()

    string_elements = [
        "GOOG4-RSA-SHA256",
        request_timestamp,
        credential_scope,
        canonical_request_hash,
    ]
    string_to_sign = "\n".join(string_elements)

    if access_token and service_account_email:
        signature = _sign_message(
            string_to_sign, access_token, service_account_email, universe_domain
        )
        signature_bytes = base64.b64decode(signature)
        signature = binascii.hexlify(signature_bytes).decode("ascii")
    else:
        signature_bytes = credentials.sign_bytes(string_to_sign.encode("ascii"))
        signature = binascii.hexlify(signature_bytes).decode("ascii")

    return "{}{}?{}&X-Goog-Signature={}".format(
        api_access_endpoint, resource, canonical_query_string, signature
    )


def get_v4_now_dtstamps():
    """Get current timestamp and datestamp in V4 valid format.

    :rtype: str, str
    :returns: Current timestamp, datestamp.
    """
    now = _NOW(_UTC).replace(tzinfo=None)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    datestamp = now.date().strftime("%Y%m%d")
    return timestamp, datestamp


def _sign_message(
    message,
    access_token,
    service_account_email,
    universe_domain=_DEFAULT_UNIVERSE_DOMAIN,
):
    """Signs a message.

    :type message: str
    :param message: The message to be signed.

    :type access_token: str
    :param access_token: Access token for a service account.


    :type service_account_email: str
    :param service_account_email: E-mail address of the service account.

    :raises: :exc:`TransportError` if an `access_token` is unauthorized.

    :rtype: str
    :returns: The signature of the message.

    """
    message = _helpers._to_bytes(message)

    method = "POST"
    url = f"https://iamcredentials.{universe_domain}/v1/projects/-/serviceAccounts/{service_account_email}:signBlob?alt=json"
    headers = {
        "Authorization": "Bearer " + access_token,
        "Content-type": "application/json",
    }
    body = json.dumps({"payload": base64.b64encode(message).decode("utf-8")})
    request = requests.Request()

    def retriable_request():
        response = request(url=url, method=method, body=body, headers=headers)
        return response

    # Apply the default retry object to the signBlob call.
    retry = DEFAULT_RETRY
    call = retry(retriable_request)
    response = call()

    if response.status != http.client.OK:
        raise exceptions.TransportError(
            f"Error calling the IAM signBytes API: {response.data}"
        )

    data = json.loads(response.data.decode("utf-8"))
    return data["signedBlob"]


def _url_encode(query_params):
    """Encode query params into URL.

    :type query_params: dict
    :param query_params: Query params to be encoded.

    :rtype: str
    :returns: URL encoded query params.
    """
    params = [
        f"{_quote_param(name)}={_quote_param(value)}"
        for name, value in query_params.items()
    ]

    return "&".join(sorted(params))


def _quote_param(param):
    """Quote query param.

    :type param: Any
    :param param: Query param to be encoded.

    :rtype: str
    :returns: URL encoded query param.
    """
    if not isinstance(param, bytes):
        param = str(param)
    return urllib.parse.quote(param, safe="~")
