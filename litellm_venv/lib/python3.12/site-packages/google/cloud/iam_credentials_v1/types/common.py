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
from __future__ import annotations

from typing import MutableMapping, MutableSequence

from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
import proto  # type: ignore

__protobuf__ = proto.module(
    package="google.iam.credentials.v1",
    manifest={
        "GenerateAccessTokenRequest",
        "GenerateAccessTokenResponse",
        "SignBlobRequest",
        "SignBlobResponse",
        "SignJwtRequest",
        "SignJwtResponse",
        "GenerateIdTokenRequest",
        "GenerateIdTokenResponse",
    },
)


class GenerateAccessTokenRequest(proto.Message):
    r"""

    Attributes:
        name (str):
            Required. The resource name of the service account for which
            the credentials are requested, in the following format:
            ``projects/-/serviceAccounts/{ACCOUNT_EMAIL_OR_UNIQUEID}``.
            The ``-`` wildcard character is required; replacing it with
            a project ID is invalid.
        delegates (MutableSequence[str]):
            The sequence of service accounts in a delegation chain. Each
            service account must be granted the
            ``roles/iam.serviceAccountTokenCreator`` role on its next
            service account in the chain. The last service account in
            the chain must be granted the
            ``roles/iam.serviceAccountTokenCreator`` role on the service
            account that is specified in the ``name`` field of the
            request.

            The delegates must have the following format:
            ``projects/-/serviceAccounts/{ACCOUNT_EMAIL_OR_UNIQUEID}``.
            The ``-`` wildcard character is required; replacing it with
            a project ID is invalid.
        scope (MutableSequence[str]):
            Required. Code to identify the scopes to be
            included in the OAuth 2.0 access token. See
            https://developers.google.com/identity/protocols/googlescopes
            for more information.
            At least one value required.
        lifetime (google.protobuf.duration_pb2.Duration):
            The desired lifetime duration of the access
            token in seconds. Must be set to a value less
            than or equal to 3600 (1 hour). If a value is
            not specified, the token's lifetime will be set
            to a default value of one hour.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    delegates: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=2,
    )
    scope: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=4,
    )
    lifetime: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=7,
        message=duration_pb2.Duration,
    )


class GenerateAccessTokenResponse(proto.Message):
    r"""

    Attributes:
        access_token (str):
            The OAuth 2.0 access token.
        expire_time (google.protobuf.timestamp_pb2.Timestamp):
            Token expiration time.
            The expiration time is always set.
    """

    access_token: str = proto.Field(
        proto.STRING,
        number=1,
    )
    expire_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )


class SignBlobRequest(proto.Message):
    r"""

    Attributes:
        name (str):
            Required. The resource name of the service account for which
            the credentials are requested, in the following format:
            ``projects/-/serviceAccounts/{ACCOUNT_EMAIL_OR_UNIQUEID}``.
            The ``-`` wildcard character is required; replacing it with
            a project ID is invalid.
        delegates (MutableSequence[str]):
            The sequence of service accounts in a delegation chain. Each
            service account must be granted the
            ``roles/iam.serviceAccountTokenCreator`` role on its next
            service account in the chain. The last service account in
            the chain must be granted the
            ``roles/iam.serviceAccountTokenCreator`` role on the service
            account that is specified in the ``name`` field of the
            request.

            The delegates must have the following format:
            ``projects/-/serviceAccounts/{ACCOUNT_EMAIL_OR_UNIQUEID}``.
            The ``-`` wildcard character is required; replacing it with
            a project ID is invalid.
        payload (bytes):
            Required. The bytes to sign.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    delegates: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=3,
    )
    payload: bytes = proto.Field(
        proto.BYTES,
        number=5,
    )


class SignBlobResponse(proto.Message):
    r"""

    Attributes:
        key_id (str):
            The ID of the key used to sign the blob.
        signed_blob (bytes):
            The signed blob.
    """

    key_id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    signed_blob: bytes = proto.Field(
        proto.BYTES,
        number=4,
    )


class SignJwtRequest(proto.Message):
    r"""

    Attributes:
        name (str):
            Required. The resource name of the service account for which
            the credentials are requested, in the following format:
            ``projects/-/serviceAccounts/{ACCOUNT_EMAIL_OR_UNIQUEID}``.
            The ``-`` wildcard character is required; replacing it with
            a project ID is invalid.
        delegates (MutableSequence[str]):
            The sequence of service accounts in a delegation chain. Each
            service account must be granted the
            ``roles/iam.serviceAccountTokenCreator`` role on its next
            service account in the chain. The last service account in
            the chain must be granted the
            ``roles/iam.serviceAccountTokenCreator`` role on the service
            account that is specified in the ``name`` field of the
            request.

            The delegates must have the following format:
            ``projects/-/serviceAccounts/{ACCOUNT_EMAIL_OR_UNIQUEID}``.
            The ``-`` wildcard character is required; replacing it with
            a project ID is invalid.
        payload (str):
            Required. The JWT payload to sign: a JSON
            object that contains a JWT Claims Set.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    delegates: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=3,
    )
    payload: str = proto.Field(
        proto.STRING,
        number=5,
    )


class SignJwtResponse(proto.Message):
    r"""

    Attributes:
        key_id (str):
            The ID of the key used to sign the JWT.
        signed_jwt (str):
            The signed JWT.
    """

    key_id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    signed_jwt: str = proto.Field(
        proto.STRING,
        number=2,
    )


class GenerateIdTokenRequest(proto.Message):
    r"""

    Attributes:
        name (str):
            Required. The resource name of the service account for which
            the credentials are requested, in the following format:
            ``projects/-/serviceAccounts/{ACCOUNT_EMAIL_OR_UNIQUEID}``.
            The ``-`` wildcard character is required; replacing it with
            a project ID is invalid.
        delegates (MutableSequence[str]):
            The sequence of service accounts in a delegation chain. Each
            service account must be granted the
            ``roles/iam.serviceAccountTokenCreator`` role on its next
            service account in the chain. The last service account in
            the chain must be granted the
            ``roles/iam.serviceAccountTokenCreator`` role on the service
            account that is specified in the ``name`` field of the
            request.

            The delegates must have the following format:
            ``projects/-/serviceAccounts/{ACCOUNT_EMAIL_OR_UNIQUEID}``.
            The ``-`` wildcard character is required; replacing it with
            a project ID is invalid.
        audience (str):
            Required. The audience for the token, such as
            the API or account that this token grants access
            to.
        include_email (bool):
            Include the service account email in the token. If set to
            ``true``, the token will contain ``email`` and
            ``email_verified`` claims.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    delegates: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=2,
    )
    audience: str = proto.Field(
        proto.STRING,
        number=3,
    )
    include_email: bool = proto.Field(
        proto.BOOL,
        number=4,
    )


class GenerateIdTokenResponse(proto.Message):
    r"""

    Attributes:
        token (str):
            The OpenId Connect ID token.
    """

    token: str = proto.Field(
        proto.STRING,
        number=1,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
