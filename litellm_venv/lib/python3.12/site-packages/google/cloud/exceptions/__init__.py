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

# pylint: disable=invalid-name
# pylint recognizies all of these aliases as constants and thinks they have
# invalid names.

"""Custom exceptions for :mod:`google.cloud` package."""

# Avoid the grpc and google.cloud.grpc collision.
from __future__ import absolute_import

from google.api_core import exceptions

try:
    from grpc._channel import _Rendezvous
except ImportError:  # pragma: NO COVER
    _Rendezvous = None

GrpcRendezvous = _Rendezvous
"""Exception class raised by gRPC stable."""

# Aliases to moved classes.
GoogleCloudError = exceptions.GoogleAPICallError
Redirection = exceptions.Redirection
MovedPermanently = exceptions.MovedPermanently
NotModified = exceptions.NotModified
TemporaryRedirect = exceptions.TemporaryRedirect
ResumeIncomplete = exceptions.ResumeIncomplete
ClientError = exceptions.ClientError
BadRequest = exceptions.BadRequest
Unauthorized = exceptions.Unauthorized
Forbidden = exceptions.Forbidden
NotFound = exceptions.NotFound
MethodNotAllowed = exceptions.MethodNotAllowed
Conflict = exceptions.Conflict
LengthRequired = exceptions.LengthRequired
PreconditionFailed = exceptions.PreconditionFailed
RequestRangeNotSatisfiable = exceptions.RequestRangeNotSatisfiable
TooManyRequests = exceptions.TooManyRequests
ServerError = exceptions.ServerError
InternalServerError = exceptions.InternalServerError
MethodNotImplemented = exceptions.MethodNotImplemented
BadGateway = exceptions.BadGateway
ServiceUnavailable = exceptions.ServiceUnavailable
GatewayTimeout = exceptions.GatewayTimeout
from_http_status = exceptions.from_http_status
from_http_response = exceptions.from_http_response
