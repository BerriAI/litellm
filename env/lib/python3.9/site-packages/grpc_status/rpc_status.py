# Copyright 2018 The gRPC Authors
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
"""Reference implementation for status mapping in gRPC Python."""

import collections
import sys

from google.rpc import status_pb2
import grpc

from ._common import GRPC_DETAILS_METADATA_KEY
from ._common import code_to_grpc_status_code


class _Status(
    collections.namedtuple("_Status", ("code", "details", "trailing_metadata")),
    grpc.Status,
):
    pass


def from_call(call):
    """Returns a google.rpc.status.Status message corresponding to a given grpc.Call.

    This is an EXPERIMENTAL API.

    Args:
      call: A grpc.Call instance.

    Returns:
      A google.rpc.status.Status message representing the status of the RPC.

    Raises:
      ValueError: If the gRPC call's code or details are inconsistent with the
        status code and message inside of the google.rpc.status.Status.
    """
    if call.trailing_metadata() is None:
        return None
    for key, value in call.trailing_metadata():
        if key == GRPC_DETAILS_METADATA_KEY:
            rich_status = status_pb2.Status.FromString(value)
            if call.code().value[0] != rich_status.code:
                raise ValueError(
                    "Code in Status proto (%s) doesn't match status code (%s)"
                    % (code_to_grpc_status_code(rich_status.code), call.code())
                )
            if call.details() != rich_status.message:
                raise ValueError(
                    "Message in Status proto (%s) doesn't match status details"
                    " (%s)" % (rich_status.message, call.details())
                )
            return rich_status
    return None


def to_status(status):
    """Convert a google.rpc.status.Status message to grpc.Status.

    This is an EXPERIMENTAL API.

    Args:
      status: a google.rpc.status.Status message representing the non-OK status
        to terminate the RPC with and communicate it to the client.

    Returns:
      A grpc.Status instance representing the input google.rpc.status.Status message.
    """
    return _Status(
        code=code_to_grpc_status_code(status.code),
        details=status.message,
        trailing_metadata=(
            (GRPC_DETAILS_METADATA_KEY, status.SerializeToString()),
        ),
    )


__all__ = [
    "from_call",
    "to_status",
]

if sys.version_info[0] >= 3 and sys.version_info[1] >= 6:
    from . import _async as aio  # pylint: disable=unused-import

    __all__.append("aio")
