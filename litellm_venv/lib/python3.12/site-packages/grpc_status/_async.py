# Copyright 2020 The gRPC Authors
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

from google.rpc import status_pb2
from grpc.experimental import aio

from ._common import GRPC_DETAILS_METADATA_KEY
from ._common import code_to_grpc_status_code


async def from_call(call: aio.Call):
    """Returns a google.rpc.status.Status message from a given grpc.aio.Call.

    This is an EXPERIMENTAL API.

    Args:
      call: An grpc.aio.Call instance.

    Returns:
      A google.rpc.status.Status message representing the status of the RPC.
    """
    code = await call.code()
    details = await call.details()
    trailing_metadata = await call.trailing_metadata()
    if trailing_metadata is None:
        return None
    for key, value in trailing_metadata:
        if key == GRPC_DETAILS_METADATA_KEY:
            rich_status = status_pb2.Status.FromString(value)
            if code.value[0] != rich_status.code:
                raise ValueError(
                    "Code in Status proto (%s) doesn't match status code (%s)"
                    % (code_to_grpc_status_code(rich_status.code), code)
                )
            if details != rich_status.message:
                raise ValueError(
                    "Message in Status proto (%s) doesn't match status details"
                    " (%s)" % (rich_status.message, details)
                )
            return rich_status
    return None


__all__ = [
    "from_call",
]
