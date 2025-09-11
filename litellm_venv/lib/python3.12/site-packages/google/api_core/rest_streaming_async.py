# Copyright 2024 Google LLC
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

"""Helpers for asynchronous server-side streaming in REST."""

from typing import Union

import proto

try:
    import google.auth.aio.transport
except ImportError as e:  # pragma: NO COVER
    raise ImportError(
        "`google-api-core[async_rest]` is required to use asynchronous rest streaming. "
        "Install the `async_rest` extra of `google-api-core` using "
        "`pip install google-api-core[async_rest]`."
    ) from e

import google.protobuf.message
from google.api_core._rest_streaming_base import BaseResponseIterator


class AsyncResponseIterator(BaseResponseIterator):
    """Asynchronous Iterator over REST API responses.

    Args:
        response (google.auth.aio.transport.Response): An API response object.
        response_message_cls (Union[proto.Message, google.protobuf.message.Message]): A response
        class expected to be returned from an API.

    Raises:
        ValueError:
            - If `response_message_cls` is not a subclass of `proto.Message` or `google.protobuf.message.Message`.
    """

    def __init__(
        self,
        response: google.auth.aio.transport.Response,
        response_message_cls: Union[proto.Message, google.protobuf.message.Message],
    ):
        self._response = response
        self._chunk_size = 1024
        # TODO(https://github.com/googleapis/python-api-core/issues/703): mypy does not recognize the abstract content
        # method as an async generator as it looks for the `yield` keyword in the implementation.
        # Given that the abstract method is not implemented, mypy fails to recognize it as an async generator.
        # mypy warnings are silenced until the linked issue is resolved.
        self._response_itr = self._response.content(self._chunk_size).__aiter__()  # type: ignore
        super(AsyncResponseIterator, self).__init__(
            response_message_cls=response_message_cls
        )

    async def __aenter__(self):
        return self

    async def cancel(self):
        """Cancel existing streaming operation."""
        await self._response.close()

    async def __anext__(self):
        while not self._ready_objs:
            try:
                chunk = await self._response_itr.__anext__()
                chunk = chunk.decode("utf-8")
                self._process_chunk(chunk)
            except StopAsyncIteration as e:
                if self._level > 0:
                    raise ValueError("i Unfinished stream: %s" % self._obj)
                raise e
            except ValueError as e:
                raise e
        return self._grab()

    def __aiter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Cancel existing async streaming operation."""
        await self._response.close()
