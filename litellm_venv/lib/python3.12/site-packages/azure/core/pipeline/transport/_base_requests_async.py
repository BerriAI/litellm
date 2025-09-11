# --------------------------------------------------------------------------
#
# Copyright (c) Microsoft Corporation. All rights reserved.
#
# The MIT License (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the ""Software""), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#
# --------------------------------------------------------------------------
from typing import Optional, Type
from types import TracebackType
from ._requests_basic import RequestsTransport
from ._base_async import AsyncHttpTransport


class RequestsAsyncTransportBase(RequestsTransport, AsyncHttpTransport):  # type: ignore
    async def _retrieve_request_data(self, request):
        if hasattr(request.data, "__aiter__"):
            # Need to consume that async generator, since requests can't do anything with it
            # That's not ideal, but a list is our only choice. Memory not optimal here,
            # but providing an async generator to a requests based transport is not optimal too
            new_data = []
            async for part in request.data:
                new_data.append(part)
            data_to_send = iter(new_data)
        else:
            data_to_send = request.data
        return data_to_send

    async def __aenter__(self):
        return super(RequestsAsyncTransportBase, self).__enter__()

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]] = None,
        exc_value: Optional[BaseException] = None,
        traceback: Optional[TracebackType] = None,
    ):
        return super(RequestsAsyncTransportBase, self).__exit__(
            exc_type, exc_value, traceback
        )
