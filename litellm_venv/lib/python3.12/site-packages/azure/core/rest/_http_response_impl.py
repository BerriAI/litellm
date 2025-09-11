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
from json import loads
from typing import Any, Optional, Iterator, MutableMapping, Callable
from http.client import HTTPResponse as _HTTPResponse
from ._helpers import (
    get_charset_encoding,
    decode_to_text,
)
from ..exceptions import (
    HttpResponseError,
    ResponseNotReadError,
    StreamConsumedError,
    StreamClosedError,
)
from ._rest_py3 import (
    _HttpResponseBase,
    HttpResponse as _HttpResponse,
    HttpRequest as _HttpRequest,
)
from ..utils._utils import case_insensitive_dict
from ..utils._pipeline_transport_rest_shared import (
    _pad_attr_name,
    BytesIOSocket,
    _decode_parts_helper,
    _get_raw_parts_helper,
    _parts_helper,
)


class _HttpResponseBackcompatMixinBase:
    """Base Backcompat mixin for responses.

    This mixin is used by both sync and async HttpResponse
    backcompat mixins.
    """

    def __getattr__(self, attr):
        backcompat_attrs = [
            "body",
            "internal_response",
            "block_size",
            "stream_download",
        ]
        attr = _pad_attr_name(attr, backcompat_attrs)
        return self.__getattribute__(attr)

    def __setattr__(self, attr, value):
        backcompat_attrs = [
            "block_size",
            "internal_response",
            "request",
            "status_code",
            "headers",
            "reason",
            "content_type",
            "stream_download",
        ]
        attr = _pad_attr_name(attr, backcompat_attrs)
        super(_HttpResponseBackcompatMixinBase, self).__setattr__(attr, value)

    def _body(self):
        """DEPRECATED: Get the response body.
        This is deprecated and will be removed in a later release.
        You should get it through the `content` property instead

        :return: The response body.
        :rtype: bytes
        """
        self.read()
        return self.content

    def _decode_parts(self, message, http_response_type, requests):
        """Helper for _decode_parts.

        Rebuild an HTTP response from pure string.

        :param message: The body as an email.Message type
        :type message: ~email.message.Message
        :param http_response_type: The type of response to build
        :type http_response_type: type
        :param requests: A list of requests to process
        :type requests: list[~azure.core.rest.HttpRequest]
        :return: A list of responses
        :rtype: list[~azure.core.rest.HttpResponse]
        """

        def _deserialize_response(
            http_response_as_bytes, http_request, http_response_type
        ):
            local_socket = BytesIOSocket(http_response_as_bytes)
            response = _HTTPResponse(local_socket, method=http_request.method)
            response.begin()
            return http_response_type(request=http_request, internal_response=response)

        return _decode_parts_helper(
            self,
            message,
            http_response_type or RestHttpClientTransportResponse,
            requests,
            _deserialize_response,
        )

    def _get_raw_parts(self, http_response_type=None):
        """Helper for get_raw_parts

        Assuming this body is multipart, return the iterator or parts.

        If parts are application/http use http_response_type or HttpClientTransportResponse
        as envelope.

        :param http_response_type: The type of response to build
        :type http_response_type: type
        :return: An iterator of responses
        :rtype: Iterator[~azure.core.rest.HttpResponse]
        """
        return _get_raw_parts_helper(
            self, http_response_type or RestHttpClientTransportResponse
        )

    def _stream_download(self, pipeline, **kwargs):
        """DEPRECATED: Generator for streaming request body data.
        This is deprecated and will be removed in a later release.
        You should use `iter_bytes` or `iter_raw` instead.

        :param pipeline: The pipeline object
        :type pipeline: ~azure.core.pipeline.Pipeline
        :return: An iterator for streaming request body data.
        :rtype: iterator[bytes]
        """
        return self._stream_download_generator(pipeline, self, **kwargs)


class HttpResponseBackcompatMixin(_HttpResponseBackcompatMixinBase):
    """Backcompat mixin for sync HttpResponses"""

    def __getattr__(self, attr):
        backcompat_attrs = ["parts"]
        attr = _pad_attr_name(attr, backcompat_attrs)
        return super(HttpResponseBackcompatMixin, self).__getattr__(attr)

    def parts(self):
        """DEPRECATED: Assuming the content-type is multipart/mixed, will return the parts as an async iterator.
        This is deprecated and will be removed in a later release.

        :rtype: Iterator
        :return: The parts of the response
        :raises ValueError: If the content is not multipart/mixed
        """
        return _parts_helper(self)


class _HttpResponseBaseImpl(
    _HttpResponseBase, _HttpResponseBackcompatMixinBase
):  # pylint: disable=too-many-instance-attributes
    """Base Implementation class for azure.core.rest.HttpRespone and azure.core.rest.AsyncHttpResponse

    Since the rest responses are abstract base classes, we need to implement them for each of our transport
    responses. This is the base implementation class shared by HttpResponseImpl and AsyncHttpResponseImpl.
    The transport responses will be built on top of HttpResponseImpl and AsyncHttpResponseImpl

    :keyword request: The request that led to the response
    :type request: ~azure.core.rest.HttpRequest
    :keyword any internal_response: The response we get directly from the transport. For example, for our requests
     transport, this will be a requests.Response.
    :keyword optional[int] block_size: The block size we are using in our transport
    :keyword int status_code: The status code of the response
    :keyword str reason: The HTTP reason
    :keyword str content_type: The content type of the response
    :keyword MutableMapping[str, str] headers: The response headers
    :keyword Callable stream_download_generator: The stream download generator that we use to stream the response.
    """

    def __init__(self, **kwargs) -> None:
        super(_HttpResponseBaseImpl, self).__init__()
        self._request = kwargs.pop("request")
        self._internal_response = kwargs.pop("internal_response")
        self._block_size: int = kwargs.pop("block_size", None) or 4096
        self._status_code: int = kwargs.pop("status_code")
        self._reason: str = kwargs.pop("reason")
        self._content_type: str = kwargs.pop("content_type")
        self._headers: MutableMapping[str, str] = kwargs.pop("headers")
        self._stream_download_generator: Callable = kwargs.pop(
            "stream_download_generator"
        )
        self._is_closed = False
        self._is_stream_consumed = False
        self._json = None  # this is filled in ContentDecodePolicy, when we deserialize
        self._content: Optional[bytes] = None
        self._text: Optional[str] = None

    @property
    def request(self) -> _HttpRequest:
        """The request that resulted in this response.

        :rtype: ~azure.core.rest.HttpRequest
        :return: The request that resulted in this response.
        """
        return self._request

    @property
    def url(self) -> str:
        """The URL that resulted in this response.

        :rtype: str
        :return: The URL that resulted in this response.
        """
        return self.request.url

    @property
    def is_closed(self) -> bool:
        """Whether the network connection has been closed yet.

        :rtype: bool
        :return: Whether the network connection has been closed yet.
        """
        return self._is_closed

    @property
    def is_stream_consumed(self) -> bool:
        """Whether the stream has been consumed.

        :rtype: bool
        :return: Whether the stream has been consumed.
        """
        return self._is_stream_consumed

    @property
    def status_code(self) -> int:
        """The status code of this response.

        :rtype: int
        :return: The status code of this response.
        """
        return self._status_code

    @property
    def headers(self) -> MutableMapping[str, str]:
        """The response headers.

        :rtype: MutableMapping[str, str]
        :return: The response headers.
        """
        return self._headers

    @property
    def content_type(self) -> Optional[str]:
        """The content type of the response.

        :rtype: optional[str]
        :return: The content type of the response.
        """
        return self._content_type

    @property
    def reason(self) -> str:
        """The reason phrase for this response.

        :rtype: str
        :return: The reason phrase for this response.
        """
        return self._reason

    @property
    def encoding(self) -> Optional[str]:
        """Returns the response encoding.

        :return: The response encoding. We either return the encoding set by the user,
         or try extracting the encoding from the response's content type. If all fails,
         we return `None`.
        :rtype: optional[str]
        """
        try:
            return self._encoding
        except AttributeError:
            self._encoding: Optional[str] = get_charset_encoding(self)
            return self._encoding

    @encoding.setter
    def encoding(self, value: str) -> None:
        """Sets the response encoding.

        :param str value: Sets the response encoding.
        """
        self._encoding = value
        self._text = None  # clear text cache
        self._json = None  # clear json cache as well

    def text(self, encoding: Optional[str] = None) -> str:
        """Returns the response body as a string

        :param optional[str] encoding: The encoding you want to decode the text with. Can
         also be set independently through our encoding property
        :return: The response's content decoded as a string.
        :rtype: str
        """
        if encoding:
            return decode_to_text(encoding, self.content)
        if self._text:
            return self._text
        self._text = decode_to_text(self.encoding, self.content)
        return self._text

    def json(self) -> Any:
        """Returns the whole body as a json object.

        :return: The JSON deserialized response body
        :rtype: any
        :raises json.decoder.JSONDecodeError or ValueError (in python 2.7) if object is not JSON decodable:
        """
        # this will trigger errors if response is not read in
        self.content  # pylint: disable=pointless-statement
        if not self._json:
            self._json = loads(self.text())
        return self._json

    def _stream_download_check(self):
        if self.is_stream_consumed:
            raise StreamConsumedError(self)
        if self.is_closed:
            raise StreamClosedError(self)

        self._is_stream_consumed = True

    def raise_for_status(self) -> None:
        """Raises an HttpResponseError if the response has an error status code.

        If response is good, does nothing.
        """
        if self.status_code >= 400:
            raise HttpResponseError(response=self)

    @property
    def content(self) -> bytes:
        """Return the response's content in bytes.

        :return: The response's content in bytes.
        :rtype: bytes
        """
        if self._content is None:
            raise ResponseNotReadError(self)
        return self._content

    def __repr__(self) -> str:
        content_type_str = (
            ", Content-Type: {}".format(self.content_type) if self.content_type else ""
        )
        return "<HttpResponse: {} {}{}>".format(
            self.status_code, self.reason, content_type_str
        )


class HttpResponseImpl(
    _HttpResponseBaseImpl, _HttpResponse, HttpResponseBackcompatMixin
):
    """HttpResponseImpl built on top of our HttpResponse protocol class.

    Since ~azure.core.rest.HttpResponse is an abstract base class, we need to
    implement HttpResponse for each of our transports. This is an implementation
    that each of the sync transport responses can be built on.

    :keyword request: The request that led to the response
    :type request: ~azure.core.rest.HttpRequest
    :keyword any internal_response: The response we get directly from the transport. For example, for our requests
     transport, this will be a requests.Response.
    :keyword optional[int] block_size: The block size we are using in our transport
    :keyword int status_code: The status code of the response
    :keyword str reason: The HTTP reason
    :keyword str content_type: The content type of the response
    :keyword MutableMapping[str, str] headers: The response headers
    :keyword Callable stream_download_generator: The stream download generator that we use to stream the response.
    """

    def __enter__(self) -> "HttpResponseImpl":
        return self

    def close(self) -> None:
        if not self.is_closed:
            self._is_closed = True
            self._internal_response.close()

    def __exit__(self, *args) -> None:
        self.close()

    def _set_read_checks(self):
        self._is_stream_consumed = True
        self.close()

    def read(self) -> bytes:
        """Read the response's bytes.

        :return: The response's bytes
        :rtype: bytes
        """
        if self._content is None:
            self._content = b"".join(self.iter_bytes())
        self._set_read_checks()
        return self.content

    def iter_bytes(self, **kwargs) -> Iterator[bytes]:
        """Iterates over the response's bytes. Will decompress in the process.

        :return: An iterator of bytes from the response
        :rtype: Iterator[str]
        """
        if self._content is not None:
            chunk_size = self._block_size
            for i in range(0, len(self.content), chunk_size):
                yield self.content[i : i + chunk_size]
        else:
            self._stream_download_check()
            yield from self._stream_download_generator(
                response=self,
                pipeline=None,
                decompress=True,
            )
        self.close()

    def iter_raw(self, **kwargs) -> Iterator[bytes]:
        """Iterates over the response's bytes. Will not decompress in the process.

        :return: An iterator of bytes from the response
        :rtype: Iterator[str]
        """
        self._stream_download_check()
        yield from self._stream_download_generator(
            response=self, pipeline=None, decompress=False
        )
        self.close()


class _RestHttpClientTransportResponseBackcompatBaseMixin(
    _HttpResponseBackcompatMixinBase
):
    def body(self):
        if self._content is None:
            self._content = self.internal_response.read()
        return self.content


class _RestHttpClientTransportResponseBase(
    _HttpResponseBaseImpl, _RestHttpClientTransportResponseBackcompatBaseMixin
):
    def __init__(self, **kwargs):
        internal_response = kwargs.pop("internal_response")
        headers = case_insensitive_dict(internal_response.getheaders())
        super(_RestHttpClientTransportResponseBase, self).__init__(
            internal_response=internal_response,
            status_code=internal_response.status,
            reason=internal_response.reason,
            headers=headers,
            content_type=headers.get("Content-Type"),
            stream_download_generator=None,
            **kwargs
        )


class RestHttpClientTransportResponse(
    _RestHttpClientTransportResponseBase, HttpResponseImpl
):
    """Create a Rest HTTPResponse from an http.client response."""

    def iter_bytes(self, **kwargs):
        raise TypeError("We do not support iter_bytes for this transport response")

    def iter_raw(self, **kwargs):
        raise TypeError("We do not support iter_raw for this transport response")

    def read(self):
        if self._content is None:
            self._content = self._internal_response.read()
        return self._content
