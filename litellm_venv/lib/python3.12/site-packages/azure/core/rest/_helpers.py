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
from __future__ import annotations
import copy
import codecs
import email.message
from json import dumps
from typing import (
    Optional,
    Union,
    Mapping,
    Sequence,
    Tuple,
    IO,
    Any,
    Iterable,
    MutableMapping,
    AsyncIterable,
    cast,
    Dict,
    TYPE_CHECKING,
)
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
from azure.core.serialization import AzureJSONEncoder
from ..utils._pipeline_transport_rest_shared import (
    _format_parameters_helper,
    _pad_attr_name,
    _prepare_multipart_body_helper,
    _serialize_request,
    _format_data_helper,
    get_file_items,
)

if TYPE_CHECKING:
    # This avoid a circular import
    from ._rest_py3 import HttpRequest

################################### TYPES SECTION #########################

binary_type = str
PrimitiveData = Optional[Union[str, int, float, bool]]

ParamsType = Mapping[str, Union[PrimitiveData, Sequence[PrimitiveData]]]

FileContent = Union[str, bytes, IO[str], IO[bytes]]

FileType = Union[
    # file (or bytes)
    FileContent,
    # (filename, file (or bytes))
    Tuple[Optional[str], FileContent],
    # (filename, file (or bytes), content_type)
    Tuple[Optional[str], FileContent, Optional[str]],
]

FilesType = Union[Mapping[str, FileType], Sequence[Tuple[str, FileType]]]

ContentTypeBase = Union[str, bytes, Iterable[bytes]]
ContentType = Union[str, bytes, Iterable[bytes], AsyncIterable[bytes]]

DataType = Optional[Union[bytes, Dict[str, Union[str, int]]]]

########################### HELPER SECTION #################################


def _verify_data_object(name, value):
    if not isinstance(name, str):
        raise TypeError(
            "Invalid type for data name. Expected str, got {}: {}".format(
                type(name), name
            )
        )
    if value is not None and not isinstance(value, (str, bytes, int, float)):
        raise TypeError(
            "Invalid type for data value. Expected primitive type, got {}: {}".format(
                type(name), name
            )
        )


def set_urlencoded_body(data, has_files):
    body = {}
    default_headers = {}
    for f, d in data.items():
        if not d:
            continue
        if isinstance(d, list):
            for item in d:
                _verify_data_object(f, item)
        else:
            _verify_data_object(f, d)
        body[f] = d
    if not has_files:
        # little hacky, but for files we don't send a content type with
        # boundary so requests / aiohttp etc deal with it
        default_headers["Content-Type"] = "application/x-www-form-urlencoded"
    return default_headers, body


def set_multipart_body(files: FilesType):
    formatted_files = [
        (f, _format_data_helper(d)) for f, d in get_file_items(files) if d is not None
    ]
    return {}, dict(formatted_files) if isinstance(files, Mapping) else formatted_files


def set_xml_body(content):
    headers = {}
    bytes_content = ET.tostring(content, encoding="utf8")
    body = bytes_content.replace(b"encoding='utf8'", b"encoding='utf-8'")
    if body:
        headers["Content-Length"] = str(len(body))
    return headers, body


def set_content_body(
    content: Any,
) -> Tuple[MutableMapping[str, str], Optional[ContentTypeBase]]:
    headers: MutableMapping[str, str] = {}

    if isinstance(content, ET.Element):
        # XML body
        return set_xml_body(content)
    if isinstance(content, (str, bytes)):
        headers = {}
        body = content
        if isinstance(content, str):
            headers["Content-Type"] = "text/plain"
        if body:
            headers["Content-Length"] = str(len(body))
        return headers, body
    if any(hasattr(content, attr) for attr in ["read", "__iter__", "__aiter__"]):
        return headers, content
    raise TypeError(
        "Unexpected type for 'content': '{}'. ".format(type(content))
        + "We expect 'content' to either be str, bytes, a open file-like object or an iterable/asynciterable."
    )


def set_json_body(json: Any) -> Tuple[Dict[str, str], Any]:
    headers = {"Content-Type": "application/json"}
    if hasattr(json, "read"):
        content_headers, body = set_content_body(json)
        headers.update(content_headers)
    else:
        body = dumps(json, cls=AzureJSONEncoder)
        headers.update({"Content-Length": str(len(body))})
    return headers, body


def lookup_encoding(encoding: str) -> bool:
    # including check for whether encoding is known taken from httpx
    try:
        codecs.lookup(encoding)
        return True
    except LookupError:
        return False


def get_charset_encoding(response) -> Optional[str]:
    content_type = response.headers.get("Content-Type")

    if not content_type:
        return None
    # https://peps.python.org/pep-0594/#cgi
    m = email.message.Message()
    m["content-type"] = content_type
    encoding = cast(str, m.get_param("charset"))  # -> utf-8
    if encoding is None or not lookup_encoding(encoding):
        return None
    return encoding


def decode_to_text(encoding: Optional[str], content: bytes) -> str:
    if not content:
        return ""
    if encoding == "utf-8":
        encoding = "utf-8-sig"
    if encoding:
        return content.decode(encoding)
    return codecs.getincrementaldecoder("utf-8-sig")(errors="replace").decode(content)


class HttpRequestBackcompatMixin:
    def __getattr__(self, attr: str) -> Any:
        backcompat_attrs = [
            "files",
            "data",
            "multipart_mixed_info",
            "query",
            "body",
            "format_parameters",
            "set_streamed_data_body",
            "set_text_body",
            "set_xml_body",
            "set_json_body",
            "set_formdata_body",
            "set_bytes_body",
            "set_multipart_mixed",
            "prepare_multipart_body",
            "serialize",
        ]
        attr = _pad_attr_name(attr, backcompat_attrs)
        return self.__getattribute__(attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        backcompat_attrs = [
            "multipart_mixed_info",
            "files",
            "data",
            "body",
        ]
        attr = _pad_attr_name(attr, backcompat_attrs)
        super(HttpRequestBackcompatMixin, self).__setattr__(attr, value)

    @property
    def _multipart_mixed_info(
        self,
    ) -> Optional[Tuple[Sequence[Any], Sequence[Any], str, Dict[str, Any]]]:
        """DEPRECATED: Information used to make multipart mixed requests.
        This is deprecated and will be removed in a later release.

        :rtype: tuple
        :return: (requests, policies, boundary, kwargs)
        """
        try:
            return self._multipart_mixed_info_val
        except AttributeError:
            return None

    @_multipart_mixed_info.setter
    def _multipart_mixed_info(
        self, val: Optional[Tuple[Sequence[Any], Sequence[Any], str, Dict[str, Any]]]
    ):
        """DEPRECATED: Set information to make multipart mixed requests.
        This is deprecated and will be removed in a later release.

        :param tuple val: (requests, policies, boundary, kwargs)
        """
        self._multipart_mixed_info_val = val

    @property
    def _query(self) -> Dict[str, Any]:
        """DEPRECATED: Query parameters passed in by user
        This is deprecated and will be removed in a later release.

        :rtype: dict
        :return: Query parameters
        """
        query = urlparse(self.url).query
        if query:
            return {p[0]: p[-1] for p in [p.partition("=") for p in query.split("&")]}
        return {}

    @property
    def _body(self) -> DataType:
        """DEPRECATED: Body of the request. You should use the `content` property instead
        This is deprecated and will be removed in a later release.

        :rtype: bytes
        :return: Body of the request
        """
        return self._data

    @_body.setter
    def _body(self, val: DataType) -> None:
        """DEPRECATED: Set the body of the request
        This is deprecated and will be removed in a later release.

        :param bytes val: Body of the request
        """
        self._data = val

    def _format_parameters(self, params: MutableMapping[str, str]) -> None:
        """DEPRECATED: Format the query parameters
        This is deprecated and will be removed in a later release.
        You should pass the query parameters through the kwarg `params`
        instead.

        :param dict params: Query parameters
        """
        _format_parameters_helper(self, params)

    def _set_streamed_data_body(self, data):
        """DEPRECATED: Set the streamed request body.
        This is deprecated and will be removed in a later release.
        You should pass your stream content through the `content` kwarg instead

        :param data: Streamed data
        :type data: bytes or iterable
        """
        if not isinstance(data, binary_type) and not any(
            hasattr(data, attr) for attr in ["read", "__iter__", "__aiter__"]
        ):
            raise TypeError(
                "A streamable data source must be an open file-like object or iterable."
            )
        headers = self._set_body(content=data)
        self._files = None
        self.headers.update(headers)

    def _set_text_body(self, data):
        """DEPRECATED: Set the text body
        This is deprecated and will be removed in a later release.
        You should pass your text content through the `content` kwarg instead

        :param str data: Text data
        """
        headers = self._set_body(content=data)
        self.headers.update(headers)
        self._files = None

    def _set_xml_body(self, data):
        """DEPRECATED: Set the xml body.
        This is deprecated and will be removed in a later release.
        You should pass your xml content through the `content` kwarg instead

        :param data: XML data
        :type data: xml.etree.ElementTree.Element
        """
        headers = self._set_body(content=data)
        self.headers.update(headers)
        self._files = None

    def _set_json_body(self, data):
        """DEPRECATED: Set the json request body.
        This is deprecated and will be removed in a later release.
        You should pass your json content through the `json` kwarg instead

        :param data: JSON data
        :type data: dict
        """
        headers = self._set_body(json=data)
        self.headers.update(headers)
        self._files = None

    def _set_formdata_body(self, data=None):
        """DEPRECATED: Set the formrequest body.
        This is deprecated and will be removed in a later release.
        You should pass your stream content through the `files` kwarg instead

        :param data: Form data
        :type data: dict
        """
        if data is None:
            data = {}
        content_type = self.headers.pop("Content-Type", None) if self.headers else None

        if content_type and content_type.lower() == "application/x-www-form-urlencoded":
            headers = self._set_body(data=data)
            self._files = None
        else:  # Assume "multipart/form-data"
            headers = self._set_body(files=data)
            self._data = None
        self.headers.update(headers)

    def _set_bytes_body(self, data):
        """DEPRECATED: Set the bytes request body.
        This is deprecated and will be removed in a later release.
        You should pass your bytes content through the `content` kwarg instead

        :param bytes data: Bytes data
        """
        headers = self._set_body(content=data)
        # we don't want default Content-Type
        # in 2.7, byte strings are still strings, so they get set with text/plain content type

        headers.pop("Content-Type", None)
        self.headers.update(headers)
        self._files = None

    def _set_multipart_mixed(self, *requests: HttpRequest, **kwargs: Any) -> None:
        """DEPRECATED: Set the multipart mixed info.
        This is deprecated and will be removed in a later release.

        :param requests: Requests to be sent in the multipart request
        :type requests: list[HttpRequest]
        """
        self.multipart_mixed_info: Tuple[
            Sequence[HttpRequest], Sequence[Any], str, Dict[str, Any]
        ] = (
            requests,
            kwargs.pop("policies", []),
            kwargs.pop("boundary", None),
            kwargs,
        )

    def _prepare_multipart_body(self, content_index=0):
        """DEPRECATED: Prepare your request body for multipart requests.
        This is deprecated and will be removed in a later release.

        :param int content_index: The index of the request to be sent in the multipart request
        :returns: The updated index after all parts in this request have been added.
        :rtype: int
        """
        return _prepare_multipart_body_helper(self, content_index)

    def _serialize(self):
        """DEPRECATED: Serialize this request using application/http spec.
        This is deprecated and will be removed in a later release.

        :rtype: bytes
        :return: The serialized request
        """
        return _serialize_request(self)

    def _add_backcompat_properties(self, request, memo):
        """While deepcopying, we also need to add the private backcompat attrs.

        :param HttpRequest request: The request to copy from
        :param dict memo: The memo dict used by deepcopy
        """
        request._multipart_mixed_info = (
            copy.deepcopy(  # pylint: disable=protected-access
                self._multipart_mixed_info, memo
            )
        )
