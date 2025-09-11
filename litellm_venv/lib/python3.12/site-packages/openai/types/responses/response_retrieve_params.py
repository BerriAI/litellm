# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import List, Union
from typing_extensions import Literal, Required, TypedDict

from .response_includable import ResponseIncludable

__all__ = [
    "ResponseRetrieveParamsBase",
    "ResponseRetrieveParamsNonStreaming",
    "ResponseRetrieveParamsStreaming",
]


class ResponseRetrieveParamsBase(TypedDict, total=False):
    include: List[ResponseIncludable]
    """Additional fields to include in the response.

    See the `include` parameter for Response creation above for more information.
    """

    include_obfuscation: bool
    """When true, stream obfuscation will be enabled.

    Stream obfuscation adds random characters to an `obfuscation` field on streaming
    delta events to normalize payload sizes as a mitigation to certain side-channel
    attacks. These obfuscation fields are included by default, but add a small
    amount of overhead to the data stream. You can set `include_obfuscation` to
    false to optimize for bandwidth if you trust the network links between your
    application and the OpenAI API.
    """

    starting_after: int
    """The sequence number of the event after which to start streaming."""


class ResponseRetrieveParamsNonStreaming(ResponseRetrieveParamsBase, total=False):
    stream: Literal[False]
    """
    If set to true, the model response data will be streamed to the client as it is
    generated using
    [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format).
    See the
    [Streaming section below](https://platform.openai.com/docs/api-reference/responses-streaming)
    for more information.
    """


class ResponseRetrieveParamsStreaming(ResponseRetrieveParamsBase):
    stream: Required[Literal[True]]
    """
    If set to true, the model response data will be streamed to the client as it is
    generated using
    [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format).
    See the
    [Streaming section below](https://platform.openai.com/docs/api-reference/responses-streaming)
    for more information.
    """


ResponseRetrieveParams = Union[
    ResponseRetrieveParamsNonStreaming, ResponseRetrieveParamsStreaming
]
