# File generated from our OpenAPI spec by Stainless.

from __future__ import annotations

from typing import Union
from typing_extensions import Literal, Required, TypedDict

__all__ = ["FineTuneListEventsParamsBase", "FineTuneListEventsParamsNonStreaming", "FineTuneListEventsParamsStreaming"]


class FineTuneListEventsParamsBase(TypedDict, total=False):
    pass


class FineTuneListEventsParamsNonStreaming(FineTuneListEventsParamsBase):
    stream: Literal[False]
    """Whether to stream events for the fine-tune job.

    If set to true, events will be sent as data-only
    [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
    as they become available. The stream will terminate with a `data: [DONE]`
    message when the job is finished (succeeded, cancelled, or failed).

    If set to false, only events generated so far will be returned.
    """


class FineTuneListEventsParamsStreaming(FineTuneListEventsParamsBase):
    stream: Required[Literal[True]]
    """Whether to stream events for the fine-tune job.

    If set to true, events will be sent as data-only
    [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
    as they become available. The stream will terminate with a `data: [DONE]`
    message when the job is finished (succeeded, cancelled, or failed).

    If set to false, only events generated so far will be returned.
    """


FineTuneListEventsParams = Union[FineTuneListEventsParamsNonStreaming, FineTuneListEventsParamsStreaming]
