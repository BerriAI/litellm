# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union, Iterable
from typing_extensions import Literal, Required, TypeAlias, TypedDict

__all__ = [
    "ResponseOutputTextParam",
    "Annotation",
    "AnnotationFileCitation",
    "AnnotationURLCitation",
    "AnnotationContainerFileCitation",
    "AnnotationFilePath",
    "Logprob",
    "LogprobTopLogprob",
]


class AnnotationFileCitation(TypedDict, total=False):
    file_id: Required[str]
    """The ID of the file."""

    filename: Required[str]
    """The filename of the file cited."""

    index: Required[int]
    """The index of the file in the list of files."""

    type: Required[Literal["file_citation"]]
    """The type of the file citation. Always `file_citation`."""


class AnnotationURLCitation(TypedDict, total=False):
    end_index: Required[int]
    """The index of the last character of the URL citation in the message."""

    start_index: Required[int]
    """The index of the first character of the URL citation in the message."""

    title: Required[str]
    """The title of the web resource."""

    type: Required[Literal["url_citation"]]
    """The type of the URL citation. Always `url_citation`."""

    url: Required[str]
    """The URL of the web resource."""


class AnnotationContainerFileCitation(TypedDict, total=False):
    container_id: Required[str]
    """The ID of the container file."""

    end_index: Required[int]
    """The index of the last character of the container file citation in the message."""

    file_id: Required[str]
    """The ID of the file."""

    filename: Required[str]
    """The filename of the container file cited."""

    start_index: Required[int]
    """The index of the first character of the container file citation in the message."""

    type: Required[Literal["container_file_citation"]]
    """The type of the container file citation. Always `container_file_citation`."""


class AnnotationFilePath(TypedDict, total=False):
    file_id: Required[str]
    """The ID of the file."""

    index: Required[int]
    """The index of the file in the list of files."""

    type: Required[Literal["file_path"]]
    """The type of the file path. Always `file_path`."""


Annotation: TypeAlias = Union[
    AnnotationFileCitation,
    AnnotationURLCitation,
    AnnotationContainerFileCitation,
    AnnotationFilePath,
]


class LogprobTopLogprob(TypedDict, total=False):
    token: Required[str]

    bytes: Required[Iterable[int]]

    logprob: Required[float]


class Logprob(TypedDict, total=False):
    token: Required[str]

    bytes: Required[Iterable[int]]

    logprob: Required[float]

    top_logprobs: Required[Iterable[LogprobTopLogprob]]


class ResponseOutputTextParam(TypedDict, total=False):
    annotations: Required[Iterable[Annotation]]
    """The annotations of the text output."""

    text: Required[str]
    """The text output from the model."""

    type: Required[Literal["output_text"]]
    """The type of the output text. Always `output_text`."""

    logprobs: Iterable[Logprob]
