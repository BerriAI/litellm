# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import List, Union, Iterable, Optional
from typing_extensions import Literal, Required, TypedDict

from ...shared_params.metadata import Metadata
from .conversation_item_with_reference_param import ConversationItemWithReferenceParam

__all__ = ["ResponseCreateEventParam", "Response", "ResponseTool"]


class ResponseTool(TypedDict, total=False):
    description: str
    """
    The description of the function, including guidance on when and how to call it,
    and guidance about what to tell the user when calling (if anything).
    """

    name: str
    """The name of the function."""

    parameters: object
    """Parameters of the function in JSON Schema."""

    type: Literal["function"]
    """The type of the tool, i.e. `function`."""


class Response(TypedDict, total=False):
    conversation: Union[str, Literal["auto", "none"]]
    """Controls which conversation the response is added to.

    Currently supports `auto` and `none`, with `auto` as the default value. The
    `auto` value means that the contents of the response will be added to the
    default conversation. Set this to `none` to create an out-of-band response which
    will not add items to default conversation.
    """

    input: Iterable[ConversationItemWithReferenceParam]
    """Input items to include in the prompt for the model.

    Using this field creates a new context for this Response instead of using the
    default conversation. An empty array `[]` will clear the context for this
    Response. Note that this can include references to items from the default
    conversation.
    """

    instructions: str
    """The default system instructions (i.e.

    system message) prepended to model calls. This field allows the client to guide
    the model on desired responses. The model can be instructed on response content
    and format, (e.g. "be extremely succinct", "act friendly", "here are examples of
    good responses") and on audio behavior (e.g. "talk quickly", "inject emotion
    into your voice", "laugh frequently"). The instructions are not guaranteed to be
    followed by the model, but they provide guidance to the model on the desired
    behavior.

    Note that the server sets default instructions which will be used if this field
    is not set and are visible in the `session.created` event at the start of the
    session.
    """

    max_response_output_tokens: Union[int, Literal["inf"]]
    """
    Maximum number of output tokens for a single assistant response, inclusive of
    tool calls. Provide an integer between 1 and 4096 to limit output tokens, or
    `inf` for the maximum available tokens for a given model. Defaults to `inf`.
    """

    metadata: Optional[Metadata]
    """Set of 16 key-value pairs that can be attached to an object.

    This can be useful for storing additional information about the object in a
    structured format, and querying for objects via API or the dashboard.

    Keys are strings with a maximum length of 64 characters. Values are strings with
    a maximum length of 512 characters.
    """

    modalities: List[Literal["text", "audio"]]
    """The set of modalities the model can respond with.

    To disable audio, set this to ["text"].
    """

    output_audio_format: Literal["pcm16", "g711_ulaw", "g711_alaw"]
    """The format of output audio. Options are `pcm16`, `g711_ulaw`, or `g711_alaw`."""

    temperature: float
    """Sampling temperature for the model, limited to [0.6, 1.2]. Defaults to 0.8."""

    tool_choice: str
    """How the model chooses tools.

    Options are `auto`, `none`, `required`, or specify a function, like
    `{"type": "function", "function": {"name": "my_function"}}`.
    """

    tools: Iterable[ResponseTool]
    """Tools (functions) available to the model."""

    voice: Union[
        str,
        Literal["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"],
    ]
    """The voice the model uses to respond.

    Voice cannot be changed during the session once the model has responded with
    audio at least once. Current voice options are `alloy`, `ash`, `ballad`,
    `coral`, `echo`, `sage`, `shimmer`, and `verse`.
    """


class ResponseCreateEventParam(TypedDict, total=False):
    type: Required[Literal["response.create"]]
    """The event type, must be `response.create`."""

    event_id: str
    """Optional client-generated ID used to identify this event."""

    response: Response
    """Create a new Realtime response with these parameters"""
