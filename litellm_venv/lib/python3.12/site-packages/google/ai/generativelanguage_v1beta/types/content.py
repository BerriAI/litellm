# -*- coding: utf-8 -*-
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
#
from __future__ import annotations

from typing import MutableMapping, MutableSequence

from google.protobuf import struct_pb2  # type: ignore
import proto  # type: ignore

__protobuf__ = proto.module(
    package="google.ai.generativelanguage.v1beta",
    manifest={
        "Type",
        "Content",
        "Part",
        "Blob",
        "FileData",
        "Tool",
        "ToolConfig",
        "FunctionCallingConfig",
        "FunctionDeclaration",
        "FunctionCall",
        "FunctionResponse",
        "Schema",
        "GroundingPassage",
        "GroundingPassages",
    },
)


class Type(proto.Enum):
    r"""Type contains the list of OpenAPI data types as defined by
    https://spec.openapis.org/oas/v3.0.3#data-types

    Values:
        TYPE_UNSPECIFIED (0):
            Not specified, should not be used.
        STRING (1):
            String type.
        NUMBER (2):
            Number type.
        INTEGER (3):
            Integer type.
        BOOLEAN (4):
            Boolean type.
        ARRAY (5):
            Array type.
        OBJECT (6):
            Object type.
    """
    TYPE_UNSPECIFIED = 0
    STRING = 1
    NUMBER = 2
    INTEGER = 3
    BOOLEAN = 4
    ARRAY = 5
    OBJECT = 6


class Content(proto.Message):
    r"""The base structured datatype containing multi-part content of a
    message.

    A ``Content`` includes a ``role`` field designating the producer of
    the ``Content`` and a ``parts`` field containing multi-part data
    that contains the content of the message turn.

    Attributes:
        parts (MutableSequence[google.ai.generativelanguage_v1beta.types.Part]):
            Ordered ``Parts`` that constitute a single message. Parts
            may have different MIME types.
        role (str):
            Optional. The producer of the content. Must
            be either 'user' or 'model'.
            Useful to set for multi-turn conversations,
            otherwise can be left blank or unset.
    """

    parts: MutableSequence["Part"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="Part",
    )
    role: str = proto.Field(
        proto.STRING,
        number=2,
    )


class Part(proto.Message):
    r"""A datatype containing media that is part of a multi-part ``Content``
    message.

    A ``Part`` consists of data which has an associated datatype. A
    ``Part`` can only contain one of the accepted types in
    ``Part.data``.

    A ``Part`` must have a fixed IANA MIME type identifying the type and
    subtype of the media if the ``inline_data`` field is filled with raw
    bytes.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        text (str):
            Inline text.

            This field is a member of `oneof`_ ``data``.
        inline_data (google.ai.generativelanguage_v1beta.types.Blob):
            Inline media bytes.

            This field is a member of `oneof`_ ``data``.
        function_call (google.ai.generativelanguage_v1beta.types.FunctionCall):
            A predicted ``FunctionCall`` returned from the model that
            contains a string representing the
            ``FunctionDeclaration.name`` with the arguments and their
            values.

            This field is a member of `oneof`_ ``data``.
        function_response (google.ai.generativelanguage_v1beta.types.FunctionResponse):
            The result output of a ``FunctionCall`` that contains a
            string representing the ``FunctionDeclaration.name`` and a
            structured JSON object containing any output from the
            function is used as context to the model.

            This field is a member of `oneof`_ ``data``.
        file_data (google.ai.generativelanguage_v1beta.types.FileData):
            URI based data.

            This field is a member of `oneof`_ ``data``.
    """

    text: str = proto.Field(
        proto.STRING,
        number=2,
        oneof="data",
    )
    inline_data: "Blob" = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="data",
        message="Blob",
    )
    function_call: "FunctionCall" = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof="data",
        message="FunctionCall",
    )
    function_response: "FunctionResponse" = proto.Field(
        proto.MESSAGE,
        number=5,
        oneof="data",
        message="FunctionResponse",
    )
    file_data: "FileData" = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="data",
        message="FileData",
    )


class Blob(proto.Message):
    r"""Raw media bytes.

    Text should not be sent as raw bytes, use the 'text' field.

    Attributes:
        mime_type (str):
            The IANA standard MIME type of the source
            data. Accepted types include: "image/png",
            "image/jpeg", "image/heic", "image/heif",
            "image/webp".
        data (bytes):
            Raw bytes for media formats.
    """

    mime_type: str = proto.Field(
        proto.STRING,
        number=1,
    )
    data: bytes = proto.Field(
        proto.BYTES,
        number=2,
    )


class FileData(proto.Message):
    r"""URI based data.

    Attributes:
        mime_type (str):
            Optional. The IANA standard MIME type of the
            source data.
        file_uri (str):
            Required. URI.
    """

    mime_type: str = proto.Field(
        proto.STRING,
        number=1,
    )
    file_uri: str = proto.Field(
        proto.STRING,
        number=2,
    )


class Tool(proto.Message):
    r"""Tool details that the model may use to generate response.

    A ``Tool`` is a piece of code that enables the system to interact
    with external systems to perform an action, or set of actions,
    outside of knowledge and scope of the model.

    Attributes:
        function_declarations (MutableSequence[google.ai.generativelanguage_v1beta.types.FunctionDeclaration]):
            Optional. A list of ``FunctionDeclarations`` available to
            the model that can be used for function calling.

            The model or system does not execute the function. Instead
            the defined function may be returned as a
            [FunctionCall][content.part.function_call] with arguments to
            the client side for execution. The model may decide to call
            a subset of these functions by populating
            [FunctionCall][content.part.function_call] in the response.
            The next conversation turn may contain a
            [FunctionResponse][content.part.function_response] with the
            [content.role] "function" generation context for the next
            model turn.
    """

    function_declarations: MutableSequence["FunctionDeclaration"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="FunctionDeclaration",
    )


class ToolConfig(proto.Message):
    r"""The Tool configuration containing parameters for specifying ``Tool``
    use in the request.

    Attributes:
        function_calling_config (google.ai.generativelanguage_v1beta.types.FunctionCallingConfig):
            Optional. Function calling config.
    """

    function_calling_config: "FunctionCallingConfig" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="FunctionCallingConfig",
    )


class FunctionCallingConfig(proto.Message):
    r"""Configuration for specifying function calling behavior.

    Attributes:
        mode (google.ai.generativelanguage_v1beta.types.FunctionCallingConfig.Mode):
            Optional. Specifies the mode in which
            function calling should execute. If unspecified,
            the default value will be set to AUTO.
        allowed_function_names (MutableSequence[str]):
            Optional. A set of function names that, when provided,
            limits the functions the model will call.

            This should only be set when the Mode is ANY. Function names
            should match [FunctionDeclaration.name]. With mode set to
            ANY, model will predict a function call from the set of
            function names provided.
    """

    class Mode(proto.Enum):
        r"""Defines the execution behavior for function calling by
        defining the execution mode.

        Values:
            MODE_UNSPECIFIED (0):
                Unspecified function calling mode. This value
                should not be used.
            AUTO (1):
                Default model behavior, model decides to
                predict either a function call or a natural
                language repspose.
            ANY (2):
                Model is constrained to always predicting a function call
                only. If "allowed_function_names" are set, the predicted
                function call will be limited to any one of
                "allowed_function_names", else the predicted function call
                will be any one of the provided "function_declarations".
            NONE (3):
                Model will not predict any function call.
                Model behavior is same as when not passing any
                function declarations.
        """
        MODE_UNSPECIFIED = 0
        AUTO = 1
        ANY = 2
        NONE = 3

    mode: Mode = proto.Field(
        proto.ENUM,
        number=1,
        enum=Mode,
    )
    allowed_function_names: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=2,
    )


class FunctionDeclaration(proto.Message):
    r"""Structured representation of a function declaration as defined by
    the `OpenAPI 3.03
    specification <https://spec.openapis.org/oas/v3.0.3>`__. Included in
    this declaration are the function name and parameters. This
    FunctionDeclaration is a representation of a block of code that can
    be used as a ``Tool`` by the model and executed by the client.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        name (str):
            Required. The name of the function.
            Must be a-z, A-Z, 0-9, or contain underscores
            and dashes, with a maximum length of 63.
        description (str):
            Required. A brief description of the
            function.
        parameters (google.ai.generativelanguage_v1beta.types.Schema):
            Optional. Describes the parameters to this
            function. Reflects the Open API 3.03 Parameter
            Object string Key: the name of the parameter.
            Parameter names are case sensitive. Schema
            Value: the Schema defining the type used for the
            parameter.

            This field is a member of `oneof`_ ``_parameters``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    description: str = proto.Field(
        proto.STRING,
        number=2,
    )
    parameters: "Schema" = proto.Field(
        proto.MESSAGE,
        number=3,
        optional=True,
        message="Schema",
    )


class FunctionCall(proto.Message):
    r"""A predicted ``FunctionCall`` returned from the model that contains a
    string representing the ``FunctionDeclaration.name`` with the
    arguments and their values.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        name (str):
            Required. The name of the function to call.
            Must be a-z, A-Z, 0-9, or contain underscores
            and dashes, with a maximum length of 63.
        args (google.protobuf.struct_pb2.Struct):
            Optional. The function parameters and values
            in JSON object format.

            This field is a member of `oneof`_ ``_args``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    args: struct_pb2.Struct = proto.Field(
        proto.MESSAGE,
        number=2,
        optional=True,
        message=struct_pb2.Struct,
    )


class FunctionResponse(proto.Message):
    r"""The result output from a ``FunctionCall`` that contains a string
    representing the ``FunctionDeclaration.name`` and a structured JSON
    object containing any output from the function is used as context to
    the model. This should contain the result of a\ ``FunctionCall``
    made based on model prediction.

    Attributes:
        name (str):
            Required. The name of the function to call.
            Must be a-z, A-Z, 0-9, or contain underscores
            and dashes, with a maximum length of 63.
        response (google.protobuf.struct_pb2.Struct):
            Required. The function response in JSON
            object format.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    response: struct_pb2.Struct = proto.Field(
        proto.MESSAGE,
        number=2,
        message=struct_pb2.Struct,
    )


class Schema(proto.Message):
    r"""The ``Schema`` object allows the definition of input and output data
    types. These types can be objects, but also primitives and arrays.
    Represents a select subset of an `OpenAPI 3.0 schema
    object <https://spec.openapis.org/oas/v3.0.3#schema>`__.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        type_ (google.ai.generativelanguage_v1beta.types.Type):
            Required. Data type.
        format_ (str):
            Optional. The format of the data. This is
            used only for primitive datatypes. Supported
            formats:

             for NUMBER type: float, double
             for INTEGER type: int32, int64
        description (str):
            Optional. A brief description of the
            parameter. This could contain examples of use.
            Parameter description may be formatted as
            Markdown.
        nullable (bool):
            Optional. Indicates if the value may be null.
        enum (MutableSequence[str]):
            Optional. Possible values of the element of Type.STRING with
            enum format. For example we can define an Enum Direction as
            : {type:STRING, format:enum, enum:["EAST", NORTH", "SOUTH",
            "WEST"]}
        items (google.ai.generativelanguage_v1beta.types.Schema):
            Optional. Schema of the elements of
            Type.ARRAY.

            This field is a member of `oneof`_ ``_items``.
        properties (MutableMapping[str, google.ai.generativelanguage_v1beta.types.Schema]):
            Optional. Properties of Type.OBJECT.
        required (MutableSequence[str]):
            Optional. Required properties of Type.OBJECT.
    """

    type_: "Type" = proto.Field(
        proto.ENUM,
        number=1,
        enum="Type",
    )
    format_: str = proto.Field(
        proto.STRING,
        number=2,
    )
    description: str = proto.Field(
        proto.STRING,
        number=3,
    )
    nullable: bool = proto.Field(
        proto.BOOL,
        number=4,
    )
    enum: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=5,
    )
    items: "Schema" = proto.Field(
        proto.MESSAGE,
        number=6,
        optional=True,
        message="Schema",
    )
    properties: MutableMapping[str, "Schema"] = proto.MapField(
        proto.STRING,
        proto.MESSAGE,
        number=7,
        message="Schema",
    )
    required: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=8,
    )


class GroundingPassage(proto.Message):
    r"""Passage included inline with a grounding configuration.

    Attributes:
        id (str):
            Identifier for the passage for attributing
            this passage in grounded answers.
        content (google.ai.generativelanguage_v1beta.types.Content):
            Content of the passage.
    """

    id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    content: "Content" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="Content",
    )


class GroundingPassages(proto.Message):
    r"""A repeated list of passages.

    Attributes:
        passages (MutableSequence[google.ai.generativelanguage_v1beta.types.GroundingPassage]):
            List of passages.
    """

    passages: MutableSequence["GroundingPassage"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="GroundingPassage",
    )


__all__ = tuple(sorted(__protobuf__.manifest))
