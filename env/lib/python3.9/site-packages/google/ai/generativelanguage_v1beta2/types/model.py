# -*- coding: utf-8 -*-
# Copyright 2023 Google LLC
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

import proto  # type: ignore

__protobuf__ = proto.module(
    package="google.ai.generativelanguage.v1beta2",
    manifest={
        "Model",
    },
)


class Model(proto.Message):
    r"""Information about a Generative Language Model.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        name (str):
            Required. The resource name of the ``Model``.

            Format: ``models/{model}`` with a ``{model}`` naming
            convention of:

            -  "{base_model_id}-{version}"

            Examples:

            -  ``models/chat-bison-001``
        base_model_id (str):
            Required. The name of the base model, pass this to the
            generation request.

            Examples:

            -  ``chat-bison``
        version (str):
            Required. The version number of the model.

            This represents the major version
        display_name (str):
            The human-readable name of the model. E.g.
            "Chat Bison".
            The name can be up to 128 characters long and
            can consist of any UTF-8 characters.
        description (str):
            A short description of the model.
        input_token_limit (int):
            Maximum number of input tokens allowed for
            this model.
        output_token_limit (int):
            Maximum number of output tokens available for
            this model.
        supported_generation_methods (MutableSequence[str]):
            The model's supported generation methods.

            The method names are defined as Pascal case strings, such as
            ``generateMessage`` which correspond to API methods.
        temperature (float):
            Controls the randomness of the output.

            Values can range over ``[0.0,1.0]``, inclusive. A value
            closer to ``1.0`` will produce responses that are more
            varied, while a value closer to ``0.0`` will typically
            result in less surprising responses from the model. This
            value specifies default to be used by the backend while
            making the call to the model.

            This field is a member of `oneof`_ ``_temperature``.
        top_p (float):
            For Nucleus sampling.

            Nucleus sampling considers the smallest set of tokens whose
            probability sum is at least ``top_p``. This value specifies
            default to be used by the backend while making the call to
            the model.

            This field is a member of `oneof`_ ``_top_p``.
        top_k (int):
            For Top-k sampling.

            Top-k sampling considers the set of ``top_k`` most probable
            tokens. This value specifies default to be used by the
            backend while making the call to the model.

            This field is a member of `oneof`_ ``_top_k``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    base_model_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    version: str = proto.Field(
        proto.STRING,
        number=3,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=4,
    )
    description: str = proto.Field(
        proto.STRING,
        number=5,
    )
    input_token_limit: int = proto.Field(
        proto.INT32,
        number=6,
    )
    output_token_limit: int = proto.Field(
        proto.INT32,
        number=7,
    )
    supported_generation_methods: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=8,
    )
    temperature: float = proto.Field(
        proto.FLOAT,
        number=9,
        optional=True,
    )
    top_p: float = proto.Field(
        proto.FLOAT,
        number=10,
        optional=True,
    )
    top_k: int = proto.Field(
        proto.INT32,
        number=11,
        optional=True,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
