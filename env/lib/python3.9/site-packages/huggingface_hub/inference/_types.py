# coding=utf-8
# Copyright 2023-present, the HuggingFace Inc. team.
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
from typing import TYPE_CHECKING, List

from ..utils._typing import TypedDict


if TYPE_CHECKING:
    from PIL import Image


class ClassificationOutput(TypedDict):
    """Dictionary containing the output of a [`~InferenceClient.audio_classification`] and  [`~InferenceClient.image_classification`] task.

    Args:
        label (`str`):
            The label predicted by the model.
        score (`float`):
            The score of the label predicted by the model.
    """

    label: str
    score: float


class ConversationalOutputConversation(TypedDict):
    """Dictionary containing the "conversation" part of a [`~InferenceClient.conversational`] task.

    Args:
        generated_responses (`List[str]`):
            A list of the responses from the model.
        past_user_inputs (`List[str]`):
            A list of the inputs from the user. Must be the same length as `generated_responses`.
    """

    generated_responses: List[str]
    past_user_inputs: List[str]


class ConversationalOutput(TypedDict):
    """Dictionary containing the output of a  [`~InferenceClient.conversational`] task.

    Args:
        generated_text (`str`):
            The last response from the model.
        conversation (`ConversationalOutputConversation`):
            The past conversation.
        warnings (`List[str]`):
            A list of warnings associated with the process.
    """

    conversation: ConversationalOutputConversation
    generated_text: str
    warnings: List[str]


class ImageSegmentationOutput(TypedDict):
    """Dictionary containing information about a [`~InferenceClient.image_segmentation`] task. In practice, image segmentation returns a
    list of `ImageSegmentationOutput` with 1 item per mask.

    Args:
        label (`str`):
            The label corresponding to the mask.
        mask (`Image`):
            An Image object representing the mask predicted by the model.
        score (`float`):
            The score associated with the label for this mask.
    """

    label: str
    mask: "Image"
    score: float
