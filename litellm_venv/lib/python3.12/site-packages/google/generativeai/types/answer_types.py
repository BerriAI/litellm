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
from __future__ import annotations

from typing import Union

import google.ai.generativelanguage as glm

__all__ = ["Answer"]

FinishReason = glm.Candidate.FinishReason

FinishReasonOptions = Union[int, str, FinishReason]

_FINISH_REASONS: dict[FinishReasonOptions, FinishReason] = {
    FinishReason.FINISH_REASON_UNSPECIFIED: FinishReason.FINISH_REASON_UNSPECIFIED,
    0: FinishReason.FINISH_REASON_UNSPECIFIED,
    "finish_reason_unspecified": FinishReason.FINISH_REASON_UNSPECIFIED,
    "unspecified": FinishReason.FINISH_REASON_UNSPECIFIED,
    FinishReason.STOP: FinishReason.STOP,
    1: FinishReason.STOP,
    "finish_reason_stop": FinishReason.STOP,
    "stop": FinishReason.STOP,
    FinishReason.MAX_TOKENS: FinishReason.MAX_TOKENS,
    2: FinishReason.MAX_TOKENS,
    "finish_reason_max_tokens": FinishReason.MAX_TOKENS,
    "max_tokens": FinishReason.MAX_TOKENS,
    FinishReason.SAFETY: FinishReason.SAFETY,
    3: FinishReason.SAFETY,
    "finish_reason_safety": FinishReason.SAFETY,
    "safety": FinishReason.SAFETY,
    FinishReason.RECITATION: FinishReason.RECITATION,
    4: FinishReason.RECITATION,
    "finish_reason_recitation": FinishReason.RECITATION,
    "recitation": FinishReason.RECITATION,
    FinishReason.OTHER: FinishReason.OTHER,
    5: FinishReason.OTHER,
    "finish_reason_other": FinishReason.OTHER,
    "other": FinishReason.OTHER,
}


def to_finish_reason(x: FinishReasonOptions) -> FinishReason:
    if isinstance(x, str):
        x = x.lower()
    return _FINISH_REASONS[x]
