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
"""Type definitions for the models service."""
from __future__ import annotations

from collections.abc import Mapping
import csv
import dataclasses
import datetime
import json
import pathlib
import re
from typing import Any, Iterable, TypedDict, Union
import urllib.request

import google.ai.generativelanguage as glm
from google.generativeai import string_utils

__all__ = [
    "Model",
    "ModelNameOptions",
    "AnyModelNameOptions",
    "BaseModelNameOptions",
    "TunedModelNameOptions",
    "ModelsIterable",
    "TunedModel",
    "TunedModelState",
]

TunedModelState = glm.TunedModel.State

TunedModelStateOptions = Union[None, str, int, TunedModelState]

# fmt: off
_TUNED_MODEL_STATES: dict[TunedModelStateOptions, TunedModelState] = {
    TunedModelState.ACTIVE: TunedModelState.ACTIVE,
    int(TunedModelState.ACTIVE): TunedModelState.ACTIVE,
    "active": TunedModelState.ACTIVE,

    TunedModelState.CREATING: TunedModelState.CREATING,
    int(TunedModelState.CREATING): TunedModelState.CREATING,
    "creating": TunedModelState.CREATING,

    TunedModelState.FAILED: TunedModelState.FAILED,
    int(TunedModelState.FAILED): TunedModelState.FAILED,
    "failed": TunedModelState.FAILED,

    TunedModelState.STATE_UNSPECIFIED: TunedModelState.STATE_UNSPECIFIED,
    int(TunedModelState.STATE_UNSPECIFIED): TunedModelState.STATE_UNSPECIFIED,
    "state_unspecified": TunedModelState.STATE_UNSPECIFIED,
    "unspecified": TunedModelState.STATE_UNSPECIFIED,
    None: TunedModelState.STATE_UNSPECIFIED,
}
# fmt: on


def to_tuned_model_state(x: TunedModelStateOptions) -> TunedModelState:
    if isinstance(x, str):
        x = x.lower()
    return _TUNED_MODEL_STATES[x]


@string_utils.prettyprint
@dataclasses.dataclass
class Model:
    """A dataclass representation of a `glm.Model`.

    Attributes:
        name: The resource name of the `Model`. Format: `models/{model}` with a `{model}` naming
           convention of: "{base_model_id}-{version}". For example: `models/chat-bison-001`.
        base_model_id: The base name of the model. For example: `chat-bison`.
        version:  The major version number of the model. For example: `001`.
        display_name: The human-readable name of the model. E.g. `"Chat Bison"`. The name can be up
           to 128 characters long and can consist of any UTF-8 characters.
        description: A short description of the model.
        input_token_limit: Maximum number of input tokens allowed for this model.
        output_token_limit: Maximum number of output tokens available for this model.
        supported_generation_methods: lists which methods are supported by the model. The method
          names are defined as Pascal case strings, such as `generateMessage` which correspond to
          API methods.
    """

    name: str
    base_model_id: str
    version: str
    display_name: str
    description: str
    input_token_limit: int
    output_token_limit: int
    supported_generation_methods: list[str]
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None


def _fix_microseconds(match):
    # microseconds needs exactly 6 digits
    fraction = float(match.group(0))
    return f".{int(round(fraction*1e6)):06d}"


def idecode_time(parent: dict["str", Any], name: str):
    time = parent.pop(name, None)
    if time is not None:
        if "." in time:
            time = re.sub(r"\.\d+", _fix_microseconds, time)
            dt = datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%fZ")
        else:
            dt = datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%SZ")

        dt = dt.replace(tzinfo=datetime.timezone.utc)
        parent[name] = dt


def decode_tuned_model(tuned_model: glm.TunedModel | dict["str", Any]) -> TunedModel:
    if isinstance(tuned_model, glm.TunedModel):
        tuned_model = type(tuned_model).to_dict(tuned_model)  # pytype: disable=attribute-error
    tuned_model["state"] = to_tuned_model_state(tuned_model.pop("state", None))

    base_model = tuned_model.pop("base_model", None)
    tuned_model_source = tuned_model.pop("tuned_model_source", None)
    if base_model is not None:
        tuned_model["base_model"] = base_model
        tuned_model["source_model"] = base_model
    elif tuned_model_source is not None:
        tuned_model["base_model"] = tuned_model_source["base_model"]
        tuned_model["source_model"] = tuned_model_source["tuned_model"]

    idecode_time(tuned_model, "create_time")
    idecode_time(tuned_model, "update_time")

    task = tuned_model.pop("tuning_task", None)
    if task is not None:
        hype = task.pop("hyperparameters", None)
        if hype is not None:
            hype = Hyperparameters(**hype)
            task["hyperparameters"] = hype

        idecode_time(task, "start_time")
        idecode_time(task, "complete_time")

        snapshots = task.pop("snapshots", None)
        if snapshots is not None:
            for snap in snapshots:
                idecode_time(snap, "compute_time")
            task["snapshots"] = snapshots
        task = TuningTask(**task)
        tuned_model["tuning_task"] = task
    return TunedModel(**tuned_model)


@string_utils.prettyprint
@dataclasses.dataclass
class TunedModel:
    """A dataclass representation of a `glm.TunedModel`."""

    name: str | None = None
    source_model: str | None = None
    base_model: str | None = None
    display_name: str = ""
    description: str = ""
    temperature: float | None = None
    top_p: float | None = None
    top_k: float | None = None
    state: TunedModelState = TunedModelState.STATE_UNSPECIFIED
    create_time: datetime.datetime | None = None
    update_time: datetime.datetime | None = None
    tuning_task: TuningTask | None = None


@string_utils.prettyprint
@dataclasses.dataclass
class TuningTask:
    start_time: datetime.datetime | None = None
    complete_time: datetime.datetime | None = None
    snapshots: list[TuningSnapshot] = dataclasses.field(default_factory=list)
    hyperparameters: Hyperparameters | None = None


class TuningExampleDict(TypedDict):
    text_input: str
    output: str


TuningExampleOptions = Union[TuningExampleDict, glm.TuningExample, tuple[str, str], list[str]]

# TODO(markdaoust): gs:// URLS? File-type argument for files without extension?
TuningDataOptions = Union[
    pathlib.Path,
    str,
    glm.Dataset,
    Mapping[str, Iterable[str]],
    Iterable[TuningExampleOptions],
]


def encode_tuning_data(
    data: TuningDataOptions, input_key="text_input", output_key="output"
) -> glm.Dataset:
    if isinstance(data, glm.Dataset):
        return data

    if isinstance(data, str):
        # Strings are either URLs or system paths.
        if re.match(r"^\w+://\S+$", data):
            data = _normalize_url(data)
        else:
            # Normalize system paths to use pathlib
            data = pathlib.Path(data)

    if isinstance(data, (str, pathlib.Path)):
        if isinstance(data, str):
            f = urllib.request.urlopen(data)
            # csv needs strings, json does not.
            content = (line.decode("utf-8") for line in f)
        else:
            f = data.open("r")
            content = f

        if str(data).lower().endswith(".json"):
            with f:
                data = json.load(f)
        else:
            with f:
                data = csv.DictReader(content)
                return _convert_iterable(data, input_key, output_key)

    if hasattr(data, "keys"):
        return _convert_dict(data, input_key, output_key)
    else:
        return _convert_iterable(data, input_key, output_key)


def _normalize_url(url: str) -> str:
    sheet_base = "https://docs.google.com/spreadsheets"
    if url.startswith(sheet_base):
        # Normalize google-sheets URLs to download the csv.
        id_match = re.match(f"{sheet_base}/d/[^/]+", url)
        if id_match is None:
            raise ValueError("Incomplete Google Sheets URL: {data}")

        if tab_match := re.search(r"gid=(\d+)", url):
            tab_param = f"&gid={tab_match.group(1)}"
        else:
            tab_param = ""

        url = f"{id_match.group(0)}/export?format=csv{tab_param}"

    return url


def _convert_dict(data, input_key, output_key):
    new_data = list()

    try:
        inputs = data[input_key]
    except KeyError as e:
        raise KeyError(f'input_key is "{input_key}", but data has keys: {sorted(data.keys())}')

    try:
        outputs = data[output_key]
    except KeyError as e:
        raise KeyError(f'output_key is "{output_key}", but data has keys: {sorted(data.keys())}')

    for i, o in zip(inputs, outputs):
        new_data.append(glm.TuningExample({"text_input": str(i), "output": str(o)}))
    return glm.Dataset(examples=glm.TuningExamples(examples=new_data))


def _convert_iterable(data, input_key, output_key):
    new_data = list()
    for example in data:
        example = encode_tuning_example(example, input_key, output_key)
        new_data.append(example)
    return glm.Dataset(examples=glm.TuningExamples(examples=new_data))


def encode_tuning_example(example: TuningExampleOptions, input_key, output_key):
    if isinstance(example, glm.TuningExample):
        return example
    elif isinstance(example, (tuple, list)):
        a, b = example
        example = glm.TuningExample(text_input=a, output=b)
    else:  # dict
        example = glm.TuningExample(text_input=example[input_key], output=example[output_key])
    return example


@string_utils.prettyprint
@dataclasses.dataclass
class TuningSnapshot:
    step: int
    epoch: int
    mean_score: float
    compute_time: datetime.datetime


@string_utils.prettyprint
@dataclasses.dataclass
class Hyperparameters:
    epoch_count: int = 0
    batch_size: int = 0
    learning_rate: float = 0.0


BaseModelNameOptions = Union[str, Model, glm.Model]
TunedModelNameOptions = Union[str, TunedModel, glm.TunedModel]
AnyModelNameOptions = Union[str, Model, glm.Model, TunedModel, glm.TunedModel]
ModelNameOptions = AnyModelNameOptions


def make_model_name(name: AnyModelNameOptions):
    if isinstance(name, (Model, glm.Model, TunedModel, glm.TunedModel)):
        name = name.name  # pytype: disable=attribute-error
    elif isinstance(name, str):
        name = name
    else:
        raise TypeError("Expected: str, Model, or TunedModel")

    if not (name.startswith("models/") or name.startswith("tunedModels/")):
        raise ValueError("Model names should start with `models/` or `tunedModels/`, got: {name}")

    return name


ModelsIterable = Iterable[Model]
TunedModelsIterable = Iterable[TunedModel]


@string_utils.prettyprint
@dataclasses.dataclass
class TokenCount:
    """A dataclass representation of a `glm.TokenCountResponse`.

    Attributes:
        token_count: The number of tokens returned by the model's tokenizer for the `input_text`.
        token_count_limit:
    """

    token_count: int
    token_count_limit: int

    def over_limit(self):
        return self.token_count > self.token_count_limit
