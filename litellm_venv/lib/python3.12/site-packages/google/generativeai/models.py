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

import typing
from typing import Any, Literal

import google.ai.generativelanguage as glm
from google.generativeai import operations
from google.generativeai.client import get_default_model_client
from google.generativeai.types import model_types
from google.api_core import operation
from google.api_core import protobuf_helpers
from google.protobuf import field_mask_pb2
from google.generativeai.utils import flatten_update_paths


def get_model(
    name: model_types.AnyModelNameOptions,
    *,
    client=None,
    request_options: dict[str, Any] | None = None,
) -> model_types.Model | model_types.TunedModel:
    """Given a model name, fetch the `types.Model` or `types.TunedModel` object.

    ```
    import pprint
    model = genai.get_tuned_model(model_name):
    pprint.pprint(model)
    ```

    Args:
        name: The name of the model to fetch.
        client: The client to use.
        request_options: Options for the request.

    Returns:
        A `types.Model` or `types.TunedModel` object.
    """
    name = model_types.make_model_name(name)
    if name.startswith("models/"):
        return get_base_model(name, client=client, request_options=request_options)
    elif name.startswith("tunedModels/"):
        return get_tuned_model(name, client=client, request_options=request_options)
    else:
        raise ValueError("Model names must start with `models/` or `tunedModels/`")


def get_base_model(
    name: model_types.BaseModelNameOptions,
    *,
    client=None,
    request_options: dict[str, Any] | None = None,
) -> model_types.Model:
    """Get the `types.Model` for the given base model name.

    ```
    import pprint
    model = genai.get_model('models/chat-bison-001'):
    pprint.pprint(model)
    ```

    Args:
        name: The name of the model to fetch.
        client: The client to use.
        request_options: Options for the request.

    Returns:
        A `types.Model`.
    """
    if request_options is None:
        request_options = {}

    if client is None:
        client = get_default_model_client()

    name = model_types.make_model_name(name)
    if not name.startswith("models/"):
        raise ValueError(f"Base model names must start with `models/`, got: {name}")

    result = client.get_model(name=name, **request_options)
    result = type(result).to_dict(result)
    return model_types.Model(**result)


def get_tuned_model(
    name: model_types.TunedModelNameOptions,
    *,
    client=None,
    request_options: dict[str, Any] | None = None,
) -> model_types.TunedModel:
    """Get the `types.TunedModel` for the given tuned model name.

    ```
    import pprint
    model = genai.get_tuned_model('tunedModels/my-model-1234'):
    pprint.pprint(model)
    ```

    Args:
        name: The name of the model to fetch.
        client: The client to use.
        request_options: Options for the request.

    Returns:
        A `types.TunedModel`.
    """
    if request_options is None:
        request_options = {}

    if client is None:
        client = get_default_model_client()

    name = model_types.make_model_name(name)

    if not name.startswith("tunedModels/"):
        raise ValueError("Tuned model names must start with `tunedModels/`")

    result = client.get_tuned_model(name=name, **request_options)

    return model_types.decode_tuned_model(result)


def get_base_model_name(
    model: model_types.AnyModelNameOptions, client: glm.ModelServiceClient | None = None
):
    if isinstance(model, str):
        if model.startswith("tunedModels/"):
            model = get_model(model, client=client)
            base_model = model.base_model
        else:
            base_model = model
    elif isinstance(model, model_types.TunedModel):
        base_model = model.base_model
    elif isinstance(model, model_types.Model):
        base_model = model.name
    elif isinstance(model, glm.Model):
        base_model = model.name
    elif isinstance(model, glm.TunedModel):
        base_model = getattr(model, "base_model", None)
        if not base_model:
            base_model = model.tuned_model_source.base_model
    else:
        raise TypeError(f"Cannot understand model: {model}")

    return base_model


def list_models(
    *,
    page_size: int | None = 50,
    client: glm.ModelServiceClient | None = None,
    request_options: dict[str, Any] | None = None,
) -> model_types.ModelsIterable:
    """Lists available models.

    ```
    import pprint
    for model in genai.list_models():
        pprint.pprint(model)
    ```

    Args:
        page_size: How many `types.Models` to fetch per page (api call).
        client: You may pass a `glm.ModelServiceClient` instead of using the default client.
        request_options: Options for the request.

    Yields:
        `types.Model` objects.

    """
    if request_options is None:
        request_options = {}

    if client is None:
        client = get_default_model_client()

    for model in client.list_models(page_size=page_size, **request_options):
        model = type(model).to_dict(model)
        yield model_types.Model(**model)


def list_tuned_models(
    *,
    page_size: int | None = 50,
    client: glm.ModelServiceClient | None = None,
    request_options: dict[str, Any] | None = None,
) -> model_types.TunedModelsIterable:
    """Lists available models.

    ```
    import pprint
    for model in genai.list_tuned_models():
        pprint.pprint(model)
    ```

    Args:
        page_size: How many `types.Models` to fetch per page (api call).
        client: You may pass a `glm.ModelServiceClient` instead of using the default client.
        request_options: Options for the request.

    Yields:
        `types.TunedModel` objects.
    """
    if request_options is None:
        request_options = {}

    if client is None:
        client = get_default_model_client()

    for model in client.list_tuned_models(
        page_size=page_size,
        **request_options,
    ):
        model = type(model).to_dict(model)
        yield model_types.decode_tuned_model(model)


def create_tuned_model(
    source_model: model_types.AnyModelNameOptions,
    training_data: model_types.TuningDataOptions,
    *,
    id: str | None = None,
    display_name: str | None = None,
    description: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    epoch_count: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    input_key: str = "text_input",
    output_key: str = "output",
    client: glm.ModelServiceClient | None = None,
    request_options: dict[str, Any] | None = None,
) -> operations.CreateTunedModelOperation:
    """Launches a tuning job to create a TunedModel.

    Since tuning a model can take significant time, this API doesn't wait for the tuning to complete.
    Instead, it returns a `google.api_core.operation.Operation` object that lets you check on the
    status of the tuning job, or wait for it to complete, and check the result.

    After the job completes you can either find the resulting `TunedModel` object in
    `Operation.result()` or `palm.list_tuned_models` or `palm.get_tuned_model(model_id)`.

    ```
    my_id = "my-tuned-model-id"
    operation = palm.create_tuned_model(
      id = my_id,
      source_model="models/text-bison-001",
      training_data=[{'text_input': 'example input', 'output': 'example output'},...]
    )
    tuned_model=operation.result()      # Wait for tuning to finish

    palm.generate_text(f"tunedModels/{my_id}", prompt="...")
    ```

    Args:
        source_model: The name of the model to tune.
        training_data: The dataset to tune the model on. This must be either:
          * A `glm.Dataset`, or
          * An `Iterable` of:
            *`glm.TuningExample`,
            * {'text_input': text_input, 'output': output} dicts, or
            * `(text_input, output)` tuples.
          * A `Mapping` of `Iterable[str]` - use `input_key` and `output_key` to choose which
            columns to use as the input/output
          * A csv file (will be read with `pd.read_csv` and handles as a `Mapping`
            above). This can be:
            * A local path as a `str` or `pathlib.Path`.
            * A url for a csv file.
            * The url of a Google Sheets file.
          * A JSON file - Its contents will be handled either as an `Iterable` or `Mapping`
            above. This can be:
            * A local path as a `str` or `pathlib.Path`.
        id: The model identifier, used to refer to the model in the API
          `tunedModels/{id}`. Must be unique.
        display_name: A human-readable name for display.
        description: A description of the tuned model.
        temperature: The default temperature for the tuned model, see `types.Model` for details.
        top_p: The default `top_p` for the model, see `types.Model` for details.
        top_k: The default `top_k` for the model, see `types.Model` for details.
        epoch_count: The number of tuning epochs to run. An epoch is a pass over the whole dataset.
        batch_size: The number of examples to use in each training batch.
        learning_rate: The step size multiplier for the gradient updates.
        client: Which client to use.
        request_options: Options for the request.

    Returns:
        A [`google.api_core.operation.Operation`](https://googleapis.dev/python/google-api-core/latest/operation.html)
    """
    if request_options is None:
        request_options = {}

    if client is None:
        client = get_default_model_client()

    source_model_name = model_types.make_model_name(source_model)
    base_model_name = get_base_model_name(source_model)
    if source_model_name.startswith("models/"):
        source_model = {"base_model": source_model_name}
    elif source_model_name.startswith("tunedModels/"):
        source_model = {
            "tuned_model_source": {
                "tuned_model": source_model_name,
                "base_model": base_model_name,
            }
        }
    else:
        ValueError(f"Not understood: `{source_model=}`")

    training_data = model_types.encode_tuning_data(
        training_data, input_key=input_key, output_key=output_key
    )

    hyperparameters = glm.Hyperparameters(
        epoch_count=epoch_count,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    tuning_task = glm.TuningTask(
        training_data=training_data,
        hyperparameters=hyperparameters,
    )

    tuned_model = glm.TunedModel(
        **source_model,
        display_name=display_name,
        description=description,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        tuning_task=tuning_task,
    )
    operation = client.create_tuned_model(
        dict(tuned_model_id=id, tuned_model=tuned_model), **request_options
    )

    return operations.CreateTunedModelOperation.from_core_operation(operation)


@typing.overload
def update_tuned_model(
    tuned_model: glm.TunedModel,
    updates: None = None,
    *,
    client: glm.ModelServiceClient | None = None,
    request_options: dict[str, Any] | None = None,
) -> model_types.TunedModel:
    pass


@typing.overload
def update_tuned_model(
    tuned_model: str,
    updates: dict[str, Any],
    *,
    client: glm.ModelServiceClient | None = None,
    request_options: dict[str, Any] | None = None,
) -> model_types.TunedModel:
    pass


def update_tuned_model(
    tuned_model: str | glm.TunedModel,
    updates: dict[str, Any] | None = None,
    *,
    client: glm.ModelServiceClient | None = None,
    request_options: dict[str, Any] | None = None,
) -> model_types.TunedModel:
    """Push updates to the tuned model. Only certain attributes are updatable."""
    if request_options is None:
        request_options = {}

    if client is None:
        client = get_default_model_client()

    if isinstance(tuned_model, str):
        name = tuned_model
        if not isinstance(updates, dict):
            raise TypeError(
                "When calling `update_tuned_model(name:str, updates: dict)`,\n"
                "`updates` must be a `dict`.\n"
                f"got: {type(updates)}"
            )
        tuned_model = client.get_tuned_model(name=name, **request_options)

        updates = flatten_update_paths(updates)
        field_mask = field_mask_pb2.FieldMask()
        for path in updates.keys():
            field_mask.paths.append(path)
        for path, value in updates.items():
            _apply_update(tuned_model, path, value)
    elif isinstance(tuned_model, glm.TunedModel):
        if updates is not None:
            raise ValueError(
                "When calling `update_tuned_model(tuned_model:glm.TunedModel, updates=None)`,"
                "`updates` must not be set."
            )

        name = tuned_model.name
        was = client.get_tuned_model(name=name)
        field_mask = protobuf_helpers.field_mask(was._pb, tuned_model._pb)
    else:
        raise TypeError(
            "For `update_tuned_model(tuned_model:dict|glm.TunedModel)`,"
            f"`tuned_model` must be a `dict` or a `glm.TunedModel`. Got a: `{type(tuned_model)}`"
        )

    result = client.update_tuned_model(
        glm.UpdateTunedModelRequest(tuned_model=tuned_model, update_mask=field_mask),
        **request_options,
    )
    return model_types.decode_tuned_model(result)


def _apply_update(thing, path, value):
    parts = path.split(".")
    for part in parts[:-1]:
        thing = getattr(thing, part)
    setattr(thing, parts[-1], value)


def delete_tuned_model(
    tuned_model: model_types.TunedModelNameOptions,
    client: glm.ModelServiceClient | None = None,
    request_options: dict[str, Any] | None = None,
) -> None:
    if request_options is None:
        request_options = {}

    if client is None:
        client = get_default_model_client()

    name = model_types.make_model_name(tuned_model)
    client.delete_tuned_model(name=name, **request_options)
