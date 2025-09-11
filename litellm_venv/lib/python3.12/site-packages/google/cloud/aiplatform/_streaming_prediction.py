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
"""Streaming prediction functions."""

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Sequence

from google.cloud.aiplatform_v1.services import prediction_service
from google.cloud.aiplatform_v1.types import (
    prediction_service as prediction_service_types,
)
from google.cloud.aiplatform_v1.types import (
    types as aiplatform_types,
)


def value_to_tensor(value: Any) -> aiplatform_types.Tensor:
    """Converts a Python value to `Tensor`.

    Args:
        value: A value to convert

    Returns:
        A `Tensor` object
    """
    if value is None:
        return aiplatform_types.Tensor()
    elif isinstance(value, int):
        return aiplatform_types.Tensor(int_val=[value])
    elif isinstance(value, float):
        return aiplatform_types.Tensor(float_val=[value])
    elif isinstance(value, bool):
        return aiplatform_types.Tensor(bool_val=[value])
    elif isinstance(value, str):
        return aiplatform_types.Tensor(string_val=[value])
    elif isinstance(value, bytes):
        return aiplatform_types.Tensor(bytes_val=[value])
    elif isinstance(value, list):
        return aiplatform_types.Tensor(list_val=[value_to_tensor(x) for x in value])
    elif isinstance(value, dict):
        return aiplatform_types.Tensor(
            struct_val={k: value_to_tensor(v) for k, v in value.items()}
        )
    raise TypeError(f"Unsupported value type {type(value)}")


def tensor_to_value(tensor_pb: aiplatform_types.Tensor) -> Any:
    """Converts `Tensor` to a Python value.

    Args:
        tensor_pb: A `Tensor` object

    Returns:
        A corresponding Python object
    """
    list_of_fields = tensor_pb.ListFields()
    if not list_of_fields:
        return None
    descriptor, value = tensor_pb.ListFields()[0]
    if descriptor.name == "list_val":
        return [tensor_to_value(x) for x in value]
    elif descriptor.name == "struct_val":
        return {k: tensor_to_value(v) for k, v in value.items()}
    if not isinstance(value, Sequence):
        raise TypeError(f"Unexpected non-list tensor value {value}")
    if len(value) == 1:
        return value[0]
    else:
        return value


def predict_stream_of_tensor_lists_from_single_tensor_list(
    prediction_service_client: prediction_service.PredictionServiceClient,
    endpoint_name: str,
    tensor_list: List[aiplatform_types.Tensor],
    parameters_tensor: Optional[aiplatform_types.Tensor] = None,
) -> Iterator[List[aiplatform_types.Tensor]]:
    """Predicts a stream of lists of `Tensor` objects from a single list of `Tensor` objects.

    Args:
        tensor_list: Model input as a list of `Tensor` objects.
        parameters_tensor: Optional. Prediction parameters in `Tensor` form.
        prediction_service_client: A PredictionServiceClient object.
        endpoint_name: Resource name of Endpoint or PublisherModel.

    Yields:
        A generator of model prediction `Tensor` lists.
    """
    request = prediction_service_types.StreamingPredictRequest(
        endpoint=endpoint_name,
        inputs=tensor_list,
        parameters=parameters_tensor,
    )
    for response in prediction_service_client.server_streaming_predict(request=request):
        yield response.outputs


async def predict_stream_of_tensor_lists_from_single_tensor_list_async(
    prediction_service_async_client: prediction_service.PredictionServiceAsyncClient,
    endpoint_name: str,
    tensor_list: List[aiplatform_types.Tensor],
    parameters_tensor: Optional[aiplatform_types.Tensor] = None,
) -> AsyncIterator[List[aiplatform_types.Tensor]]:
    """Asynchronously predicts a stream of lists of `Tensor` objects from a single list of `Tensor` objects.

    Args:
        tensor_list: Model input as a list of `Tensor` objects.
        parameters_tensor: Optional. Prediction parameters in `Tensor` form.
        prediction_service_async_client: A PredictionServiceAsyncClient object.
        endpoint_name: Resource name of Endpoint or PublisherModel.

    Yields:
        A generator of model prediction `Tensor` lists.
    """
    request = prediction_service_types.StreamingPredictRequest(
        endpoint=endpoint_name,
        inputs=tensor_list,
        parameters=parameters_tensor,
    )
    async for response in await prediction_service_async_client.server_streaming_predict(
        request=request
    ):
        yield response.outputs


def predict_stream_of_dict_lists_from_single_dict_list(
    prediction_service_client: prediction_service.PredictionServiceClient,
    endpoint_name: str,
    dict_list: List[Dict[str, Any]],
    parameters: Optional[Dict[str, Any]] = None,
) -> Iterator[List[Dict[str, Any]]]:
    """Predicts a stream of lists of dicts from a stream of lists of dicts.

    Args:
        dict_list: Model input as a list of `dict` objects.
        parameters: Optional. Prediction parameters `dict` form.
        prediction_service_client: A PredictionServiceClient object.
        endpoint_name: Resource name of Endpoint or PublisherModel.

    Yields:
        A generator of model prediction dict lists.
    """
    tensor_list = [value_to_tensor(d) for d in dict_list]
    parameters_tensor = value_to_tensor(parameters) if parameters else None
    for tensor_list in predict_stream_of_tensor_lists_from_single_tensor_list(
        prediction_service_client=prediction_service_client,
        endpoint_name=endpoint_name,
        tensor_list=tensor_list,
        parameters_tensor=parameters_tensor,
    ):
        yield [tensor_to_value(tensor._pb) for tensor in tensor_list]


async def predict_stream_of_dict_lists_from_single_dict_list_async(
    prediction_service_async_client: prediction_service.PredictionServiceAsyncClient,
    endpoint_name: str,
    dict_list: List[Dict[str, Any]],
    parameters: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[List[Dict[str, Any]]]:
    """Asynchronously predicts a stream of lists of dicts from a stream of lists of dicts.

    Args:
        dict_list: Model input as a list of `dict` objects.
        parameters: Optional. Prediction parameters `dict` form.
        prediction_service_async_client: A PredictionServiceAsyncClient object.
        endpoint_name: Resource name of Endpoint or PublisherModel.

    Yields:
        A generator of model prediction dict lists.
    """
    tensor_list = [value_to_tensor(d) for d in dict_list]
    parameters_tensor = value_to_tensor(parameters) if parameters else None
    async for tensor_list in predict_stream_of_tensor_lists_from_single_tensor_list_async(
        prediction_service_async_client=prediction_service_async_client,
        endpoint_name=endpoint_name,
        tensor_list=tensor_list,
        parameters_tensor=parameters_tensor,
    ):
        yield [tensor_to_value(tensor._pb) for tensor in tensor_list]


def predict_stream_of_dicts_from_single_dict(
    prediction_service_client: prediction_service.PredictionServiceClient,
    endpoint_name: str,
    instance: Dict[str, Any],
    parameters: Optional[Dict[str, Any]] = None,
) -> Iterator[Dict[str, Any]]:
    """Predicts a stream of dicts from a single instance dict.

    Args:
        instance: A single input instance `dict`.
        parameters: Optional. Prediction parameters `dict`.
        prediction_service_client: A PredictionServiceClient object.
        endpoint_name: Resource name of Endpoint or PublisherModel.

    Yields:
        A generator of model prediction dicts.
    """
    for dict_list in predict_stream_of_dict_lists_from_single_dict_list(
        prediction_service_client=prediction_service_client,
        endpoint_name=endpoint_name,
        dict_list=[instance],
        parameters=parameters,
    ):
        if len(dict_list) > 1:
            raise ValueError(
                f"Expected to receive a single output, but got {dict_list}"
            )
        yield dict_list[0]


async def predict_stream_of_dicts_from_single_dict_async(
    prediction_service_async_client: prediction_service.PredictionServiceAsyncClient,
    endpoint_name: str,
    instance: Dict[str, Any],
    parameters: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Asynchronously predicts a stream of dicts from a single instance dict.

    Args:
        instance: A single input instance `dict`.
        parameters: Optional. Prediction parameters `dict`.
        prediction_service_async_client: A PredictionServiceAsyncClient object.
        endpoint_name: Resource name of Endpoint or PublisherModel.

    Yields:
        A generator of model prediction dicts.
    """
    async for dict_list in predict_stream_of_dict_lists_from_single_dict_list_async(
        prediction_service_async_client=prediction_service_async_client,
        endpoint_name=endpoint_name,
        dict_list=[instance],
        parameters=parameters,
    ):
        if len(dict_list) > 1:
            raise ValueError(
                f"Expected to receive a single output, but got {dict_list}"
            )
        yield dict_list[0]
