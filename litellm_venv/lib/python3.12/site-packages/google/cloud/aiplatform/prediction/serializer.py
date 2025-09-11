# -*- coding: utf-8 -*-

# Copyright 2022 Google LLC
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

from abc import ABC, abstractmethod
import json
from typing import Any, Optional

try:
    from fastapi import HTTPException
except ImportError:
    raise ImportError(
        "FastAPI is not installed and is required to build model servers. "
        'Please install the SDK using `pip install "google-cloud-aiplatform[prediction]>=1.16.0"`.'
    )

from google.cloud.aiplatform.constants import prediction as prediction_constants
from google.cloud.aiplatform.prediction import handler_utils


APPLICATION_JSON = "application/json"


class Serializer(ABC):
    """Interface to implement serialization and deserialization for prediction."""

    @staticmethod
    @abstractmethod
    def deserialize(data: Any, content_type: Optional[str]) -> Any:
        """Deserializes the request data. Invoked before predict.

        Args:
            data (Any):
                Required. The request data sent to the application.
            content_type (str):
                Optional. The specified content type of the request.
        """
        pass

    @staticmethod
    @abstractmethod
    def serialize(prediction: Any, accept: Optional[str]) -> Any:
        """Serializes the prediction results. Invoked after predict.

        Args:
            prediction (Any):
                Required. The generated prediction to be sent back to clients.
            accept (str):
                Optional. The specified content type of the response.
        """
        pass


class DefaultSerializer(Serializer):
    """Default serializer for serialization and deserialization for prediction."""

    @staticmethod
    def deserialize(data: Any, content_type: Optional[str]) -> Any:
        """Deserializes the request data. Invoked before predict.

        Args:
            data (Any):
                Required. The request data sent to the application.
            content_type (str):
                Optional. The specified content type of the request.

        Raises:
            HTTPException: If Json deserialization failed or the specified content type is not
                supported.
        """
        if content_type == APPLICATION_JSON:
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"JSON deserialization failed for the request data: {data}.\n"
                        'To specify a different type, please set the "content-type" header '
                        "in the request.\nCurrently supported content-type in DefaultSerializer: "
                        f'"{APPLICATION_JSON}".'
                    ),
                )
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported content type of the request: {content_type}.\n"
                    f'Currently supported content-type in DefaultSerializer: "{APPLICATION_JSON}".'
                ),
            )

    @staticmethod
    def serialize(prediction: Any, accept: Optional[str]) -> Any:
        """Serializes the prediction results. Invoked after predict.

        Args:
            prediction (Any):
                Required. The generated prediction to be sent back to clients.
            accept (str):
                Optional. The specified content type of the response.

        Raises:
            HTTPException: If Json serialization failed or the specified accept is not supported.
        """
        accept_dict = handler_utils.parse_accept_header(accept)

        if (
            APPLICATION_JSON in accept_dict
            or prediction_constants.ANY_ACCEPT_TYPE in accept_dict
        ):
            try:
                return json.dumps(prediction)
            except TypeError:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"JSON serialization failed for the prediction result: {prediction}.\n"
                        'To specify a different type, please set the "accept" header '
                        "in the request.\nCurrently supported accept in DefaultSerializer: "
                        f'"{APPLICATION_JSON}".'
                    ),
                )
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported accept of the response: {accept}.\n"
                    f'Currently supported accept in DefaultSerializer: "{APPLICATION_JSON}".'
                ),
            )
