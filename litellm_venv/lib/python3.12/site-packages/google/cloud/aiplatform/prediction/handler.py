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
import logging
from typing import Optional, Type
import traceback

try:
    from fastapi import HTTPException
    from fastapi import Request
    from fastapi import Response
except ImportError:
    raise ImportError(
        "FastAPI is not installed and is required to build model servers. "
        'Please install the SDK using `pip install "google-cloud-aiplatform[prediction]>=1.16.0"`.'
    )

from google.cloud.aiplatform.prediction import handler_utils
from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.prediction.serializer import DefaultSerializer


class Handler(ABC):
    """Interface for Handler class to handle prediction requests."""

    @abstractmethod
    def __init__(
        self,
        artifacts_uri: str,
        predictor: Optional[Type[Predictor]] = None,
    ):
        """Initializes a Handler instance.

        Args:
            artifacts_uri (str):
                Required. The value of the environment variable AIP_STORAGE_URI.
            predictor (Type[Predictor]):
                Optional. The Predictor class this handler uses to initiate predictor
                instance if given.
        """
        pass

    @abstractmethod
    def handle(self, request: Request) -> Response:
        """Handles a prediction request.

        Args:
            request (Request):
                The request sent to the application.

        Returns:
            The response of the prediction request.
        """
        pass


class PredictionHandler(Handler):
    """Default prediction handler for the prediction requests sent to the application."""

    def __init__(
        self,
        artifacts_uri: str,
        predictor: Optional[Type[Predictor]] = None,
    ):
        """Initializes a Handler instance.

        Args:
            artifacts_uri (str):
                Required. The value of the environment variable AIP_STORAGE_URI.
            predictor (Type[Predictor]):
                Optional. The Predictor class this handler uses to initiate predictor
                instance if given.

        Raises:
            ValueError: If predictor is None.
        """
        if predictor is None:
            raise ValueError(
                "PredictionHandler must have a predictor class passed to the init function."
            )

        self._predictor = predictor()
        self._predictor.load(artifacts_uri)

    async def handle(self, request: Request) -> Response:
        """Handles a prediction request.

        Args:
            request (Request):
                Required. The prediction request sent to the application.

        Returns:
            The response of the prediction request.

        Raises:
            HTTPException: If any exception is thrown from predictor object.
        """
        request_body = await request.body()
        content_type = handler_utils.get_content_type_from_headers(request.headers)
        prediction_input = DefaultSerializer.deserialize(request_body, content_type)

        try:
            prediction_results = self._predictor.postprocess(
                self._predictor.predict(self._predictor.preprocess(prediction_input))
            )
        except HTTPException:
            raise
        except Exception as exception:
            error_message = (
                "The following exception has occurred: {}. Arguments: {}.".format(
                    type(exception).__name__, exception.args
                )
            )
            logging.info(
                "{}\\nTraceback: {}".format(error_message, traceback.format_exc())
            )

            # Converts all other exceptions to HTTPException.
            raise HTTPException(status_code=500, detail=error_message)

        accept = handler_utils.get_accept_from_headers(request.headers)
        data = DefaultSerializer.serialize(prediction_results, accept)
        return Response(content=data, media_type=accept)
