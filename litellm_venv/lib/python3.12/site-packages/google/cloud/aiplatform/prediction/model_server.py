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

import importlib
import logging
import multiprocessing
import os
import sys
import traceback

try:
    from fastapi import FastAPI
    from fastapi import HTTPException
    from fastapi import Request
    from fastapi import Response
except ImportError:
    raise ImportError(
        "FastAPI is not installed and is required to run model servers. "
        'Please install the SDK using `pip install "google-cloud-aiplatform[prediction]>=1.16.0"`.'
    )

try:
    import uvicorn
except ImportError:
    raise ImportError(
        "Uvicorn is not installed and is required to run fastapi applications. "
        'Please install the SDK using `pip install "google-cloud-aiplatform[prediction]>=1.16.0"`.'
    )

from google.cloud.aiplatform.constants import prediction
from google.cloud.aiplatform import version


class CprModelServer:
    """Model server to do custom prediction routines."""

    def __init__(self):
        """Initializes a fastapi application and sets the configs.

        Raises:
            ValueError: If either HANDLER_MODULE or HANDLER_CLASS is not set in the
                environment variables. Or if any of AIP_HTTP_PORT, AIP_HEALTH_ROUTE,
                and AIP_PREDICT_ROUTE is not set in the environment variables.
        """
        self._init_logging()

        if "HANDLER_MODULE" not in os.environ or "HANDLER_CLASS" not in os.environ:
            raise ValueError(
                "Both of the environment variables, HANDLER_MODULE and HANDLER_CLASS "
                "need to be specified."
            )
        handler_module = importlib.import_module(os.environ.get("HANDLER_MODULE"))
        handler_class = getattr(handler_module, os.environ.get("HANDLER_CLASS"))
        self.is_default_handler = (
            handler_module == "google.cloud.aiplatform.prediction.handler"
        )

        predictor_class = None
        if "PREDICTOR_MODULE" in os.environ:
            predictor_module = importlib.import_module(
                os.environ.get("PREDICTOR_MODULE")
            )
            predictor_class = getattr(
                predictor_module, os.environ.get("PREDICTOR_CLASS")
            )

        self.handler = handler_class(
            os.environ.get("AIP_STORAGE_URI"), predictor=predictor_class
        )

        if "AIP_HTTP_PORT" not in os.environ:
            raise ValueError(
                "The environment variable AIP_HTTP_PORT needs to be specified."
            )
        if (
            "AIP_HEALTH_ROUTE" not in os.environ
            or "AIP_PREDICT_ROUTE" not in os.environ
        ):
            raise ValueError(
                "Both of the environment variables AIP_HEALTH_ROUTE and "
                "AIP_PREDICT_ROUTE need to be specified."
            )
        self.http_port = int(os.environ.get("AIP_HTTP_PORT"))
        self.health_route = os.environ.get("AIP_HEALTH_ROUTE")
        self.predict_route = os.environ.get("AIP_PREDICT_ROUTE")

        self.app = FastAPI()
        self.app.add_api_route(
            path=self.health_route,
            endpoint=self.health,
            methods=["GET"],
        )
        self.app.add_api_route(
            path=self.predict_route,
            endpoint=self.predict,
            methods=["POST"],
        )

    async def __call__(self, scope, receive, send):
        await self.app(scope, receive, send)

    def _init_logging(self):
        """Initializes the logging config."""
        logging.basicConfig(
            format="%(asctime)s: %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p",
            level=logging.INFO,
            stream=sys.stdout,
        )

    def health(self):
        """Executes a health check."""
        return {}

    async def predict(self, request: Request) -> Response:
        """Executes a prediction.

        Args:
            request (Request):
                Required. The prediction request.

        Returns:
            The response containing prediction results.

        Raises:
            HTTPException: If the handle function of the handler raises any exceptions.
        """
        try:
            return await self.handler.handle(request)
        except HTTPException:
            # Raises exception if it's a HTTPException.
            raise
        except Exception as exception:
            error_message = "An exception {} occurred. Arguments: {}.".format(
                type(exception).__name__, exception.args
            )
            logging.info(
                "{}\\nTraceback: {}".format(error_message, traceback.format_exc())
            )

            # Converts all other exceptions to HTTPException.
            if self.is_default_handler:
                raise HTTPException(
                    status_code=500,
                    detail=error_message,
                    headers={
                        prediction.CUSTOM_PREDICTION_ROUTINES_SERVER_ERROR_HEADER_KEY: version.__version__
                    },
                )
            raise HTTPException(status_code=500, detail=error_message)


def set_number_of_workers_from_env() -> None:
    """Sets the number of model server workers used by Uvicorn in the environment variable.

    The number of model server workers will be set as WEB_CONCURRENCY in the environment
    variables.
    The default number of model server workers is the number of cores.
    The following environment variables will adjust the number of workers:
        VERTEX_CPR_WEB_CONCURRENCY:
            The number of the workers. This will overwrite the number calculated by the other
            variables, min(VERTEX_CPR_WORKERS_PER_CORE * number_of_cores, VERTEX_CPR_MAX_WORKERS).
        VERTEX_CPR_WORKERS_PER_CORE:
            The number of the workers per core. The default is 1.
        VERTEX_CPR_MAX_WORKERS:
            The maximum number of workers can be used given the value of VERTEX_CPR_WORKERS_PER_CORE
            and the number of cores.
    """
    workers_per_core_str = os.getenv("VERTEX_CPR_WORKERS_PER_CORE", "1")
    max_workers_str = os.getenv("VERTEX_CPR_MAX_WORKERS")
    use_max_workers = None
    if max_workers_str:
        use_max_workers = int(max_workers_str)
    web_concurrency_str = os.getenv("VERTEX_CPR_WEB_CONCURRENCY")

    if not web_concurrency_str:
        cores = multiprocessing.cpu_count()
        workers_per_core = float(workers_per_core_str)
        default_web_concurrency = workers_per_core * cores
        web_concurrency = max(int(default_web_concurrency), 2)
        if use_max_workers:
            web_concurrency = min(web_concurrency, use_max_workers)
        web_concurrency_str = str(web_concurrency)
    os.environ["WEB_CONCURRENCY"] = web_concurrency_str
    logging.warning(
        f'Set the number of model server workers to {os.environ["WEB_CONCURRENCY"]}.'
    )


if __name__ == "__main__":
    set_number_of_workers_from_env()
    uvicorn.run(
        "google.cloud.aiplatform.prediction.model_server:CprModelServer",
        host="0.0.0.0",
        port=int(os.environ.get("AIP_HTTP_PORT")),
        factory=True,
    )
