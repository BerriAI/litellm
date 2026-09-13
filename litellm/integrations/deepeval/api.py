# duplicate -> https://github.com/confident-ai/deepeval/blob/main/deepeval/confident/api.py
import logging
from enum import Enum
from typing import Final

import httpx

from litellm._logging import verbose_logger

DEEPEVAL_BASE_URL: Final = "https://deepeval.confident-ai.com"
DEEPEVAL_BASE_URL_EU: Final = "https://eu.deepeval.confident-ai.com"
API_BASE_URL: Final = "https://api.confident-ai.com"
API_BASE_URL_EU: Final = "https://eu.api.confident-ai.com"
retryable_exceptions: Final = httpx.HTTPError

from litellm.llms.custom_httpx.http_handler import (
    HTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)


def log_retry_error(details):
    exception: Final = details.get("exception")
    tries: Final = details.get("tries")
    if exception:
        logging.error("Confident AI Error: %s. Retrying: %s time(s)...", exception, tries)
    else:
        logging.error("Retrying: %s time(s)...", tries)


class HttpMethods(Enum):
    GET = "GET"
    POST = "POST"
    DELETE = "DELETE"
    PUT = "PUT"


class Endpoints(Enum):
    DATASET_ENDPOINT = "/v1/dataset"
    TEST_RUN_ENDPOINT = "/v1/test-run"
    TRACING_ENDPOINT = "/v1/tracing"
    EVENT_ENDPOINT = "/v1/event"
    FEEDBACK_ENDPOINT = "/v1/feedback"
    PROMPT_ENDPOINT = "/v1/prompt"
    RECOMMEND_ENDPOINT = "/v1/recommend-metrics"
    EVALUATE_ENDPOINT = "/evaluate"
    GUARD_ENDPOINT = "/guard"
    GUARDRAILS_ENDPOINT = "/guardrails"
    BASELINE_ATTACKS_ENDPOINT = "/generate-baseline-attacks"


class Api:
    def __init__(self, api_key: str, base_url=None):
        self.api_key = api_key
        self._headers = {
            "Content-Type": "application/json",
            # "User-Agent": "Python/Requests",
            "CONFIDENT_API_KEY": api_key,
        }
        # using the global non-eu variable for base url
        self.base_api_url = base_url or API_BASE_URL
        self.sync_http_handler = HTTPHandler()
        self.async_http_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.LoggingCallback)

    def _http_request(self, method: str, url: str, headers=None, json=None, params=None):
        if method != "POST":
            raise Exception("Only POST requests are supported")
        try:
            self.sync_http_handler.post(
                url=url,
                headers=headers,
                json=json,
                params=params,
            )
        except httpx.HTTPStatusError as e:
            raise Exception(f"DeepEval logging error: {e.response.text}")
        except Exception as e:
            raise e

    def send_request(self, method: HttpMethods, endpoint: Endpoints, body=None, params=None):
        url: Final = f"{self.base_api_url}{endpoint.value}"
        res: Final = self._http_request(
            method=method.value,
            url=url,
            headers=self._headers,
            json=body,
            params=params,
        )

        if res.status_code == 200:
            try:
                return res.json()
            except ValueError:
                return res.text
        else:
            verbose_logger.debug(res.json())
            raise Exception(res.json().get("error", res.text))

    async def a_send_request(self, method: HttpMethods, endpoint: Endpoints, body=None, params=None):
        if method != HttpMethods.POST:
            raise Exception("Only POST requests are supported")

        url: Final = f"{self.base_api_url}{endpoint.value}"
        try:
            await self.async_http_handler.post(
                url=url,
                headers=self._headers,
                json=body,
                params=params,
            )
        except httpx.HTTPStatusError as e:
            raise Exception(f"DeepEval logging error: {e.response.text}")
        except Exception as e:
            raise e
