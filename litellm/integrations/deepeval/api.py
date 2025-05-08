# duplicate -> https://github.com/confident-ai/deepeval/blob/main/deepeval/confident/api.py
import logging
import aiohttp
import requests
from enum import Enum
from litellm._logging import verbose_logger
from tenacity import (
    retry,
    wait_exponential_jitter,
    retry_if_exception_type,
    RetryCallState,
)

DEEPEVAL_BASE_URL = "https://deepeval.confident-ai.com"
DEEPEVAL_BASE_URL_EU = "https://eu.deepeval.confident-ai.com"
API_BASE_URL = "https://api.confident-ai.com"
API_BASE_URL_EU = "https://eu.api.confident-ai.com"
retryable_exceptions = requests.exceptions.SSLError


def log_retry_error(retry_state: RetryCallState):
    exception = retry_state.outcome.exception()
    logging.error(
        f"Confident AI Error: {exception}. Retrying: {retry_state.attempt_number} time(s)..."
    )


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

    @staticmethod
    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    def _http_request(method: str, url: str, headers=None, json=None, params=None):
        session = requests.Session()
        return session.request(
            method=method,
            url=url,
            headers=headers,
            json=json,
            params=params,
            verify=True,  # SSL verification is always enabled
        )

    def send_request(
        self, method: HttpMethods, endpoint: Endpoints, body=None, params=None
    ):
        url = f"{self.base_api_url}{endpoint.value}"
        res = self._http_request(
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
        elif res.status_code == 409 and body:
            message = res.json().get("message", "Conflict occurred.")

            # Prompt the user for action
            user_input = (
                input(
                    f"{message} Would you like to overwrite it? [y/N] or change the alias [c]: "
                )
                .strip()
                .lower()
            )

            if user_input == "y":
                body["overwrite"] = True
                return self.send_request(method, endpoint, body)
            elif user_input == "c":
                new_alias = input("Enter a new alias: ").strip()
                body["alias"] = new_alias
                return self.send_request(method, endpoint, body)
            else:
                verbose_logger.debug("Aborted.")
                return None
        else:
            verbose_logger.debug(res.json())
            raise Exception(res.json().get("error", res.text))

    async def a_send_request(
        self, method: HttpMethods, endpoint: Endpoints, body=None, params=None
    ):
        url = f"{self.base_api_url}{endpoint.value}"
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=method.value,
                url=url,
                headers=self._headers,
                json=body,
                params=params,
                ssl=True,  # SSL verification enabled
            ) as res:
                if res.status == 200:
                    try:
                        return await res.json()
                    except aiohttp.ContentTypeError:
                        return await res.text()
                elif res.status == 409 and body:
                    message = (await res.json()).get("message", "Conflict occurred.")

                    user_input = (
                        input(
                            f"{message} Would you like to overwrite it? [y/N] or change the alias [c]: "
                        )
                        .strip()
                        .lower()
                    )

                    if user_input == "y":
                        body["overwrite"] = True
                        return await self.a_send_request(method, endpoint, body)
                    elif user_input == "c":
                        new_alias = input("Enter a new alias: ").strip()
                        body["alias"] = new_alias
                        return await self.a_send_request(method, endpoint, body)
                    else:
                        verbose_logger.debug("Aborted.")
                        return None
                else:
                    try:
                        error_data = await res.json()
                        error_message = error_data.get("error", await res.text())
                    except aiohttp.ContentTypeError:
                        error_message = await res.text()
                    raise Exception(error_message)
