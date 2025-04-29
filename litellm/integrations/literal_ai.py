#### What this does ####
# This file contains the LiteralAILogger class which is used to log steps to the LiteralAI observability platform.
import asyncio
import os
import uuid
from typing import List, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.llms.custom_httpx.http_handler import (
    HTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.utils import StandardLoggingPayload


class LiteralAILogger(CustomBatchLogger):
    def __init__(
        self,
        literalai_api_key=None,
        literalai_api_url="https://cloud.getliteral.ai",
        env=None,
        **kwargs,
    ):
        self.literalai_api_url = os.getenv("LITERAL_API_URL") or literalai_api_url
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": literalai_api_key or os.getenv("LITERAL_API_KEY"),
            "x-client-name": "litellm",
        }
        if env:
            self.headers["x-env"] = env
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        self.sync_http_handler = HTTPHandler()
        batch_size = os.getenv("LITERAL_BATCH_SIZE", None)
        self.flush_lock = asyncio.Lock()
        super().__init__(
            **kwargs,
            flush_lock=self.flush_lock,
            batch_size=int(batch_size) if batch_size else None,
        )

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            verbose_logger.debug(
                "Literal AI Layer Logging - kwargs: %s, response_obj: %s",
                kwargs,
                response_obj,
            )
            data = self._prepare_log_data(kwargs, response_obj, start_time, end_time)
            self.log_queue.append(data)
            verbose_logger.debug(
                "Literal AI logging: queue length %s, batch size %s",
                len(self.log_queue),
                self.batch_size,
            )
            if len(self.log_queue) >= self.batch_size:
                self._send_batch()
        except Exception:
            verbose_logger.exception(
                "Literal AI Layer Error - error logging success event."
            )

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        verbose_logger.info("Literal AI Failure Event Logging!")
        try:
            data = self._prepare_log_data(kwargs, response_obj, start_time, end_time)
            self.log_queue.append(data)
            verbose_logger.debug(
                "Literal AI logging: queue length %s, batch size %s",
                len(self.log_queue),
                self.batch_size,
            )
            if len(self.log_queue) >= self.batch_size:
                self._send_batch()
        except Exception:
            verbose_logger.exception(
                "Literal AI Layer Error - error logging failure event."
            )

    def _send_batch(self):
        if not self.log_queue:
            return

        url = f"{self.literalai_api_url}/api/graphql"
        query = self._steps_query_builder(self.log_queue)
        variables = self._steps_variables_builder(self.log_queue)
        try:
            response = self.sync_http_handler.post(
                url=url,
                json={
                    "query": query,
                    "variables": variables,
                },
                headers=self.headers,
            )

            if response.status_code >= 300:
                verbose_logger.error(
                    f"Literal AI Error: {response.status_code} - {response.text}"
                )
            else:
                verbose_logger.debug(
                    f"Batch of {len(self.log_queue)} runs successfully created"
                )
        except Exception:
            verbose_logger.exception("Literal AI Layer Error")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            verbose_logger.debug(
                "Literal AI Async Layer Logging - kwargs: %s, response_obj: %s",
                kwargs,
                response_obj,
            )
            data = self._prepare_log_data(kwargs, response_obj, start_time, end_time)
            self.log_queue.append(data)
            verbose_logger.debug(
                "Literal AI logging: queue length %s, batch size %s",
                len(self.log_queue),
                self.batch_size,
            )
            if len(self.log_queue) >= self.batch_size:
                await self.flush_queue()
        except Exception:
            verbose_logger.exception(
                "Literal AI Layer Error - error logging async success event."
            )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        verbose_logger.info("Literal AI Failure Event Logging!")
        try:
            data = self._prepare_log_data(kwargs, response_obj, start_time, end_time)
            self.log_queue.append(data)
            verbose_logger.debug(
                "Literal AI logging: queue length %s, batch size %s",
                len(self.log_queue),
                self.batch_size,
            )
            if len(self.log_queue) >= self.batch_size:
                await self.flush_queue()
        except Exception:
            verbose_logger.exception(
                "Literal AI Layer Error - error logging async failure event."
            )

    async def async_send_batch(self):
        if not self.log_queue:
            return

        url = f"{self.literalai_api_url}/api/graphql"
        query = self._steps_query_builder(self.log_queue)
        variables = self._steps_variables_builder(self.log_queue)

        try:
            response = await self.async_httpx_client.post(
                url=url,
                json={
                    "query": query,
                    "variables": variables,
                },
                headers=self.headers,
            )
            if response.status_code >= 300:
                verbose_logger.error(
                    f"Literal AI Error: {response.status_code} - {response.text}"
                )
            else:
                verbose_logger.debug(
                    f"Batch of {len(self.log_queue)} runs successfully created"
                )
        except httpx.HTTPStatusError as e:
            verbose_logger.exception(
                f"Literal AI HTTP Error: {e.response.status_code} - {e.response.text}"
            )
        except Exception:
            verbose_logger.exception("Literal AI Layer Error")

    def _prepare_log_data(self, kwargs, response_obj, start_time, end_time) -> dict:
        logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object", None
        )

        if logging_payload is None:
            raise ValueError("standard_logging_object not found in kwargs")
        clean_metadata = logging_payload["metadata"]
        metadata = kwargs.get("litellm_params", {}).get("metadata", {})

        settings = logging_payload["model_parameters"]
        messages = logging_payload["messages"]
        response = logging_payload["response"]
        choices: List = []
        if isinstance(response, dict) and "choices" in response:
            choices = response["choices"]
        message_completion = choices[0]["message"] if choices else None
        prompt_id = None
        variables = None

        if messages and isinstance(messages, list) and isinstance(messages[0], dict):
            for message in messages:
                if literal_prompt := getattr(message, "__literal_prompt__", None):
                    prompt_id = literal_prompt.get("prompt_id")
                    variables = literal_prompt.get("variables")
                    message["uuid"] = literal_prompt.get("uuid")
                    message["templated"] = True

        tools = settings.pop("tools", None)

        step = {
            "id": metadata.get("step_id", str(uuid.uuid4())),
            "error": logging_payload["error_str"],
            "name": kwargs.get("model", ""),
            "threadId": metadata.get("literalai_thread_id", None),
            "parentId": metadata.get("literalai_parent_id", None),
            "rootRunId": metadata.get("literalai_root_run_id", None),
            "input": None,
            "output": None,
            "type": "llm",
            "tags": metadata.get("tags", metadata.get("literalai_tags", None)),
            "startTime": str(start_time),
            "endTime": str(end_time),
            "metadata": clean_metadata,
            "generation": {
                "inputTokenCount": logging_payload["prompt_tokens"],
                "outputTokenCount": logging_payload["completion_tokens"],
                "tokenCount": logging_payload["total_tokens"],
                "promptId": prompt_id,
                "variables": variables,
                "provider": kwargs.get("custom_llm_provider", "litellm"),
                "model": kwargs.get("model", ""),
                "duration": (end_time - start_time).total_seconds(),
                "settings": settings,
                "messages": messages,
                "messageCompletion": message_completion,
                "tools": tools,
            },
        }
        return step

    def _steps_query_variables_builder(self, steps):
        generated = ""
        for id in range(len(steps)):
            generated += f"""$id_{id}: String!
            $threadId_{id}: String
            $rootRunId_{id}: String
            $type_{id}: StepType
            $startTime_{id}: DateTime
            $endTime_{id}: DateTime
            $error_{id}: String
            $input_{id}: Json
            $output_{id}: Json
            $metadata_{id}: Json
            $parentId_{id}: String
            $name_{id}: String
            $tags_{id}: [String!]
            $generation_{id}: GenerationPayloadInput
            $scores_{id}: [ScorePayloadInput!]
            $attachments_{id}: [AttachmentPayloadInput!]
            """
        return generated

    def _steps_ingest_steps_builder(self, steps):
        generated = ""
        for id in range(len(steps)):
            generated += f"""
        step{id}: ingestStep(
            id: $id_{id}
            threadId: $threadId_{id}
            rootRunId: $rootRunId_{id}
            startTime: $startTime_{id}
            endTime: $endTime_{id}
            type: $type_{id}
            error: $error_{id}
            input: $input_{id}
            output: $output_{id}
            metadata: $metadata_{id}
            parentId: $parentId_{id}
            name: $name_{id}
            tags: $tags_{id}
            generation: $generation_{id}
            scores: $scores_{id}
            attachments: $attachments_{id}
        ) {{
            ok
            message
        }}
    """
        return generated

    def _steps_query_builder(self, steps):
        return f"""
        mutation AddStep({self._steps_query_variables_builder(steps)}) {{
        {self._steps_ingest_steps_builder(steps)}
        }}
        """

    def _steps_variables_builder(self, steps):
        def serialize_step(event, id):
            result = {}

            for key, value in event.items():
                # Only keep the keys that are not None to avoid overriding existing values
                if value is not None:
                    result[f"{key}_{id}"] = value

            return result

        variables = {}
        for i in range(len(steps)):
            step = steps[i]
            variables.update(serialize_step(step, i))
        return variables
