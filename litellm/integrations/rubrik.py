#### What this does ####
#    On success, logs events to Rubrik
import asyncio
import os
import httpx
import random
import traceback
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)

from litellm.types.utils import StandardLoggingPayload


class RubrikLogger(CustomBatchLogger):
    def __init__(self, **kwargs):
        self.flush_lock = asyncio.Lock()
        super().__init__(**kwargs, flush_lock=self.flush_lock)
        
        # Instance variables
        verbose_logger.debug("initializing rubrik logger")
        self.sampling_rate: float = (
            float(os.getenv("RUBRIK_SAMPLING_RATE"))  # type: ignore
            if os.getenv("RUBRIK_SAMPLING_RATE") is not None
            and os.getenv("RUBRIK_SAMPLING_RATE").strip().isdigit()  # type: ignore
            else 1.0
        )

        self.key = os.getenv("RUBRIK_API_KEY")
        _batch_size = (
            os.getenv("RUBRIK_BATCH_SIZE", None)
        )

        if _batch_size:
            # Batch size has a default of 512
            # Queue will be flushed when the queue reaches this size or when 
            # the periodic interval is triggered (every 5 seconds by default)
            self.batch_size = int(_batch_size)

        _webhook_url = os.getenv("RUBRIK_WEBHOOK_URL")

        if _webhook_url is None:
            raise ValueError("environment variable RUBRIK_WEBHOOK_URL not set")

        if _webhook_url.endswith("/"):
            _webhook_url = _webhook_url[:-1]

        self.webhook_endpoint = f"{_webhook_url}/litellm/batch"

        # Cache the httpx client
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )

        self.log_queue = []
        asyncio.create_task(self.periodic_flush())

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        standard_logging_payload: StandardLoggingPayload = kwargs["standard_logging_object"]
        random_sample = random.random()
        if random_sample > self.sampling_rate:
            verbose_logger.info(
                "Skipping Rubrik logging. Sampling rate={}, random_sample={}".format(
                    self.sampling_rate, random_sample
                )
            )
            return  # Skip logging
        # If the request is an anthropic request, the system prompt _might_ be in kwargs["system"]
        if "system" in kwargs:
            # Insert the system prompt at the beginning of the messages list
            # This is a list of dictionaries
            system_prompt_msg_list = kwargs["system"]
            try: 
                # check if system prompt is not empty
                if system_prompt_msg_list: 
                    system_scaffold = {"role": "system", "content": system_prompt_msg_list}
                    # If the messages are a list 
                    if type(standard_logging_payload.messages) is list: # type: ignore
                        standard_logging_payload.messages.insert(0, system_scaffold) # type: ignore 
                    # If the messages are a dictionary or a string -> transform the messages into a list?
                    elif type(standard_logging_payload.messages) is dict or type(standard_logging_payload.messages) is str: # type: ignore
                        # Transform the messages into a list
                        standard_logging_payload.messages = [standard_logging_payload.messages, system_scaffold] # type: ignore
            except Exception as e: 
                verbose_logger.debug(f"Error adding system prompt to messages: {e}")
        
        self.log_queue.append(standard_logging_payload)
        
        if len(self.log_queue) >= self.batch_size: 
            await self.flush_queue()

    async def _log_batch_to_rubrik(self, data): 
        try: 
            headers = {"Content-Type": "application/json"}

            if self.key:
                headers["Authorization"] = f"Bearer {self.key}"

            response = await self.async_httpx_client.post(
                url=self.webhook_endpoint, 
                content={"data": data},
                headers=headers, 
            ) 
            response.raise_for_status()

            if response.status_code >= 300:
                verbose_logger.error(
                    f"Rubrik Error: {response.status_code} - {response.text}"
                )
        except httpx.HTTPStatusError as e:
            verbose_logger.exception(
                f"Rubrik HTTP Error: {e.response.status_code} - {e.response.text}"
            )
        except Exception:
            verbose_logger.exception(
                f"Rubrik Layer Error - {traceback.format_exc()}"
            )

    async def async_send_batch(self):
        """
        Handles sending batches of responses to Rubrik
        """
        if not self.log_queue:
            return

        await self._log_batch_to_rubrik(
            data=self.log_queue,
        )

    def _send_batch(self):
        """Calls async_send_batch in an event loop"""
        if not self.log_queue:
            return

        try:
            # Try to get the existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, create a task
                asyncio.create_task(self.async_send_batch())
            else:
                # If no event loop is running, run the coroutine directly
                loop.run_until_complete(self.async_send_batch())
        except RuntimeError:
            # If we can't get an event loop, create a new one
            asyncio.run(self.async_send_batch())

