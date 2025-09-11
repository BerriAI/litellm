import json
import logging
import os
import threading
import time
from queue import Empty, Queue
from typing import Any, List, Optional

import backoff

try:
    import pydantic.v1 as pydantic
except ImportError:
    import pydantic

from langfuse.parse_error import handle_exception
from langfuse.request import APIError, LangfuseClient
from langfuse.Sampler import Sampler
from langfuse.serializer import EventSerializer
from langfuse.types import MaskFunction

from .media_manager import MediaManager

MAX_EVENT_SIZE_BYTES = int(os.environ.get("LANGFUSE_MAX_EVENT_SIZE_BYTES", 1_000_000))
MAX_BATCH_SIZE_BYTES = int(os.environ.get("LANGFUSE_MAX_BATCH_SIZE_BYTES", 2_500_000))


class IngestionMetadata(pydantic.BaseModel):
    batch_size: int
    sdk_integration: str
    sdk_name: str
    sdk_version: str
    public_key: str


class IngestionConsumer(threading.Thread):
    _log = logging.getLogger("langfuse")
    _ingestion_queue: Queue
    _identifier: int
    _client: LangfuseClient
    _flush_at: int
    _flush_interval: float
    _max_retries: int
    _public_key: str
    _sdk_name: str
    _sdk_version: str
    _sdk_integration: str
    _mask: Optional[MaskFunction]
    _sampler: Sampler
    _media_manager: MediaManager

    def __init__(
        self,
        *,
        ingestion_queue: Queue,
        identifier: int,
        client: LangfuseClient,
        flush_at: int,
        flush_interval: float,
        max_retries: int,
        public_key: str,
        media_manager: MediaManager,
        sdk_name: str,
        sdk_version: str,
        sdk_integration: str,
        sample_rate: float,
        mask: Optional[MaskFunction] = None,
    ):
        """Create a consumer thread."""
        super().__init__()
        # It's important to set running in the constructor: if we are asked to
        # pause immediately after construction, we might set running to True in
        # run() *after* we set it to False in pause... and keep running
        # forever.
        self.running = True
        # Make consumer a daemon thread so that it doesn't block program exit
        self.daemon = True
        self._ingestion_queue = ingestion_queue
        self._identifier = identifier
        self._client = client
        self._flush_at = flush_at
        self._flush_interval = flush_interval
        self._max_retries = max_retries
        self._public_key = public_key
        self._sdk_name = sdk_name
        self._sdk_version = sdk_version
        self._sdk_integration = sdk_integration
        self._mask = mask
        self._sampler = Sampler(sample_rate)
        self._media_manager = media_manager

    def _next(self):
        """Return the next batch of items to upload."""
        events = []

        start_time = time.monotonic()
        total_size = 0

        while len(events) < self._flush_at:
            elapsed = time.monotonic() - start_time
            if elapsed >= self._flush_interval:
                break
            try:
                event = self._ingestion_queue.get(
                    block=True, timeout=self._flush_interval - elapsed
                )

                # convert pydantic models to dicts
                if "body" in event and isinstance(event["body"], pydantic.BaseModel):
                    event["body"] = event["body"].dict(exclude_none=True)

                # sample event
                if not self._sampler.sample_event(event):
                    self._ingestion_queue.task_done()

                    continue

                # handle multimodal data
                self._media_manager.process_media_in_event(event)

                # truncate item if it exceeds size limit
                item_size = self._truncate_item_in_place(
                    event=event,
                    max_size=MAX_EVENT_SIZE_BYTES,
                    log_message="<truncated due to size exceeding limit>",
                )

                # apply mask
                self._apply_mask_in_place(event)

                # check for serialization errors
                try:
                    json.dumps(event, cls=EventSerializer)
                except Exception as e:
                    self._log.error(f"Error serializing item, skipping: {e}")
                    self._ingestion_queue.task_done()

                    continue

                events.append(event)

                total_size += item_size
                if total_size >= MAX_BATCH_SIZE_BYTES:
                    self._log.debug("hit batch size limit (size: %d)", total_size)
                    break

            except Empty:
                break

            except Exception as e:
                self._log.warning(
                    "Failed to process event in IngestionConsumer, skipping",
                    exc_info=e,
                )
                self._ingestion_queue.task_done()

        self._log.debug(
            "~%d items in the Langfuse queue", self._ingestion_queue.qsize()
        )

        return events

    def _truncate_item_in_place(
        self,
        *,
        event: Any,
        max_size: int,
        log_message: Optional[str] = None,
    ) -> int:
        """Truncate the item in place to fit within the size limit."""
        item_size = self._get_item_size(event)
        self._log.debug(f"item size {item_size}")

        if item_size > max_size:
            self._log.warning(
                "Item exceeds size limit (size: %s), dropping input / output / metadata of item until it fits.",
                item_size,
            )

            if "body" in event:
                drop_candidates = ["input", "output", "metadata"]
                sorted_field_sizes = sorted(
                    [
                        (
                            field,
                            self._get_item_size((event["body"][field]))
                            if field in event["body"]
                            else 0,
                        )
                        for field in drop_candidates
                    ],
                    key=lambda x: x[1],
                )

                # drop the largest field until the item size is within the limit
                for _ in range(len(sorted_field_sizes)):
                    field_to_drop, size_to_drop = sorted_field_sizes.pop()

                    if field_to_drop not in event["body"]:
                        continue

                    event["body"][field_to_drop] = log_message
                    item_size -= size_to_drop

                    self._log.debug(
                        f"Dropped field {field_to_drop}, new item size {item_size}"
                    )

                    if item_size <= max_size:
                        break

            # if item does not have body or input/output fields, drop the event
            if "body" not in event or (
                "input" not in event["body"] and "output" not in event["body"]
            ):
                self._log.warning(
                    "Item does not have body or input/output fields, dropping item."
                )
                self._ingestion_queue.task_done()
                return 0

        return self._get_item_size(event)

    def _get_item_size(self, item: Any) -> int:
        """Return the size of the item in bytes."""
        return len(json.dumps(item, cls=EventSerializer).encode())

    def _apply_mask_in_place(self, event: dict):
        """Apply the mask function to the event. This is done in place."""
        if not self._mask:
            return

        body = event["body"] if "body" in event else {}
        for key in ("input", "output"):
            if key in body:
                try:
                    body[key] = self._mask(data=body[key])
                except Exception as e:
                    self._log.error(f"Mask function failed with error: {e}")
                    body[key] = "<fully masked due to failed mask function>"

    def run(self):
        """Run the consumer."""
        self._log.debug("consumer is running...")
        while self.running:
            self.upload()

    def upload(self):
        """Upload the next batch of items, return whether successful."""
        batch = self._next()
        if len(batch) == 0:
            return

        try:
            self._upload_batch(batch)
        except Exception as e:
            handle_exception(e)
        finally:
            # mark items as acknowledged from queue
            for _ in batch:
                self._ingestion_queue.task_done()

    def pause(self):
        """Pause the consumer."""
        self.running = False

    def _upload_batch(self, batch: List[Any]):
        self._log.debug("uploading batch of %d items", len(batch))

        metadata = IngestionMetadata(
            batch_size=len(batch),
            sdk_integration=self._sdk_integration,
            sdk_name=self._sdk_name,
            sdk_version=self._sdk_version,
            public_key=self._public_key,
        ).dict()

        @backoff.on_exception(
            backoff.expo, Exception, max_tries=self._max_retries, logger=None
        )
        def execute_task_with_backoff(batch: List[Any]):
            try:
                self._client.batch_post(batch=batch, metadata=metadata)
            except Exception as e:
                if (
                    isinstance(e, APIError)
                    and 400 <= int(e.status) < 500
                    and int(e.status) != 429  # retry if rate-limited
                ):
                    return

                raise e

        execute_task_with_backoff(batch)
        self._log.debug("successfully uploaded batch of %d events", len(batch))
