import logging
import os
import warnings
from typing import Any, Callable, List, Optional, Union

import httpx

from langfuse.client import Langfuse, StatefulSpanClient, StatefulTraceClient, StateType


class LangfuseBaseCallbackHandler:
    log = logging.getLogger("langfuse")

    def __init__(
        self,
        *,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        debug: bool = False,
        stateful_client: Optional[
            Union[StatefulTraceClient, StatefulSpanClient]
        ] = None,
        update_stateful_client: bool = False,
        version: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        trace_name: Optional[str] = None,
        release: Optional[str] = None,
        metadata: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        threads: Optional[int] = None,
        flush_at: Optional[int] = None,
        flush_interval: Optional[int] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[int] = None,
        enabled: Optional[bool] = None,
        httpx_client: Optional[httpx.Client] = None,
        sdk_integration: str,
        sample_rate: Optional[float] = None,
        mask: Optional[Callable] = None,
        environment: Optional[str] = None,
    ) -> None:
        self.version = version
        self.session_id = session_id
        self.user_id = user_id
        self.trace_name = trace_name
        self.release = release
        self.metadata = metadata
        self.tags = tags

        self.root_span = None
        self.update_stateful_client = update_stateful_client
        self.langfuse = None

        prio_public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
        prio_secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
        prio_host = host or os.environ.get(
            "LANGFUSE_HOST", "https://cloud.langfuse.com"
        )

        prio_sample_rate = (
            sample_rate
            if sample_rate is not None
            else float(os.environ.get("LANGFUSE_SAMPLE_RATE", 1.0))
        )

        if stateful_client and isinstance(stateful_client, StatefulTraceClient):
            self.trace = stateful_client
            self._task_manager = stateful_client.task_manager

            return

        elif stateful_client and isinstance(stateful_client, StatefulSpanClient):
            self.root_span = stateful_client
            self.trace = StatefulTraceClient(
                stateful_client.client,
                stateful_client.trace_id,
                StateType.TRACE,
                stateful_client.trace_id,
                stateful_client.task_manager,
            )
            self._task_manager = stateful_client.task_manager

            return

        args = {
            "public_key": prio_public_key,
            "secret_key": prio_secret_key,
            "host": prio_host,
            "debug": debug,
        }

        if release is not None:
            args["release"] = release
        if threads is not None:
            args["threads"] = threads
        if flush_at is not None:
            args["flush_at"] = flush_at
        if flush_interval is not None:
            args["flush_interval"] = flush_interval
        if max_retries is not None:
            args["max_retries"] = max_retries
        if timeout is not None:
            args["timeout"] = timeout
        if enabled is not None:
            args["enabled"] = enabled
        if httpx_client is not None:
            args["httpx_client"] = httpx_client
        if prio_sample_rate is not None:
            args["sample_rate"] = prio_sample_rate
        if mask is not None:
            args["mask"] = mask
        if environment is not None:
            args["environment"] = environment

        args["sdk_integration"] = sdk_integration

        self.langfuse = Langfuse(**args)
        self.trace: Optional[StatefulTraceClient] = None
        self._task_manager = self.langfuse.task_manager

    def get_trace_id(self):
        """This method is deprecated and will be removed in a future version as it is not concurrency-safe.
        Please refer to the [documentation](https://langfuse.com/docs/integrations/langchain/get-started#interoperability) on how to use interop with the Langfuse SDK to get the id of a trace.

        Returns:
            The ID of the current/last trace or None if no trace is available.
        """
        warnings.warn(
            "get_trace_id is deprecated, create a trace for this handler instead. See interop documentation of this integration for more information.",
            DeprecationWarning,
        )
        return self.trace.id if self.trace else None

    def get_trace_url(self):
        """This method is deprecated and will be removed in a future version as it is not concurrency-safe.
        Please refer to the [documentation](https://langfuse.com/docs/tracing/url) for more information.

        Returns:
            The URL of the current/last trace or None if no trace is available.
        """
        warnings.warn(
            "get_trace_url is deprecated, create a trace for this handler instead. See interop documentation of this integration for more information.",
            DeprecationWarning,
        )
        return self.trace.get_trace_url() if self.trace else None

    def flush(self):
        if self.trace is not None:
            self.trace.task_manager.flush()
        elif self.root_span is not None:
            self.root_span.task_manager.flush()
        else:
            self.log.debug("There was no trace yet, hence no flushing possible.")

    def auth_check(self):
        if self.langfuse is not None:
            return self.langfuse.auth_check()
        elif self.trace is not None:
            projects = self.trace.client.projects.get()
            if len(projects.data) == 0:
                raise Exception("No projects found for the keys.")
            return True
        elif self.root_span is not None:
            projects = self.root_span.client.projects.get()
            if len(projects) == 0:
                raise Exception("No projects found for the keys.")
            return True

        return False
