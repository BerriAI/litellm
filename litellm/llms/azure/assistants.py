from typing import Any, Coroutine, Dict, Iterable, Literal, Optional, Union

import httpx
from openai import AsyncAzureOpenAI, AzureOpenAI
from typing_extensions import overload

from ...types.llms.openai import (
    Assistant,
    AssistantEventHandler,
    AssistantStreamManager,
    AssistantToolParam,
    AsyncAssistantEventHandler,
    AsyncAssistantStreamManager,
    AsyncCursorPage,
    OpenAICreateThreadParamsMessage,
    OpenAIMessage,
    Run,
    SyncCursorPage,
    Thread,
)
from .common_utils import BaseAzureLLM


class AzureAssistantsAPI(BaseAzureLLM):
    def __init__(self) -> None:
        super().__init__()

    def get_azure_client(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AzureOpenAI] = None,
        litellm_params: Optional[dict] = None,
    ) -> AzureOpenAI:
        if client is None:
            azure_client_params = self.initialize_azure_sdk_client(
                litellm_params=litellm_params or {},
                api_key=api_key,
                api_base=api_base,
                model_name="",
                api_version=api_version,
                is_async=False,
            )
            azure_openai_client = AzureOpenAI(**azure_client_params)  # type: ignore
        else:
            azure_openai_client = client

        return azure_openai_client

    def async_get_azure_client(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AsyncAzureOpenAI] = None,
        litellm_params: Optional[dict] = None,
    ) -> AsyncAzureOpenAI:
        if client is None:
            azure_client_params = self.initialize_azure_sdk_client(
                litellm_params=litellm_params or {},
                api_key=api_key,
                api_base=api_base,
                model_name="",
                api_version=api_version,
                is_async=True,
            )

            azure_openai_client = AsyncAzureOpenAI(**azure_client_params)
            # azure_openai_client = AsyncAzureOpenAI(**data)  # type: ignore
        else:
            azure_openai_client = client

        return azure_openai_client

    ### ASSISTANTS ###

    async def async_get_assistants(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AsyncAzureOpenAI],
        litellm_params: Optional[dict] = None,
    ) -> AsyncCursorPage[Assistant]:
        azure_openai_client = self.async_get_azure_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
            litellm_params=litellm_params,
        )

        response = await azure_openai_client.beta.assistants.list()

        return response

    # fmt: off

    @overload
    def get_assistants(
        self, 
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AsyncAzureOpenAI],
        aget_assistants: Literal[True], 
    ) -> Coroutine[None, None, AsyncCursorPage[Assistant]]:
        ...

    @overload
    def get_assistants(
        self, 
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AzureOpenAI],
        aget_assistants: Optional[Literal[False]], 
    ) -> SyncCursorPage[Assistant]: 
        ...

    # fmt: on

    def get_assistants(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client=None,
        aget_assistants=None,
        litellm_params: Optional[dict] = None,
    ):
        if aget_assistants is not None and aget_assistants is True:
            return self.async_get_assistants(
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                azure_ad_token=azure_ad_token,
                timeout=timeout,
                max_retries=max_retries,
                client=client,
                litellm_params=litellm_params,
            )
        azure_openai_client = self.get_azure_client(
            api_key=api_key,
            api_base=api_base,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
            api_version=api_version,
            litellm_params=litellm_params,
        )

        response = azure_openai_client.beta.assistants.list()

        return response

    ### MESSAGES ###

    async def a_add_message(
        self,
        thread_id: str,
        message_data: dict,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AsyncAzureOpenAI] = None,
        litellm_params: Optional[dict] = None,
    ) -> OpenAIMessage:
        openai_client = self.async_get_azure_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
            litellm_params=litellm_params,
        )

        thread_message: OpenAIMessage = await openai_client.beta.threads.messages.create(  # type: ignore
            thread_id, **message_data  # type: ignore
        )

        response_obj: Optional[OpenAIMessage] = None
        if getattr(thread_message, "status", None) is None:
            thread_message.status = "completed"
            response_obj = OpenAIMessage(**thread_message.dict())
        else:
            response_obj = OpenAIMessage(**thread_message.dict())
        return response_obj

    # fmt: off

    @overload
    def add_message(
        self,
        thread_id: str,
        message_data: dict,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AsyncAzureOpenAI],
        a_add_message: Literal[True],
        litellm_params: Optional[dict] = None,
    ) -> Coroutine[None, None, OpenAIMessage]:
        ...

    @overload
    def add_message(
        self,
        thread_id: str,
        message_data: dict,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AzureOpenAI],
        a_add_message: Optional[Literal[False]],
        litellm_params: Optional[dict] = None,
    ) -> OpenAIMessage:
        ...

    # fmt: on

    def add_message(
        self,
        thread_id: str,
        message_data: dict,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client=None,
        a_add_message: Optional[bool] = None,
        litellm_params: Optional[dict] = None,
    ):
        if a_add_message is not None and a_add_message is True:
            return self.a_add_message(
                thread_id=thread_id,
                message_data=message_data,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                azure_ad_token=azure_ad_token,
                timeout=timeout,
                max_retries=max_retries,
                client=client,
                litellm_params=litellm_params,
            )
        openai_client = self.get_azure_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
        )

        thread_message: OpenAIMessage = openai_client.beta.threads.messages.create(  # type: ignore
            thread_id, **message_data  # type: ignore
        )

        response_obj: Optional[OpenAIMessage] = None
        if getattr(thread_message, "status", None) is None:
            thread_message.status = "completed"
            response_obj = OpenAIMessage(**thread_message.dict())
        else:
            response_obj = OpenAIMessage(**thread_message.dict())
        return response_obj

    async def async_get_messages(
        self,
        thread_id: str,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AsyncAzureOpenAI] = None,
        litellm_params: Optional[dict] = None,
    ) -> AsyncCursorPage[OpenAIMessage]:
        openai_client = self.async_get_azure_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
            litellm_params=litellm_params,
        )

        response = await openai_client.beta.threads.messages.list(thread_id=thread_id)

        return response

    # fmt: off

    @overload
    def get_messages(
        self,
        thread_id: str,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AsyncAzureOpenAI],
        aget_messages: Literal[True],
        litellm_params: Optional[dict] = None,
    ) -> Coroutine[None, None, AsyncCursorPage[OpenAIMessage]]:
        ...

    @overload
    def get_messages(
        self,
        thread_id: str,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AzureOpenAI],
        aget_messages: Optional[Literal[False]],
        litellm_params: Optional[dict] = None,
    ) -> SyncCursorPage[OpenAIMessage]:
        ...

    # fmt: on

    def get_messages(
        self,
        thread_id: str,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client=None,
        aget_messages=None,
        litellm_params: Optional[dict] = None,
    ):
        if aget_messages is not None and aget_messages is True:
            return self.async_get_messages(
                thread_id=thread_id,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                azure_ad_token=azure_ad_token,
                timeout=timeout,
                max_retries=max_retries,
                client=client,
                litellm_params=litellm_params,
            )
        openai_client = self.get_azure_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
            litellm_params=litellm_params,
        )

        response = openai_client.beta.threads.messages.list(thread_id=thread_id)

        return response

    ### THREADS ###

    async def async_create_thread(
        self,
        metadata: Optional[dict],
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AsyncAzureOpenAI],
        messages: Optional[Iterable[OpenAICreateThreadParamsMessage]],
        litellm_params: Optional[dict] = None,
    ) -> Thread:
        openai_client = self.async_get_azure_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
            litellm_params=litellm_params,
        )

        data = {}
        if messages is not None:
            data["messages"] = messages  # type: ignore
        if metadata is not None:
            data["metadata"] = metadata  # type: ignore

        message_thread = await openai_client.beta.threads.create(**data)  # type: ignore

        return Thread(**message_thread.dict())

    # fmt: off

    @overload
    def create_thread(
        self,
        metadata: Optional[dict],
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        messages: Optional[Iterable[OpenAICreateThreadParamsMessage]],
        client: Optional[AsyncAzureOpenAI],
        acreate_thread: Literal[True],
        litellm_params: Optional[dict] = None,
    ) -> Coroutine[None, None, Thread]:
        ...

    @overload
    def create_thread(
        self,
        metadata: Optional[dict],
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        messages: Optional[Iterable[OpenAICreateThreadParamsMessage]],
        client: Optional[AzureOpenAI],
        acreate_thread: Optional[Literal[False]],
        litellm_params: Optional[dict] = None,
    ) -> Thread:
        ...

    # fmt: on

    def create_thread(
        self,
        metadata: Optional[dict],
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        messages: Optional[Iterable[OpenAICreateThreadParamsMessage]],
        client=None,
        acreate_thread=None,
        litellm_params: Optional[dict] = None,
    ):
        """
        Here's an example:
        ```
        from litellm.llms.openai.openai import OpenAIAssistantsAPI, MessageData

        # create thread
        message: MessageData = {"role": "user", "content": "Hey, how's it going?"}
        openai_api.create_thread(messages=[message])
        ```
        """
        if acreate_thread is not None and acreate_thread is True:
            return self.async_create_thread(
                metadata=metadata,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                azure_ad_token=azure_ad_token,
                timeout=timeout,
                max_retries=max_retries,
                client=client,
                messages=messages,
                litellm_params=litellm_params,
            )
        azure_openai_client = self.get_azure_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
            litellm_params=litellm_params,
        )

        data = {}
        if messages is not None:
            data["messages"] = messages  # type: ignore
        if metadata is not None:
            data["metadata"] = metadata  # type: ignore

        message_thread = azure_openai_client.beta.threads.create(**data)  # type: ignore

        return Thread(**message_thread.dict())

    async def async_get_thread(
        self,
        thread_id: str,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AsyncAzureOpenAI],
        litellm_params: Optional[dict] = None,
    ) -> Thread:
        openai_client = self.async_get_azure_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
            litellm_params=litellm_params,
        )

        response = await openai_client.beta.threads.retrieve(thread_id=thread_id)

        return Thread(**response.dict())

    # fmt: off

    @overload
    def get_thread(
        self,
        thread_id: str,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AsyncAzureOpenAI],
        aget_thread: Literal[True],
        litellm_params: Optional[dict] = None,
    ) -> Coroutine[None, None, Thread]:
        ...

    @overload
    def get_thread(
        self,
        thread_id: str,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AzureOpenAI],
        aget_thread: Optional[Literal[False]],
        litellm_params: Optional[dict] = None,
    ) -> Thread:
        ...

    # fmt: on

    def get_thread(
        self,
        thread_id: str,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client=None,
        aget_thread=None,
        litellm_params: Optional[dict] = None,
    ):
        if aget_thread is not None and aget_thread is True:
            return self.async_get_thread(
                thread_id=thread_id,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                azure_ad_token=azure_ad_token,
                timeout=timeout,
                max_retries=max_retries,
                client=client,
                litellm_params=litellm_params,
            )
        openai_client = self.get_azure_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
            litellm_params=litellm_params,
        )

        response = openai_client.beta.threads.retrieve(thread_id=thread_id)

        return Thread(**response.dict())

    # def delete_thread(self):
    #     pass

    ### RUNS ###

    async def arun_thread(
        self,
        thread_id: str,
        assistant_id: str,
        additional_instructions: Optional[str],
        instructions: Optional[str],
        metadata: Optional[Dict],
        model: Optional[str],
        stream: Optional[bool],
        tools: Optional[Iterable[AssistantToolParam]],
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AsyncAzureOpenAI],
        litellm_params: Optional[dict] = None,
    ) -> Run:
        openai_client = self.async_get_azure_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            client=client,
            litellm_params=litellm_params,
        )

        response = await openai_client.beta.threads.runs.create_and_poll(  # type: ignore
            thread_id=thread_id,
            assistant_id=assistant_id,
            additional_instructions=additional_instructions,
            instructions=instructions,
            metadata=metadata,  # type: ignore
            model=model,
            tools=tools,
        )

        return response

    def async_run_thread_stream(
        self,
        client: AsyncAzureOpenAI,
        thread_id: str,
        assistant_id: str,
        additional_instructions: Optional[str],
        instructions: Optional[str],
        metadata: Optional[Dict],
        model: Optional[str],
        tools: Optional[Iterable[AssistantToolParam]],
        event_handler: Optional[AssistantEventHandler],
        litellm_params: Optional[dict] = None,
    ) -> AsyncAssistantStreamManager[AsyncAssistantEventHandler]:
        data: Dict[str, Any] = {
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            "additional_instructions": additional_instructions,
            "instructions": instructions,
            "metadata": metadata,
            "model": model,
            "tools": tools,
        }
        if event_handler is not None:
            data["event_handler"] = event_handler
        return client.beta.threads.runs.stream(**data)  # type: ignore

    def run_thread_stream(
        self,
        client: AzureOpenAI,
        thread_id: str,
        assistant_id: str,
        additional_instructions: Optional[str],
        instructions: Optional[str],
        metadata: Optional[Dict],
        model: Optional[str],
        tools: Optional[Iterable[AssistantToolParam]],
        event_handler: Optional[AssistantEventHandler],
        litellm_params: Optional[dict] = None,
    ) -> AssistantStreamManager[AssistantEventHandler]:
        data: Dict[str, Any] = {
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            "additional_instructions": additional_instructions,
            "instructions": instructions,
            "metadata": metadata,
            "model": model,
            "tools": tools,
        }
        if event_handler is not None:
            data["event_handler"] = event_handler
        return client.beta.threads.runs.stream(**data)  # type: ignore

    # fmt: off

    @overload
    def run_thread(
        self,
        thread_id: str,
        assistant_id: str,
        additional_instructions: Optional[str],
        instructions: Optional[str],
        metadata: Optional[Dict],
        model: Optional[str],
        stream: Optional[bool],
        tools: Optional[Iterable[AssistantToolParam]],
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AsyncAzureOpenAI],
        arun_thread: Literal[True],
    ) -> Coroutine[None, None, Run]:
        ...

    @overload
    def run_thread(
        self,
        thread_id: str,
        assistant_id: str,
        additional_instructions: Optional[str],
        instructions: Optional[str],
        metadata: Optional[Dict],
        model: Optional[str],
        stream: Optional[bool],
        tools: Optional[Iterable[AssistantToolParam]],
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AzureOpenAI],
        arun_thread: Optional[Literal[False]],
    ) -> Run:
        ...

    # fmt: on

    def run_thread(
        self,
        thread_id: str,
        assistant_id: str,
        additional_instructions: Optional[str],
        instructions: Optional[str],
        metadata: Optional[Dict],
        model: Optional[str],
        stream: Optional[bool],
        tools: Optional[Iterable[AssistantToolParam]],
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client=None,
        arun_thread=None,
        event_handler: Optional[AssistantEventHandler] = None,
        litellm_params: Optional[dict] = None,
    ):
        if arun_thread is not None and arun_thread is True:
            if stream is not None and stream is True:
                azure_client = self.async_get_azure_client(
                    api_key=api_key,
                    api_base=api_base,
                    api_version=api_version,
                    azure_ad_token=azure_ad_token,
                    timeout=timeout,
                    max_retries=max_retries,
                    client=client,
                    litellm_params=litellm_params,
                )
                return self.async_run_thread_stream(
                    client=azure_client,
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                    additional_instructions=additional_instructions,
                    instructions=instructions,
                    metadata=metadata,
                    model=model,
                    tools=tools,
                    event_handler=event_handler,
                    litellm_params=litellm_params,
                )
            return self.arun_thread(
                thread_id=thread_id,
                assistant_id=assistant_id,
                additional_instructions=additional_instructions,
                instructions=instructions,
                metadata=metadata,  # type: ignore
                model=model,
                stream=stream,
                tools=tools,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                azure_ad_token=azure_ad_token,
                timeout=timeout,
                max_retries=max_retries,
                client=client,
                litellm_params=litellm_params,
            )
        openai_client = self.get_azure_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
            litellm_params=litellm_params,
        )

        if stream is not None and stream is True:
            return self.run_thread_stream(
                client=openai_client,
                thread_id=thread_id,
                assistant_id=assistant_id,
                additional_instructions=additional_instructions,
                instructions=instructions,
                metadata=metadata,
                model=model,
                tools=tools,
                event_handler=event_handler,
                litellm_params=litellm_params,
            )

        response = openai_client.beta.threads.runs.create_and_poll(  # type: ignore
            thread_id=thread_id,
            assistant_id=assistant_id,
            additional_instructions=additional_instructions,
            instructions=instructions,
            metadata=metadata,  # type: ignore
            model=model,
            tools=tools,
        )

        return response

    # Create Assistant
    async def async_create_assistants(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AsyncAzureOpenAI],
        create_assistant_data: dict,
        litellm_params: Optional[dict] = None,
    ) -> Assistant:
        azure_openai_client = self.async_get_azure_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
            litellm_params=litellm_params,
        )

        response = await azure_openai_client.beta.assistants.create(
            **create_assistant_data
        )
        return response

    def create_assistants(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        create_assistant_data: dict,
        client=None,
        async_create_assistants=None,
        litellm_params: Optional[dict] = None,
    ):
        if async_create_assistants is not None and async_create_assistants is True:
            return self.async_create_assistants(
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                azure_ad_token=azure_ad_token,
                timeout=timeout,
                max_retries=max_retries,
                client=client,
                create_assistant_data=create_assistant_data,
                litellm_params=litellm_params,
            )
        azure_openai_client = self.get_azure_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
            litellm_params=litellm_params,
        )

        response = azure_openai_client.beta.assistants.create(**create_assistant_data)
        return response

    # Delete Assistant
    async def async_delete_assistant(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AsyncAzureOpenAI],
        assistant_id: str,
        litellm_params: Optional[dict] = None,
    ):
        azure_openai_client = self.async_get_azure_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
            litellm_params=litellm_params,
        )

        response = await azure_openai_client.beta.assistants.delete(
            assistant_id=assistant_id
        )
        return response

    def delete_assistant(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        azure_ad_token: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        assistant_id: str,
        async_delete_assistants: Optional[bool] = None,
        client=None,
        litellm_params: Optional[dict] = None,
    ):
        if async_delete_assistants is not None and async_delete_assistants is True:
            return self.async_delete_assistant(
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                azure_ad_token=azure_ad_token,
                timeout=timeout,
                max_retries=max_retries,
                client=client,
                assistant_id=assistant_id,
                litellm_params=litellm_params,
            )
        azure_openai_client = self.get_azure_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
            litellm_params=litellm_params,
        )

        response = azure_openai_client.beta.assistants.delete(assistant_id=assistant_id)
        return response
