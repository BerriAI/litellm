#### ASSISTANTS API ####
from typing import Any, Iterable, List, Literal, Optional

from openai import AsyncOpenAI, OpenAI

import litellm
from litellm.assistants.main import AssistantDeleted
from litellm.types.llms.openai import (
    Assistant,
    AssistantToolParam,
    AsyncCursorPage,
    Attachment,
    Batch,
    CreateFileRequest,
    FileContentRequest,
    FileObject,
    FileTypes,
    HttpxBinaryResponseContent,
    OpenAIMessage,
    Run,
    Thread,
)
from litellm.types.router import AssistantsTypedDict


class AssistantsAPIRouter:
    def __init__(self, assistants_config: Optional[AssistantsTypedDict] = None):
        self.assistants_config = assistants_config

    async def acreate_assistants(
        self,
        custom_llm_provider: Optional[Literal["openai", "azure"]] = None,
        client: Optional[AsyncOpenAI] = None,
        **kwargs,
    ) -> Assistant:
        if custom_llm_provider is None:
            if self.assistants_config is not None:
                custom_llm_provider = self.assistants_config["custom_llm_provider"]
                kwargs.update(self.assistants_config["litellm_params"])
            else:
                raise Exception(
                    "'custom_llm_provider' must be set. Either via:\n `Router(assistants_config={'custom_llm_provider': ..})` \nor\n `router.arun_thread(custom_llm_provider=..)`"
                )

        return await litellm.acreate_assistants(
            custom_llm_provider=custom_llm_provider, client=client, **kwargs
        )

    async def adelete_assistant(
        self,
        custom_llm_provider: Optional[Literal["openai", "azure"]] = None,
        client: Optional[AsyncOpenAI] = None,
        **kwargs,
    ) -> AssistantDeleted:
        if custom_llm_provider is None:
            if self.assistants_config is not None:
                custom_llm_provider = self.assistants_config["custom_llm_provider"]
                kwargs.update(self.assistants_config["litellm_params"])
            else:
                raise Exception(
                    "'custom_llm_provider' must be set. Either via:\n `Router(assistants_config={'custom_llm_provider': ..})` \nor\n `router.arun_thread(custom_llm_provider=..)`"
                )

        return await litellm.adelete_assistant(
            custom_llm_provider=custom_llm_provider, client=client, **kwargs
        )

    async def aget_assistants(
        self,
        custom_llm_provider: Optional[Literal["openai", "azure"]] = None,
        client: Optional[AsyncOpenAI] = None,
        **kwargs,
    ) -> AsyncCursorPage[Assistant]:
        if custom_llm_provider is None:
            if self.assistants_config is not None:
                custom_llm_provider = self.assistants_config["custom_llm_provider"]
                kwargs.update(self.assistants_config["litellm_params"])
            else:
                raise Exception(
                    "'custom_llm_provider' must be set. Either via:\n `Router(assistants_config={'custom_llm_provider': ..})` \nor\n `router.arun_thread(custom_llm_provider=..)`"
                )

        return await litellm.aget_assistants(
            custom_llm_provider=custom_llm_provider, client=client, **kwargs
        )

    async def acreate_thread(
        self,
        custom_llm_provider: Optional[Literal["openai", "azure"]] = None,
        client: Optional[AsyncOpenAI] = None,
        **kwargs,
    ) -> Thread:
        if custom_llm_provider is None:
            if self.assistants_config is not None:
                custom_llm_provider = self.assistants_config["custom_llm_provider"]
                kwargs.update(self.assistants_config["litellm_params"])
            else:
                raise Exception(
                    "'custom_llm_provider' must be set. Either via:\n `Router(assistants_config={'custom_llm_provider': ..})` \nor\n `router.arun_thread(custom_llm_provider=..)`"
                )
        return await litellm.acreate_thread(
            custom_llm_provider=custom_llm_provider, client=client, **kwargs
        )

    async def aget_thread(
        self,
        thread_id: str,
        custom_llm_provider: Optional[Literal["openai", "azure"]] = None,
        client: Optional[AsyncOpenAI] = None,
        **kwargs,
    ) -> Thread:
        if custom_llm_provider is None:
            if self.assistants_config is not None:
                custom_llm_provider = self.assistants_config["custom_llm_provider"]
                kwargs.update(self.assistants_config["litellm_params"])
            else:
                raise Exception(
                    "'custom_llm_provider' must be set. Either via:\n `Router(assistants_config={'custom_llm_provider': ..})` \nor\n `router.arun_thread(custom_llm_provider=..)`"
                )
        return await litellm.aget_thread(
            custom_llm_provider=custom_llm_provider,
            thread_id=thread_id,
            client=client,
            **kwargs,
        )

    async def a_add_message(
        self,
        thread_id: str,
        role: Literal["user", "assistant"],
        content: str,
        attachments: Optional[List[Attachment]] = None,
        metadata: Optional[dict] = None,
        custom_llm_provider: Optional[Literal["openai", "azure"]] = None,
        client: Optional[AsyncOpenAI] = None,
        **kwargs,
    ) -> OpenAIMessage:
        if custom_llm_provider is None:
            if self.assistants_config is not None:
                custom_llm_provider = self.assistants_config["custom_llm_provider"]
                kwargs.update(self.assistants_config["litellm_params"])
            else:
                raise Exception(
                    "'custom_llm_provider' must be set. Either via:\n `Router(assistants_config={'custom_llm_provider': ..})` \nor\n `router.arun_thread(custom_llm_provider=..)`"
                )

        return await litellm.a_add_message(
            custom_llm_provider=custom_llm_provider,
            thread_id=thread_id,
            role=role,
            content=content,
            attachments=attachments,
            metadata=metadata,
            client=client,
            **kwargs,
        )

    async def aget_messages(
        self,
        thread_id: str,
        custom_llm_provider: Optional[Literal["openai", "azure"]] = None,
        client: Optional[AsyncOpenAI] = None,
        **kwargs,
    ) -> AsyncCursorPage[OpenAIMessage]:
        if custom_llm_provider is None:
            if self.assistants_config is not None:
                custom_llm_provider = self.assistants_config["custom_llm_provider"]
                kwargs.update(self.assistants_config["litellm_params"])
            else:
                raise Exception(
                    "'custom_llm_provider' must be set. Either via:\n `Router(assistants_config={'custom_llm_provider': ..})` \nor\n `router.arun_thread(custom_llm_provider=..)`"
                )
        return await litellm.aget_messages(
            custom_llm_provider=custom_llm_provider,
            thread_id=thread_id,
            client=client,
            **kwargs,
        )

    async def arun_thread(
        self,
        thread_id: str,
        assistant_id: str,
        custom_llm_provider: Optional[Literal["openai", "azure"]] = None,
        additional_instructions: Optional[str] = None,
        instructions: Optional[str] = None,
        metadata: Optional[dict] = None,
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        tools: Optional[Iterable[AssistantToolParam]] = None,
        client: Optional[Any] = None,
        **kwargs,
    ) -> Run:

        if custom_llm_provider is None:
            if self.assistants_config is not None:
                custom_llm_provider = self.assistants_config["custom_llm_provider"]
                kwargs.update(self.assistants_config["litellm_params"])
            else:
                raise Exception(
                    "'custom_llm_provider' must be set. Either via:\n `Router(assistants_config={'custom_llm_provider': ..})` \nor\n `router.arun_thread(custom_llm_provider=..)`"
                )

        return await litellm.arun_thread(
            custom_llm_provider=custom_llm_provider,
            thread_id=thread_id,
            assistant_id=assistant_id,
            additional_instructions=additional_instructions,
            instructions=instructions,
            metadata=metadata,
            model=model,
            stream=stream,
            tools=tools,
            client=client,
            **kwargs,
        )

    #### [END] ASSISTANTS API ####
