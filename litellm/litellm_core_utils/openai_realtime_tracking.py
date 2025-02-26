import asyncio
import datetime
import os
from typing import Any, Optional
from litellm.integrations.langfuse.langfuse import LangFuseLogger
from litellm.litellm_core_utils.litellm_logging import Logging, ServiceTraceIDCache
from litellm.litellm_core_utils.specialty_caches.dynamic_logging_cache import (
    DynamicLoggingCache,
)

from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import LiteLLM_UserTable
from litellm.types.utils import (
    Usage,
    PromptTokensDetailsWrapper,
    CompletionTokensDetailsWrapper,
)


class OpenAIRealtimeCostTracking:
    def __init__(
        self,
        user_id: Optional[str],
        token: Optional[str],
        end_user_id: Optional[str] = None,
        team_id: Optional[str] = None,
        response: Any = None,
        org_id: Optional[str] = None,
        kwargs: Optional[dict] = None,
    ):
        """
        Spend logs table [api_key, total_tokens, prompt_tokens, completion_tokens, model_id, user, ] missing
        """
        from litellm.proxy.utils import hash_token
        from litellm.proxy.proxy_server import UserAPIKeyCacheTTLEnum

        self.user_id: Optional[str] = user_id
        self.user_api_key_cache: DualCache = DualCache(
            default_in_memory_ttl=UserAPIKeyCacheTTLEnum.in_memory_cache_ttl.value
        )
        self.token: Optional[str] = token
        self.end_user_id: Optional[str] = end_user_id
        self.team_id: Optional[str] = team_id
        self.response: Optional[Any] = response
        self.org_id: Optional[str] = org_id
        if token is not None and isinstance(token, str) and token.startswith("sk-"):
            self.hashed_token = hash_token(token=token)
        else:
            self.hashed_token = token
        self.prisma_client: Optional[Any] = self.get_prisma_client()
        self.kwargs = kwargs

        now: datetime.datetime = datetime.datetime.now(datetime.timezone.utc)
        self.start_time: datetime.datetime = now
        self.end_time: datetime.datetime = now

        self.set_token_counts()
        self.response_cost: float = sum(self.get_calculated_cost())

    def update_langfuse(
        self,
        langFuseLogger: LangFuseLogger,
        in_memory_trace_id_cache: ServiceTraceIDCache,
        in_memory_dynamic_logger_cache: DynamicLoggingCache,
        logging_obj: Logging,
    ):
        from litellm.integrations.langfuse.langfuse_handler import LangFuseHandler

        kwargs = {}

        for k, v in self.kwargs.items():
            if k == "original_response":
                continue
            kwargs[k] = v

        langfuse_logger_to_use = LangFuseHandler.get_langfuse_logger_for_request(
            globalLangfuseLogger=langFuseLogger,
            standard_callback_dynamic_params=logging_obj.standard_callback_dynamic_params,
            in_memory_dynamic_logger_cache=in_memory_dynamic_logger_cache,
        )

        if langfuse_logger_to_use is None:
            return

        kwargs["response_cost"] = self.response_cost
        _response = langfuse_logger_to_use.log_event_on_langfuse(
            kwargs=kwargs,
            response_obj=self.response,
            start_time=self.start_time,
            end_time=self.end_time,
            user_id=kwargs.get("user", None),
        )

        if _response is None or not isinstance(_response, dict):
            return

        _trace_id = _response.get("trace_id", None)

        if _trace_id is None:
            return

        in_memory_trace_id_cache.set_cache(
            litellm_call_id=logging_obj.litellm_call_id,
            service_name="langfuse",
            trace_id=_trace_id,
        )

    def get_prisma_client(
        self,
    ) -> Optional[Any]:
        from litellm.proxy.proxy_server import proxy_logging_obj
        from litellm.proxy.utils import PrismaClient

        db_url = os.getenv("DATABASE_URL")
        if db_url is None:
            return

        prisma_client = PrismaClient(
            database_url=db_url, proxy_logging_obj=proxy_logging_obj
        )
        return prisma_client

    def get_calculated_cost(self) -> float:
        """
        Audio tokens calculated separatelly
        """
        from litellm.llms.openai.cost_calculation import cost_per_token

        model = self.kwargs["model"]
        usage = self.get_usage_obj()
        cost = cost_per_token(model, usage)

        return cost

    def get_usage_obj(
        self,
    ) -> Usage:
        prompt_token_details = PromptTokensDetailsWrapper(
            audio_tokens=self.response["usage"]["prompt_audio_tokens"]
        )
        completion_token_details = CompletionTokensDetailsWrapper(
            audio_tokens=self.response["usage"]["completion_audio_tokens"]
        )

        return Usage(
            prompt_tokens=self.response["usage"]["prompt_tokens"],
            completion_tokens=self.response["usage"]["completion_tokens"],
            total_tokens=self.response["usage"]["total_tokens"],
            prompt_tokens_details=prompt_token_details,
            completion_tokens_details=completion_token_details,
        )

    def set_token_counts(self):
        realtime_usage = self.response.get("response", {}).get("usage", {})
        input_token_details = realtime_usage.get("input_token_details", {})
        output_token_details = realtime_usage.get("input_token_details", {})

        prompt_text_tokens = input_token_details.get("text_tokens")
        completion_text_tokens = output_token_details.get("text_tokens")

        prompt_audio_tokens = input_token_details.get("audio_tokens")
        completion_audio_tokens = output_token_details.get("audio_tokens")

        self.response["usage"] = {
            "completion_tokens": completion_text_tokens,
            "prompt_tokens": prompt_text_tokens,
            "prompt_audio_tokens": prompt_audio_tokens,
            "completion_audio_tokens": completion_audio_tokens,
            "total_tokens": realtime_usage.get("total_tokens"),
        }

    async def update_database(self):
        if self.prisma_client is None:
            return

        from litellm.proxy.proxy_server import disable_spend_logs
        from litellm.proxy.utils import update_spend
        from litellm.proxy.proxy_server import proxy_logging_obj

        update_db_funcs = [
            self.update_user_db,
            self.update_key_db,
            self.update_team_db,
            self.update_org_db,
        ]

        for func in update_db_funcs:
            asyncio.create_task(func())
        if not disable_spend_logs:
            await self.insert_spend_log_to_db()

        await self.prisma_client.connect()
        await update_spend(
            prisma_client=self.prisma_client,
            db_writer_client=None,
            proxy_logging_obj=proxy_logging_obj,
        )

    async def update_user_db(self):
        import litellm
        from litellm.proxy.proxy_server import litellm_proxy_budget_name

        existing_user_obj = await self.user_api_key_cache.async_get_cache(
            key=self.user_id
        )
        if existing_user_obj is not None and isinstance(existing_user_obj, dict):
            existing_user_obj = LiteLLM_UserTable(**existing_user_obj)

        user_ids = [
            self.user_id,
        ]
        if litellm.max_budget > 0:
            user_ids.append(litellm_proxy_budget_name)

        for _id in user_ids:
            if _id is None:
                continue

            self.prisma_client.user_list_transactons[_id] = (
                self.response_cost
                + self.prisma_client.user_list_transactons.get(_id, 0)
            )

        if self.end_user_id is None:
            return

        self.prisma_client.end_user_list_transactons[self.end_user_id] = (
            self.response_cost
            + self.prisma_client.end_user_list_transactons.get(self.end_user_id, 0)
        )

    async def update_key_db(self):
        if self.hashed_token is None:
            return

        self.prisma_client.key_list_transactons[self.hashed_token] = (
            self.response_cost
            + self.prisma_client.key_list_transactons.get(self.hashed_token, 0)
        )

    async def insert_spend_log_to_db(self):
        from litellm.proxy.proxy_server import _set_spend_logs_payload
        from litellm.proxy.spend_tracking.spend_tracking_utils import (
            get_logging_payload,
        )

        payload = get_logging_payload(
            kwargs=self.kwargs,
            response_obj=self.response,
            start_time=self.start_time,
            end_time=self.end_time,
        )
        payload["spend"] = self.response_cost

        self.prisma_client = _set_spend_logs_payload(
            payload=payload,
            spend_logs_url=os.getenv("SPEND_LOGS_URL"),
            prisma_client=self.prisma_client,
        )

    async def update_team_db(self):
        if self.team_id is None:
            return

        self.prisma_client.team_list_transactons[self.team_id] = (
            self.response_cost
            + self.prisma_client.team_list_transactons.get(self.team_id, 0)
        )
        team_member_key = f"team_id::{self.team_id}::user_id::{self.user_id}"

        self.prisma_client.team_member_list_transactons[team_member_key] = (
            self.response_cost
            + self.prisma_client.team_member_list_transactons.get(team_member_key, 0)
        )

    async def update_org_db(self):
        if self.org_id is None:
            return

        self.prisma_client.org_list_transactons[self.org_id] = (
            self.response_cost
            + self.prisma_client.org_list_transactons.get(self.org_id, 0)
        )

    @staticmethod
    def model_is_openai_realtime(model: Optional[str] = None):
        model = model if isinstance(model, str) else ""
        open_ai_realtime = [
            "gpt-4o-realtime-preview-2024-10-01",
            "gpt-4o-realtime-preview",
            "gpt-4o-realtime-preview-2024-12-17",
            "gpt-4o-mini-realtime-preview",
            "gpt-4o-mini-realtime-preview-2024-12-17",
        ]
        return model in open_ai_realtime
