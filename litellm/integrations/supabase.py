#### What this does ####
#    On success + failure, log events to Supabase

import os
import traceback
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from typing import Final, Protocol

import litellm


class _SupabaseQuery(Protocol):
    def execute(self) -> Iterable[object]: ...


class _SupabaseTable(Protocol):
    def insert(self, json: Mapping[str, object]) -> _SupabaseQuery: ...

    def upsert(self, json: Mapping[str, object], *, on_conflict: str) -> _SupabaseQuery: ...


class _SupabaseClient(Protocol):
    def table(self, table_name: str) -> _SupabaseTable: ...


class Supabase:
    # Class variables or attributes
    supabase_table_name = "request_logs"

    def __init__(self):
        # Instance variables
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        try:
            import supabase
        except ImportError as e:
            raise ImportError(
                "Package 'supabase' is missing. Please install it at build time using 'pip install supabase'."
            ) from e

        if self.supabase_url is None or self.supabase_key is None:
            raise ValueError(
                "LiteLLM Error, trying to use Supabase but url or key not passed. Create a table and set `litellm.supabase_url=<your-url>` and `litellm.supabase_key=<your-key>`"
            )
        self.supabase_client: _SupabaseClient = supabase.create_client(self.supabase_url, self.supabase_key)

    def input_log_event(
        self,
        model: object,
        messages: object,
        end_user: object,
        litellm_call_id: object,
        print_verbose: Callable[[str], object],
    ) -> None:
        try:
            print_verbose(f"Supabase Logging - Enters input logging function for model {model}")
            supabase_data_obj: Final = {
                "model": model,
                "messages": messages,
                "end_user": end_user,
                "status": "initiated",
                "litellm_call_id": litellm_call_id,
            }
            data, count = self.supabase_client.table(self.supabase_table_name).insert(supabase_data_obj).execute()
            print_verbose(f"data: {data}")
        except Exception:
            print_verbose(f"Supabase Logging Error - {traceback.format_exc()}")

    def log_event(
        self,
        model: object,
        messages: object,
        end_user: object,
        response_obj,
        start_time: datetime,
        end_time,
        litellm_call_id: object,
        print_verbose: Callable[[str], object],
    ) -> None:
        try:
            print_verbose(f"Supabase Logging - Enters logging function for model {model}, response_obj: {response_obj}")

            total_cost = litellm.completion_cost(completion_response=response_obj)

            response_time: Final = (end_time - start_time).total_seconds()
            if "choices" in response_obj:
                supabase_data_obj = {
                    "response_time": response_time,
                    "model": response_obj["model"],
                    "total_cost": total_cost,
                    "messages": messages,
                    "response": response_obj["choices"][0]["message"]["content"],
                    "end_user": end_user,
                    "litellm_call_id": litellm_call_id,
                    "status": "success",
                }
                print_verbose(f"Supabase Logging - final data object: {supabase_data_obj}")
                data, count = (
                    self.supabase_client.table(self.supabase_table_name)
                    .upsert(supabase_data_obj, on_conflict="litellm_call_id")
                    .execute()
                )
            elif "error" in response_obj:
                if "Unable to map your input to a model." in response_obj["error"]:
                    total_cost = 0
                supabase_data_obj = {
                    "response_time": response_time,
                    "model": response_obj["model"],
                    "total_cost": total_cost,
                    "messages": messages,
                    "error": response_obj["error"],
                    "end_user": end_user,
                    "litellm_call_id": litellm_call_id,
                    "status": "failure",
                }
                print_verbose(f"Supabase Logging - final data object: {supabase_data_obj}")
                data, count = (
                    self.supabase_client.table(self.supabase_table_name)
                    .upsert(supabase_data_obj, on_conflict="litellm_call_id")
                    .execute()
                )

        except Exception:
            print_verbose(f"Supabase Logging Error - {traceback.format_exc()}")
