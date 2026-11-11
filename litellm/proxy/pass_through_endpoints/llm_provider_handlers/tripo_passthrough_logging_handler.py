import asyncio
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Executor
from datetime import datetime
from functools import partial
from typing import Final
from urllib.parse import quote

import httpx
from pydantic import ValidationError

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import get_standard_logging_object_payload
from litellm.litellm_core_utils.thread_pool_executor import executor
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.transcribe_passthrough_logging_handler import (
    PassThroughLogDispatch,
)
from litellm.types.passthrough_endpoints.tripo import (
    TRIPO_MAX_POLLING_ATTEMPTS,
    TRIPO_POLLING_INTERVAL,
    TRIPO_TERMINAL_STATUSES,
    TripoResponse,
    TripoTaskData,
)
from litellm.types.utils import StandardPassThroughResponseObject


def tripo_task_creation(response: httpx.Response) -> bool:
    parts: Final = response.request.url.path.strip('/').split('/')
    return (
        response.request.method == 'POST'
        and response.is_success
        and len(parts) >= 2
        and parts[0] == 'v3'
        and parts[1] != 'tasks'
    )


def submitted_tripo_task(response_body: object) -> TripoTaskData | None:
    try:
        parsed: Final = TripoResponse.model_validate(response_body)
    except ValidationError:
        return None
    if parsed.code != 0 or parsed.data is None or not parsed.data.task_id:
        return None
    return parsed.data


def tripo_task_cost(task: TripoTaskData | None, cost_per_credit: float | None) -> float | None:
    if task is None or task.credits_consumed is None or cost_per_credit is None:
        return None
    return float(task.credits_consumed) * cost_per_credit


def tripo_cost_per_credit(model: str) -> float | None:
    for candidate in (model, 'tripo/task'):
        try:
            model_info: Final = litellm.get_model_info(model=candidate, custom_llm_provider='tripo')
        except Exception:
            continue
        cost: Final = model_info.get('cost_per_credit')
        if isinstance(cost, (int, float)):
            return float(cost)
    return None


def poll_tripo_task(
    url: httpx.URL,
    authorization: str,
    task_id: str,
    client_factory: Callable[[], httpx.Client] = httpx.Client,
    sleep: Callable[[float], None] = time.sleep,
    max_attempts: int = TRIPO_MAX_POLLING_ATTEMPTS,
) -> TripoTaskData | None:
    target: Final = url.copy_with(path=f'/v3/tasks/{quote(task_id, safe="")}', query=None, fragment=None)
    with client_factory() as client:
        for attempt in range(max_attempts):
            try:
                response: Final = client.get(
                    target, headers={'Authorization': authorization}, timeout=30, follow_redirects=False
                )
                parsed: Final = TripoResponse.model_validate_json(response.content) if response.is_success else None
            except (httpx.HTTPError, ValidationError):
                verbose_proxy_logger.warning('Tripo task polling failed; cost remains unknown')
                return None
            if parsed is None or parsed.code != 0 or parsed.data is None:
                verbose_proxy_logger.warning('Tripo task polling returned an invalid response; cost remains unknown')
                return None
            if parsed.data.task_id is not None and parsed.data.task_id != task_id:
                verbose_proxy_logger.warning('Tripo task polling returned a different task; cost remains unknown')
                return None
            if parsed.data.status in TRIPO_TERMINAL_STATUSES:
                return parsed.data
            if attempt + 1 < max_attempts:
                sleep(TRIPO_POLLING_INTERVAL)
    verbose_proxy_logger.warning('Tripo task polling exhausted its attempts; cost remains unknown')
    return None


class TripoPassthroughLoggingHandler:
    def __init__(
        self,
        pool: Executor = executor,
        client_factory: Callable[[], httpx.Client] = httpx.Client,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.pool: Final = pool
        self.client_factory: Final = client_factory
        self.sleep: Final = sleep
        self.pending: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: retains running background tasks until completion

    def schedule(
        self,
        response: httpx.Response,
        response_body: object,
        logging_obj: LiteLLMLoggingObj,
        start_time: datetime,
        request_body: Mapping[str, object],
        dispatch: PassThroughLogDispatch,
        kwargs: Mapping[str, object],
        billing_model: str | None = None,
    ) -> bool:
        if not tripo_task_creation(response):
            return False
        task: Final = submitted_tripo_task(response_body)
        if task is None:
            return False
        model: Final = request_body.get('model')
        pending: Final = asyncio.create_task(
            self._poll_and_log(
                response.request.url,
                response.request.headers.get('authorization', ''),
                task.task_id,
                billing_model or (model if isinstance(model, str) else 'tripo/task'),
                logging_obj,
                start_time,
                dispatch,
                kwargs,
            )
        )
        self.pending.add(pending)
        pending.add_done_callback(self.pending.discard)
        return True

    async def _poll_and_log(
        self,
        url: httpx.URL,
        authorization: str,
        task_id: str,
        model: str,
        logging_obj: LiteLLMLoggingObj,
        start_time: datetime,
        dispatch: PassThroughLogDispatch,
        kwargs: Mapping[str, object],
    ) -> None:
        try:
            task: Final = await asyncio.get_running_loop().run_in_executor(
                self.pool,
                partial(poll_tripo_task, url, authorization, task_id, self.client_factory, self.sleep),
            )
            cost: Final = tripo_task_cost(task, tripo_cost_per_credit(model))
            if cost is None:
                verbose_proxy_logger.warning('Tripo task finished without known cost')
            end_time: Final = datetime.now()
            logged: Final = StandardPassThroughResponseObject(
                response=task.model_dump(mode='json') if task is not None else {'task_id': task_id}
            )
            logging_obj.model_call_details.update(model=model, custom_llm_provider='tripo', response_cost=cost)
            payload: Final = {
                **kwargs,
                'model': model,
                'custom_llm_provider': 'tripo',
                'response_cost': cost,
                'litellm_call_id': logging_obj.litellm_call_id,
                'litellm_params': kwargs.get('litellm_params') or logging_obj.model_call_details.get('litellm_params') or {},
            }
            standard_logging_object: Final = get_standard_logging_object_payload(
                kwargs=payload,
                init_response_obj=logged,
                start_time=start_time,
                end_time=end_time,
                logging_obj=logging_obj,
                status='success',
            )
            await dispatch(
                logging_obj=logging_obj,
                standard_logging_response_object=logged,
                result='',
                start_time=start_time,
                end_time=end_time,
                cache_hit=False,
                **{**payload, 'standard_logging_object': standard_logging_object},
            )
        except asyncio.CancelledError:
            verbose_proxy_logger.warning('Tripo billing task cancelled; its cost may be unrecorded')
            raise
        except Exception:
            verbose_proxy_logger.warning('Tripo background billing failed; its cost may be unrecorded')
