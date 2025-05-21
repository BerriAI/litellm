import enum
import uuid
from typing import List, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class ServiceMetrics(enum.Enum):
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"


class ServiceTypes(str, enum.Enum):
    """
    Enum for litellm + litellm-adjacent services (redis/postgres/etc.)
    """

    REDIS = "redis"
    DB = "postgres"
    BATCH_WRITE_TO_DB = "batch_write_to_db"
    RESET_BUDGET_JOB = "reset_budget_job"
    LITELLM = "self"
    ROUTER = "router"
    AUTH = "auth"
    PROXY_PRE_CALL = "proxy_pre_call"
    POD_LOCK_MANAGER = "pod_lock_manager"

    """
    Operational metrics for DB Transaction Queues
    """
    # daily spend update queue - actual transaction events
    IN_MEMORY_DAILY_SPEND_UPDATE_QUEUE = "in_memory_daily_spend_update_queue"
    REDIS_DAILY_SPEND_UPDATE_QUEUE = "redis_daily_spend_update_queue"
    REDIS_DAILY_TEAM_SPEND_UPDATE_QUEUE = "redis_daily_team_spend_update_queue"
    REDIS_DAILY_TAG_SPEND_UPDATE_QUEUE = "redis_daily_tag_spend_update_queue"
    # spend update queue - current spend of key, user, team
    IN_MEMORY_SPEND_UPDATE_QUEUE = "in_memory_spend_update_queue"
    REDIS_SPEND_UPDATE_QUEUE = "redis_spend_update_queue"


class ServiceConfig(TypedDict):
    """
    Configuration for services and their metrics
    """

    metrics: List[ServiceMetrics]  # What metrics this service should support


"""
Metric types to use for each service

- REDIS only needs Counter, Histogram
- Pod Lock Manager only needs a gauge metric
"""
DEFAULT_SERVICE_CONFIGS = {
    ServiceTypes.REDIS.value: {
        "metrics": [ServiceMetrics.COUNTER, ServiceMetrics.HISTOGRAM]
    },
    ServiceTypes.DB.value: {
        "metrics": [ServiceMetrics.COUNTER, ServiceMetrics.HISTOGRAM]
    },
    ServiceTypes.BATCH_WRITE_TO_DB.value: {
        "metrics": [ServiceMetrics.COUNTER, ServiceMetrics.HISTOGRAM]
    },
    ServiceTypes.RESET_BUDGET_JOB.value: {
        "metrics": [ServiceMetrics.COUNTER, ServiceMetrics.HISTOGRAM]
    },
    ServiceTypes.LITELLM.value: {
        "metrics": [ServiceMetrics.COUNTER, ServiceMetrics.HISTOGRAM]
    },
    ServiceTypes.ROUTER.value: {
        "metrics": [ServiceMetrics.COUNTER, ServiceMetrics.HISTOGRAM]
    },
    ServiceTypes.AUTH.value: {
        "metrics": [ServiceMetrics.COUNTER, ServiceMetrics.HISTOGRAM]
    },
    ServiceTypes.PROXY_PRE_CALL.value: {
        "metrics": [ServiceMetrics.COUNTER, ServiceMetrics.HISTOGRAM]
    },
    # Operational metrics for DB Transaction Queues
    ServiceTypes.POD_LOCK_MANAGER.value: {"metrics": [ServiceMetrics.GAUGE]},
    ServiceTypes.IN_MEMORY_DAILY_SPEND_UPDATE_QUEUE.value: {
        "metrics": [ServiceMetrics.GAUGE]
    },
    ServiceTypes.REDIS_DAILY_SPEND_UPDATE_QUEUE.value: {
        "metrics": [ServiceMetrics.GAUGE]
    },
    ServiceTypes.IN_MEMORY_SPEND_UPDATE_QUEUE.value: {
        "metrics": [ServiceMetrics.GAUGE]
    },
    ServiceTypes.REDIS_SPEND_UPDATE_QUEUE.value: {"metrics": [ServiceMetrics.GAUGE]},
}


class ServiceEventMetadata(TypedDict, total=False):
    """
    The metadata logged during service success/failure

    Add any extra fields you expect to access in the service_success_hook/service_failure_hook
    """

    # Dynamically control gauge labels and values
    gauge_labels: Optional[str]
    gauge_value: Optional[float]


class ServiceLoggerPayload(BaseModel):
    """
    The payload logged during service success/failure
    """

    is_error: bool = Field(description="did an error occur")
    error: Optional[str] = Field(None, description="what was the error")
    service: ServiceTypes = Field(description="who is this for? - postgres/redis")
    duration: float = Field(description="How long did the request take?")
    call_type: str = Field(description="The call of the service, being made")
    event_metadata: Optional[dict] = Field(
        description="The metadata logged during service success/failure"
    )

    def to_json(self, **kwargs):
        try:
            return self.model_dump(**kwargs)  # noqa
        except Exception as e:
            # if using pydantic v1
            return self.dict(**kwargs)
