from datetime import datetime
from typing import Optional, TYPE_CHECKING
from uuid import uuid4

from redis import Redis

if TYPE_CHECKING:
    from redis.client import Pipeline

from .job import Job
from .registry import BaseRegistry, StartedJobRegistry
from .utils import current_timestamp, now


def get_key(job_id: str) -> str:
    return 'rq:executions:%s' % job_id


class Execution:
    """Class to represent an execution of a job."""

    def __init__(self, id: str, job_id: str, connection: Redis, created_at: Optional[datetime] = None):
        self.id = id
        self.job_id = job_id
        self.connection = connection
        self.created_at = created_at if created_at else now()
    
    @property
    def composite_key(self):
        return f'{self.job_id}:{self.id}'

    @classmethod
    def from_composite_key(cls, composite_key: str, connection: Redis) -> 'Execution':
        job_id, id = composite_key.split(':')
        return cls(id=id, job_id=job_id, connection=connection)

    @classmethod
    def create(cls, job: Job) -> 'Execution':
        id = uuid4().hex
        return cls(id=id, job_id=job.id, connection=job.connection, created_at=now())


class ExecutionRegistry(BaseRegistry):
    """Class to represent a registry of executions."""
    key_template = 'rq:executions:{0}'

    def __init__(self, job: Job):
        self.connection = job.connection
        self.job = job

    def add(self, execution: Execution, pipeline: 'Pipeline', ttl=0, xx: bool = False) -> Any:
        """Register an execution to registry with expiry time of now + ttl, unless it's -1 which is set to +inf

        Args:
            execution (Execution): The Execution to add
            ttl (int, optional): The time to live. Defaults to 0.
            pipeline (Optional[Pipeline], optional): The Redis Pipeline. Defaults to None.
            xx (bool, optional): .... Defaults to False.

        Returns:
            result (int): The ZADD command result
        """
        score = ttl if ttl < 0 else current_timestamp() + ttl
        if score == -1:
            score = '+inf'
        return pipeline.zadd(self.key, {execution.id: score}, xx=xx)
