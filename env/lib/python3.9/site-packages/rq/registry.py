import calendar
import logging
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, List, Optional, Type, Union

from rq.serializers import resolve_serializer

from .timeouts import BaseDeathPenalty, UnixSignalDeathPenalty

if TYPE_CHECKING:
    from redis import Redis
    from redis.client import Pipeline

from .connections import resolve_connection
from .defaults import DEFAULT_FAILURE_TTL
from .exceptions import AbandonedJobError, InvalidJobOperation, NoSuchJobError
from .job import Job, JobStatus
from .queue import Queue
from .utils import as_text, backend_class, current_timestamp

logger = logging.getLogger("rq.registry")


class BaseRegistry:
    """
    Base implementation of a job registry, implemented in Redis sorted set.
    Each job is stored as a key in the registry, scored by expiration time
    (unix timestamp).
    """

    job_class = Job
    death_penalty_class = UnixSignalDeathPenalty
    key_template = 'rq:registry:{0}'

    def __init__(
        self,
        name: str = 'default',
        connection: Optional['Redis'] = None,
        job_class: Optional[Type['Job']] = None,
        queue: Optional['Queue'] = None,
        serializer: Any = None,
        death_penalty_class: Optional[Type[BaseDeathPenalty]] = None,
    ):
        if queue:
            self.name = queue.name
            self.connection = queue.connection or resolve_connection()
            self.serializer = queue.serializer
        else:
            self.name = name
            self.connection = connection or resolve_connection()
            self.serializer = resolve_serializer(serializer)

        self.key = self.key_template.format(self.name)
        self.job_class = backend_class(self, 'job_class', override=job_class)
        self.death_penalty_class = backend_class(self, 'death_penalty_class', override=death_penalty_class)

    def __len__(self):
        """Returns the number of jobs in this registry"""
        return self.count

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.connection.connection_pool.connection_kwargs == other.connection.connection_pool.connection_kwargs
        )

    def __contains__(self, item: Union[str, 'Job']) -> bool:
        """
        Returns a boolean indicating registry contains the given
        job instance or job id.

        Args:
            item (Union[str, Job]): A Job ID or a Job.
        """
        job_id = item
        if isinstance(item, self.job_class):
            job_id = item.id
        return self.connection.zscore(self.key, job_id) is not None

    @property
    def count(self) -> int:
        """Returns the number of jobs in this registry

        Returns:
            int: _description_
        """
        self.cleanup()
        return self.connection.zcard(self.key)

    def add(self, job: 'Job', ttl=0, pipeline: Optional['Pipeline'] = None, xx: bool = False) -> int:
        """Adds a job to a registry with expiry time of now + ttl, unless it's -1 which is set to +inf

        Args:
            job (Job): The Job to add
            ttl (int, optional): The time to live. Defaults to 0.
            pipeline (Optional[Pipeline], optional): The Redis Pipeline. Defaults to None.
            xx (bool, optional): .... Defaults to False.

        Returns:
            result (int): The ZADD command result
        """
        score = ttl if ttl < 0 else current_timestamp() + ttl
        if score == -1:
            score = '+inf'
        if pipeline is not None:
            return pipeline.zadd(self.key, {job.id: score}, xx=xx)

        return self.connection.zadd(self.key, {job.id: score}, xx=xx)

    def remove(self, job: 'Job', pipeline: Optional['Pipeline'] = None, delete_job: bool = False):
        """Removes job from registry and deletes it if `delete_job == True`

        Args:
            job (Job): The Job to remove from the registry
            pipeline (Optional[Pipeline], optional): The Redis Pipeline. Defaults to None.
            delete_job (bool, optional): If should delete the job.. Defaults to False.
        """
        connection = pipeline if pipeline is not None else self.connection
        job_id = job.id if isinstance(job, self.job_class) else job
        result = connection.zrem(self.key, job_id)
        if delete_job:
            if isinstance(job, self.job_class):
                job_instance = job
            else:
                job_instance = Job.fetch(job_id, connection=connection, serializer=self.serializer)
            job_instance.delete()
        return result

    def get_expired_job_ids(self, timestamp: Optional[float] = None):
        """Returns job ids whose score are less than current timestamp.

        Returns ids for jobs with an expiry time earlier than timestamp,
        specified as seconds since the Unix epoch. timestamp defaults to call
        time if unspecified.
        """
        score = timestamp if timestamp is not None else current_timestamp()
        expired_jobs = self.connection.zrangebyscore(self.key, 0, score)
        return [as_text(job_id) for job_id in expired_jobs]

    def get_job_ids(self, start: int = 0, end: int = -1):
        """Returns list of all job ids.

        Args:
            start (int, optional): _description_. Defaults to 0.
            end (int, optional): _description_. Defaults to -1.

        Returns:
            _type_: _description_
        """
        self.cleanup()
        return [as_text(job_id) for job_id in self.connection.zrange(self.key, start, end)]

    def get_queue(self):
        """Returns Queue object associated with this registry."""
        return Queue(self.name, connection=self.connection, serializer=self.serializer)

    def get_expiration_time(self, job: 'Job') -> datetime:
        """Returns job's expiration time.

        Args:
            job (Job): The Job to get the expiration
        """
        score = self.connection.zscore(self.key, job.id)
        return datetime.utcfromtimestamp(score)

    def requeue(self, job_or_id: Union['Job', str], at_front: bool = False) -> 'Job':
        """Requeues the job with the given job ID.

        Args:
            job_or_id (Union[&#39;Job&#39;, str]): The Job or the Job ID
            at_front (bool, optional): If the Job should be put at the front of the queue. Defaults to False.

        Raises:
            InvalidJobOperation: If nothing is returned from the `ZREM` operation.

        Returns:
            Job: The Requeued Job.
        """
        if isinstance(job_or_id, self.job_class):
            job = job_or_id
            serializer = job.serializer
        else:
            serializer = self.serializer
            job = self.job_class.fetch(job_or_id, connection=self.connection, serializer=serializer)

        result = self.connection.zrem(self.key, job.id)
        if not result:
            raise InvalidJobOperation

        with self.connection.pipeline() as pipeline:
            queue = Queue(job.origin, connection=self.connection, job_class=self.job_class, serializer=serializer)
            job.started_at = None
            job.ended_at = None
            job._exc_info = ''
            job.save()
            job = queue._enqueue_job(job, pipeline=pipeline, at_front=at_front)
            pipeline.execute()
        return job


class StartedJobRegistry(BaseRegistry):
    """
    Registry of currently executing jobs. Each queue maintains a
    StartedJobRegistry. Jobs in this registry are ones that are currently
    being executed.

    Jobs are added to registry right before they are executed and removed
    right after completion (success or failure).
    """

    key_template = 'rq:wip:{0}'

    def cleanup(self, timestamp: Optional[float] = None):
        """Remove abandoned jobs from registry and add them to FailedJobRegistry.

        Removes jobs with an expiry time earlier than timestamp, specified as
        seconds since the Unix epoch. timestamp defaults to call time if
        unspecified. Removed jobs are added to the global failed job queue.

        Args:
            timestamp (datetime): The datetime to use as the limit.
        """
        score = timestamp if timestamp is not None else current_timestamp()
        job_ids = self.get_expired_job_ids(score)

        if job_ids:
            failed_job_registry = FailedJobRegistry(self.name, self.connection, serializer=self.serializer)

            with self.connection.pipeline() as pipeline:
                for job_id in job_ids:
                    try:
                        job = self.job_class.fetch(job_id, connection=self.connection, serializer=self.serializer)
                    except NoSuchJobError:
                        continue

                    job.execute_failure_callback(
                        self.death_penalty_class, AbandonedJobError, AbandonedJobError(), traceback.extract_stack()
                    )

                    retry = job.retries_left and job.retries_left > 0

                    if retry:
                        queue = self.get_queue()
                        job.retry(queue, pipeline)

                    else:
                        exc_string = f"due to {AbandonedJobError.__name__}"
                        logger.warning(
                            f'{self.__class__.__name__} cleanup: Moving job to {FailedJobRegistry.__name__} '
                            f'({exc_string})'
                        )
                        job.set_status(JobStatus.FAILED)
                        job._exc_info = f"Moved to {FailedJobRegistry.__name__}, {exc_string}, at {datetime.now()}"
                        job.save(pipeline=pipeline, include_meta=False)
                        job.cleanup(ttl=-1, pipeline=pipeline)
                        failed_job_registry.add(job, job.failure_ttl)

                pipeline.zremrangebyscore(self.key, 0, score)
                pipeline.execute()

        return job_ids


class FinishedJobRegistry(BaseRegistry):
    """
    Registry of jobs that have been completed. Jobs are added to this
    registry after they have successfully completed for monitoring purposes.
    """

    key_template = 'rq:finished:{0}'

    def cleanup(self, timestamp: Optional[float] = None):
        """Remove expired jobs from registry.

        Removes jobs with an expiry time earlier than timestamp, specified as
        seconds since the Unix epoch. timestamp defaults to call time if
        unspecified.
        """
        score = timestamp if timestamp is not None else current_timestamp()
        self.connection.zremrangebyscore(self.key, 0, score)


class FailedJobRegistry(BaseRegistry):
    """
    Registry of containing failed jobs.
    """

    key_template = 'rq:failed:{0}'

    def cleanup(self, timestamp: Optional[float] = None):
        """Remove expired jobs from registry.

        Removes jobs with an expiry time earlier than timestamp, specified as
        seconds since the Unix epoch. timestamp defaults to call time if
        unspecified.
        """
        score = timestamp if timestamp is not None else current_timestamp()
        self.connection.zremrangebyscore(self.key, 0, score)

    def add(
        self,
        job: 'Job',
        ttl=None,
        exc_string: str = '',
        pipeline: Optional['Pipeline'] = None,
        _save_exc_to_job: bool = False,
    ):
        """
        Adds a job to a registry with expiry time of now + ttl.
        `ttl` defaults to DEFAULT_FAILURE_TTL if not specified.
        """
        if ttl is None:
            ttl = DEFAULT_FAILURE_TTL
        score = ttl if ttl < 0 else current_timestamp() + ttl

        if pipeline:
            p = pipeline
        else:
            p = self.connection.pipeline()

        job._exc_info = exc_string
        job.save(pipeline=p, include_meta=False, include_result=_save_exc_to_job)
        job.cleanup(ttl=ttl, pipeline=p)
        p.zadd(self.key, {job.id: score})

        if not pipeline:
            p.execute()


class DeferredJobRegistry(BaseRegistry):
    """
    Registry of deferred jobs (waiting for another job to finish).
    """

    key_template = 'rq:deferred:{0}'

    def cleanup(self):
        """This method is only here to prevent errors because this method is
        automatically called by `count()` and `get_job_ids()` methods
        implemented in BaseRegistry."""
        pass


class ScheduledJobRegistry(BaseRegistry):
    """
    Registry of scheduled jobs.
    """

    key_template = 'rq:scheduled:{0}'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The underlying implementation of get_jobs_to_enqueue() is
        # the same as get_expired_job_ids, but get_expired_job_ids() doesn't
        # make sense in this context
        self.get_jobs_to_enqueue = self.get_expired_job_ids

    def schedule(self, job: 'Job', scheduled_datetime, pipeline: Optional['Pipeline'] = None):
        """
        Adds job to registry, scored by its execution time (in UTC).
        If datetime has no tzinfo, it will assume localtimezone.
        """
        # If datetime has no timezone, assume server's local timezone
        if not scheduled_datetime.tzinfo:
            tz = timezone(timedelta(seconds=-(time.timezone if time.daylight == 0 else time.altzone)))
            scheduled_datetime = scheduled_datetime.replace(tzinfo=tz)

        timestamp = calendar.timegm(scheduled_datetime.utctimetuple())
        return self.connection.zadd(self.key, {job.id: timestamp})

    def cleanup(self):
        """This method is only here to prevent errors because this method is
        automatically called by `count()` and `get_job_ids()` methods
        implemented in BaseRegistry."""
        pass

    def remove_jobs(self, timestamp: Optional[datetime] = None, pipeline: Optional['Pipeline'] = None):
        """Remove jobs whose timestamp is in the past from registry.

        Args:
            timestamp (Optional[datetime], optional): The timestamp. Defaults to None.
            pipeline (Optional[Pipeline], optional): The Redis pipeline. Defaults to None.
        """
        connection = pipeline if pipeline is not None else self.connection
        score = timestamp if timestamp is not None else current_timestamp()
        return connection.zremrangebyscore(self.key, 0, score)

    def get_jobs_to_schedule(self, timestamp: Optional[datetime] = None, chunk_size: int = 1000) -> List[str]:
        """Get's a list of job IDs that should be scheduled.

        Args:
            timestamp (Optional[datetime], optional): _description_. Defaults to None.
            chunk_size (int, optional): _description_. Defaults to 1000.

        Returns:
            jobs (List[str]): A list of Job ids
        """
        score = timestamp if timestamp is not None else current_timestamp()
        jobs_to_schedule = self.connection.zrangebyscore(self.key, 0, score, start=0, num=chunk_size)
        return [as_text(job_id) for job_id in jobs_to_schedule]

    def get_scheduled_time(self, job_or_id: Union['Job', str]) -> datetime:
        """Returns datetime (UTC) at which job is scheduled to be enqueued

        Args:
            job_or_id (Union[Job, str]): The Job instance or Job ID

        Raises:
            NoSuchJobError: If the job was not found

        Returns:
            datetime (datetime): The scheduled time as datetime object
        """
        if isinstance(job_or_id, self.job_class):
            job_id = job_or_id.id
        else:
            job_id = job_or_id

        score = self.connection.zscore(self.key, job_id)
        if not score:
            raise NoSuchJobError

        return datetime.fromtimestamp(score, tz=timezone.utc)


class CanceledJobRegistry(BaseRegistry):
    key_template = 'rq:canceled:{0}'

    def get_expired_job_ids(self, timestamp: Optional[datetime] = None):
        raise NotImplementedError

    def cleanup(self):
        """This method is only here to prevent errors because this method is
        automatically called by `count()` and `get_job_ids()` methods
        implemented in BaseRegistry."""
        pass


def clean_registries(queue: 'Queue'):
    """Cleans StartedJobRegistry, FinishedJobRegistry and FailedJobRegistry of a queue.

    Args:
        queue (Queue): The queue to clean
    """
    registry = FinishedJobRegistry(
        name=queue.name, connection=queue.connection, job_class=queue.job_class, serializer=queue.serializer
    )
    registry.cleanup()
    registry = StartedJobRegistry(
        name=queue.name, connection=queue.connection, job_class=queue.job_class, serializer=queue.serializer
    )
    registry.cleanup()

    registry = FailedJobRegistry(
        name=queue.name, connection=queue.connection, job_class=queue.job_class, serializer=queue.serializer
    )
    registry.cleanup()
