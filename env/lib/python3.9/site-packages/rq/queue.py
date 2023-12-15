import logging
import sys
import traceback
import uuid
import warnings
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from functools import total_ordering
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union

from redis import WatchError

from .timeouts import BaseDeathPenalty, UnixSignalDeathPenalty

if TYPE_CHECKING:
    from redis import Redis
    from redis.client import Pipeline

    from .job import Retry

from .connections import resolve_connection
from .defaults import DEFAULT_RESULT_TTL
from .dependency import Dependency
from .exceptions import DequeueTimeout, NoSuchJobError
from .job import Callback, Job, JobStatus
from .logutils import blue, green
from .serializers import resolve_serializer
from .types import FunctionReferenceType, JobDependencyType
from .utils import as_text, backend_class, compact, get_version, import_attribute, parse_timeout, utcnow

logger = logging.getLogger("rq.queue")


class EnqueueData(
    namedtuple(
        'EnqueueData',
        [
            "func",
            "args",
            "kwargs",
            "timeout",
            "result_ttl",
            "ttl",
            "failure_ttl",
            "description",
            "depends_on",
            "job_id",
            "at_front",
            "meta",
            "retry",
            "on_success",
            "on_failure",
            "on_stopped",
        ],
    )
):
    """Helper type to use when calling enqueue_many
    NOTE: Does not support `depends_on` yet.
    """

    __slots__ = ()


@total_ordering
class Queue:
    job_class: Type['Job'] = Job
    death_penalty_class: Type[BaseDeathPenalty] = UnixSignalDeathPenalty
    DEFAULT_TIMEOUT: int = 180  # Default timeout seconds.
    redis_queue_namespace_prefix: str = 'rq:queue:'
    redis_queues_keys: str = 'rq:queues'

    @classmethod
    def all(
        cls,
        connection: Optional['Redis'] = None,
        job_class: Optional[Type['Job']] = None,
        serializer=None,
        death_penalty_class: Optional[Type[BaseDeathPenalty]] = None,
    ) -> List['Queue']:
        """Returns an iterable of all Queues.

        Args:
            connection (Optional[Redis], optional): The Redis Connection. Defaults to None.
            job_class (Optional[Job], optional): The Job class to use. Defaults to None.
            serializer (optional): The serializer to use. Defaults to None.
            death_penalty_class (Optional[Job], optional): The Death Penalty class to use. Defaults to None.

        Returns:
            queues (List[Queue]): A list of all queues.
        """
        connection = connection or resolve_connection()

        def to_queue(queue_key: Union[bytes, str]):
            return cls.from_queue_key(
                as_text(queue_key),
                connection=connection,
                job_class=job_class,
                serializer=serializer,
                death_penalty_class=death_penalty_class,
            )

        all_registerd_queues = connection.smembers(cls.redis_queues_keys)
        all_queues = [to_queue(rq_key) for rq_key in all_registerd_queues if rq_key]
        return all_queues

    @classmethod
    def from_queue_key(
        cls,
        queue_key: str,
        connection: Optional['Redis'] = None,
        job_class: Optional[Type['Job']] = None,
        serializer: Any = None,
        death_penalty_class: Optional[Type[BaseDeathPenalty]] = None,
    ) -> 'Queue':
        """Returns a Queue instance, based on the naming conventions for naming
        the internal Redis keys.  Can be used to reverse-lookup Queues by their
        Redis keys.

        Args:
            queue_key (str): The queue key
            connection (Optional[Redis], optional): Redis connection. Defaults to None.
            job_class (Optional[Job], optional): Job class. Defaults to None.
            serializer (Any, optional): Serializer. Defaults to None.
            death_penalty_class (Optional[BaseDeathPenalty], optional): Death penalty class. Defaults to None.

        Raises:
            ValueError: If the queue_key doesn't start with the defined prefix

        Returns:
            queue (Queue): The Queue object
        """
        prefix = cls.redis_queue_namespace_prefix
        if not queue_key.startswith(prefix):
            raise ValueError('Not a valid RQ queue key: {0}'.format(queue_key))
        name = queue_key[len(prefix) :]
        return cls(
            name,
            connection=connection,
            job_class=job_class,
            serializer=serializer,
            death_penalty_class=death_penalty_class,
        )

    @classmethod
    def get_intermediate_queue_key(cls, key: str) -> str:
        """Returns the intermediate queue key for a given queue key.

        Args:
            key (str): The queue key

        Returns:
            str: The intermediate queue key
        """
        return f'{key}:intermediate'

    def __init__(
        self,
        name: str = 'default',
        default_timeout: Optional[int] = None,
        connection: Optional['Redis'] = None,
        is_async: bool = True,
        job_class: Optional[Union[str, Type['Job']]] = None,
        serializer: Any = None,
        death_penalty_class: Type[BaseDeathPenalty] = UnixSignalDeathPenalty,
        **kwargs,
    ):
        """Initializes a Queue object.

        Args:
            name (str, optional): The queue name. Defaults to 'default'.
            default_timeout (Optional[int], optional): Queue's default timeout. Defaults to None.
            connection (Optional[Redis], optional): Redis connection. Defaults to None.
            is_async (bool, optional): Whether jobs should run "async" (using the worker).
                If `is_async` is false, jobs will run on the same process from where it was called. Defaults to True.
            job_class (Union[str, 'Job', optional): Job class or a string referencing the Job class path.
                Defaults to None.
            serializer (Any, optional): Serializer. Defaults to None.
            death_penalty_class (Type[BaseDeathPenalty, optional): Job class or a string referencing the Job class path.
                Defaults to UnixSignalDeathPenalty.
        """
        self.connection = connection or resolve_connection()
        prefix = self.redis_queue_namespace_prefix
        self.name = name
        self._key = '{0}{1}'.format(prefix, name)
        self._default_timeout = parse_timeout(default_timeout) or self.DEFAULT_TIMEOUT
        self._is_async = is_async
        self.log = logger

        if 'async' in kwargs:
            self._is_async = kwargs['async']
            warnings.warn('The `async` keyword is deprecated. Use `is_async` instead', DeprecationWarning)

        # override class attribute job_class if one was passed
        if job_class is not None:
            if isinstance(job_class, str):
                job_class = import_attribute(job_class)
            self.job_class = job_class
        self.death_penalty_class = death_penalty_class

        self.serializer = resolve_serializer(serializer)
        self.redis_server_version: Optional[Tuple[int, int, int]] = None

    def __len__(self):
        return self.count

    def __bool__(self):
        return True

    def __iter__(self):
        yield self

    def get_redis_server_version(self) -> Tuple[int, int, int]:
        """Return Redis server version of connection

        Returns:
            redis_version (Tuple): A tuple with the parsed Redis version (eg: (5,0,0))
        """
        if not self.redis_server_version:
            self.redis_server_version = get_version(self.connection)
        return self.redis_server_version

    @property
    def key(self):
        """Returns the Redis key for this Queue."""
        return self._key

    @property
    def intermediate_queue_key(self):
        """Returns the Redis key for intermediate queue."""
        return self.get_intermediate_queue_key(self._key)

    @property
    def registry_cleaning_key(self):
        """Redis key used to indicate this queue has been cleaned."""
        return 'rq:clean_registries:%s' % self.name

    @property
    def scheduler_pid(self) -> int:
        from rq.scheduler import RQScheduler

        pid = self.connection.get(RQScheduler.get_locking_key(self.name))
        return int(pid.decode()) if pid is not None else None

    def acquire_maintenance_lock(self) -> bool:
        """Returns a boolean indicating whether a lock to clean this queue
        is acquired. A lock expires in 899 seconds (15 minutes - 1 second)

        Returns:
            lock_acquired (bool)
        """
        lock_acquired = self.connection.set(self.registry_cleaning_key, 1, nx=1, ex=899)
        if not lock_acquired:
            return False
        return lock_acquired

    def empty(self):
        """Removes all messages on the queue.
        This is currently being done using a Lua script,
        which iterates all queue messages and deletes the jobs and it's dependents.
        It registers the Lua script and calls it.
        Even though is currently being returned, this is not strictly necessary.

        Returns:
            script (...): The Lua Script is called.
        """
        script = """
            local prefix = "{0}"
            local q = KEYS[1]
            local count = 0
            while true do
                local job_id = redis.call("lpop", q)
                if job_id == false then
                    break
                end

                -- Delete the relevant keys
                redis.call("del", prefix..job_id)
                redis.call("del", prefix..job_id..":dependents")
                count = count + 1
            end
            return count
        """.format(
            self.job_class.redis_job_namespace_prefix
        ).encode(
            "utf-8"
        )
        script = self.connection.register_script(script)
        return script(keys=[self.key])

    def delete(self, delete_jobs: bool = True):
        """Deletes the queue.

        Args:
            delete_jobs (bool): If true, removes all the associated messages on the queue first.
        """
        if delete_jobs:
            self.empty()

        with self.connection.pipeline() as pipeline:
            pipeline.srem(self.redis_queues_keys, self._key)
            pipeline.delete(self._key)
            pipeline.execute()

    def is_empty(self) -> bool:
        """Returns whether the current queue is empty.

        Returns:
            is_empty (bool): Whether the queue is empty
        """
        return self.count == 0

    @property
    def is_async(self) -> bool:
        """Returns whether the current queue is async."""
        return bool(self._is_async)

    def fetch_job(self, job_id: str) -> Optional['Job']:
        """Fetch a single job by Job ID.
        If the job key is not found, will run the `remove` method, to exclude the key.
        If the job has the same name as as the current job origin, returns the Job

        Args:
            job_id (str): The Job ID

        Returns:
            job (Optional[Job]): The job if found
        """
        try:
            job = self.job_class.fetch(job_id, connection=self.connection, serializer=self.serializer)
        except NoSuchJobError:
            self.remove(job_id)
        else:
            if job.origin == self.name:
                return job

    def get_job_position(self, job_or_id: Union['Job', str]) -> Optional[int]:
        """Returns the position of a job within the queue

        Using Redis before 6.0.6 and redis-py before 3.5.4 has a complexity of
        worse than O(N) and should not be used for very long job queues. Redis
        and redis-py version afterwards should support the LPOS command
        handling job positions within Redis c implementation.

        Args:
            job_or_id (Union[Job, str]): The Job instance or Job ID

        Returns:
            _type_: _description_
        """
        job_id = job_or_id.id if isinstance(job_or_id, self.job_class) else job_or_id

        if self.get_redis_server_version() >= (6, 0, 6):
            try:
                return self.connection.lpos(self.key, job_id)
            except AttributeError:
                # not yet implemented by redis-py
                pass

        if job_id in self.job_ids:
            return self.job_ids.index(job_id)
        return None

    def get_job_ids(self, offset: int = 0, length: int = -1) -> List[str]:
        """Returns a slice of job IDs in the queue.

        Args:
            offset (int, optional): The offset. Defaults to 0.
            length (int, optional): The slice length. Defaults to -1 (last element).

        Returns:
            _type_: _description_
        """
        start = offset
        if length >= 0:
            end = offset + (length - 1)
        else:
            end = length
        job_ids = [as_text(job_id) for job_id in self.connection.lrange(self.key, start, end)]
        self.log.debug('Getting jobs for queue %s: %d found.', green(self.name), len(job_ids))
        return job_ids

    def get_jobs(self, offset: int = 0, length: int = -1) -> List['Job']:
        """Returns a slice of jobs in the queue.

        Args:
            offset (int, optional): The offset. Defaults to 0.
            length (int, optional): The slice length. Defaults to -1.

        Returns:
            _type_: _description_
        """
        job_ids = self.get_job_ids(offset, length)
        return compact([self.fetch_job(job_id) for job_id in job_ids])

    @property
    def job_ids(self) -> List[str]:
        """Returns a list of all job IDS in the queue."""
        return self.get_job_ids()

    @property
    def jobs(self) -> List['Job']:
        """Returns a list of all (valid) jobs in the queue."""
        return self.get_jobs()

    @property
    def count(self) -> int:
        """Returns a count of all messages in the queue."""
        return self.connection.llen(self.key)

    @property
    def failed_job_registry(self):
        """Returns this queue's FailedJobRegistry."""
        from rq.registry import FailedJobRegistry

        return FailedJobRegistry(queue=self, job_class=self.job_class, serializer=self.serializer)

    @property
    def started_job_registry(self):
        """Returns this queue's StartedJobRegistry."""
        from rq.registry import StartedJobRegistry

        return StartedJobRegistry(queue=self, job_class=self.job_class, serializer=self.serializer)

    @property
    def finished_job_registry(self):
        """Returns this queue's FinishedJobRegistry."""
        from rq.registry import FinishedJobRegistry

        # TODO: Why was job_class only ommited here before?  Was it intentional?
        return FinishedJobRegistry(queue=self, job_class=self.job_class, serializer=self.serializer)

    @property
    def deferred_job_registry(self):
        """Returns this queue's DeferredJobRegistry."""
        from rq.registry import DeferredJobRegistry

        return DeferredJobRegistry(queue=self, job_class=self.job_class, serializer=self.serializer)

    @property
    def scheduled_job_registry(self):
        """Returns this queue's ScheduledJobRegistry."""
        from rq.registry import ScheduledJobRegistry

        return ScheduledJobRegistry(queue=self, job_class=self.job_class, serializer=self.serializer)

    @property
    def canceled_job_registry(self):
        """Returns this queue's CanceledJobRegistry."""
        from rq.registry import CanceledJobRegistry

        return CanceledJobRegistry(queue=self, job_class=self.job_class, serializer=self.serializer)

    def remove(self, job_or_id: Union['Job', str], pipeline: Optional['Pipeline'] = None):
        """Removes Job from queue, accepts either a Job instance or ID.

        Args:
            job_or_id (Union[Job, str]): The Job instance or Job ID string.
            pipeline (Optional[Pipeline], optional): The Redis Pipeline. Defaults to None.

        Returns:
            _type_: _description_
        """
        job_id: str = job_or_id.id if isinstance(job_or_id, self.job_class) else job_or_id

        if pipeline is not None:
            return pipeline.lrem(self.key, 1, job_id)

        return self.connection.lrem(self.key, 1, job_id)

    def compact(self):
        """Removes all "dead" jobs from the queue by cycling through it,
        while guaranteeing FIFO semantics.
        """
        COMPACT_QUEUE = '{0}_compact:{1}'.format(self.redis_queue_namespace_prefix, uuid.uuid4())  # noqa

        self.connection.rename(self.key, COMPACT_QUEUE)
        while True:
            job_id = self.connection.lpop(COMPACT_QUEUE)
            if job_id is None:
                break
            if self.job_class.exists(as_text(job_id), self.connection):
                self.connection.rpush(self.key, job_id)

    def push_job_id(self, job_id: str, pipeline: Optional['Pipeline'] = None, at_front: bool = False):
        """Pushes a job ID on the corresponding Redis queue.
        'at_front' allows you to push the job onto the front instead of the back of the queue

        Args:
            job_id (str): The Job ID
            pipeline (Optional[Pipeline], optional): The Redis Pipeline to use. Defaults to None.
            at_front (bool, optional): Whether to push the job to front of the queue. Defaults to False.
        """
        connection = pipeline if pipeline is not None else self.connection
        push = connection.lpush if at_front else connection.rpush
        result = push(self.key, job_id)
        if pipeline is None:
            self.log.debug('Pushed job %s into %s, %s job(s) are in queue.', blue(job_id), green(self.name), result)
        else:
            # Pipelines do not return the number of jobs in the queue.
            self.log.debug('Pushed job %s into %s', blue(job_id), green(self.name))

    def create_job(
        self,
        func: 'FunctionReferenceType',
        args: Union[Tuple, List, None] = None,
        kwargs: Optional[Dict] = None,
        timeout: Optional[int] = None,
        result_ttl: Optional[int] = None,
        ttl: Optional[int] = None,
        failure_ttl: Optional[int] = None,
        description: Optional[str] = None,
        depends_on: Optional['JobDependencyType'] = None,
        job_id: Optional[str] = None,
        meta: Optional[Dict] = None,
        status: JobStatus = JobStatus.QUEUED,
        retry: Optional['Retry'] = None,
        *,
        on_success: Optional[Union[Callback, Callable]] = None,
        on_failure: Optional[Union[Callback, Callable]] = None,
        on_stopped: Optional[Union[Callback, Callable]] = None,
    ) -> Job:
        """Creates a job based on parameters given

        Args:
            func (FunctionReferenceType): The function referce: a callable or the path.
            args (Union[Tuple, List, None], optional): The `*args` to pass to the function. Defaults to None.
            kwargs (Optional[Dict], optional): The `**kwargs` to pass to the function. Defaults to None.
            timeout (Optional[int], optional): Function timeout. Defaults to None.
            result_ttl (Optional[int], optional): Result time to live. Defaults to None.
            ttl (Optional[int], optional): Time to live. Defaults to None.
            failure_ttl (Optional[int], optional): Failure time to live. Defaults to None.
            description (Optional[str], optional): The description. Defaults to None.
            depends_on (Optional[JobDependencyType], optional): The job dependencies. Defaults to None.
            job_id (Optional[str], optional): Job ID. Defaults to None.
            meta (Optional[Dict], optional): Job metadata. Defaults to None.
            status (JobStatus, optional): Job status. Defaults to JobStatus.QUEUED.
            retry (Optional[Retry], optional): The Retry Object. Defaults to None.
            on_success (Optional[Union[Callback, Callable[..., Any]]], optional): Callback for on success. Defaults to
                None. Callable is deprecated.
            on_failure (Optional[Union[Callback, Callable[..., Any]]], optional): Callback for on failure. Defaults to
                None. Callable is deprecated.
            on_stopped (Optional[Union[Callback, Callable[..., Any]]], optional): Callback for on stopped. Defaults to
                None. Callable is deprecated.
            pipeline (Optional[Pipeline], optional): The Redis Pipeline. Defaults to None.

        Raises:
            ValueError: If the timeout is 0
            ValueError: If the job TTL is 0 or negative

        Returns:
            Job: The created job
        """
        timeout = parse_timeout(timeout)

        if timeout is None:
            timeout = self._default_timeout
        elif timeout == 0:
            raise ValueError('0 timeout is not allowed. Use -1 for infinite timeout')

        result_ttl = parse_timeout(result_ttl)
        failure_ttl = parse_timeout(failure_ttl)

        ttl = parse_timeout(ttl)
        if ttl is not None and ttl <= 0:
            raise ValueError('Job ttl must be greater than 0')

        job = self.job_class.create(
            func,
            args=args,
            kwargs=kwargs,
            connection=self.connection,
            result_ttl=result_ttl,
            ttl=ttl,
            failure_ttl=failure_ttl,
            status=status,
            description=description,
            depends_on=depends_on,
            timeout=timeout,
            id=job_id,
            origin=self.name,
            meta=meta,
            serializer=self.serializer,
            on_success=on_success,
            on_failure=on_failure,
            on_stopped=on_stopped,
        )

        if retry:
            job.retries_left = retry.max
            job.retry_intervals = retry.intervals

        return job

    def setup_dependencies(self, job: 'Job', pipeline: Optional['Pipeline'] = None) -> 'Job':
        """If a _dependent_ job depends on any unfinished job, register all the
        _dependent_ job's dependencies instead of enqueueing it.

        `Job#fetch_dependencies` sets WATCH on all dependencies. If
        WatchError is raised in the when the pipeline is executed, that means
        something else has modified either the set of dependencies or the
        status of one of them. In this case, we simply retry.

        Args:
            job (Job): The job
            pipeline (Optional[Pipeline], optional): The Redis Pipeline. Defaults to None.

        Returns:
            job (Job): The Job
        """
        if len(job._dependency_ids) > 0:
            orig_status = job.get_status(refresh=False)
            pipe = pipeline if pipeline is not None else self.connection.pipeline()
            while True:
                try:
                    # Also calling watch even if caller
                    # passed in a pipeline since Queue#create_job
                    # is called from within this method.
                    pipe.watch(job.dependencies_key)

                    dependencies = job.fetch_dependencies(watch=True, pipeline=pipe)

                    pipe.multi()

                    for dependency in dependencies:
                        if dependency.get_status(refresh=False) != JobStatus.FINISHED:
                            # NOTE: If the following code changes local variables, those values probably have
                            # to be set back to their original values in the handling of WatchError below!
                            job.set_status(JobStatus.DEFERRED, pipeline=pipe)
                            job.register_dependency(pipeline=pipe)
                            job.save(pipeline=pipe)
                            job.cleanup(ttl=job.ttl, pipeline=pipe)
                            if pipeline is None:
                                pipe.execute()
                            return job
                    break
                except WatchError:
                    if pipeline is None:
                        # The call to job.set_status(JobStatus.DEFERRED, pipeline=pipe) above has changed the
                        # internal "_status". We have to reset it to its original value (probably QUEUED), so
                        # if during the next run no unfinished dependencies exist anymore, the job gets
                        # enqueued correctly by enqueue_call().
                        job._status = orig_status
                        continue
                    else:
                        # if pipeline comes from caller, re-raise to them
                        raise
        elif pipeline is not None:
            pipeline.multi()  # Ensure pipeline in multi mode before returning to caller
        return job

    def enqueue_call(
        self,
        func: 'FunctionReferenceType',
        args: Union[Tuple, List, None] = None,
        kwargs: Optional[Dict] = None,
        timeout: Optional[int] = None,
        result_ttl: Optional[int] = None,
        ttl: Optional[int] = None,
        failure_ttl: Optional[int] = None,
        description: Optional[str] = None,
        depends_on: Optional['JobDependencyType'] = None,
        job_id: Optional[str] = None,
        at_front: bool = False,
        meta: Optional[Dict] = None,
        retry: Optional['Retry'] = None,
        on_success: Optional[Union[Callback, Callable[..., Any]]] = None,
        on_failure: Optional[Union[Callback, Callable[..., Any]]] = None,
        on_stopped: Optional[Union[Callback, Callable[..., Any]]] = None,
        pipeline: Optional['Pipeline'] = None,
    ) -> Job:
        """Creates a job to represent the delayed function call and enqueues it.

        It is much like `.enqueue()`, except that it takes the function's args
        and kwargs as explicit arguments.  Any kwargs passed to this function
        contain options for RQ itself.

        Args:
            func (FunctionReferenceType): The reference to the function
            args (Union[Tuple, List, None], optional): THe `*args` to pass to the function. Defaults to None.
            kwargs (Optional[Dict], optional): THe `**kwargs` to pass to the function. Defaults to None.
            timeout (Optional[int], optional): Function timeout. Defaults to None.
            result_ttl (Optional[int], optional): Result time to live. Defaults to None.
            ttl (Optional[int], optional): Time to live. Defaults to None.
            failure_ttl (Optional[int], optional): Failure time to live. Defaults to None.
            description (Optional[str], optional): The job description. Defaults to None.
            depends_on (Optional[JobDependencyType], optional): The job dependencies. Defaults to None.
            job_id (Optional[str], optional): The job ID. Defaults to None.
            at_front (bool, optional): Whether to enqueue the job at the front. Defaults to False.
            meta (Optional[Dict], optional): Metadata to attach to the job. Defaults to None.
            retry (Optional[Retry], optional): Retry object. Defaults to None.
            on_success (Optional[Union[Callback, Callable[..., Any]]], optional): Callback for on success. Defaults to
                None. Callable is deprecated.
            on_failure (Optional[Union[Callback, Callable[..., Any]]], optional): Callback for on failure. Defaults to
                None. Callable is deprecated.
            on_stopped (Optional[Union[Callback, Callable[..., Any]]], optional): Callback for on stopped. Defaults to
                None. Callable is deprecated.
            pipeline (Optional[Pipeline], optional): The Redis Pipeline. Defaults to None.

        Returns:
            Job: The enqueued Job
        """

        job = self.create_job(
            func,
            args=args,
            kwargs=kwargs,
            result_ttl=result_ttl,
            ttl=ttl,
            failure_ttl=failure_ttl,
            description=description,
            depends_on=depends_on,
            job_id=job_id,
            meta=meta,
            status=JobStatus.QUEUED,
            timeout=timeout,
            retry=retry,
            on_success=on_success,
            on_failure=on_failure,
            on_stopped=on_stopped,
        )
        return self.enqueue_job(job, pipeline=pipeline, at_front=at_front)

    @staticmethod
    def prepare_data(
        func: 'FunctionReferenceType',
        args: Union[Tuple, List, None] = None,
        kwargs: Optional[Dict] = None,
        timeout: Optional[int] = None,
        result_ttl: Optional[int] = None,
        ttl: Optional[int] = None,
        failure_ttl: Optional[int] = None,
        description: Optional[str] = None,
        depends_on: Optional[List] = None,
        job_id: Optional[str] = None,
        at_front: bool = False,
        meta: Optional[Dict] = None,
        retry: Optional['Retry'] = None,
        on_success: Optional[Union[Callback, Callable]] = None,
        on_failure: Optional[Union[Callback, Callable]] = None,
        on_stopped: Optional[Union[Callback, Callable]] = None,
    ) -> EnqueueData:
        """Need this till support dropped for python_version < 3.7, where defaults can be specified for named tuples
        And can keep this logic within EnqueueData

        Args:
            func (FunctionReferenceType): The reference to the function
            args (Union[Tuple, List, None], optional): THe `*args` to pass to the function. Defaults to None.
            kwargs (Optional[Dict], optional): THe `**kwargs` to pass to the function. Defaults to None.
            timeout (Optional[int], optional): Function timeout. Defaults to None.
            result_ttl (Optional[int], optional): Result time to live. Defaults to None.
            ttl (Optional[int], optional): Time to live. Defaults to None.
            failure_ttl (Optional[int], optional): Failure time to live. Defaults to None.
            description (Optional[str], optional): The job description. Defaults to None.
            depends_on (Optional[JobDependencyType], optional): The job dependencies. Defaults to None.
            job_id (Optional[str], optional): The job ID. Defaults to None.
            at_front (bool, optional): Whether to enqueue the job at the front. Defaults to False.
            meta (Optional[Dict], optional): Metadata to attach to the job. Defaults to None.
            retry (Optional[Retry], optional): Retry object. Defaults to None.
            on_success (Optional[Union[Callback, Callable[..., Any]]], optional): Callback for on success. Defaults to
                None. Callable is deprecated.
            on_failure (Optional[Union[Callback, Callable[..., Any]]], optional): Callback for on failure. Defaults to
                None. Callable is deprecated.
            on_stopped (Optional[Union[Callback, Callable[..., Any]]], optional): Callback for on stopped. Defaults to
                None. Callable is deprecated.

        Returns:
            EnqueueData: The EnqueueData
        """
        return EnqueueData(
            func,
            args,
            kwargs,
            timeout,
            result_ttl,
            ttl,
            failure_ttl,
            description,
            depends_on,
            job_id,
            at_front,
            meta,
            retry,
            on_success,
            on_failure,
            on_stopped,
        )

    def enqueue_many(self, job_datas: List['EnqueueData'], pipeline: Optional['Pipeline'] = None) -> List[Job]:
        """Creates multiple jobs (created via `Queue.prepare_data` calls)
        to represent the delayed function calls and enqueues them.

        Args:
            job_datas (List['EnqueueData']): A List of job data
            pipeline (Optional[Pipeline], optional): The Redis Pipeline. Defaults to None.

        Returns:
            List[Job]: A list of enqueued jobs
        """
        pipe = pipeline if pipeline is not None else self.connection.pipeline()
        jobs_without_dependencies = []
        jobs_with_unmet_dependencies = []
        jobs_with_met_dependencies = []

        def get_job_kwargs(job_data, initial_status):
            return {
                "func": job_data.func,
                "args": job_data.args,
                "kwargs": job_data.kwargs,
                "result_ttl": job_data.result_ttl,
                "ttl": job_data.ttl,
                "failure_ttl": job_data.failure_ttl,
                "description": job_data.description,
                "depends_on": job_data.depends_on,
                "job_id": job_data.job_id,
                "meta": job_data.meta,
                "status": initial_status,
                "timeout": job_data.timeout,
                "retry": job_data.retry,
                "on_success": job_data.on_success,
                "on_failure": job_data.on_failure,
            }

        # Enqueue jobs without dependencies
        job_datas_without_dependencies = [job_data for job_data in job_datas if not job_data.depends_on]
        if job_datas_without_dependencies:
            jobs_without_dependencies = [
                self._enqueue_job(
                    self.create_job(**get_job_kwargs(job_data, JobStatus.QUEUED)),
                    pipeline=pipe,
                    at_front=job_data.at_front,
                )
                for job_data in job_datas_without_dependencies
            ]
            if pipeline is None:
                pipe.execute()

        job_datas_with_dependencies = [job_data for job_data in job_datas if job_data.depends_on]
        if job_datas_with_dependencies:
            # Save all jobs with dependencies as deferred
            jobs_with_dependencies = [
                self.create_job(**get_job_kwargs(job_data, JobStatus.DEFERRED))
                for job_data in job_datas_with_dependencies
            ]
            for job in jobs_with_dependencies:
                job.save(pipeline=pipe)
            if pipeline is None:
                pipe.execute()

            # Enqueue the jobs whose dependencies have been met
            jobs_with_met_dependencies, jobs_with_unmet_dependencies = Dependency.get_jobs_with_met_dependencies(
                jobs_with_dependencies, pipeline=pipe
            )
            jobs_with_met_dependencies = [
                self._enqueue_job(job, pipeline=pipe, at_front=job.enqueue_at_front)
                for job in jobs_with_met_dependencies
            ]
            if pipeline is None:
                pipe.execute()
        return jobs_without_dependencies + jobs_with_unmet_dependencies + jobs_with_met_dependencies

    def run_job(self, job: 'Job') -> Job:
        """Run the job

        Args:
            job (Job): The job to run

        Returns:
            Job: _description_
        """
        job.perform()
        result_ttl = job.get_result_ttl(default_ttl=DEFAULT_RESULT_TTL)
        with self.connection.pipeline() as pipeline:
            job._handle_success(result_ttl=result_ttl, pipeline=pipeline)
            job.cleanup(result_ttl, pipeline=pipeline)
            pipeline.execute()
        return job

    @classmethod
    def parse_args(cls, f: 'FunctionReferenceType', *args, **kwargs):
        """
        Parses arguments passed to `queue.enqueue()` and `queue.enqueue_at()`

        The function argument `f` may be any of the following:

        * A reference to a function
        * A reference to an object's instance method
        * A string, representing the location of a function (must be
          meaningful to the import context of the workers)

        Args:
            f (FunctionReferenceType): The function reference
            args (*args): function args
            kwargs (*kwargs): function kargs
        """
        if not isinstance(f, str) and f.__module__ == '__main__':
            raise ValueError('Functions from the __main__ module cannot be processed by workers')

        # Detect explicit invocations, i.e. of the form:
        #     q.enqueue(foo, args=(1, 2), kwargs={'a': 1}, job_timeout=30)
        timeout = kwargs.pop('job_timeout', None)
        description = kwargs.pop('description', None)
        result_ttl = kwargs.pop('result_ttl', None)
        ttl = kwargs.pop('ttl', None)
        failure_ttl = kwargs.pop('failure_ttl', None)
        depends_on = kwargs.pop('depends_on', None)
        job_id = kwargs.pop('job_id', None)
        at_front = kwargs.pop('at_front', False)
        meta = kwargs.pop('meta', None)
        retry = kwargs.pop('retry', None)
        on_success = kwargs.pop('on_success', None)
        on_failure = kwargs.pop('on_failure', None)
        on_stopped = kwargs.pop('on_stopped', None)
        pipeline = kwargs.pop('pipeline', None)

        if 'args' in kwargs or 'kwargs' in kwargs:
            assert args == (), 'Extra positional arguments cannot be used when using explicit args and kwargs'  # noqa
            args = kwargs.pop('args', None)
            kwargs = kwargs.pop('kwargs', None)

        return (
            f,
            timeout,
            description,
            result_ttl,
            ttl,
            failure_ttl,
            depends_on,
            job_id,
            at_front,
            meta,
            retry,
            on_success,
            on_failure,
            on_stopped,
            pipeline,
            args,
            kwargs,
        )

    def enqueue(self, f: 'FunctionReferenceType', *args, **kwargs) -> 'Job':
        """Creates a job to represent the delayed function call and enqueues it.
        Receives the same parameters accepted by the `enqueue_call` method.

        Args:
            f (FunctionReferenceType): The function reference
            args (*args): function args
            kwargs (*kwargs): function kargs

        Returns:
            job (Job): The created Job
        """
        (
            f,
            timeout,
            description,
            result_ttl,
            ttl,
            failure_ttl,
            depends_on,
            job_id,
            at_front,
            meta,
            retry,
            on_success,
            on_failure,
            on_stopped,
            pipeline,
            args,
            kwargs,
        ) = Queue.parse_args(f, *args, **kwargs)

        return self.enqueue_call(
            func=f,
            args=args,
            kwargs=kwargs,
            timeout=timeout,
            result_ttl=result_ttl,
            ttl=ttl,
            failure_ttl=failure_ttl,
            description=description,
            depends_on=depends_on,
            job_id=job_id,
            at_front=at_front,
            meta=meta,
            retry=retry,
            on_success=on_success,
            on_failure=on_failure,
            on_stopped=on_stopped,
            pipeline=pipeline,
        )

    def enqueue_at(self, datetime: datetime, f, *args, **kwargs):
        """Schedules a job to be enqueued at specified time

        Args:
            datetime (datetime): _description_
            f (_type_): _description_

        Returns:
            _type_: _description_
        """
        (
            f,
            timeout,
            description,
            result_ttl,
            ttl,
            failure_ttl,
            depends_on,
            job_id,
            at_front,
            meta,
            retry,
            on_success,
            on_failure,
            on_stopped,
            pipeline,
            args,
            kwargs,
        ) = Queue.parse_args(f, *args, **kwargs)
        job = self.create_job(
            f,
            status=JobStatus.SCHEDULED,
            args=args,
            kwargs=kwargs,
            timeout=timeout,
            result_ttl=result_ttl,
            ttl=ttl,
            failure_ttl=failure_ttl,
            description=description,
            depends_on=depends_on,
            job_id=job_id,
            meta=meta,
            retry=retry,
            on_success=on_success,
            on_failure=on_failure,
            on_stopped=on_stopped,
        )
        if at_front:
            job.enqueue_at_front = True
        return self.schedule_job(job, datetime, pipeline=pipeline)

    def schedule_job(self, job: 'Job', datetime: datetime, pipeline: Optional['Pipeline'] = None):
        """Puts job on ScheduledJobRegistry

        Args:
            job (Job): _description_
            datetime (datetime): _description_
            pipeline (Optional[Pipeline], optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        from .registry import ScheduledJobRegistry

        registry = ScheduledJobRegistry(queue=self)

        pipe = pipeline if pipeline is not None else self.connection.pipeline()

        # Add Queue key set
        pipe.sadd(self.redis_queues_keys, self.key)
        job.save(pipeline=pipe)
        registry.schedule(job, datetime, pipeline=pipe)
        if pipeline is None:
            pipe.execute()
        return job

    def enqueue_in(self, time_delta: timedelta, func: 'FunctionReferenceType', *args, **kwargs) -> 'Job':
        """Schedules a job to be executed in a given `timedelta` object

        Args:
            time_delta (timedelta): The timedelta object
            func (FunctionReferenceType): The function reference

        Returns:
            job (Job): The enqueued Job
        """
        return self.enqueue_at(datetime.now(timezone.utc) + time_delta, func, *args, **kwargs)

    def enqueue_job(self, job: 'Job', pipeline: Optional['Pipeline'] = None, at_front: bool = False) -> Job:
        """Enqueues a job for delayed execution checking dependencies.

        Args:
            job (Job): The job to enqueue
            pipeline (Optional[Pipeline], optional): The Redis pipeline to use. Defaults to None.
            at_front (bool, optional): Whether should enqueue at the front of the queue. Defaults to False.

        Returns:
            Job: The enqued job
        """
        job.origin = self.name
        job = self.setup_dependencies(job, pipeline=pipeline)
        # If we do not depend on an unfinished job, enqueue the job.
        if job.get_status(refresh=False) != JobStatus.DEFERRED:
            return self._enqueue_job(job, pipeline=pipeline, at_front=at_front)
        return job

    def _enqueue_job(self, job: 'Job', pipeline: Optional['Pipeline'] = None, at_front: bool = False) -> Job:
        """Enqueues a job for delayed execution without checking dependencies.

        If Queue is instantiated with is_async=False, job is executed immediately.

        Args:
            job (Job): The job to enqueue
            pipeline (Optional[Pipeline], optional): The Redis pipeline to use. Defaults to None.
            at_front (bool, optional): Whether should enqueue at the front of the queue. Defaults to False.

        Returns:
            Job: The enqued job
        """
        pipe = pipeline if pipeline is not None else self.connection.pipeline()

        # Add Queue key set
        pipe.sadd(self.redis_queues_keys, self.key)
        job.redis_server_version = self.get_redis_server_version()
        job.set_status(JobStatus.QUEUED, pipeline=pipe)

        job.origin = self.name
        job.enqueued_at = utcnow()

        if job.timeout is None:
            job.timeout = self._default_timeout
        job.save(pipeline=pipe)
        job.cleanup(ttl=job.ttl, pipeline=pipe)

        if self._is_async:
            self.push_job_id(job.id, pipeline=pipe, at_front=at_front)

        if pipeline is None:
            pipe.execute()

        if not self._is_async:
            job = self.run_sync(job)

        return job

    def run_sync(self, job: 'Job') -> 'Job':
        """Run a job synchronously, meaning on the same process the method was called.

        Args:
            job (Job): The job to run

        Returns:
            Job: The job instance
        """
        with self.connection.pipeline() as pipeline:
            job.prepare_for_execution('sync', pipeline)

        try:
            job = self.run_job(job)
        except:  # noqa
            with self.connection.pipeline() as pipeline:
                job.set_status(JobStatus.FAILED, pipeline=pipeline)
                exc_string = ''.join(traceback.format_exception(*sys.exc_info()))
                job._handle_failure(exc_string, pipeline)
                pipeline.execute()

            if job.failure_callback:
                job.failure_callback(job, self.connection, *sys.exc_info())  # type: ignore
        else:
            if job.success_callback:
                job.success_callback(job, self.connection, job.return_value())  # type: ignore

        return job

    def enqueue_dependents(
        self, job: 'Job', pipeline: Optional['Pipeline'] = None, exclude_job_id: Optional[str] = None
    ):
        """Enqueues all jobs in the given job's dependents set and clears it.

        When called without a pipeline, this method uses WATCH/MULTI/EXEC.
        If you pass a pipeline, only MULTI is called. The rest is up to the
        caller.

        Args:
            job (Job): The Job to enqueue the dependents
            pipeline (Optional[Pipeline], optional): The Redis Pipeline. Defaults to None.
            exclude_job_id (Optional[str], optional): Whether to exclude the job id. Defaults to None.
        """
        from .registry import DeferredJobRegistry

        pipe = pipeline if pipeline is not None else self.connection.pipeline()
        dependents_key = job.dependents_key

        while True:
            try:
                # if a pipeline is passed, the caller is responsible for calling WATCH
                # to ensure all jobs are enqueued
                if pipeline is None:
                    pipe.watch(dependents_key)

                dependent_job_ids = {as_text(_id) for _id in pipe.smembers(dependents_key)}

                # There's no dependents
                if not dependent_job_ids:
                    break

                jobs_to_enqueue = [
                    dependent_job
                    for dependent_job in self.job_class.fetch_many(
                        dependent_job_ids, connection=self.connection, serializer=self.serializer
                    )
                    if dependent_job
                    and dependent_job.dependencies_are_met(
                        parent_job=job,
                        pipeline=pipe,
                        exclude_job_id=exclude_job_id,
                    )
                    and dependent_job.get_status(refresh=False) != JobStatus.CANCELED
                ]

                pipe.multi()

                if not jobs_to_enqueue:
                    break

                for dependent in jobs_to_enqueue:
                    enqueue_at_front = dependent.enqueue_at_front or False

                    registry = DeferredJobRegistry(
                        dependent.origin, self.connection, job_class=self.job_class, serializer=self.serializer
                    )
                    registry.remove(dependent, pipeline=pipe)

                    if dependent.origin == self.name:
                        self._enqueue_job(dependent, pipeline=pipe, at_front=enqueue_at_front)
                    else:
                        queue = self.__class__(name=dependent.origin, connection=self.connection)
                        queue._enqueue_job(dependent, pipeline=pipe, at_front=enqueue_at_front)

                # Only delete dependents_key if all dependents have been enqueued
                if len(jobs_to_enqueue) == len(dependent_job_ids):
                    pipe.delete(dependents_key)
                else:
                    enqueued_job_ids = [job.id for job in jobs_to_enqueue]
                    pipe.srem(dependents_key, *enqueued_job_ids)

                if pipeline is None:
                    pipe.execute()
                break
            except WatchError:
                if pipeline is None:
                    continue
                else:
                    # if the pipeline comes from the caller, we re-raise the
                    # exception as it it the responsibility of the caller to
                    # handle it
                    raise

    def pop_job_id(self) -> Optional[str]:
        """Pops a given job ID from this Redis queue.

        Returns:
            job_id (str): The job id
        """
        return as_text(self.connection.lpop(self.key))

    @classmethod
    def lpop(cls, queue_keys: List[str], timeout: Optional[int], connection: Optional['Redis'] = None):
        """Helper method to abstract away from some Redis API details
        where LPOP accepts only a single key, whereas BLPOP
        accepts multiple.  So if we want the non-blocking LPOP, we need to
        iterate over all queues, do individual LPOPs, and return the result.

        Until Redis receives a specific method for this, we'll have to wrap it
        this way.

        The timeout parameter is interpreted as follows:
            None - non-blocking (return immediately)
             > 0 - maximum number of seconds to block

        Args:
            queue_keys (_type_): _description_
            timeout (Optional[int]): _description_
            connection (Optional[Redis], optional): _description_. Defaults to None.

        Raises:
            ValueError: If timeout of 0 was passed
            DequeueTimeout: BLPOP Timeout

        Returns:
            _type_: _description_
        """
        connection = connection or resolve_connection()
        if timeout is not None:  # blocking variant
            if timeout == 0:
                raise ValueError('RQ does not support indefinite timeouts. Please pick a timeout value > 0')
            colored_queues = ', '.join(map(str, [green(str(queue)) for queue in queue_keys]))
            logger.debug(f"Starting BLPOP operation for queues {colored_queues} with timeout of {timeout}")
            result = connection.blpop(queue_keys, timeout)
            if result is None:
                logger.debug(f"BLPOP timeout, no jobs found on queues {colored_queues}")
                raise DequeueTimeout(timeout, queue_keys)
            queue_key, job_id = result
            return queue_key, job_id
        else:  # non-blocking variant
            for queue_key in queue_keys:
                blob = connection.lpop(queue_key)
                if blob is not None:
                    return queue_key, blob
            return None

    @classmethod
    def lmove(cls, connection: 'Redis', queue_key: str, timeout: Optional[int]):
        """Similar to lpop, but accepts only a single queue key and immediately pushes
        the result to an intermediate queue.
        """
        if timeout is not None:  # blocking variant
            if timeout == 0:
                raise ValueError('RQ does not support indefinite timeouts. Please pick a timeout value > 0')
            colored_queue = green(queue_key)
            logger.debug(f"Starting BLMOVE operation for {colored_queue} with timeout of {timeout}")
            result = connection.blmove(queue_key, cls.get_intermediate_queue_key(queue_key), timeout)
            if result is None:
                logger.debug(f"BLMOVE timeout, no jobs found on {colored_queue}")
                raise DequeueTimeout(timeout, queue_key)
            return queue_key, result
        else:  # non-blocking variant
            result = connection.lmove(queue_key, cls.get_intermediate_queue_key(queue_key))
            if result is not None:
                return queue_key, result
            return None

    @classmethod
    def dequeue_any(
        cls,
        queues: List['Queue'],
        timeout: Optional[int],
        connection: Optional['Redis'] = None,
        job_class: Optional['Job'] = None,
        serializer: Any = None,
        death_penalty_class: Optional[Type[BaseDeathPenalty]] = None,
    ) -> Tuple['Job', 'Queue']:
        """Class method returning the job_class instance at the front of the given
        set of Queues, where the order of the queues is important.

        When all of the Queues are empty, depending on the `timeout` argument,
        either blocks execution of this function for the duration of the
        timeout or until new messages arrive on any of the queues, or returns
        None.

        See the documentation of cls.lpop for the interpretation of timeout.

        Args:
            queues (List[Queue]): List of queue objects
            timeout (Optional[int]): Timeout for the LPOP
            connection (Optional[Redis], optional): Redis Connection. Defaults to None.
            job_class (Optional[Type[Job]], optional): The job class. Defaults to None.
            serializer (Any, optional): Serializer to use. Defaults to None.
            death_penalty_class (Optional[Type[BaseDeathPenalty]], optional): The death penalty class. Defaults to None.

        Raises:
            e: Any exception

        Returns:
            job, queue (Tuple[Job, Queue]): A tuple of Job, Queue
        """
        job_class: Job = backend_class(cls, 'job_class', override=job_class)

        while True:
            queue_keys = [q.key for q in queues]
            if len(queue_keys) == 1 and get_version(connection) >= (6, 2, 0):
                result = cls.lmove(connection, queue_keys[0], timeout)
            else:
                result = cls.lpop(queue_keys, timeout, connection=connection)
            if result is None:
                return None
            queue_key, job_id = map(as_text, result)
            queue = cls.from_queue_key(
                queue_key,
                connection=connection,
                job_class=job_class,
                serializer=serializer,
                death_penalty_class=death_penalty_class,
            )
            try:
                job = job_class.fetch(job_id, connection=connection, serializer=serializer)
            except NoSuchJobError:
                # Silently pass on jobs that don't exist (anymore),
                # and continue in the look
                continue
            except Exception as e:
                # Attach queue information on the exception for improved error
                # reporting
                e.job_id = job_id
                e.queue = queue
                raise e
            return job, queue
        return None, None

    # Total ordering definition (the rest of the required Python methods are
    # auto-generated by the @total_ordering decorator)
    def __eq__(self, other):  # noqa
        if not isinstance(other, Queue):
            raise TypeError('Cannot compare queues to other objects')
        return self.name == other.name

    def __lt__(self, other):
        if not isinstance(other, Queue):
            raise TypeError('Cannot compare queues to other objects')
        return self.name < other.name

    def __hash__(self):  # pragma: no cover
        return hash(self.name)

    def __repr__(self):  # noqa  # pragma: no cover
        return '{0}({1!r})'.format(self.__class__.__name__, self.name)

    def __str__(self):
        return '<{0} {1}>'.format(self.__class__.__name__, self.name)
