import asyncio
import inspect
import json
import logging
import warnings
import zlib
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
from uuid import uuid4

from redis import WatchError

from .defaults import CALLBACK_TIMEOUT, UNSERIALIZABLE_RETURN_VALUE_PAYLOAD
from .timeouts import BaseDeathPenalty, JobTimeoutException

if TYPE_CHECKING:
    from redis import Redis
    from redis.client import Pipeline

    from .queue import Queue
    from .results import Result

from .connections import resolve_connection
from .exceptions import DeserializationError, InvalidJobOperation, NoSuchJobError
from .local import LocalStack
from .serializers import resolve_serializer
from .types import FunctionReferenceType, JobDependencyType
from .utils import (
    as_text,
    decode_redis_hash,
    ensure_list,
    get_call_string,
    get_version,
    import_attribute,
    parse_timeout,
    str_to_date,
    utcformat,
    utcnow,
)

logger = logging.getLogger("rq.job")


class JobStatus(str, Enum):
    """The Status of Job within its lifecycle at any given time."""

    QUEUED = 'queued'
    FINISHED = 'finished'
    FAILED = 'failed'
    STARTED = 'started'
    DEFERRED = 'deferred'
    SCHEDULED = 'scheduled'
    STOPPED = 'stopped'
    CANCELED = 'canceled'


class Dependency:
    def __init__(self, jobs: List[Union['Job', str]], allow_failure: bool = False, enqueue_at_front: bool = False):
        """The definition of a Dependency.

        Args:
            jobs (List[Union[Job, str]]): A list of Job instances or Job IDs.
                Anything different will raise a ValueError
            allow_failure (bool, optional): Whether to allow for failure when running the depency,
                meaning, the dependencies should continue running even after one of them failed.
                Defaults to False.
            enqueue_at_front (bool, optional): Whether this dependecy should be enqueued at the front of the queue.
                Defaults to False.

        Raises:
            ValueError: If the `jobs` param has anything different than `str` or `Job` class or the job list is empty
        """
        dependent_jobs = ensure_list(jobs)
        if not all(isinstance(job, Job) or isinstance(job, str) for job in dependent_jobs if job):
            raise ValueError("jobs: must contain objects of type Job and/or strings representing Job ids")
        elif len(dependent_jobs) < 1:
            raise ValueError("jobs: cannot be empty.")

        self.dependencies = dependent_jobs
        self.allow_failure = allow_failure
        self.enqueue_at_front = enqueue_at_front


UNEVALUATED = object()
"""Sentinel value to mark that some of our lazily evaluated properties have not
yet been evaluated.
"""


def cancel_job(job_id: str, connection: Optional['Redis'] = None, serializer=None, enqueue_dependents: bool = False):
    """Cancels the job with the given job ID, preventing execution.
    Use with caution. This will discard any job info (i.e. it can't be requeued later).

    Args:
        job_id (str): The Job ID
        connection (Optional[Redis], optional): The Redis Connection. Defaults to None.
        serializer (str, optional): The string of the path to the serializer to use. Defaults to None.
        enqueue_dependents (bool, optional): Whether dependents should still be enqueued. Defaults to False.
    """
    Job.fetch(job_id, connection=connection, serializer=serializer).cancel(enqueue_dependents=enqueue_dependents)


def get_current_job(connection: Optional['Redis'] = None, job_class: Optional['Job'] = None) -> Optional['Job']:
    """Returns the Job instance that is currently being executed.
    If this function is invoked from outside a job context, None is returned.

    Args:
        connection (Optional[Redis], optional): The connection to use. Defaults to None.
        job_class (Optional[Job], optional): The job class (DEPRECATED). Defaults to None.

    Returns:
        job (Optional[Job]): The current Job running
    """
    if connection:
        warnings.warn("connection argument for get_current_job is deprecated.", DeprecationWarning)
    if job_class:
        warnings.warn("job_class argument for get_current_job is deprecated.", DeprecationWarning)
    return _job_stack.top


def requeue_job(job_id: str, connection: 'Redis', serializer=None) -> 'Job':
    """Fetches a Job by ID and requeues it using the `requeue()` method.

    Args:
        job_id (str): The Job ID that should be requeued.
        connection (Redis): The Redis Connection to use
        serializer (Optional[str], optional): The serializer. Defaults to None.

    Returns:
        Job: The requeued Job object.
    """
    job = Job.fetch(job_id, connection=connection, serializer=serializer)
    return job.requeue()


class Job:
    """A Job is just a convenient datastructure to pass around job (meta) data."""

    redis_job_namespace_prefix = 'rq:job:'

    @classmethod
    def create(
        cls,
        func: FunctionReferenceType,
        args: Union[List[Any], Optional[Tuple]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        connection: Optional['Redis'] = None,
        result_ttl: Optional[int] = None,
        ttl: Optional[int] = None,
        status: Optional[JobStatus] = None,
        description: Optional[str] = None,
        depends_on: Optional[JobDependencyType] = None,
        timeout: Optional[int] = None,
        id: Optional[str] = None,
        origin: str = '',
        meta: Optional[Dict[str, Any]] = None,
        failure_ttl: Optional[int] = None,
        serializer=None,
        *,
        on_success: Optional[Union['Callback', Callable[..., Any]]] = None,  # Callable is deprecated
        on_failure: Optional[Union['Callback', Callable[..., Any]]] = None,  # Callable is deprecated
        on_stopped: Optional[Union['Callback', Callable[..., Any]]] = None,  # Callable is deprecated
    ) -> 'Job':
        """Creates a new Job instance for the given function, arguments, and
        keyword arguments.

        Args:
            func (FunctionReference): The function/method/callable for the Job. This can be
                a reference to a concrete callable or a string representing the  path of function/method to be
                imported. Effectively this is the only required attribute when creating a new Job.
            args (Union[List[Any], Optional[Tuple]], optional): A Tuple / List of positional arguments to pass the
                callable.  Defaults to None, meaning no args being passed.
            kwargs (Optional[Dict], optional): A Dictionary of keyword arguments to pass the callable.
                Defaults to None, meaning no kwargs being passed.
            connection (Optional[Redis], optional): The Redis connection to use. Defaults to None.
                This will be "resolved" using the `resolve_connection` function when initialzing the Job Class.
            result_ttl (Optional[int], optional): The amount of time in seconds the results should live.
                Defaults to None.
            ttl (Optional[int], optional): The Time To Live (TTL) for the job itself. Defaults to None.
            status (JobStatus, optional): The Job Status. Defaults to None.
            description (Optional[str], optional): The Job Description. Defaults to None.
            depends_on (Union['Dependency', List[Union['Dependency', 'Job']]], optional): What the jobs depends on.
                This accepts a variaty of different arguments including a `Dependency`, a list of `Dependency` or a
                `Job` list of `Job`. Defaults to None.
            timeout (Optional[int], optional): The amount of time in seconds that should be a hardlimit for a job
                execution. Defaults to None.
            id (Optional[str], optional): An Optional ID (str) for the Job. Defaults to None.
            origin (Optional[str], optional): The queue of origin. Defaults to None.
            meta (Optional[Dict[str, Any]], optional): Custom metadata about the job, takes a dictioanry.
                Defaults to None.
            failure_ttl (Optional[int], optional): THe time to live in seconds for failed-jobs information.
                Defaults to None.
            serializer (Optional[str], optional): The serializer class path to use. Should be a string with the import
                path for the serializer to use. eg. `mymodule.myfile.MySerializer` Defaults to None.
            on_success (Optional[Union['Callback', Callable[..., Any]]], optional): A callback to run when/if the Job
                finishes sucessfully. Defaults to None. Passing a callable is deprecated.
            on_failure (Optional[Union['Callback', Callable[..., Any]]], optional): A callback to run when/if the Job
                fails. Defaults to None. Passing a callable is deprecated.
            on_stopped (Optional[Union['Callback', Callable[..., Any]]], optional): A callback to run when/if the Job
                is stopped. Defaults to None. Passing a callable is deprecated.

        Raises:
            TypeError: If `args` is not a tuple/list
            TypeError: If `kwargs` is not a dict
            TypeError: If the `func` is something other than a string or a Callable reference
            ValueError: If `on_failure` is not a Callback or function or string
            ValueError: If `on_success` is not a Callback or function or string
            ValueError: If `on_stopped` is not a Callback or function or string

        Returns:
            Job: A job instance.
        """
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}

        if not isinstance(args, (tuple, list)):
            raise TypeError('{0!r} is not a valid args list'.format(args))
        if not isinstance(kwargs, dict):
            raise TypeError('{0!r} is not a valid kwargs dict'.format(kwargs))

        job = cls(connection=connection, serializer=serializer)
        if id is not None:
            job.set_id(id)

        if origin:
            job.origin = origin

        # Set the core job tuple properties
        job._instance = None
        if inspect.ismethod(func):
            job._instance = func.__self__
            job._func_name = func.__name__
        elif inspect.isfunction(func) or inspect.isbuiltin(func):
            job._func_name = '{0}.{1}'.format(func.__module__, func.__qualname__)
        elif isinstance(func, str):
            job._func_name = as_text(func)
        elif not inspect.isclass(func) and hasattr(func, '__call__'):  # a callable class instance
            job._instance = func
            job._func_name = '__call__'
        else:
            raise TypeError('Expected a callable or a string, but got: {0}'.format(func))
        job._args = args
        job._kwargs = kwargs

        if on_success:
            if not isinstance(on_success, Callback):
                warnings.warn(
                    'Passing a string or function for `on_success` is deprecated, pass `Callback` instead',
                    DeprecationWarning,
                )
                on_success = Callback(on_success)  # backward compatibility
            job._success_callback_name = on_success.name
            job._success_callback_timeout = on_success.timeout

        if on_failure:
            if not isinstance(on_failure, Callback):
                warnings.warn(
                    'Passing a string or function for `on_failure` is deprecated, pass `Callback` instead',
                    DeprecationWarning,
                )
                on_failure = Callback(on_failure)  # backward compatibility
            job._failure_callback_name = on_failure.name
            job._failure_callback_timeout = on_failure.timeout

        if on_stopped:
            if not isinstance(on_stopped, Callback):
                warnings.warn(
                    'Passing a string or function for `on_stopped` is deprecated, pass `Callback` instead',
                    DeprecationWarning,
                )
                on_stopped = Callback(on_stopped)  # backward compatibility
            job._stopped_callback_name = on_stopped.name
            job._stopped_callback_timeout = on_stopped.timeout

        # Extra meta data
        job.description = description or job.get_call_string()
        job.result_ttl = parse_timeout(result_ttl)
        job.failure_ttl = parse_timeout(failure_ttl)
        job.ttl = parse_timeout(ttl)
        job.timeout = parse_timeout(timeout)
        job._status = status
        job.meta = meta or {}

        # dependency could be job instance or id, or iterable thereof
        if depends_on is not None:
            depends_on = ensure_list(depends_on)
            depends_on_list = []
            for depends_on_item in depends_on:
                if isinstance(depends_on_item, Dependency):
                    # If a Dependency has enqueue_at_front or allow_failure set to True, these behaviors are used for
                    # all dependencies.
                    job.enqueue_at_front = job.enqueue_at_front or depends_on_item.enqueue_at_front
                    job.allow_dependency_failures = job.allow_dependency_failures or depends_on_item.allow_failure
                    depends_on_list.extend(depends_on_item.dependencies)
                else:
                    depends_on_list.extend(ensure_list(depends_on_item))
            job._dependency_ids = [dep.id if isinstance(dep, Job) else dep for dep in depends_on_list]

        return job

    def get_position(self) -> Optional[int]:
        """Get's the job's position on the queue

        Returns:
            position (Optional[int]): The position
        """
        from .queue import Queue

        if self.origin:
            q = Queue(name=self.origin, connection=self.connection)
            return q.get_job_position(self._id)
        return None

    def get_status(self, refresh: bool = True) -> JobStatus:
        """Gets the Job Status

        Args:
            refresh (bool, optional): Whether to refresh the Job. Defaults to True.

        Returns:
            status (JobStatus): The Job Status
        """
        if refresh:
            status = self.connection.hget(self.key, 'status')
            self._status = as_text(status) if status else None
        return self._status

    def set_status(self, status: JobStatus, pipeline: Optional['Pipeline'] = None) -> None:
        """Set's the Job Status

        Args:
            status (JobStatus): The Job Status to be set
            pipeline (Optional[Pipeline], optional): Optional Redis Pipeline to use. Defaults to None.
        """
        self._status = status
        connection: 'Redis' = pipeline if pipeline is not None else self.connection
        connection.hset(self.key, 'status', self._status)

    def get_meta(self, refresh: bool = True) -> Dict:
        """Get's the metadata for a Job, an arbitrary dictionary.

        Args:
            refresh (bool, optional): Whether to refresh. Defaults to True.

        Returns:
            meta (Dict): The dictionary of metadata
        """
        if refresh:
            meta = self.connection.hget(self.key, 'meta')
            self.meta = self.serializer.loads(meta) if meta else {}

        return self.meta

    @property
    def is_finished(self) -> bool:
        return self.get_status() == JobStatus.FINISHED

    @property
    def is_queued(self) -> bool:
        return self.get_status() == JobStatus.QUEUED

    @property
    def is_failed(self) -> bool:
        return self.get_status() == JobStatus.FAILED

    @property
    def is_started(self) -> bool:
        return self.get_status() == JobStatus.STARTED

    @property
    def is_deferred(self) -> bool:
        return self.get_status() == JobStatus.DEFERRED

    @property
    def is_canceled(self) -> bool:
        return self.get_status() == JobStatus.CANCELED

    @property
    def is_scheduled(self) -> bool:
        return self.get_status() == JobStatus.SCHEDULED

    @property
    def is_stopped(self) -> bool:
        return self.get_status() == JobStatus.STOPPED

    @property
    def _dependency_id(self):
        """Returns the first item in self._dependency_ids. Present to
        preserve compatibility with third party packages.
        """
        if self._dependency_ids:
            return self._dependency_ids[0]

    @property
    def dependency(self) -> Optional['Job']:
        """Returns a job's first dependency. To avoid repeated Redis fetches, we cache
        job.dependency as job._dependency.
        """
        if not self._dependency_ids:
            return None
        if hasattr(self, '_dependency'):
            return self._dependency
        job = self.fetch(self._dependency_ids[0], connection=self.connection, serializer=self.serializer)
        self._dependency = job
        return job

    @property
    def dependent_ids(self) -> List[str]:
        """Returns a list of ids of jobs whose execution depends on this
        job's successful execution."""
        return list(map(as_text, self.connection.smembers(self.dependents_key)))

    @property
    def func(self):
        func_name = self.func_name
        if func_name is None:
            return None

        if self.instance:
            return getattr(self.instance, func_name)

        return import_attribute(self.func_name)

    @property
    def success_callback(self):
        if self._success_callback is UNEVALUATED:
            if self._success_callback_name:
                self._success_callback = import_attribute(self._success_callback_name)
            else:
                self._success_callback = None

        return self._success_callback

    @property
    def success_callback_timeout(self) -> int:
        if self._success_callback_timeout is None:
            return CALLBACK_TIMEOUT

        return self._success_callback_timeout

    @property
    def failure_callback(self):
        if self._failure_callback is UNEVALUATED:
            if self._failure_callback_name:
                self._failure_callback = import_attribute(self._failure_callback_name)
            else:
                self._failure_callback = None

        return self._failure_callback

    @property
    def failure_callback_timeout(self) -> int:
        if self._failure_callback_timeout is None:
            return CALLBACK_TIMEOUT

        return self._failure_callback_timeout

    @property
    def stopped_callback(self):
        if self._stopped_callback is UNEVALUATED:
            if self._stopped_callback_name:
                self._stopped_callback = import_attribute(self._stopped_callback_name)
            else:
                self._stopped_callback = None

        return self._stopped_callback

    @property
    def stopped_callback_timeout(self) -> int:
        if self._stopped_callback_timeout is None:
            return CALLBACK_TIMEOUT

        return self._stopped_callback_timeout

    def _deserialize_data(self):
        """Deserializes the Job `data` into a tuple.
        This includes the `_func_name`, `_instance`, `_args` and `_kwargs`

        Raises:
            DeserializationError: Cathes any deserialization error (since serializers are generic)
        """
        try:
            self._func_name, self._instance, self._args, self._kwargs = self.serializer.loads(self.data)
        except Exception as e:
            raise DeserializationError() from e

    @property
    def data(self):
        if self._data is UNEVALUATED:
            if self._func_name is UNEVALUATED:
                raise ValueError('Cannot build the job data')

            if self._instance is UNEVALUATED:
                self._instance = None

            if self._args is UNEVALUATED:
                self._args = ()

            if self._kwargs is UNEVALUATED:
                self._kwargs = {}

            job_tuple = self._func_name, self._instance, self._args, self._kwargs
            self._data = self.serializer.dumps(job_tuple)
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self._func_name = UNEVALUATED
        self._instance = UNEVALUATED
        self._args = UNEVALUATED
        self._kwargs = UNEVALUATED

    @property
    def func_name(self):
        if self._func_name is UNEVALUATED:
            self._deserialize_data()
        return self._func_name

    @func_name.setter
    def func_name(self, value):
        self._func_name = value
        self._data = UNEVALUATED

    @property
    def instance(self):
        if self._instance is UNEVALUATED:
            self._deserialize_data()
        return self._instance

    @instance.setter
    def instance(self, value):
        self._instance = value
        self._data = UNEVALUATED

    @property
    def args(self) -> tuple:
        if self._args is UNEVALUATED:
            self._deserialize_data()
        return self._args

    @args.setter
    def args(self, value):
        self._args = value
        self._data = UNEVALUATED

    @property
    def kwargs(self):
        if self._kwargs is UNEVALUATED:
            self._deserialize_data()
        return self._kwargs

    @kwargs.setter
    def kwargs(self, value):
        self._kwargs = value
        self._data = UNEVALUATED

    @classmethod
    def exists(cls, job_id: str, connection: Optional['Redis'] = None) -> bool:
        """Checks whether a Job Hash exists for the given Job ID

        Args:
            job_id (str): The Job ID
            connection (Optional[Redis], optional): Optional connection to use. Defaults to None.

        Returns:
            job_exists (bool): Whether the Job exists
        """
        if not connection:
            connection = resolve_connection()
        job_key = cls.key_for(job_id)
        job_exists = connection.exists(job_key)
        return bool(job_exists)

    @classmethod
    def fetch(cls, id: str, connection: Optional['Redis'] = None, serializer=None) -> 'Job':
        """Fetches a persisted Job from its corresponding Redis key and instantiates it

        Args:
            id (str): The Job to fetch
            connection (Optional[&#39;Redis&#39;], optional): An optional Redis connection. Defaults to None.
            serializer (_type_, optional): The serializer to use. Defaults to None.

        Returns:
            Job: The Job instance
        """
        job = cls(id, connection=connection, serializer=serializer)
        job.refresh()
        return job

    @classmethod
    def fetch_many(cls, job_ids: Iterable[str], connection: 'Redis', serializer=None) -> List['Job']:
        """
        Bulk version of Job.fetch

        For any job_ids which a job does not exist, the corresponding item in
        the returned list will be None.

        Args:
            job_ids (Iterable[str]): A list of job ids.
            connection (Redis): Redis connection
            serializer (Callable): A serializer

        Returns:
            jobs (list[Job]): A list of Jobs instances.
        """
        with connection.pipeline() as pipeline:
            for job_id in job_ids:
                pipeline.hgetall(cls.key_for(job_id))
            results = pipeline.execute()

        jobs: List[Optional['Job']] = []
        for i, job_id in enumerate(job_ids):
            if not results[i]:
                jobs.append(None)
                continue

            job = cls(job_id, connection=connection, serializer=serializer)
            job.restore(results[i])
            jobs.append(job)

        return jobs

    def __init__(self, id: Optional[str] = None, connection: Optional['Redis'] = None, serializer=None):
        if connection:
            self.connection = connection
        else:
            self.connection = resolve_connection()
        self._id = id
        self.created_at = utcnow()
        self._data = UNEVALUATED
        self._func_name = UNEVALUATED
        self._instance = UNEVALUATED
        self._args = UNEVALUATED
        self._kwargs = UNEVALUATED
        self._success_callback_name = None
        self._success_callback = UNEVALUATED
        self._failure_callback_name = None
        self._failure_callback = UNEVALUATED
        self._stopped_callback_name = None
        self._stopped_callback = UNEVALUATED
        self.description: Optional[str] = None
        self.origin: str = ''
        self.enqueued_at: Optional[datetime] = None
        self.started_at: Optional[datetime] = None
        self.ended_at: Optional[datetime] = None
        self._result = None
        self._exc_info = None
        self.timeout: Optional[float] = None
        self._success_callback_timeout: Optional[int] = None
        self._failure_callback_timeout: Optional[int] = None
        self._stopped_callback_timeout: Optional[int] = None
        self.result_ttl: Optional[int] = None
        self.failure_ttl: Optional[int] = None
        self.ttl: Optional[int] = None
        self.worker_name: Optional[str] = None
        self._status = None
        self._dependency_ids: List[str] = []
        self.meta: Dict = {}
        self.serializer = resolve_serializer(serializer)
        self.retries_left: Optional[int] = None
        self.retry_intervals: Optional[List[int]] = None
        self.redis_server_version: Optional[Tuple[int, int, int]] = None
        self.last_heartbeat: Optional[datetime] = None
        self.allow_dependency_failures: Optional[bool] = None
        self.enqueue_at_front: Optional[bool] = None

        from .results import Result

        self._cached_result: Optional[Result] = None

    def __repr__(self):  # noqa  # pragma: no cover
        return '{0}({1!r}, enqueued_at={2!r})'.format(self.__class__.__name__, self._id, self.enqueued_at)

    def __str__(self):
        return '<{0} {1}: {2}>'.format(self.__class__.__name__, self.id, self.description)

    def __eq__(self, other):  # noqa
        return isinstance(other, self.__class__) and self.id == other.id

    def __hash__(self):  # pragma: no cover
        return hash(self.id)

    # Data access
    def get_id(self) -> str:  # noqa
        """The job ID for this job instance. Generates an ID lazily the
        first time the ID is requested.

        Returns:
            job_id (str): The Job ID
        """
        if self._id is None:
            self._id = str(uuid4())
        return self._id

    def set_id(self, value: str) -> None:
        """Sets a job ID for the given job

        Args:
            value (str): The value to set as Job ID
        """
        if not isinstance(value, str):
            raise TypeError('id must be a string, not {0}'.format(type(value)))
        self._id = value

    def heartbeat(self, timestamp: datetime, ttl: int, pipeline: Optional['Pipeline'] = None, xx: bool = False):
        """Sets the heartbeat for a job.
        It will set a hash in Redis with the `last_heartbeat` key and datetime value.
        If a Redis' pipeline is passed, it will use that, else, it will use the job's own connection.

        Args:
            timestamp (datetime): The timestamp to use
            ttl (int): The time to live
            pipeline (Optional[Pipeline], optional): Can receive a Redis' pipeline to use. Defaults to None.
            xx (bool, optional): Only sets the key if already exists. Defaults to False.
        """
        self.last_heartbeat = timestamp
        connection = pipeline if pipeline is not None else self.connection
        connection.hset(self.key, 'last_heartbeat', utcformat(self.last_heartbeat))
        self.started_job_registry.add(self, ttl, pipeline=pipeline, xx=xx)

    id = property(get_id, set_id)

    @classmethod
    def key_for(cls, job_id: str) -> bytes:
        """The Redis key that is used to store job hash under.

        Args:
            job_id (str): The Job ID

        Returns:
            redis_job_key (bytes): The Redis fully qualified key for the job
        """
        return (cls.redis_job_namespace_prefix + job_id).encode('utf-8')

    @classmethod
    def dependents_key_for(cls, job_id: str) -> str:
        """The Redis key that is used to store job dependents hash under.

        Args:
            job_id (str): The "parent" job id

        Returns:
            dependents_key (str): The dependents key
        """
        return '{0}{1}:dependents'.format(cls.redis_job_namespace_prefix, job_id)

    @property
    def key(self):
        """The Redis key that is used to store job hash under."""
        return self.key_for(self.id)

    @property
    def dependents_key(self):
        """The Redis key that is used to store job dependents hash under."""
        return self.dependents_key_for(self.id)

    @property
    def dependencies_key(self):
        return '{0}:{1}:dependencies'.format(self.redis_job_namespace_prefix, self.id)

    def fetch_dependencies(self, watch: bool = False, pipeline: Optional['Pipeline'] = None) -> List['Job']:
        """Fetch all of a job's dependencies. If a pipeline is supplied, and
        watch is true, then set WATCH on all the keys of all dependencies.

        Returned jobs will use self's connection, not the pipeline supplied.

        If a job has been deleted from redis, it is not returned.

        Args:
            watch (bool, optional): Wether to WATCH the keys. Defaults to False.
            pipeline (Optional[Pipeline]): The Redis' pipeline to use. Defaults to None.

        Returns:
            jobs (list[Job]): A list of Jobs
        """
        connection = pipeline if pipeline is not None else self.connection

        if watch and self._dependency_ids:
            connection.watch(*[self.key_for(dependency_id) for dependency_id in self._dependency_ids])

        dependencies_list = self.fetch_many(
            self._dependency_ids, connection=self.connection, serializer=self.serializer
        )
        jobs = [job for job in dependencies_list if job]
        return jobs

    @property
    def exc_info(self) -> Optional[str]:
        """
        Get the latest result and returns `exc_info` only if the latest result is a failure.
        """
        warnings.warn("job.exc_info is deprecated, use job.latest_result() instead.", DeprecationWarning)

        from .results import Result

        if self.supports_redis_streams:
            if not self._cached_result:
                self._cached_result = self.latest_result()

            if self._cached_result and self._cached_result.type == Result.Type.FAILED:
                return self._cached_result.exc_string

        return self._exc_info

    def return_value(self, refresh: bool = False) -> Optional[Any]:
        """Returns the return value of the latest execution, if it was successful

        Args:
            refresh (bool, optional): Whether to refresh the current status. Defaults to False.

        Returns:
            result (Optional[Any]): The job return value.
        """
        from .results import Result

        if refresh:
            self._cached_result = None

        if not self.supports_redis_streams:
            if self._result is not None:
                return self._result

            rv = self.connection.hget(self.key, 'result')
            if rv is not None:
                # cache the result
                self._result = self.serializer.loads(rv)
                return self._result
            return None

        if not self._cached_result:
            self._cached_result = self.latest_result()

        if self._cached_result and self._cached_result.type == Result.Type.SUCCESSFUL:
            return self._cached_result.return_value

        return None

    @property
    def result(self) -> Any:
        """Returns the return value of the job.

        Initially, right after enqueueing a job, the return value will be
        None.  But when the job has been executed, and had a return value or
        exception, this will return that value or exception.

        Note that, when the job has no return value (i.e. returns None), the
        ReadOnlyJob object is useless, as the result won't be written back to
        Redis.

        Also note that you cannot draw the conclusion that a job has _not_
        been executed when its return value is None, since return values
        written back to Redis will expire after a given amount of time (500
        seconds by default).
        """

        warnings.warn("job.result is deprecated, use job.return_value instead.", DeprecationWarning)

        from .results import Result

        if self.supports_redis_streams:
            if not self._cached_result:
                self._cached_result = self.latest_result()

            if self._cached_result and self._cached_result.type == Result.Type.SUCCESSFUL:
                return self._cached_result.return_value

        # Fallback to old behavior of getting result from job hash
        if self._result is None:
            rv = self.connection.hget(self.key, 'result')
            if rv is not None:
                # cache the result
                self._result = self.serializer.loads(rv)
        return self._result

    def results(self) -> List['Result']:
        """Returns all Result objects

        Returns:
            all_results (List[Result]): A list of 'Result' objects
        """
        from .results import Result

        return Result.all(self, serializer=self.serializer)

    def latest_result(self) -> Optional['Result']:
        """Get the latest job result.

        Returns:
            result (Result): The Result object
        """
        """Returns the latest Result object"""
        from .results import Result

        return Result.fetch_latest(self, serializer=self.serializer)

    def restore(self, raw_data) -> Any:
        """Overwrite properties with the provided values stored in Redis.

        Args:
            raw_data (_type_): The raw data to load the job data from

        Raises:
            NoSuchJobError: If there way an error getting the job data
        """
        obj = decode_redis_hash(raw_data)
        try:
            raw_data = obj['data']
        except KeyError:
            raise NoSuchJobError('Unexpected job format: {0}'.format(obj))

        try:
            self.data = zlib.decompress(raw_data)
        except zlib.error:
            # Fallback to uncompressed string
            self.data = raw_data

        self.created_at = str_to_date(obj.get('created_at'))
        self.origin = as_text(obj.get('origin')) if obj.get('origin') else ''
        self.worker_name = obj.get('worker_name').decode() if obj.get('worker_name') else None
        self.description = as_text(obj.get('description')) if obj.get('description') else None
        self.enqueued_at = str_to_date(obj.get('enqueued_at'))
        self.started_at = str_to_date(obj.get('started_at'))
        self.ended_at = str_to_date(obj.get('ended_at'))
        self.last_heartbeat = str_to_date(obj.get('last_heartbeat'))
        result = obj.get('result')
        if result:
            try:
                self._result = self.serializer.loads(result)
            except Exception:
                self._result = UNSERIALIZABLE_RETURN_VALUE_PAYLOAD
        self.timeout = parse_timeout(obj.get('timeout')) if obj.get('timeout') else None
        self.result_ttl = int(obj.get('result_ttl')) if obj.get('result_ttl') else None
        self.failure_ttl = int(obj.get('failure_ttl')) if obj.get('failure_ttl') else None
        self._status = obj.get('status').decode() if obj.get('status') else None

        if obj.get('success_callback_name'):
            self._success_callback_name = obj.get('success_callback_name').decode()

        if 'success_callback_timeout' in obj:
            self._success_callback_timeout = int(obj.get('success_callback_timeout'))

        if obj.get('failure_callback_name'):
            self._failure_callback_name = obj.get('failure_callback_name').decode()

        if 'failure_callback_timeout' in obj:
            self._failure_callback_timeout = int(obj.get('failure_callback_timeout'))

        if obj.get('stopped_callback_name'):
            self._stopped_callback_name = obj.get('stopped_callback_name').decode()

        if 'stopped_callback_timeout' in obj:
            self._stopped_callback_timeout = int(obj.get('stopped_callback_timeout'))

        dep_ids = obj.get('dependency_ids')
        dep_id = obj.get('dependency_id')  # for backwards compatibility
        self._dependency_ids = json.loads(dep_ids.decode()) if dep_ids else [dep_id.decode()] if dep_id else []
        allow_failures = obj.get('allow_dependency_failures')
        self.allow_dependency_failures = bool(int(allow_failures)) if allow_failures else None
        self.enqueue_at_front = bool(int(obj['enqueue_at_front'])) if 'enqueue_at_front' in obj else None
        self.ttl = int(obj.get('ttl')) if obj.get('ttl') else None
        try:
            self.meta = self.serializer.loads(obj.get('meta')) if obj.get('meta') else {}
        except Exception:  # depends on the serializer
            self.meta = {'unserialized': obj.get('meta', {})}

        self.retries_left = int(obj.get('retries_left')) if obj.get('retries_left') else None
        if obj.get('retry_intervals'):
            self.retry_intervals = json.loads(obj.get('retry_intervals').decode())

        raw_exc_info = obj.get('exc_info')
        if raw_exc_info:
            try:
                self._exc_info = as_text(zlib.decompress(raw_exc_info))
            except zlib.error:
                # Fallback to uncompressed string
                self._exc_info = as_text(raw_exc_info)

    # Persistence
    def refresh(self):  # noqa
        """Overwrite the current instance's properties with the values in the
        corresponding Redis key.

        Will raise a NoSuchJobError if no corresponding Redis key exists.
        """
        data = self.connection.hgetall(self.key)
        if not data:
            raise NoSuchJobError('No such job: {0}'.format(self.key))
        self.restore(data)

    def to_dict(self, include_meta: bool = True, include_result: bool = True) -> dict:
        """Returns a serialization of the current job instance

        You can exclude serializing the `meta` dictionary by setting
        `include_meta=False`.

        Args:
            include_meta (bool, optional): Whether to include the Job's metadata. Defaults to True.
            include_result (bool, optional): Whether to include the Job's result. Defaults to True.

        Returns:
            dict: The Job serialized as a dictionary
        """
        obj = {
            'created_at': utcformat(self.created_at or utcnow()),
            'data': zlib.compress(self.data),
            'success_callback_name': self._success_callback_name if self._success_callback_name else '',
            'failure_callback_name': self._failure_callback_name if self._failure_callback_name else '',
            'stopped_callback_name': self._stopped_callback_name if self._stopped_callback_name else '',
            'started_at': utcformat(self.started_at) if self.started_at else '',
            'ended_at': utcformat(self.ended_at) if self.ended_at else '',
            'last_heartbeat': utcformat(self.last_heartbeat) if self.last_heartbeat else '',
            'worker_name': self.worker_name or '',
        }

        if self.retries_left is not None:
            obj['retries_left'] = self.retries_left
        if self.retry_intervals is not None:
            obj['retry_intervals'] = json.dumps(self.retry_intervals)
        if self.origin:
            obj['origin'] = self.origin
        if self.description is not None:
            obj['description'] = self.description
        if self.enqueued_at is not None:
            obj['enqueued_at'] = utcformat(self.enqueued_at)

        if self._result is not None and include_result:
            try:
                obj['result'] = self.serializer.dumps(self._result)
            except:  # noqa
                obj['result'] = "Unserializable return value"
        if self._exc_info is not None and include_result:
            obj['exc_info'] = zlib.compress(str(self._exc_info).encode('utf-8'))
        if self.timeout is not None:
            obj['timeout'] = self.timeout
        if self._success_callback_timeout is not None:
            obj['success_callback_timeout'] = self._success_callback_timeout
        if self._failure_callback_timeout is not None:
            obj['failure_callback_timeout'] = self._failure_callback_timeout
        if self._stopped_callback_timeout is not None:
            obj['stopped_callback_timeout'] = self._stopped_callback_timeout
        if self.result_ttl is not None:
            obj['result_ttl'] = self.result_ttl
        if self.failure_ttl is not None:
            obj['failure_ttl'] = self.failure_ttl
        if self._status is not None:
            obj['status'] = self._status
        if self._dependency_ids:
            obj['dependency_id'] = self._dependency_ids[0]  # for backwards compatibility
            obj['dependency_ids'] = json.dumps(self._dependency_ids)
        if self.meta and include_meta:
            obj['meta'] = self.serializer.dumps(self.meta)
        if self.ttl:
            obj['ttl'] = self.ttl

        if self.allow_dependency_failures is not None:
            # convert boolean to integer to avoid redis.exception.DataError
            obj["allow_dependency_failures"] = int(self.allow_dependency_failures)

        if self.enqueue_at_front is not None:
            obj["enqueue_at_front"] = int(self.enqueue_at_front)

        return obj

    def save(self, pipeline: Optional['Pipeline'] = None, include_meta: bool = True, include_result: bool = True):
        """Dumps the current job instance to its corresponding Redis key.

        Exclude saving the `meta` dictionary by setting
        `include_meta=False`. This is useful to prevent clobbering
        user metadata without an expensive `refresh()` call first.

        Redis key persistence may be altered by `cleanup()` method.

        Args:
            pipeline (Optional[Pipeline], optional): The Redis' pipeline to use. Defaults to None.
            include_meta (bool, optional): Whether to include the job's metadata. Defaults to True.
            include_result (bool, optional): Whether to include the job's result. Defaults to True.
        """
        key = self.key
        connection = pipeline if pipeline is not None else self.connection

        mapping = self.to_dict(include_meta=include_meta, include_result=include_result)

        if self.get_redis_server_version() >= (4, 0, 0):
            connection.hset(key, mapping=mapping)
        else:
            connection.hmset(key, mapping)

    @property
    def supports_redis_streams(self) -> bool:
        """Only supported by Redis server >= 5.0 is required."""
        return self.get_redis_server_version() >= (5, 0, 0)

    def get_redis_server_version(self) -> Tuple[int, int, int]:
        """Return Redis server version of connection

        Returns:
            redis_server_version (Tuple[int, int, int]): The Redis version within a Tuple of integers, eg (5, 0, 9)
        """
        if self.redis_server_version is None:
            self.redis_server_version = get_version(self.connection)

        return self.redis_server_version

    def save_meta(self):
        """Stores job meta from the job instance to the corresponding Redis key."""
        meta = self.serializer.dumps(self.meta)
        self.connection.hset(self.key, 'meta', meta)

    def cancel(self, pipeline: Optional['Pipeline'] = None, enqueue_dependents: bool = False):
        """Cancels the given job, which will prevent the job from ever being
        ran (or inspected).

        This method merely exists as a high-level API call to cancel jobs
        without worrying about the internals required to implement job
        cancellation.

        You can enqueue the jobs dependents optionally,
        Same pipelining behavior as Queue.enqueue_dependents on whether or not a pipeline is passed in.

        Args:
            pipeline (Optional[Pipeline], optional): The Redis' pipeline to use. Defaults to None.
            enqueue_dependents (bool, optional): Whether to enqueue dependents jobs. Defaults to False.

        Raises:
            InvalidJobOperation: If the job has already been cancelled.
        """
        if self.is_canceled:
            raise InvalidJobOperation("Cannot cancel already canceled job: {}".format(self.get_id()))
        from .queue import Queue
        from .registry import CanceledJobRegistry

        pipe = pipeline or self.connection.pipeline()

        while True:
            try:
                q = Queue(
                    name=self.origin, connection=self.connection, job_class=self.__class__, serializer=self.serializer
                )

                self.set_status(JobStatus.CANCELED, pipeline=pipe)
                if enqueue_dependents:
                    # Only WATCH if no pipeline passed, otherwise caller is responsible
                    if pipeline is None:
                        pipe.watch(self.dependents_key)
                    q.enqueue_dependents(self, pipeline=pipeline, exclude_job_id=self.id)
                self._remove_from_registries(pipeline=pipe, remove_from_queue=True)

                registry = CanceledJobRegistry(
                    self.origin, self.connection, job_class=self.__class__, serializer=self.serializer
                )
                registry.add(self, pipeline=pipe)
                if pipeline is None:
                    pipe.execute()
                break
            except WatchError:
                if pipeline is None:
                    continue
                else:
                    # if the pipeline comes from the caller, we re-raise the
                    # exception as it is the responsibility of the caller to
                    # handle it
                    raise

    def requeue(self, at_front: bool = False) -> 'Job':
        """Requeues job

        Args:
            at_front (bool, optional): Whether the job should be requeued at the front of the queue. Defaults to False.

        Returns:
            job (Job): The requeued Job instance
        """
        return self.failed_job_registry.requeue(self, at_front=at_front)

    def _remove_from_registries(self, pipeline: Optional['Pipeline'] = None, remove_from_queue: bool = True):
        from .registry import BaseRegistry

        if remove_from_queue:
            from .queue import Queue

            q = Queue(name=self.origin, connection=self.connection, serializer=self.serializer)
            q.remove(self, pipeline=pipeline)
        registry: BaseRegistry
        if self.is_finished:
            from .registry import FinishedJobRegistry

            registry = FinishedJobRegistry(
                self.origin, connection=self.connection, job_class=self.__class__, serializer=self.serializer
            )
            registry.remove(self, pipeline=pipeline)

        elif self.is_deferred:
            from .registry import DeferredJobRegistry

            registry = DeferredJobRegistry(
                self.origin, connection=self.connection, job_class=self.__class__, serializer=self.serializer
            )
            registry.remove(self, pipeline=pipeline)

        elif self.is_started:
            from .registry import StartedJobRegistry

            registry = StartedJobRegistry(
                self.origin, connection=self.connection, job_class=self.__class__, serializer=self.serializer
            )
            registry.remove(self, pipeline=pipeline)

        elif self.is_scheduled:
            from .registry import ScheduledJobRegistry

            registry = ScheduledJobRegistry(
                self.origin, connection=self.connection, job_class=self.__class__, serializer=self.serializer
            )
            registry.remove(self, pipeline=pipeline)

        elif self.is_failed or self.is_stopped:
            self.failed_job_registry.remove(self, pipeline=pipeline)

        elif self.is_canceled:
            from .registry import CanceledJobRegistry

            registry = CanceledJobRegistry(
                self.origin, connection=self.connection, job_class=self.__class__, serializer=self.serializer
            )
            registry.remove(self, pipeline=pipeline)

    def delete(
        self, pipeline: Optional['Pipeline'] = None, remove_from_queue: bool = True, delete_dependents: bool = False
    ):
        """Cancels the job and deletes the job hash from Redis. Jobs depending
        on this job can optionally be deleted as well.

        Args:
            pipeline (Optional[Pipeline], optional): Redis' piepline. Defaults to None.
            remove_from_queue (bool, optional): Whether the job should be removed from the queue. Defaults to True.
            delete_dependents (bool, optional): Whether job dependents should also be deleted. Defaults to False.
        """
        connection = pipeline if pipeline is not None else self.connection

        self._remove_from_registries(pipeline=pipeline, remove_from_queue=remove_from_queue)

        if delete_dependents:
            self.delete_dependents(pipeline=pipeline)

        connection.delete(self.key, self.dependents_key, self.dependencies_key)

    def delete_dependents(self, pipeline: Optional['Pipeline'] = None):
        """Delete jobs depending on this job.

        Args:
            pipeline (Optional[Pipeline], optional): Redis' piepline. Defaults to None.
        """
        connection = pipeline if pipeline is not None else self.connection
        for dependent_id in self.dependent_ids:
            try:
                job = Job.fetch(dependent_id, connection=self.connection, serializer=self.serializer)
                job.delete(pipeline=pipeline, remove_from_queue=False)
            except NoSuchJobError:
                # It could be that the dependent job was never saved to redis
                pass
        connection.delete(self.dependents_key)

    # Job execution
    def perform(self) -> Any:  # noqa
        """The main execution method. Invokes the job function with the job arguments.
        This is the method that actually performs the job - it's what its called by the worker.

        Returns:
            result (Any): The job result
        """
        self.connection.persist(self.key)
        _job_stack.push(self)
        try:
            self._result = self._execute()
        finally:
            assert self is _job_stack.pop()
        return self._result

    def prepare_for_execution(self, worker_name: str, pipeline: 'Pipeline'):
        """Prepares the job for execution, setting the worker name,
        heartbeat information, status and other metadata before execution begins.

        Args:
            worker_name (str): The worker that will perform the job
            pipeline (Pipeline): The Redis' piipeline to use
        """
        self.worker_name = worker_name
        self.last_heartbeat = utcnow()
        self.started_at = self.last_heartbeat
        self._status = JobStatus.STARTED
        mapping = {
            'last_heartbeat': utcformat(self.last_heartbeat),
            'status': self._status,
            'started_at': utcformat(self.started_at),  # type: ignore
            'worker_name': worker_name,
        }
        if self.get_redis_server_version() >= (4, 0, 0):
            pipeline.hset(self.key, mapping=mapping)
        else:
            pipeline.hmset(self.key, mapping=mapping)

    def _execute(self) -> Any:
        """Actually runs the function with it's *args and **kwargs.
        It will use the `func` property, which was already resolved and ready to run at this point.
        If the function is a coroutine (it's an async function/method), then the `result`
        will have to be awaited within an event loop.

        Returns:
            result (Any): The function result
        """
        result = self.func(*self.args, **self.kwargs)
        if asyncio.iscoroutine(result):
            loop = asyncio.new_event_loop()
            coro_result = loop.run_until_complete(result)
            return coro_result
        return result

    def get_ttl(self, default_ttl: Optional[int] = None) -> Optional[int]:
        """Returns ttl for a job that determines how long a job will be
        persisted. In the future, this method will also be responsible
        for determining ttl for repeated jobs.

        Args:
            default_ttl (Optional[int]): The default time to live for the job

        Returns:
            ttl (int): The time to live
        """
        return default_ttl if self.ttl is None else self.ttl

    def get_result_ttl(self, default_ttl: int) -> int:
        """Returns ttl for a job that determines how long a jobs result will
        be persisted. In the future, this method will also be responsible
        for determining ttl for repeated jobs.

        Args:
            default_ttl (Optional[int]): The default time to live for the job result

        Returns:
            ttl (int): The time to live for the result
        """
        return default_ttl if self.result_ttl is None else self.result_ttl

    # Representation
    def get_call_string(self) -> Optional[str]:  # noqa
        """Returns a string representation of the call, formatted as a regular
        Python function invocation statement.

        Returns:
            call_repr (str): The string representation
        """
        call_repr = get_call_string(self.func_name, self.args, self.kwargs, max_length=75)
        return call_repr

    def cleanup(self, ttl: Optional[int] = None, pipeline: Optional['Pipeline'] = None, remove_from_queue: bool = True):
        """Prepare job for eventual deletion (if needed).
        This method is usually called after successful execution.
        How long we persist the job and its result depends on the value of ttl:
        - If ttl is 0, cleanup the job immediately.
        - If it's a positive number, set the job to expire in X seconds.
        - If ttl is negative, don't set an expiry to it (persist forever)

        Args:
            ttl (Optional[int], optional): Time to live. Defaults to None.
            pipeline (Optional[Pipeline], optional): Redis' pipeline. Defaults to None.
            remove_from_queue (bool, optional): Whether the job should be removed from the queue. Defaults to True.
        """
        if ttl == 0:
            self.delete(pipeline=pipeline, remove_from_queue=remove_from_queue)
        elif not ttl:
            return
        elif ttl > 0:
            connection = pipeline if pipeline is not None else self.connection
            connection.expire(self.key, ttl)
            connection.expire(self.dependents_key, ttl)
            connection.expire(self.dependencies_key, ttl)

    @property
    def started_job_registry(self):
        from .registry import StartedJobRegistry

        return StartedJobRegistry(
            self.origin, connection=self.connection, job_class=self.__class__, serializer=self.serializer
        )

    @property
    def failed_job_registry(self):
        from .registry import FailedJobRegistry

        return FailedJobRegistry(
            self.origin, connection=self.connection, job_class=self.__class__, serializer=self.serializer
        )

    @property
    def finished_job_registry(self):
        from .registry import FinishedJobRegistry

        return FinishedJobRegistry(
            self.origin, connection=self.connection, job_class=self.__class__, serializer=self.serializer
        )

    def execute_success_callback(self, death_penalty_class: Type[BaseDeathPenalty], result: Any):
        """Executes success_callback for a job.
        with timeout .

        Args:
            death_penalty_class (Type[BaseDeathPenalty]): The penalty class to use for timeout
            result (Any): The job's result.
        """
        if not self.success_callback:
            return

        logger.debug('Running success callbacks for %s', self.id)
        with death_penalty_class(self.success_callback_timeout, JobTimeoutException, job_id=self.id):
            self.success_callback(self, self.connection, result)

    def execute_failure_callback(self, death_penalty_class: Type[BaseDeathPenalty], *exc_info):
        """Executes failure_callback with possible timeout"""
        if not self.failure_callback:
            return

        logger.debug('Running failure callbacks for %s', self.id)
        try:
            with death_penalty_class(self.failure_callback_timeout, JobTimeoutException, job_id=self.id):
                self.failure_callback(self, self.connection, *exc_info)
        except Exception:  # noqa
            logger.exception(f'Job {self.id}: error while executing failure callback')
            raise

    def execute_stopped_callback(self, death_penalty_class: Type[BaseDeathPenalty]):
        """Executes stopped_callback with possible timeout"""
        logger.debug('Running stopped callbacks for %s', self.id)
        try:
            with death_penalty_class(self.stopped_callback_timeout, JobTimeoutException, job_id=self.id):
                self.stopped_callback(self, self.connection)
        except Exception:  # noqa
            logger.exception(f'Job {self.id}: error while executing stopped callback')
            raise

    def _handle_success(self, result_ttl: int, pipeline: 'Pipeline'):
        """Saves and cleanup job after successful execution"""
        # self.log.debug('Setting job %s status to finished', job.id)
        self.set_status(JobStatus.FINISHED, pipeline=pipeline)
        # Result should be saved in job hash only if server
        # doesn't support Redis streams
        include_result = not self.supports_redis_streams
        # Don't clobber user's meta dictionary!
        self.save(pipeline=pipeline, include_meta=False, include_result=include_result)
        # Result creation should eventually be moved to job.save() after support
        # for Redis < 5.0 is dropped. job.save(include_result=...) is used to test
        # for backward compatibility
        if self.supports_redis_streams:
            from .results import Result

            Result.create(self, Result.Type.SUCCESSFUL, return_value=self._result, ttl=result_ttl, pipeline=pipeline)

        if result_ttl != 0:
            finished_job_registry = self.finished_job_registry
            finished_job_registry.add(self, result_ttl, pipeline)

    def _handle_failure(self, exc_string: str, pipeline: 'Pipeline'):
        failed_job_registry = self.failed_job_registry
        # Exception should be saved in job hash if server
        # doesn't support Redis streams
        _save_exc_to_job = not self.supports_redis_streams
        failed_job_registry.add(
            self,
            ttl=self.failure_ttl,
            exc_string=exc_string,
            pipeline=pipeline,
            _save_exc_to_job=_save_exc_to_job,
        )
        if self.supports_redis_streams:
            from .results import Result

            Result.create_failure(self, self.failure_ttl, exc_string=exc_string, pipeline=pipeline)

    def get_retry_interval(self) -> int:
        """Returns the desired retry interval.
        If number of retries is bigger than length of intervals, the first
        value in the list will be used multiple times.

        Returns:
            retry_interval (int): The desired retry interval
        """
        if self.retry_intervals is None:
            return 0
        number_of_intervals = len(self.retry_intervals)
        index = max(number_of_intervals - self.retries_left, 0)
        return self.retry_intervals[index]

    def retry(self, queue: 'Queue', pipeline: 'Pipeline'):
        """Requeue or schedule this job for execution.
        If the the `retry_interval` was set on the job itself,
        it will calculate a scheduled time for the job to run, and instead
        of just regularly `enqueing` the job, it will `schedule` it.

        Args:
            queue (Queue): The queue to retry the job on
            pipeline (Pipeline): The Redis' pipeline to use
        """
        retry_interval = self.get_retry_interval()
        self.retries_left = self.retries_left - 1
        if retry_interval:
            scheduled_datetime = datetime.now(timezone.utc) + timedelta(seconds=retry_interval)
            self.set_status(JobStatus.SCHEDULED)
            queue.schedule_job(self, scheduled_datetime, pipeline=pipeline)
        else:
            queue._enqueue_job(self, pipeline=pipeline)

    def register_dependency(self, pipeline: Optional['Pipeline'] = None):
        """Jobs may have dependencies. Jobs are enqueued only if the jobs they
        depend on are successfully performed. We record this relation as
        a reverse dependency (a Redis set), with a key that looks something
        like:
        ..codeblock:python::

            rq:job:job_id:dependents = {'job_id_1', 'job_id_2'}

        This method adds the job in its dependencies' dependents sets,
        and adds the job to DeferredJobRegistry.

        Args:
            pipeline (Optional[Pipeline]): The Redis' pipeline. Defaults to None
        """
        from .registry import DeferredJobRegistry

        registry = DeferredJobRegistry(
            self.origin, connection=self.connection, job_class=self.__class__, serializer=self.serializer
        )
        registry.add(self, pipeline=pipeline)

        connection = pipeline if pipeline is not None else self.connection

        for dependency_id in self._dependency_ids:
            dependents_key = self.dependents_key_for(dependency_id)
            connection.sadd(dependents_key, self.id)
            connection.sadd(self.dependencies_key, dependency_id)

    @property
    def dependency_ids(self) -> List[bytes]:
        dependencies = self.connection.smembers(self.dependencies_key)
        return [Job.key_for(_id.decode()) for _id in dependencies]

    def dependencies_are_met(
        self,
        parent_job: Optional['Job'] = None,
        pipeline: Optional['Pipeline'] = None,
        exclude_job_id: Optional[str] = None,
    ) -> bool:
        """Returns a boolean indicating if all of this job's dependencies are `FINISHED`

        If a pipeline is passed, all dependencies are WATCHed.

        `parent_job` allows us to directly pass parent_job for the status check.
        This is useful when enqueueing the dependents of a _successful_ job -- that status of
        `FINISHED` may not be yet set in redis, but said job is indeed _done_ and this
        method is _called_ in the _stack_ of its dependents are being enqueued.

        Args:
            parent_job (Optional[Job], optional): The parent Job. Defaults to None.
            pipeline (Optional[Pipeline], optional): The Redis' pipeline. Defaults to None.
            exclude_job_id (Optional[str], optional): Whether to exclude the job id.. Defaults to None.

        Returns:
            are_met (bool): Whether the dependencies were met.
        """
        connection = pipeline if pipeline is not None else self.connection

        if pipeline is not None:
            connection.watch(*[self.key_for(dependency_id) for dependency_id in self._dependency_ids])

        dependencies_ids = {_id.decode() for _id in connection.smembers(self.dependencies_key)}

        if exclude_job_id:
            dependencies_ids.discard(exclude_job_id)
            if parent_job and parent_job.id == exclude_job_id:
                parent_job = None

        if parent_job:
            # If parent job is canceled, treat dependency as failed
            # If parent job is not finished, we should only continue
            # if this job allows parent job to fail
            dependencies_ids.discard(parent_job.id)
            if parent_job.get_status() == JobStatus.CANCELED:
                return False
            elif parent_job._status == JobStatus.FAILED and not self.allow_dependency_failures:
                return False

            # If the only dependency is parent job, dependency has been met
            if not dependencies_ids:
                return True

        with connection.pipeline() as pipeline:
            for key in dependencies_ids:
                pipeline.hget(self.key_for(key), 'status')

            dependencies_statuses = pipeline.execute()

        allowed_statuses = [JobStatus.FINISHED]
        if self.allow_dependency_failures:
            allowed_statuses.append(JobStatus.FAILED)

        return all(status.decode() in allowed_statuses for status in dependencies_statuses if status)


_job_stack = LocalStack()


class Retry:
    def __init__(self, max: int, interval: Union[int, List[int]] = 0):
        """The main object to defined Retry logics for jobs.

        Args:
            max (int): The max number of times a job should be retried
            interval (Union[int, List[int]], optional): The interval between retries.
                Can be a positive number (int) or a list of ints. Defaults to 0 (meaning no interval between retries).

        Raises:
            ValueError: If the `max` argument is lower than 1
            ValueError: If the interval param is negative or the list contains negative numbers
        """
        super().__init__()
        if max < 1:
            raise ValueError('max: please enter a value greater than 0')

        if isinstance(interval, int):
            if interval < 0:
                raise ValueError('interval: negative numbers are not allowed')
            intervals = [interval]
        elif isinstance(interval, Iterable):
            for i in interval:
                if i < 0:
                    raise ValueError('interval: negative numbers are not allowed')
            intervals = interval

        self.max = max
        self.intervals = intervals


class Callback:
    def __init__(self, func: Union[str, Callable[..., Any]], timeout: Optional[Any] = None):
        if not isinstance(func, str) and not inspect.isfunction(func) and not inspect.isbuiltin(func):
            raise ValueError('Callback `func` must be a string or function')

        self.func = func
        self.timeout = parse_timeout(timeout) if timeout else CALLBACK_TIMEOUT

    @property
    def name(self) -> str:
        if isinstance(self.func, str):
            return self.func
        return '{0}.{1}'.format(self.func.__module__, self.func.__qualname__)
