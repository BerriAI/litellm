import logging
import os
import signal
import time
import traceback
from datetime import datetime
from enum import Enum
from multiprocessing import Process
from typing import List, Set

from redis import ConnectionPool, Redis

from .connections import parse_connection
from .defaults import DEFAULT_LOGGING_DATE_FORMAT, DEFAULT_LOGGING_FORMAT, DEFAULT_SCHEDULER_FALLBACK_PERIOD
from .job import Job
from .logutils import setup_loghandlers
from .queue import Queue
from .registry import ScheduledJobRegistry
from .serializers import resolve_serializer
from .utils import current_timestamp, parse_names

SCHEDULER_KEY_TEMPLATE = 'rq:scheduler:%s'
SCHEDULER_LOCKING_KEY_TEMPLATE = 'rq:scheduler-lock:%s'


class SchedulerStatus(str, Enum):
    STARTED = 'started'
    WORKING = 'working'
    STOPPED = 'stopped'


class RQScheduler:
    # STARTED: scheduler has been started but sleeping
    # WORKING: scheduler is in the midst of scheduling jobs
    # STOPPED: scheduler is in stopped condition

    Status = SchedulerStatus

    def __init__(
        self,
        queues,
        connection: Redis,
        interval=1,
        logging_level=logging.INFO,
        date_format=DEFAULT_LOGGING_DATE_FORMAT,
        log_format=DEFAULT_LOGGING_FORMAT,
        serializer=None,
    ):
        self._queue_names = set(parse_names(queues))
        self._acquired_locks: Set[str] = set()
        self._scheduled_job_registries: List[ScheduledJobRegistry] = []
        self.lock_acquisition_time = None
        self._connection_class, self._pool_class, self._pool_kwargs = parse_connection(connection)
        self.serializer = resolve_serializer(serializer)

        self._connection = None
        self.interval = interval
        self._stop_requested = False
        self._status = self.Status.STOPPED
        self._process = None
        self.log = logging.getLogger(__name__)
        setup_loghandlers(
            level=logging_level,
            name=__name__,
            log_format=log_format,
            date_format=date_format,
        )

    @property
    def connection(self):
        if self._connection:
            return self._connection
        self._connection = self._connection_class(
            connection_pool=ConnectionPool(connection_class=self._pool_class, **self._pool_kwargs)
        )
        return self._connection

    @property
    def acquired_locks(self):
        return self._acquired_locks

    @property
    def status(self):
        return self._status

    @property
    def should_reacquire_locks(self):
        """Returns True if lock_acquisition_time is longer than 10 minutes ago"""
        if self._queue_names == self.acquired_locks:
            return False
        if not self.lock_acquisition_time:
            return True
        return (datetime.now() - self.lock_acquisition_time).total_seconds() > DEFAULT_SCHEDULER_FALLBACK_PERIOD

    def acquire_locks(self, auto_start=False):
        """Returns names of queue it successfully acquires lock on"""
        successful_locks = set()
        pid = os.getpid()
        self.log.debug('Trying to acquire locks for %s', ', '.join(self._queue_names))
        for name in self._queue_names:
            if self.connection.set(self.get_locking_key(name), pid, nx=True, ex=self.interval + 60):
                successful_locks.add(name)

        # Always reset _scheduled_job_registries when acquiring locks
        self._scheduled_job_registries = []
        self._acquired_locks = self._acquired_locks.union(successful_locks)
        self.lock_acquisition_time = datetime.now()

        # If auto_start is requested and scheduler is not started,
        # run self.start()
        if self._acquired_locks and auto_start:
            if not self._process or not self._process.is_alive():
                self.start()

        return successful_locks

    def prepare_registries(self, queue_names: str = None):
        """Prepare scheduled job registries for use"""
        self._scheduled_job_registries = []
        if not queue_names:
            queue_names = self._acquired_locks
        for name in queue_names:
            self._scheduled_job_registries.append(
                ScheduledJobRegistry(name, connection=self.connection, serializer=self.serializer)
            )

    @classmethod
    def get_locking_key(cls, name: str):
        """Returns scheduler key for a given queue name"""
        return SCHEDULER_LOCKING_KEY_TEMPLATE % name

    def enqueue_scheduled_jobs(self):
        """Enqueue jobs whose timestamp is in the past"""
        self._status = self.Status.WORKING

        if not self._scheduled_job_registries and self._acquired_locks:
            self.prepare_registries()

        for registry in self._scheduled_job_registries:
            timestamp = current_timestamp()

            # TODO: try to use Lua script to make get_jobs_to_schedule()
            # and remove_jobs() atomic
            job_ids = registry.get_jobs_to_schedule(timestamp)

            if not job_ids:
                continue

            queue = Queue(registry.name, connection=self.connection, serializer=self.serializer)

            with self.connection.pipeline() as pipeline:
                jobs = Job.fetch_many(job_ids, connection=self.connection, serializer=self.serializer)
                for job in jobs:
                    if job is not None:
                        queue._enqueue_job(job, pipeline=pipeline, at_front=bool(job.enqueue_at_front))
                        registry.remove(job, pipeline=pipeline)
                pipeline.execute()
        self._status = self.Status.STARTED

    def _install_signal_handlers(self):
        """Installs signal handlers for handling SIGINT and SIGTERM
        gracefully.
        """
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)

    def request_stop(self, signum=None, frame=None):
        """Toggle self._stop_requested that's checked on every loop"""
        self._stop_requested = True

    def heartbeat(self):
        """Updates the TTL on scheduler keys and the locks"""
        self.log.debug('Scheduler sending heartbeat to %s', ', '.join(self.acquired_locks))
        if len(self._acquired_locks) > 1:
            with self.connection.pipeline() as pipeline:
                for name in self._acquired_locks:
                    key = self.get_locking_key(name)
                    pipeline.expire(key, self.interval + 60)
                pipeline.execute()
        elif self._acquired_locks:
            key = self.get_locking_key(next(iter(self._acquired_locks)))
            self.connection.expire(key, self.interval + 60)

    def stop(self):
        self.log.info('Scheduler stopping, releasing locks for %s...', ', '.join(self._acquired_locks))
        self.release_locks()
        self._status = self.Status.STOPPED

    def release_locks(self):
        """Release acquired locks"""
        keys = [self.get_locking_key(name) for name in self._acquired_locks]
        self.connection.delete(*keys)
        self._acquired_locks = set()

    def start(self):
        self._status = self.Status.STARTED
        # Redis instance can't be pickled across processes so we need to
        # clean this up before forking
        self._connection = None
        self._process = Process(target=run, args=(self,), name='Scheduler')
        self._process.start()
        return self._process

    def work(self):
        self._install_signal_handlers()

        while True:
            if self._stop_requested:
                self.stop()
                break

            if self.should_reacquire_locks:
                self.acquire_locks()

            self.enqueue_scheduled_jobs()
            self.heartbeat()
            time.sleep(self.interval)


def run(scheduler):
    scheduler.log.info('Scheduler for %s started with PID %s', ', '.join(scheduler._queue_names), os.getpid())
    try:
        scheduler.work()
    except:  # noqa
        scheduler.log.error('Scheduler [PID %s] raised an exception.\n%s', os.getpid(), traceback.format_exc())
        raise
    scheduler.log.info('Scheduler with PID %d has stopped', os.getpid())
