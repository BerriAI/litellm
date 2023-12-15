import importlib
import os
import sys
import time
from ast import literal_eval
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import partial, update_wrapper
from json import JSONDecodeError, loads
from shutil import get_terminal_size

import click
from redis import Redis
from redis.sentinel import Sentinel

from rq.defaults import (
    DEFAULT_CONNECTION_CLASS,
    DEFAULT_DEATH_PENALTY_CLASS,
    DEFAULT_JOB_CLASS,
    DEFAULT_QUEUE_CLASS,
    DEFAULT_SERIALIZER_CLASS,
    DEFAULT_WORKER_CLASS,
)
from rq.logutils import setup_loghandlers
from rq.utils import import_attribute, parse_timeout
from rq.worker import WorkerStatus

red = partial(click.style, fg='red')
green = partial(click.style, fg='green')
yellow = partial(click.style, fg='yellow')


def read_config_file(module):
    """Reads all UPPERCASE variables defined in the given module file."""
    settings = importlib.import_module(module)
    return dict([(k, v) for k, v in settings.__dict__.items() if k.upper() == k])


def get_redis_from_config(settings, connection_class=Redis):
    """Returns a StrictRedis instance from a dictionary of settings.
    To use redis sentinel, you must specify a dictionary in the configuration file.
    Example of a dictionary with keys without values:
    SENTINEL = {'INSTANCES':, 'SOCKET_TIMEOUT':, 'USERNAME':, 'PASSWORD':, 'DB':, 'MASTER_NAME':, 'SENTINEL_KWARGS':}
    """
    if settings.get('REDIS_URL') is not None:
        return connection_class.from_url(settings['REDIS_URL'])

    elif settings.get('SENTINEL') is not None:
        instances = settings['SENTINEL'].get('INSTANCES', [('localhost', 26379)])
        master_name = settings['SENTINEL'].get('MASTER_NAME', 'mymaster')

        connection_kwargs = {
            'db': settings['SENTINEL'].get('DB', 0),
            'username': settings['SENTINEL'].get('USERNAME', None),
            'password': settings['SENTINEL'].get('PASSWORD', None),
            'socket_timeout': settings['SENTINEL'].get('SOCKET_TIMEOUT', None),
            'ssl': settings['SENTINEL'].get('SSL', False),
        }
        connection_kwargs.update(settings['SENTINEL'].get('CONNECTION_KWARGS', {}))
        sentinel_kwargs = settings['SENTINEL'].get('SENTINEL_KWARGS', {})

        sn = Sentinel(instances, sentinel_kwargs=sentinel_kwargs, **connection_kwargs)
        return sn.master_for(master_name)

    ssl = settings.get('REDIS_SSL', False)
    if isinstance(ssl, str):
        if ssl.lower() in ['y', 'yes', 't', 'true']:
            ssl = True
        elif ssl.lower() in ['n', 'no', 'f', 'false', '']:
            ssl = False
        else:
            raise ValueError('REDIS_SSL is a boolean and must be "True" or "False".')

    kwargs = {
        'host': settings.get('REDIS_HOST', 'localhost'),
        'port': settings.get('REDIS_PORT', 6379),
        'db': settings.get('REDIS_DB', 0),
        'password': settings.get('REDIS_PASSWORD', None),
        'ssl': ssl,
        'ssl_ca_certs': settings.get('REDIS_SSL_CA_CERTS', None),
        'ssl_cert_reqs': settings.get('REDIS_SSL_CERT_REQS', 'required'),
    }

    return connection_class(**kwargs)


def pad(s, pad_to_length):
    """Pads the given string to the given length."""
    return ('%-' + '%ds' % pad_to_length) % (s,)


def get_scale(x):
    """Finds the lowest scale where x <= scale."""
    scales = [20, 50, 100, 200, 400, 600, 800, 1000]
    for scale in scales:
        if x <= scale:
            return scale
    return x


def state_symbol(state):
    symbols = {
        WorkerStatus.BUSY: red('busy'),
        WorkerStatus.IDLE: green('idle'),
        WorkerStatus.SUSPENDED: yellow('suspended'),
    }
    try:
        return symbols[state]
    except KeyError:
        return state


def show_queues(queues, raw, by_queue, queue_class, worker_class):
    num_jobs = 0
    termwidth = get_terminal_size().columns
    chartwidth = min(20, termwidth - 20)

    max_count = 0
    counts = dict()
    for q in queues:
        count = q.count
        counts[q] = count
        max_count = max(max_count, count)
    scale = get_scale(max_count)
    ratio = chartwidth * 1.0 / scale

    for q in queues:
        count = counts[q]
        if not raw:
            chart = green('|' + '█' * int(ratio * count))
            line = '%-12s %s %d, %d executing, %d finished, %d failed' % (
                q.name,
                chart,
                count,
                q.started_job_registry.count,
                q.finished_job_registry.count,
                q.failed_job_registry.count,
            )
        else:
            line = 'queue %s %d, %d executing, %d finished, %d failed' % (
                q.name,
                count,
                q.started_job_registry.count,
                q.finished_job_registry.count,
                q.failed_job_registry.count,
            )
        click.echo(line)

        num_jobs += count

    # print summary when not in raw mode
    if not raw:
        click.echo('%d queues, %d jobs total' % (len(queues), num_jobs))


def show_workers(queues, raw, by_queue, queue_class, worker_class):
    workers = set()
    if queues:
        for queue in queues:
            for worker in worker_class.all(queue=queue):
                workers.add(worker)
    else:
        for worker in worker_class.all():
            workers.add(worker)

    if not by_queue:
        for worker in workers:
            queue_names = ', '.join(worker.queue_names())
            name = '%s (%s %s %s)' % (worker.name, worker.hostname, worker.ip_address, worker.pid)
            if not raw:
                line = '%s: %s %s. jobs: %d finished, %d failed' % (
                    name,
                    state_symbol(worker.get_state()),
                    queue_names,
                    worker.successful_job_count,
                    worker.failed_job_count,
                )
                click.echo(line)
            else:
                line = 'worker %s %s %s. jobs: %d finished, %d failed' % (
                    name,
                    worker.get_state(),
                    queue_names,
                    worker.successful_job_count,
                    worker.failed_job_count,
                )
                click.echo(line)

    else:
        # Display workers by queue
        queue_dict = {}
        for queue in queues:
            queue_dict[queue] = worker_class.all(queue=queue)

        if queue_dict:
            max_length = max(len(q.name) for q, in queue_dict.keys())
        else:
            max_length = 0

        for queue in queue_dict:
            if queue_dict[queue]:
                queues_str = ", ".join(
                    sorted(map(lambda w: '%s (%s)' % (w.name, state_symbol(w.get_state())), queue_dict[queue]))
                )
            else:
                queues_str = '–'
            click.echo('%s %s' % (pad(queue.name + ':', max_length + 1), queues_str))

    if not raw:
        click.echo('%d workers, %d queues' % (len(workers), len(queues)))


def show_both(queues, raw, by_queue, queue_class, worker_class):
    show_queues(queues, raw, by_queue, queue_class, worker_class)
    if not raw:
        click.echo('')
    show_workers(queues, raw, by_queue, queue_class, worker_class)
    if not raw:
        click.echo('')
        import datetime

        click.echo('Updated: %s' % datetime.datetime.now())


def refresh(interval, func, *args):
    while True:
        if interval:
            click.clear()
        func(*args)
        if interval:
            time.sleep(interval)
        else:
            break


def setup_loghandlers_from_args(verbose, quiet, date_format, log_format):
    if verbose and quiet:
        raise RuntimeError("Flags --verbose and --quiet are mutually exclusive.")

    if verbose:
        level = 'DEBUG'
    elif quiet:
        level = 'WARNING'
    else:
        level = 'INFO'
    setup_loghandlers(level, date_format=date_format, log_format=log_format)


def parse_function_arg(argument, arg_pos):
    class ParsingMode(Enum):
        PLAIN_TEXT = 0
        JSON = 1
        LITERAL_EVAL = 2

    keyword = None
    if argument.startswith(':'):  # no keyword, json
        mode = ParsingMode.JSON
        value = argument[1:]
    elif argument.startswith('%'):  # no keyword, literal_eval
        mode = ParsingMode.LITERAL_EVAL
        value = argument[1:]
    else:
        index = argument.find('=')
        if index > 0:
            if ':' in argument and argument.index(':') + 1 == index:  # keyword, json
                mode = ParsingMode.JSON
                keyword = argument[: index - 1]
            elif '%' in argument and argument.index('%') + 1 == index:  # keyword, literal_eval
                mode = ParsingMode.LITERAL_EVAL
                keyword = argument[: index - 1]
            else:  # keyword, text
                mode = ParsingMode.PLAIN_TEXT
                keyword = argument[:index]
            value = argument[index + 1 :]
        else:  # no keyword, text
            mode = ParsingMode.PLAIN_TEXT
            value = argument

    if value.startswith('@'):
        try:
            with open(value[1:], 'r') as file:
                value = file.read()
        except FileNotFoundError:
            raise click.FileError(value[1:], 'Not found')

    if mode == ParsingMode.JSON:  # json
        try:
            value = loads(value)
        except JSONDecodeError:
            raise click.BadParameter('Unable to parse %s as JSON.' % (keyword or '%s. non keyword argument' % arg_pos))
    elif mode == ParsingMode.LITERAL_EVAL:  # literal_eval
        try:
            value = literal_eval(value)
        except Exception:
            raise click.BadParameter(
                'Unable to eval %s as Python object. See '
                'https://docs.python.org/3/library/ast.html#ast.literal_eval'
                % (keyword or '%s. non keyword argument' % arg_pos)
            )

    return keyword, value


def parse_function_args(arguments):
    args = []
    kwargs = {}

    for argument in arguments:
        keyword, value = parse_function_arg(argument, len(args) + 1)
        if keyword is not None:
            if keyword in kwargs:
                raise click.BadParameter('You can\'t specify multiple values for the same keyword.')
            kwargs[keyword] = value
        else:
            args.append(value)
    return args, kwargs


def parse_schedule(schedule_in, schedule_at):
    if schedule_in is not None:
        if schedule_at is not None:
            raise click.BadArgumentUsage('You can\'t specify both --schedule-in and --schedule-at')
        return datetime.now(timezone.utc) + timedelta(seconds=parse_timeout(schedule_in))
    elif schedule_at is not None:
        return datetime.strptime(schedule_at, '%Y-%m-%dT%H:%M:%S')


class CliConfig:
    """A helper class to be used with click commands, to handle shared options"""

    def __init__(
        self,
        url=None,
        config=None,
        worker_class=DEFAULT_WORKER_CLASS,
        job_class=DEFAULT_JOB_CLASS,
        death_penalty_class=DEFAULT_DEATH_PENALTY_CLASS,
        queue_class=DEFAULT_QUEUE_CLASS,
        connection_class=DEFAULT_CONNECTION_CLASS,
        path=None,
        *args,
        **kwargs,
    ):
        self._connection = None
        self.url = url
        self.config = config

        if path:
            for pth in path:
                sys.path.append(pth)

        try:
            self.worker_class = import_attribute(worker_class)
        except (ImportError, AttributeError) as exc:
            raise click.BadParameter(str(exc), param_hint='--worker-class')
        try:
            self.job_class = import_attribute(job_class)
        except (ImportError, AttributeError) as exc:
            raise click.BadParameter(str(exc), param_hint='--job-class')

        try:
            self.death_penalty_class = import_attribute(death_penalty_class)
        except (ImportError, AttributeError) as exc:
            raise click.BadParameter(str(exc), param_hint='--death-penalty-class')

        try:
            self.queue_class = import_attribute(queue_class)
        except (ImportError, AttributeError) as exc:
            raise click.BadParameter(str(exc), param_hint='--queue-class')

        try:
            self.connection_class = import_attribute(connection_class)
        except (ImportError, AttributeError) as exc:
            raise click.BadParameter(str(exc), param_hint='--connection-class')

    @property
    def connection(self):
        if self._connection is None:
            if self.url:
                self._connection = self.connection_class.from_url(self.url)
            elif self.config:
                settings = read_config_file(self.config) if self.config else {}
                self._connection = get_redis_from_config(settings, self.connection_class)
            else:
                self._connection = get_redis_from_config(os.environ, self.connection_class)
        return self._connection


shared_options = [
    click.option('--url', '-u', envvar='RQ_REDIS_URL', help='URL describing Redis connection details.'),
    click.option('--config', '-c', envvar='RQ_CONFIG', help='Module containing RQ settings.'),
    click.option(
        '--worker-class', '-w', envvar='RQ_WORKER_CLASS', default=DEFAULT_WORKER_CLASS, help='RQ Worker class to use'
    ),
    click.option('--job-class', '-j', envvar='RQ_JOB_CLASS', default=DEFAULT_JOB_CLASS, help='RQ Job class to use'),
    click.option('--queue-class', envvar='RQ_QUEUE_CLASS', default=DEFAULT_QUEUE_CLASS, help='RQ Queue class to use'),
    click.option(
        '--connection-class',
        envvar='RQ_CONNECTION_CLASS',
        default=DEFAULT_CONNECTION_CLASS,
        help='Redis client class to use',
    ),
    click.option('--path', '-P', default=['.'], help='Specify the import path.', multiple=True),
    click.option(
        '--serializer',
        '-S',
        default=DEFAULT_SERIALIZER_CLASS,
        help='Path to serializer, defaults to rq.serializers.DefaultSerializer',
    ),
]


def pass_cli_config(func):
    # add all the shared options to the command
    for option in shared_options:
        func = option(func)

    # pass the cli config object into the command
    def wrapper(*args, **kwargs):
        ctx = click.get_current_context()
        cli_config = CliConfig(**kwargs)
        return ctx.invoke(func, cli_config, *args[1:], **kwargs)

    return update_wrapper(wrapper, func)
