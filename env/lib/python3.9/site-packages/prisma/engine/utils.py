from __future__ import annotations

import os
import sys
import time
import socket
import logging
import subprocess
from pathlib import Path
from typing import NoReturn, Dict, Type, Any

from . import errors
from .. import errors as prisma_errors

from .. import config
from ..http_abstract import AbstractResponse
from ..utils import DEBUG_GENERATOR, time_since
from ..binaries import platform


log: logging.Logger = logging.getLogger(__name__)

ERROR_MAPPING: Dict[str, Type[Exception]] = {
    'P2002': prisma_errors.UniqueViolationError,
    'P2003': prisma_errors.ForeignKeyViolationError,
    'P2009': prisma_errors.FieldNotFoundError,
    'P2010': prisma_errors.RawQueryError,
    'P2012': prisma_errors.MissingRequiredValueError,
    'P2019': prisma_errors.InputError,
    'P2021': prisma_errors.TableNotFoundError,
    'P2025': prisma_errors.RecordNotFoundError,
}

META_ERROR_MAPPING: dict[str, type[Exception]] = {
    'UnknownArgument': prisma_errors.FieldNotFoundError,
    'UnknownInputField': prisma_errors.FieldNotFoundError,
    'UnknownSelectionField': prisma_errors.FieldNotFoundError,
}


def query_engine_name() -> str:
    return f'prisma-query-engine-{platform.check_for_extension(platform.binary_platform())}'


def _resolve_from_binary_paths(binary_paths: dict[str, str]) -> Path | None:
    if config.binary_platform is not None:
        return Path(binary_paths[config.binary_platform])

    paths = [Path(p) for p in binary_paths.values()]

    # fast path for when there are no `binaryTargets` defined
    if len(paths) == 1:
        return paths[0]

    for path in paths:
        # we only want to resolve to binary's that we can run on the current machine.
        # because of the `binaryTargets` option some of the binaries we are given may
        # not be targeting the same architecture as the current machine
        if path.exists() and _can_execute_binary(path):
            return path

    # none of the given paths existed or they target a different architecture
    return None


def _can_execute_binary(path: Path) -> bool:
    proc = subprocess.run([str(path), '--version'], check=False)
    log.debug(
        'Executable check for %s exited with code: %s', path, proc.returncode
    )
    return proc.returncode == 0


def ensure(binary_paths: dict[str, str]) -> Path:
    start_time = time.monotonic()
    file = None
    force_version = not DEBUG_GENERATOR
    name = query_engine_name()
    local_path = Path.cwd().joinpath(name)
    global_path = config.binary_cache_dir.joinpath(name)
    file_from_paths = _resolve_from_binary_paths(binary_paths)

    log.debug('Expecting local query engine %s', local_path)
    log.debug('Expecting global query engine %s', global_path)

    binary = os.environ.get('PRISMA_QUERY_ENGINE_BINARY')
    if binary:
        log.debug('PRISMA_QUERY_ENGINE_BINARY is defined, using %s', binary)

        if not Path(binary).exists():
            raise errors.BinaryNotFoundError(
                'PRISMA_QUERY_ENGINE_BINARY was provided, '
                f'but no query engine was found at {binary}'
            )

        file = Path(binary)
        force_version = False
    elif local_path.exists():
        file = local_path
        log.debug('Query engine found in the working directory')
    elif file_from_paths is not None and file_from_paths.exists():
        file = file_from_paths
        log.debug(
            'Query engine found from the Prisma CLI generated path: %s',
            file_from_paths,
        )
    elif global_path.exists():
        file = global_path
        log.debug('Query engine found in the global path')

    if not file:
        if file_from_paths is not None:
            expected = f'{local_path}, {global_path} or {file_from_paths} to exist but none'
        else:
            expected = f'{local_path} or {global_path} to exist but neither'

        raise errors.BinaryNotFoundError(
            f'Expected {expected} were found or could not be executed.\n'
            + 'Try running prisma py fetch'
        )

    log.debug('Using Query Engine binary at %s', file)

    start_version = time.monotonic()
    process = subprocess.run(
        [str(file.absolute()), '--version'], stdout=subprocess.PIPE, check=True
    )
    log.debug('Version check took %s', time_since(start_version))

    version = (
        str(process.stdout, sys.getdefaultencoding())
        .replace('query-engine', '')
        .strip()
    )
    log.debug('Using query engine version %s', version)

    if force_version and version != config.expected_engine_version:
        raise errors.MismatchedVersionsError(
            expected=config.expected_engine_version, got=version
        )

    log.debug('Using query engine at %s', file)
    log.debug('Ensuring query engine took: %s', time_since(start_time))
    return file


def get_open_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return int(port)


def handle_response_errors(resp: AbstractResponse[Any], data: Any) -> NoReturn:
    for error in data:
        try:
            base_error_message = error.get('error', '')
            user_facing = error.get('user_facing_error', {})
            code = user_facing.get('error_code')
            if code is None:
                continue

            # TODO: the order of these if statements is important because
            # the P2009 code can be returned for both: missing a required value
            # and an unknown field error. As we want to explicitly handle
            # the missing a required value error then we need to check for that first.
            # As we can only check for this error by searching the message then this
            # comes with performance concerns.
            message = user_facing.get('message', '')

            if code == 'P2028':
                if base_error_message.startswith('Transaction already closed'):
                    raise prisma_errors.TransactionExpiredError(
                        base_error_message
                    )
                raise prisma_errors.TransactionError(message)

            if 'A value is required but not set' in message:
                raise prisma_errors.MissingRequiredValueError(error)

            exc: type[Exception] | None = None

            kind = user_facing.get('meta', {}).get('kind')
            if kind is not None:
                exc = META_ERROR_MAPPING.get(kind)

            if exc is None:
                exc = ERROR_MAPPING.get(code)

            if exc is not None:
                raise exc(error)
        except (KeyError, TypeError) as err:
            log.debug(
                'Ignoring error while constructing specialized error %s', err
            )
            continue

    try:
        raise prisma_errors.DataError(data[0])
    except (IndexError, TypeError):
        pass

    raise errors.EngineRequestError(
        resp, f'Could not process erroneous response: {data}'
    )
