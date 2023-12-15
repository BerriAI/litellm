from __future__ import annotations

import re
import os
import sys
import shutil
import logging
import subprocess
from pathlib import Path
from abc import ABC, abstractmethod
from typing import IO, Union, Any, Mapping, cast
from typing_extensions import Literal

from .. import config
from .._proxy import LazyProxy
from ..binaries import platform
from ..errors import PrismaError
from .._compat import nodejs, get_args


log: logging.Logger = logging.getLogger(__name__)
File = Union[int, IO[Any]]
Target = Literal['node', 'npm']

# taken from https://github.com/prisma/prisma/blob/main/package.json
MIN_NODE_VERSION = (14, 17)

# mapped the node version above from https://nodejs.org/en/download/releases/
MIN_NPM_VERSION = (6, 14)

# we only care about the first two entries in the version number
VERSION_RE = re.compile(r'v?(\d+)(?:\.?(\d+))')


# TODO: remove the possibility to get mismatched paths for `node` and `npm`


class UnknownTargetError(PrismaError):
    def __init__(self, *, target: str) -> None:
        super().__init__(
            f'Unknown target: {target}; Valid choices are: {", ".join(get_args(cast(type, Target)))}'
        )


# TODO: add tests for this error
class MissingNodejsBinError(PrismaError):
    def __init__(self) -> None:
        super().__init__(
            'Attempted to access a function that requires the `nodejs-bin` package to be installed but it is not.'
        )


class Strategy(ABC):
    resolver: Literal['nodejs-bin', 'global', 'nodeenv']

    # TODO: support more options
    def run(
        self,
        *args: str,
        check: bool = False,
        cwd: Path | None = None,
        stdout: File | None = None,
        stderr: File | None = None,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        """Call the underlying Node.js binary.

        The interface for this function is very similar to `subprocess.run()`.
        """
        return self.__run__(
            *args,
            check=check,
            cwd=cwd,
            stdout=stdout,
            stderr=stderr,
            env=_update_path_env(
                env=env,
                target_bin=self.target_bin,
            ),
        )

    @abstractmethod
    def __run__(
        self,
        *args: str,
        check: bool = False,
        cwd: Path | None = None,
        stdout: File | None = None,
        stderr: File | None = None,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        """Call the underlying Node.js binary.

        This should not be directly accessed, the `run()` function should be used instead.
        """

    @property
    @abstractmethod
    def target_bin(self) -> Path:
        """Property containing the location of the `bin` directory for the resolved node installation.

        This is used to dynamically alter the `PATH` environment variable to give the appearance that Node
        is installed globally on the machine as this is a requirement of Prisma's installation step, see this
        comment for more context: https://github.com/RobertCraigie/prisma-client-py/pull/454#issuecomment-1280059779
        """
        ...


class NodeBinaryStrategy(Strategy):
    target: Target
    resolver: Literal['global', 'nodeenv']

    def __init__(
        self,
        *,
        path: Path,
        target: Target,
        resolver: Literal['global', 'nodeenv'],
    ) -> None:
        self.path = path
        self.target = target
        self.resolver = resolver

    @property
    def target_bin(self) -> Path:
        return self.path.parent

    def __run__(
        self,
        *args: str,
        check: bool = False,
        cwd: Path | None = None,
        stdout: File | None = None,
        stderr: File | None = None,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        path = str(self.path.absolute())
        log.debug('Executing binary at %s with args: %s', path, args)
        return subprocess.run(
            [path, *args],
            check=check,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )

    @classmethod
    def resolve(cls, target: Target) -> NodeBinaryStrategy:
        path = None
        if config.use_global_node:
            path = _get_global_binary(target)

        if path is not None:
            return NodeBinaryStrategy(
                path=path,
                target=target,
                resolver='global',
            )

        return NodeBinaryStrategy.from_nodeenv(target)

    @classmethod
    def from_nodeenv(cls, target: Target) -> NodeBinaryStrategy:
        cache_dir = config.nodeenv_cache_dir.absolute()
        if cache_dir.exists():
            log.debug(
                'Skipping nodeenv installation as it already exists at %s',
                cache_dir,
            )
        else:
            log.debug('Installing nodeenv to %s', cache_dir)
            try:
                subprocess.run(
                    [
                        sys.executable,
                        '-m',
                        'nodeenv',
                        str(cache_dir),
                        *config.nodeenv_extra_args,
                    ],
                    check=True,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                )
            except Exception as exc:
                print(
                    'nodeenv installation failed; You may want to try installing `nodejs-bin` as it is more reliable.',
                    file=sys.stderr,
                )
                raise exc

        if not cache_dir.exists():
            raise RuntimeError(
                'Could not install nodeenv to the expected directory; See the output above for more details.'
            )

        # TODO: what hapens on cygwin?
        if platform.name() == 'windows':
            bin_dir = cache_dir / 'Scripts'
            if target == 'node':
                path = bin_dir / 'node.exe'
            else:
                path = bin_dir / f'{target}.cmd'
        else:
            path = cache_dir / 'bin' / target

        if target == 'npm':
            return cls(path=path, resolver='nodeenv', target=target)
        elif target == 'node':
            return cls(path=path, resolver='nodeenv', target=target)
        else:
            raise UnknownTargetError(target=target)


class NodeJSPythonStrategy(Strategy):
    target: Target
    resolver: Literal['nodejs-bin']

    def __init__(self, *, target: Target) -> None:
        self.target = target
        self.resolver = 'nodejs-bin'

    def __run__(
        self,
        *args: str,
        check: bool = False,
        cwd: Path | None = None,
        stdout: File | None = None,
        stderr: File | None = None,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        if nodejs is None:
            raise MissingNodejsBinError()

        func = None
        if self.target == 'node':
            func = nodejs.node.run
        elif self.target == 'npm':
            func = nodejs.npm.run
        else:
            raise UnknownTargetError(target=self.target)

        return cast(
            'subprocess.CompletedProcess[bytes]',
            func(
                args,
                check=check,
                cwd=cwd,
                env=env,
                stdout=stdout,
                stderr=stderr,
            ),
        )

    @property
    def node_path(self) -> Path:
        """Returns the path to the `node` binary"""
        if nodejs is None:
            raise MissingNodejsBinError()

        return Path(nodejs.node.path)

    @property
    def target_bin(self) -> Path:
        return Path(self.node_path).parent


Node = Union[NodeJSPythonStrategy, NodeBinaryStrategy]


def resolve(target: Target) -> Node:
    if target not in {'node', 'npm'}:
        raise UnknownTargetError(target=target)

    if config.use_nodejs_bin:
        log.debug('Checking if nodejs-bin is installed')
        if nodejs is not None:
            log.debug('Using nodejs-bin with version: %s', nodejs.node_version)
            return NodeJSPythonStrategy(target=target)

    return NodeBinaryStrategy.resolve(target)


def _update_path_env(
    *,
    env: Mapping[str, str] | None,
    target_bin: Path,
    sep: str = os.pathsep,
) -> dict[str, str]:
    """Returns a modified version of `os.environ` with the `PATH` environment variable updated
    to include the location of the downloaded Node binaries.
    """
    if env is None:
        env = dict(os.environ)

    log.debug('Attempting to preprend %s to the PATH', target_bin)
    assert target_bin.exists(), 'Target `bin` directory does not exist'

    path = env.get('PATH', '') or os.environ.get('PATH', '')
    if path:
        # handle the case where the PATH already starts with the separator (this probably shouldn't happen)
        if path.startswith(sep):
            path = f'{target_bin.absolute()}{path}'
        else:
            path = f'{target_bin.absolute()}{sep}{path}'
    else:
        # handle the case where there is no PATH set (unlikely / impossible to actually happen?)
        path = str(target_bin.absolute())

    log.debug('Using PATH environment variable: %s', path)
    return {**env, 'PATH': path}


def _get_global_binary(target: Target) -> Path | None:
    """Returns the path to a globally installed binary.

    This also ensures that the binary is of the right version.
    """
    log.debug('Checking for global target binary: %s', target)

    which = shutil.which(target)
    if which is None:
        log.debug('Global target binary: %s not found', target)
        return None

    log.debug('Found global binary at: %s', which)

    path = Path(which)
    if not path.exists():
        log.debug('Global binary does not exist at: %s', which)
        return None

    if not _should_use_binary(target=target, path=path):
        return None

    log.debug('Using global %s binary at %s', target, path)
    return path


def _should_use_binary(target: Target, path: Path) -> bool:
    """Call the binary at `path` with a `--version` flag to check if it matches our minimum version requirements.

    This only applies to the global node installation as:

    - the minimum version of `nodejs-bin` is higher than our requirement
    - `nodeenv` defaults to the latest stable version of node
    """
    if target == 'node':
        min_version = MIN_NODE_VERSION
    elif target == 'npm':
        min_version = MIN_NPM_VERSION
    else:
        raise UnknownTargetError(target=target)

    version = _get_binary_version(target, path)
    if version is None:
        log.debug(
            'Could not resolve %s version, ignoring global %s installation',
            target,
            target,
        )
        return False

    if version < min_version:
        log.debug(
            'Global %s version (%s) is lower than the minimum required version (%s), ignoring',
            target,
            version,
            min_version,
        )
        return False

    return True


def _get_binary_version(target: Target, path: Path) -> tuple[int, ...] | None:
    proc = subprocess.run(
        [str(path), '--version'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log.debug('%s version check exited with code %s', target, proc.returncode)

    output = proc.stdout.decode('utf-8').rstrip('\n')
    log.debug('%s version check output: %s', target, output)

    match = VERSION_RE.search(output)
    if not match:
        return None

    version = tuple(int(value) for value in match.groups())
    log.debug('%s version check returning %s', target, version)
    return version


class LazyBinaryProxy(LazyProxy[Node]):
    target: Target

    def __init__(self, target: Target) -> None:
        super().__init__()
        self.target = target

    def __load__(self) -> Node:
        return resolve(self.target)


npm = LazyBinaryProxy('npm').__as_proxied__()
node = LazyBinaryProxy('node').__as_proxied__()
