import errno
import importlib.util
import os
import pkgutil
from importlib import import_module
from types import ModuleType
from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import Optional
from typing import Type

from ..exceptions import InvalidFile


def import_types_from_package(
    root: ModuleType,
    filter: Callable[[Any], bool],
) -> Iterable[Type]:
    output: List[Type] = []
    modules = get_modules_from_package(root)

    for module_path in modules:
        module = import_module(module_path)
        output.extend(import_types_from_module(module, filter))

    return output


def import_types_from_module(
    module: ModuleType,
    filter: Callable[[Any], bool],
) -> Iterable[Type]:
    output = []
    for name in dir(module):
        if name.startswith("_"):
            continue

        attribute = getattr(module, name)
        if filter(attribute):
            continue

        output.append(attribute)

    return output


def import_modules_from_package(
    root: ModuleType,
    filter: Callable[[str], bool],
) -> Iterable[ModuleType]:
    output = []
    modules = get_modules_from_package(root)

    # NOTE: It should be auto-sorted, but let's just do it for sanity sake.
    # This sorting is required for performing upgrades in order.
    for module_path in sorted(modules):
        if filter(module_path):
            continue

        output.append(import_module(module_path))

    return output


def import_file_as_module(filename: str, name: Optional[str] = None) -> ModuleType:
    """
    NOTE(2020-11-09|domanchi): We're essentially executing arbitrary code here, so some thoughts
    should be recorded as to the security of this feature. This should not add any additional
    security risk, given the following assumptions hold true:

      1. detect-secrets is not used in an environment that has privileged access (more
         than the current user), OR
      2. detect-secrets (when running in a privileged context) does not accept arbitrary
         user input that feeds into this function (e.g. custom plugins).

    The first assumption should be rather self-explanatory: if you are running detect-secrets
    in a context that has the same permissions as you, you can import any code you want, since
    this acts more of a utility function than a security flaw. If you're going to do it *anyway*,
    let's just make your life easier.

    The second assumption should also be pretty straight-forward: don't trust user input,
    especially if it's going to be executed as that privileged user, unless you want a privilege
    escalation vulnerability. detect-secrets is not going to do any sanitization of user input
    for you.
    """
    if not os.path.exists(filename):
        # Source: https://stackoverflow.com/a/36077407
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

    if not name:
        # NOTE: After several trial and error attempts, I could not discern the importance
        # of this field, in this context. Hence, I don't think it matters that much.
        name = os.path.splitext(os.path.basename(filename))[0]

    # Source: https://stackoverflow.com/a/67692/13340678
    spec = importlib.util.spec_from_file_location(name, filename)
    if not spec:
        raise InvalidFile

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    module.__path__ = os.path.abspath(filename)  # type: ignore

    return module


def get_modules_from_package(root: ModuleType) -> Iterable[str]:
    return [
        module
        for _, module, is_package in pkgutil.walk_packages(
            root.__path__,
            prefix=f"{root.__name__}.",
        )
        if not is_package
    ]
