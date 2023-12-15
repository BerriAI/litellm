from typing import TYPE_CHECKING, Any, Callable, List, TypeVar, Union

if TYPE_CHECKING:
    from .job import Dependency, Job


FunctionReferenceType = TypeVar('FunctionReferenceType', str, Callable[..., Any])
"""Custom type definition for what a `func` is in the context of a job.
A `func` can be a string with the function import path (eg.: `myfile.mymodule.myfunc`)
or a direct callable (function/method).
"""


JobDependencyType = TypeVar('JobDependencyType', 'Dependency', 'Job', str, List[Union['Dependency', 'Job']])
"""Custom type definition for a job dependencies.
A simple helper definition for the `depends_on` parameter when creating a job.
"""
