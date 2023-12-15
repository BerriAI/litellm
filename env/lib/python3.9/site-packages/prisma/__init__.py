# -*- coding: utf-8 -*-

__title__ = 'prisma'
__author__ = 'RobertCraigie'
__license__ = 'APACHE'
__copyright__ = 'Copyright 2020-2023 RobertCraigie'
__version__ = '0.11.0'

from typing import TYPE_CHECKING

from ._config import config as config
from .utils import setup_logging
from . import errors as errors
from .validator import *
from ._types import PrismaMethod as PrismaMethod
from ._metrics import (
    Metric as Metric,
    Metrics as Metrics,
    MetricHistogram as MetricHistogram,
)


try:
    from .client import *
    from .fields import *
    from . import (
        models as models,
        partials as partials,
        types as types,
        bases as bases,
    )
except ModuleNotFoundError:
    # code has not been generated yet
    # TODO: this could swallow unexpected errors

    # we only define this magic function when the client has not
    # been generated for performance reasons and we hide it from
    # type checkers so that they don't incidentally disable type
    # checking due to the dynamic nature of the `__getattr__`
    # magic dunder method.
    if not TYPE_CHECKING:

        def __getattr__(name: str):
            try:
                return globals()[name]
            except KeyError:
                # TODO: support checking for 'models' here too
                if name in {'Prisma', 'Client'}:
                    # TODO: remove this frame from the stack trace
                    raise RuntimeError(
                        "The Client hasn't been generated yet, "
                        'you must run `prisma generate` before you can use the client.\n'
                        'See https://prisma-client-py.readthedocs.io/en/stable/reference/troubleshooting/#client-has-not-been-generated-yet'
                    ) from None

                # leaves handling of this potential error to Python as per PEP 562
                raise AttributeError()


setup_logging()
