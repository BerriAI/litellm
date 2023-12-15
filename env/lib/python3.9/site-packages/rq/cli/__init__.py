# ruff: noqa: F401 I001
from .cli import main

# TODO: the following imports can be removed when we drop the `rqinfo` and
# `rqworkers` commands in favor of just shipping the `rq` command.
from .cli import info, worker
