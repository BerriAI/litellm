"""
Instance of the LiteLLM callback.
"""

from .callback import Callback
from .writer_factory import build_writer


_writer = build_writer()

callback = Callback(_writer)
