"""
The Anthropic integration instruments the Anthropic Python library to traces for requests made to the models for messages.

All traces submitted from the Anthropic integration are tagged by:

- ``service``, ``env``, ``version``: see the `Unified Service Tagging docs <https://docs.datadoghq.com/getting_started/tagging/unified_service_tagging>`_.
- ``anthropic.request.model``: Anthropic model used in the request.
- ``anthropic.request.api_key``: Anthropic API key used to make the request (obfuscated to match the Anthropic UI representation ``sk-...XXXX`` where ``XXXX`` is the last 4 digits of the key).
- ``anthropic.request.parameters``: Parameters used in anthropic package call.


(beta) Prompt and Completion Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prompt texts and completion content for the ``Messages.create`` endpoint are collected in span tags with a default sampling rate of ``1.0``.
These tags will have truncation applied if the text exceeds the configured character limit.


Enabling
~~~~~~~~

The Anthropic integration is enabled automatically when you use
:ref:`ddtrace-run<ddtracerun>` or :ref:`import ddtrace.auto<ddtraceauto>`.

Note that these commands also enable the ``httpx`` integration which traces HTTP requests from the Anthropic library.

Alternatively, use :func:`patch() <ddtrace.patch>` to manually enable the Anthropic integration::

    from ddtrace import config, patch

    patch(anthropic=True)


Global Configuration
~~~~~~~~~~~~~~~~~~~~

.. py:data:: ddtrace.config.anthropic["service"]

   The service name reported by default for Anthropic requests.

   Alternatively, you can set this option with the ``DD_SERVICE`` or ``DD_ANTHROPIC_SERVICE`` environment
   variables.

   Default: ``DD_SERVICE``


.. py:data:: (beta) ddtrace.config.anthropic["span_char_limit"]

   Configure the maximum number of characters for the following data within span tags:

   - Message inputs and completions

   Text exceeding the maximum number of characters is truncated to the character limit
   and has ``...`` appended to the end.

   Alternatively, you can set this option with the ``DD_ANTHROPIC_SPAN_CHAR_LIMIT`` environment
   variable.

   Default: ``128``


.. py:data:: (beta) ddtrace.config.anthropic["span_prompt_completion_sample_rate"]

   Configure the sample rate for the collection of prompts and completions as span tags.

   Alternatively, you can set this option with the ``DD_ANTHROPIC_SPAN_PROMPT_COMPLETION_SAMPLE_RATE`` environment
   variable.

   Default: ``1.0``


Instance Configuration
~~~~~~~~~~~~~~~~~~~~~~

To configure the Anthropic integration on a per-instance basis use the
``Pin`` API::

    import anthropic
    from ddtrace import Pin, config

    Pin.override(anthropic, service="my-anthropic-service")
"""  # noqa: E501
from ddtrace.internal.utils.importlib import require_modules


required_modules = ["anthropic"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.anthropic.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001

        from ddtrace.contrib.internal.anthropic.patch import get_version
        from ddtrace.contrib.internal.anthropic.patch import patch
        from ddtrace.contrib.internal.anthropic.patch import unpatch

        __all__ = ["patch", "unpatch", "get_version"]
