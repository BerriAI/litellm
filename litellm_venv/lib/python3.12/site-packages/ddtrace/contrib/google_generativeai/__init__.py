"""
The Gemini integration instruments the Google Gemini Python API to traces for requests made to Google models.

All traces submitted from the Gemini integration are tagged by:

- ``service``, ``env``, ``version``: see the `Unified Service Tagging docs <https://docs.datadoghq.com/getting_started/tagging/unified_service_tagging>`_.
- ``google_generativeai.request.model``: Google model used in the request.
- ``google_generativeai.request.api_key``: Google Gemini API key used to make the request (obfuscated to match the Google AI Studio UI representation ``...XXXX`` where ``XXXX`` is the last 4 digits of the key).


(beta) Prompt and Completion Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prompt texts and completion content for the ``generateContent`` endpoint are collected in span tags with a default sampling rate of ``1.0``.
These tags will have truncation applied if the text exceeds the configured character limit.


Enabling
~~~~~~~~

The Gemini integration is enabled automatically when you use
:ref:`ddtrace-run<ddtracerun>` or :ref:`import ddtrace.auto<ddtraceauto>`.

Alternatively, use :func:`patch() <ddtrace.patch>` to manually enable the Gemini integration::

    from ddtrace import config, patch

    patch(google_generativeai=True)


Global Configuration
~~~~~~~~~~~~~~~~~~~~

.. py:data:: ddtrace.config.google_generativeai["service"]

   The service name reported by default for Gemini requests.

   Alternatively, you can set this option with the ``DD_SERVICE`` or ``DD_GOOGLE_GENERATIVEAI_SERVICE`` environment
   variables.

   Default: ``DD_SERVICE``


.. py:data:: (beta) ddtrace.config.google_generativeai["span_char_limit"]

   Configure the maximum number of characters for the following data within span tags:

   - Text inputs and completions

   Text exceeding the maximum number of characters is truncated to the character limit
   and has ``...`` appended to the end.

   Alternatively, you can set this option with the ``DD_GOOGLE_GENERATIVEAI_SPAN_CHAR_LIMIT`` environment
   variable.

   Default: ``128``


.. py:data:: (beta) ddtrace.config.google_generativeai["span_prompt_completion_sample_rate"]

   Configure the sample rate for the collection of prompts and completions as span tags.

   Alternatively, you can set this option with the ``DD_GOOGLE_GENERATIVEAI_SPAN_PROMPT_COMPLETION_SAMPLE_RATE`` environment
   variable.

   Default: ``1.0``


Instance Configuration
~~~~~~~~~~~~~~~~~~~~~~

To configure the Gemini integration on a per-instance basis use the
``Pin`` API::

    import google.generativeai as genai
    from ddtrace import Pin, config

    Pin.override(genai, service="my-gemini-service")
"""  # noqa: E501
from ddtrace.internal.utils.importlib import require_modules


required_modules = ["google.generativeai"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        from ..internal.google_generativeai.patch import get_version
        from ..internal.google_generativeai.patch import patch
        from ..internal.google_generativeai.patch import unpatch

        __all__ = ["patch", "unpatch", "get_version"]
