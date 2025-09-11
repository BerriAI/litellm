"""
The Vertex AI integration instruments the Vertex Generative AI SDK for Python for requests made to Google models.

All traces submitted from the Vertex AI integration are tagged by:

- ``service``, ``env``, ``version``: see the `Unified Service Tagging docs <https://docs.datadoghq.com/getting_started/tagging/unified_service_tagging>`_.
- ``vertexai.request.provider``: LLM provider used in the request (e.g. ``google`` for Google models).
- ``vertexai.request.model``: Google model used in the request.

(beta) Prompt and Completion Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prompt texts and completion content are collected in span tags with a default sampling rate of ``1.0``
for the following methods:

- ``generate_content/generate_content_async`` of the GenerativeModel class
- ``send_message/send_message_async`` of the ChatSession class

These tags will have truncation applied if the text exceeds the configured character limit.


Enabling
~~~~~~~~

The Vertex AI integration is enabled automatically when you use
:ref:`ddtrace-run<ddtracerun>` or :ref:`import ddtrace.auto<ddtraceauto>`.

Alternatively, use :func:`patch() <ddtrace.patch>` to manually enable the Vertex AI integration::

    from ddtrace import config, patch

    patch(vertexai=True)


Global Configuration
~~~~~~~~~~~~~~~~~~~~

.. py:data:: ddtrace.config.vertexai["service"]

   The service name reported by default for Vertex AI requests.

   Alternatively, you can set this option with the ``DD_SERVICE`` or ``DD_VERTEXAI_SERVICE`` environment
   variables.

   Default: ``DD_SERVICE``


.. py:data:: (beta) ddtrace.config.vertexai["span_char_limit"]

   Configure the maximum number of characters for the following data within span tags:

   - Text inputs and completions

   Text exceeding the maximum number of characters is truncated to the character limit
   and has ``...`` appended to the end.

   Alternatively, you can set this option with the ``DD_VERTEXAI_SPAN_CHAR_LIMIT`` environment
   variable.

   Default: ``128``


.. py:data:: (beta) ddtrace.config.vertexai["span_prompt_completion_sample_rate"]

   Configure the sample rate for the collection of prompts and completions as span tags.

   Alternatively, you can set this option with the ``DD_VERTEXAI_SPAN_PROMPT_COMPLETION_SAMPLE_RATE`` environment
   variable.

   Default: ``1.0``


Instance Configuration
~~~~~~~~~~~~~~~~~~~~~~

To configure the Vertex AI integration on a per-instance basis use the
``Pin`` API::

    import vertexai
    from ddtrace import Pin, config

    Pin.override(vertexai, service="my-vertexai-service")
"""  # noqa: E501

from ddtrace.internal.utils.importlib import require_modules


required_modules = ["vertexai"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        from ddtrace.contrib.internal.vertexai.patch import get_version
        from ddtrace.contrib.internal.vertexai.patch import patch
        from ddtrace.contrib.internal.vertexai.patch import unpatch

        __all__ = ["patch", "unpatch", "get_version"]
