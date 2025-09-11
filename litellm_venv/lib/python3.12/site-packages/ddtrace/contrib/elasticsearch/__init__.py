"""
The Elasticsearch integration will trace Elasticsearch queries.

Enabling
~~~~~~~~

The elasticsearch integration is enabled automatically when using
:ref:`ddtrace-run<ddtracerun>` or :ref:`import ddtrace.auto<ddtraceauto>`.

Or use :func:`patch()<ddtrace.patch>` to manually enable the integration::

    from ddtrace import patch
    from elasticsearch import Elasticsearch

    patch(elasticsearch=True)
    # This will report spans with the default instrumentation
    es = Elasticsearch(port=ELASTICSEARCH_CONFIG['port'])
    # Example of instrumented query
    es.indices.create(index='books', ignore=400)

    # Use a pin to specify metadata related to this client
    es = Elasticsearch(port=ELASTICSEARCH_CONFIG['port'])
    Pin.override(es.transport, service='elasticsearch-videos')
    es.indices.create(index='videos', ignore=400)

OpenSearch is also supported (`opensearch-py`)::

    from ddtrace import patch
    from opensearchpy import OpenSearch

    patch(elasticsearch=True)
    os = OpenSearch()
    # Example of instrumented query
    os.indices.create(index='books', ignore=400)


Configuration
~~~~~~~~~~~~~

.. py:data:: ddtrace.config.elasticsearch['service']

   The service name reported for your elasticsearch app.


Example::

    from ddtrace import config

    # Override service name
    config.elasticsearch['service'] = 'custom-service-name'
"""
from ddtrace.contrib.internal.elasticsearch.patch import get_version
from ddtrace.contrib.internal.elasticsearch.patch import get_versions
from ddtrace.contrib.internal.elasticsearch.patch import patch


__all__ = ["patch", "get_version", "get_versions"]
