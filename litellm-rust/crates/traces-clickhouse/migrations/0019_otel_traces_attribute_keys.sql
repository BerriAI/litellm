ALTER TABLE {database}.otel_traces
    ADD INDEX IF NOT EXISTS idx_span_attribute_keys mapKeys(SpanAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    ADD INDEX IF NOT EXISTS idx_resource_attribute_keys mapKeys(ResourceAttributes) TYPE bloom_filter(0.01) GRANULARITY 1
