# Kafka request logging

Export LiteLLM's `StandardLoggingPayload` as UTF-8 JSON Kafka records. The same
configuration works with Apache Kafka and managed Kafka services, including
[Alibaba Cloud ApsaraMQ for Kafka](oss-tables.md).

## Install and configure

```sh
pip install 'litellm[proxy,kafka]'
```

Create the topic before starting the proxy. Give its producer permission to
describe the topic and write records. Idempotent production is enabled by
default, so check your broker's authorization requirements for idempotent writes.

```yaml
litellm_settings:
  callbacks: ["kafka"]
  kafka_params:
    bootstrap_servers: os.environ/KAFKA_BOOTSTRAP_SERVERS
    topic: litellm_events
    security_protocol: SASL_SSL
    sasl_mechanism: SCRAM-SHA-256
    sasl_username: os.environ/KAFKA_SASL_USERNAME
    sasl_password: os.environ/KAFKA_SASL_PASSWORD
    ssl_ca_location: /etc/kafka/ca.pem
    turn_off_message_logging: true
```

Use `callbacks` to export both successful and failed calls. For just one outcome,
use `success_callback` or `failure_callback`. Streaming calls export a completed
request event, rather than a record for every output chunk. The integration uses
the existing standard event adapter, including when V2 logging is enabled.

The SDK also supports the named callback.

```python
import litellm

litellm.kafka_params = {
    "bootstrap_servers": "localhost:9092",
    "topic": "litellm_events",
    "turn_off_message_logging": True,
}
litellm.callbacks = ["kafka"]
```

For explicit SDK lifecycle management, construct
`litellm.integrations.kafka.KafkaLogger`, put that instance in `litellm.callbacks`,
and call `logger.close()` after draining your application's logging tasks.
`await logger.flush_queue()` waits for delivery of records already handed to the
producer, within `flush_timeout`. It does not drain callbacks still waiting in
LiteLLM's logging worker.

Explicit configuration takes precedence over environment defaults. The supported
defaults are `KAFKA_BOOTSTRAP_SERVERS`, `KAFKA_TOPIC`, `KAFKA_SECURITY_PROTOCOL`,
`KAFKA_SASL_MECHANISM`, `KAFKA_SASL_USERNAME`, `KAFKA_SASL_PASSWORD`,
`KAFKA_SSL_CA_LOCATION`, `KAFKA_SSL_CERTIFICATE_LOCATION`, `KAFKA_SSL_KEY_LOCATION`
and `KAFKA_SSL_KEY_PASSWORD`. Configuration values can reference
`os.environ/VARIABLE_NAME`. Missing required settings fail proxy initialization.
Broker availability and broker-side authentication are checked asynchronously
when the first record is produced.

## Transport and resource limits

| Setting | Default | Meaning |
| --- | --- | --- |
| `security_protocol` | `PLAINTEXT` | `PLAINTEXT`, `SSL`, `SASL_PLAINTEXT` or `SASL_SSL` |
| `sasl_mechanism` | `PLAIN` | `PLAIN`, `SCRAM-SHA-256` or `SCRAM-SHA-512` |
| `ssl_ca_location` | system trust | PEM CA file or directory |
| `ssl_certificate_location`, `ssl_key_location` | unset | PEM client certificate and key for mutual TLS |
| `ssl_key_password` | unset | Password for an encrypted client key |
| `client_id` | `litellm` | Kafka producer client identifier |
| `compression_type` | `gzip` | `none`, `gzip`, `snappy`, `lz4` or `zstd` |
| `linger_ms` | `20` | Producer batching delay in milliseconds |
| `batch_size` | `65536` | Producer batch size in bytes |
| `queue_max_messages` | `10000` | Maximum queued producer records per worker |
| `queue_max_kbytes` | `65536` | Maximum queued producer value bytes in KiB per worker |
| `message_max_bytes` | `1000000` | Client record size limit, align with broker and topic limits |
| `request_timeout_ms` | `10000` | Broker acknowledgement timeout |
| `delivery_timeout_ms` | `30000` | Total delivery deadline including retries and batching |
| `enable_idempotence` | `true` | Broker deduplication of retries within a producer session |
| `flush_timeout` | `5.0` | Maximum seconds for a flush or close |

TLS verifies both the certificate chain and broker hostname. SASL passwords are
passed to the Kafka client and masked in configuration representations. The
integration does not support Kerberos or OAuth token refresh.

Each process lazily creates one reusable producer per logger and a daemon thread
that handles delivery reports. The native Kafka client batches, compresses and
retries asynchronously. Async callbacks move payload preparation and enqueueing
to a thread. They do not wait for broker acknowledgements. Serialization,
redaction and buffering still use CPU and memory, so benchmark representative
payload sizes and concurrency before rollout. Queue limits apply per producer,
so multiply them by the number of proxy workers. These limits supplement the
existing LiteLLM logging worker limits.

This exporter is best effort. A full queue, oversized record, expired delivery,
invalid record or failed producer drops the event without changing the model
response. Warnings contain fixed reason codes and cumulative drop counts, with
rate limiting. Failures also use LiteLLM's callback failure metric when enabled.
An explicit logger exposes `statistics.accepted`, `statistics.delivered` and
`statistics.dropped`. Accepted means queued locally, delivered means acknowledged
by the broker. Dropped includes pre-enqueue and delivery failures.

Graceful proxy shutdown drains logging work, then closes Kafka producers within
the remaining shutdown budget. Process crashes, forced termination or a broker
outage longer than the delivery deadline can lose records. Idempotence does not
provide a durable outbox or end-to-end exactly-once delivery. Consumers should
retain Kafka coordinates and deduplicate replayed records.

## Analysis fields and existing persistence

The Kafka value is the existing standard payload, without a vendor-specific
envelope. Its key is `litellm_call_id`, falling back to `id`. Important paths are
`metadata.user_api_key_alias`, `metadata.user_api_key_hash`,
`metadata.user_api_key_team_id`, `prompt_tokens`, `completion_tokens`,
`total_tokens`, `response_cost`, `status`, `startTime` and `endTime`.
`metadata.usage_object` carries provider-specific usage details where available.
An SDK call without proxy virtual-key context can have no alias or team fields.
Failures and interrupted streams can have missing or incomplete usage. Preserve
that distinction in warehouse reconciliation.

`turn_off_message_logging`, per-request redaction and
`standard_logging_payload_excluded_fields` use the existing LiteLLM helpers.
Redaction applies to the Kafka copy without changing another destination's event.
For analysis that only needs attribution and usage, turn off message logging.
Review custom metadata before exporting it.

Kafka logging runs alongside existing PostgreSQL spend persistence and other
destinations. It does not change spending, budgets or database configuration.
Keeping PostgreSQL request rows is the default. The existing
`general_settings.disable_spend_logs: true` can disable detailed request spend
rows, while key/user/team spend updates continue. That option is a separate
operator decision, and it can remove database-backed request history. Kafka
delivery and PostgreSQL writes are not one transaction.
