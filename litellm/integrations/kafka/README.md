# Kafka request logging

Export LiteLLM's `StandardLoggingPayload` as UTF-8 JSON Kafka records. The same
transport settings work with Apache Kafka and managed Kafka-compatible services.

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
    "security_protocol": "PLAINTEXT",
    "topic": "litellm_events",
    "turn_off_message_logging": True,
}
litellm.callbacks = ["kafka"]
```

For explicit SDK lifecycle management, construct
`litellm.integrations.kafka.KafkaLogger`, put that instance in `litellm.callbacks`,
and call `logger.close()` after draining your application's logging tasks.
Async applications can use `await logger.aclose()` without occupying their
default executor. `await logger.flush_queue()` drains already submitted Kafka
preparation jobs and asks the producer to flush, sharing one `flush_timeout`
budget. It does not drain callbacks still waiting in LiteLLM's logging worker.
The SDK exit handler drains the existing logging worker before closing Kafka.

Explicit configuration takes precedence over environment defaults. The supported
defaults are `KAFKA_BOOTSTRAP_SERVERS`, `KAFKA_TOPIC`, `KAFKA_SECURITY_PROTOCOL`,
`KAFKA_SASL_MECHANISM`, `KAFKA_SASL_USERNAME`, `KAFKA_SASL_PASSWORD`,
`KAFKA_SSL_CA_LOCATION`, `KAFKA_SSL_ENDPOINT_IDENTIFICATION_ALGORITHM`,
`KAFKA_SSL_CERTIFICATE_LOCATION`, `KAFKA_SSL_KEY_LOCATION` and `KAFKA_SSL_KEY_PASSWORD`.
Configuration values can reference
`os.environ/VARIABLE_NAME`. Missing required settings fail proxy initialization.
Broker availability and broker-side authentication are checked asynchronously
when the first record is produced.

## Transport and resource limits

| Setting | Default | Meaning |
| --- | --- | --- |
| `security_protocol` | `SSL` | `PLAINTEXT`, `SSL`, `SASL_PLAINTEXT` or `SASL_SSL` |
| `sasl_mechanism` | `PLAIN` | `PLAIN`, `SCRAM-SHA-256` or `SCRAM-SHA-512` |
| `ssl_ca_location` | system trust | PEM CA file or directory |
| `ssl_endpoint_identification_algorithm` | `https` | Broker hostname verification, or explicit `none` for a legacy certificate |
| `ssl_certificate_location`, `ssl_key_location` | unset | PEM client certificate and key for mutual TLS |
| `ssl_key_password` | unset | Password for an encrypted client key |
| `client_id` | `litellm` | Kafka producer client identifier |
| `compression_type` | `gzip` | `none`, `gzip`, `snappy`, `lz4` or `zstd` |
| `linger_ms` | `20` | Producer batching delay in milliseconds |
| `batch_size` | `65536` | Producer batch size in bytes |
| `queue_max_messages` | `10000` | Maximum queued producer records per worker |
| `callback_queue_max_messages` | `128` | Maximum waiting preparation jobs, in addition to one active job |
| `queue_max_kbytes` | `65536` | Maximum queued producer value bytes in KiB per worker |
| `message_max_bytes` | `1000000` | Client record size limit, align with broker and topic limits |
| `request_timeout_ms` | `10000` | Broker acknowledgement timeout |
| `retry_backoff_ms` | `100` | Initial native produce retry backoff |
| `retry_backoff_max_ms` | `1000` | Maximum native produce retry backoff |
| `delivery_timeout_ms` | `30000` | Total delivery deadline including retries and batching |
| `enable_idempotence` | `true` | Broker deduplication of retries within a producer session |
| `flush_timeout` | `5.0` | Maximum seconds for a flush or close |

TLS is enabled by default and verifies both the certificate chain and broker
hostname. Prefer certificates that match every advertised broker hostname. For
a legacy certificate, an operator can explicitly set
`ssl_endpoint_identification_algorithm: none`. This keeps certificate-chain
verification enabled but removes hostname identity checks, reducing protection
against broker impersonation under a trusted CA. LiteLLM never chooses this
setting automatically for a provider or after a failed handshake.
Set `PLAINTEXT` or `SASL_PLAINTEXT` explicitly only when unencrypted
transport is acceptable for your deployment. The localhost SDK example opts in
to plaintext for a local development broker. SASL passwords are
passed to the Kafka client and masked in configuration representations. The
integration does not support Kerberos or OAuth token refresh.

Each logger uses one bounded queue and a daemon worker for preparation, producer
operations and delivery callbacks. Both sync and async logging callbacks enqueue
without waiting for preparation, broker acknowledgements or the application's
default executor. A full preparation queue rejects the new event. The worker
lazily creates a reusable producer and preserves the submitting callback's
context.

The native client batches, compresses and retries retriable transport/broker
errors within `delivery_timeout_ms`. There is no application retry loop around
`produce()` and no wait for broker acknowledgements in request callbacks. A full
native queue drops the new event instead of blocking or retrying the enqueue.
Authentication, authorization and other terminal failures are not retried by
LiteLLM. Configure both delivery deadlines and backoff for your service's
throttling behavior.

Serialization, redaction and buffering still use CPU and memory, so benchmark
representative payload sizes and concurrency. The preparation limit counts
Python jobs, while `queue_max_kbytes` bounds serialized values in the native
queue. Neither limit represents total process memory. Limits apply per logger
and worker and supplement the existing LiteLLM logging worker limits.

This exporter is best effort. A full queue, oversized record, expired delivery,
invalid record or failed producer drops the event without changing the model
response. Warnings contain fixed reason codes and cumulative drop counts, with
rate limiting. Failures also use LiteLLM's callback failure metric when enabled.
An explicit logger exposes `statistics.accepted`, `statistics.delivered` and
`statistics.dropped`. Accepted means queued in the native producer, delivered
means acknowledged by the broker, and dropped includes pre-enqueue and terminal
delivery failures. `statistics.delivery_failed` counts that latter subset.
`statistics.unconfirmed` records messages still
unacknowledged at the final shutdown flush, or outstanding messages when that
flush fails. It is not proof that the broker never received those messages.

Graceful proxy shutdown drains logging work and finishes the existing PostgreSQL
writer cleanup before closing Kafka. Preparation drain and final producer flush
share the remaining shutdown budget. Neither producer construction nor the
application's default executor can hold up Kafka's asynchronous close. Client
cleanup that outlives the deadline continues on its daemon thread. A timed-out
close does not claim all messages were delivered. Process crashes, forced termination or a broker
outage longer than the delivery deadline can lose records. Idempotence does not
provide a durable outbox or end-to-end exactly-once delivery. Consumers should
retain Kafka coordinates and deduplicate replayed records.

## Managed services

Use the provider console's exact bootstrap endpoints and make every advertised
broker reachable. Create the topic and grant its producer describe, write and
any required idempotent-write permissions. Common authentication mappings are
shown below. Check the authentication settings for your instance.

| Service | Protocol | Authentication |
| --- | --- | --- |
| Confluent Cloud | `SASL_SSL` | `PLAIN`, Kafka API key and secret |
| Amazon MSK with SCRAM | `SASL_SSL` | `SCRAM-SHA-512`, associated SCRAM credentials |
| Azure Event Hubs | `SASL_SSL` | `PLAIN`, literal `$ConnectionString` username and SAS connection string |
| Aiven client certificates | `SSL` | CA, client certificate and private key |
| Alibaba Cloud Kafka | `SASL_SSL` | Console-selected PLAIN or SCRAM credentials and official CA bundle |

For legacy broker certificates that do not match their advertised hostname,
prefer a corrected certificate. If the operator accepts the loss of hostname
identity checks, explicitly set `ssl_endpoint_identification_algorithm: none`
and configure the official CA bundle. CA verification remains enabled. See the
[Alibaba Cloud Python example](https://help.aliyun.com/zh/apsaramq-for-kafka/cloud-message-queue-for-kafka/developer-reference/use-the-sdk-for-python-to-send-and-receive-messages).
Kafka export does not create downstream lake tables or run ingestion jobs.

## Analysis fields and existing persistence

See the [message format reference](message-format.md) for a JSON example,
field descriptions, nullable values, redaction and field-exclusion scope.

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
