# Alibaba Cloud Kafka and OSS Tables

Use the [Kafka exporter](README.md) to send request events to ApsaraMQ for Kafka,
then ingest that topic into OSS Tables. The proxy only connects to Kafka. Table
commits, file sizing and lake maintenance run downstream.

## Connect to Alibaba Cloud Kafka

Obtain bootstrap endpoints, the SASL mechanism and credentials from the instance
console. Create a topic such as `litellm_events`, and grant the producer identity
the required topic and idempotent-write permissions. Allow network access from
the proxy to every advertised broker. Mount the instance's PEM CA bundle in the
proxy container.

```yaml
litellm_settings:
  callbacks: ["kafka"]
  kafka_params:
    bootstrap_servers: os.environ/KAFKA_BOOTSTRAP_SERVERS
    topic: litellm_events
    security_protocol: SASL_SSL
    sasl_mechanism: PLAIN
    sasl_username: os.environ/KAFKA_SASL_USERNAME
    sasl_password: os.environ/KAFKA_SASL_PASSWORD
    ssl_ca_location: /etc/kafka/instance-ca.pem
    turn_off_message_logging: true
    queue_max_messages: 10000
    queue_max_kbytes: 65536
    delivery_timeout_ms: 30000
    flush_timeout: 5
```

Use the console's exact endpoint and mechanism. `SCRAM-SHA-256` is also supported.
For a VPC SASL endpoint without TLS, use `SASL_PLAINTEXT`. The default VPC endpoint
can use `PLAINTEXT` without SASL credentials. See the
[endpoint comparison](https://www.alibabacloud.com/help/en/apsaramq-for-kafka/cloud-message-queue-for-kafka/product-overview/comparison-among-endpoints).

Certificate and hostname verification stay enabled. Older instances and SDK
examples may assume older OpenSSL clients or disable hostname checks. Check the
instance certificate generation, advertised hostname and CA bundle with your
current client, following the
[SDK compatibility guidance](https://www.alibabacloud.com/help/en/apsaramq-for-kafka/cloud-message-queue-for-kafka/developer-reference/sdk-overview).
Upgrade legacy certificates or use an appropriate supported endpoint when
needed. Matching SASL settings alone does not establish compatibility with a
legacy public SSL endpoint. LiteLLM needs only Kafka connection settings, without
an Alibaba Cloud vendor SDK.

## Ingest into OSS Tables

Alibaba Cloud documents managed Kafka-to-OSS Tables ingestion. As of 2026-11-19,
its documented scope is Kafka Serverless 3.7.0.0 or newer in Hangzhou, Singapore
and Shanghai. Check current availability and limits in the
[managed ingestion guide](https://help.aliyun.com/zh/oss/user-guide/data-ingestion-into-the-data-lake).

Create an OSS Table Bucket, enable the instance's lake ingestion and Schema
Registry, authorize the service and associate the bucket catalog. Enable
ingestion for `litellm_events` and select an append archive. Start with raw
archive to preserve complete events, including variable provider usage and
custom metadata. Choose string or raw value conversion for the consumer. Retain
topic, partition and offset with the archived value for replay and reconciliation.

For structured archive, the exporter sends plain JSON without a schema wire
header. Register a compatible JSON schema and use `by_latest_schema` instead of
`by_schema_id`. Validate success, error and streaming samples before enabling it.
Keep optional identity and usage fields nullable. Nested metadata and provider
usage may need downstream normalization or raw archive. A latest-schema change
can affect outstanding messages, so test compatibility and retain replayable
source data. Monitor rejected records and choose an explicit ingestion error
policy.

If managed ingestion is unavailable, run a separate Kafka consumer, Flink or
Spark job writing Iceberg through the supported catalog. Commit consumer offsets
after the table commit and make reprocessing idempotent. The
[PyIceberg guide](https://help.aliyun.com/zh/oss/user-guide/pyiceberg-access-oss-tables)
describes catalog access. Uploading JSON to an ordinary OSS bucket does not
create an Iceberg table commit.

## Build the analysis table

Extract stable columns and keep the raw event for schema changes.

| Analysis column | JSON event path |
| --- | --- |
| Request correlation | `litellm_call_id`, `id` |
| Key alias | `metadata.user_api_key_alias` |
| Key identity | `metadata.user_api_key_hash` |
| Team | `metadata.user_api_key_team_id` |
| Input/output/total tokens | `prompt_tokens`, `completion_tokens`, `total_tokens` |
| Provider usage details | `metadata.usage_object` |
| Cost and outcome | `response_cost`, `status`, `error_information` |
| Model and timing | `model`, `custom_llm_provider`, `startTime`, `endTime` |

After landing the raw JSON value as `event_json`, this Spark SQL projection
produces a daily report. Substitute your catalog and raw table name. Deduplicate
physical replay by topic/partition/offset first. Handle logical duplicates using
request correlation and your application's event semantics, rather than assuming
response ID alone is unique across all routes and retries.

```sql
WITH events AS (
  SELECT
    to_date(from_unixtime(
      CAST(get_json_object(event_json, '$.startTime') AS DOUBLE)
    )) AS request_day,
    get_json_object(event_json, '$.metadata.user_api_key_alias') AS key_alias,
    get_json_object(event_json, '$.status') AS status,
    CAST(get_json_object(event_json, '$.total_tokens') AS BIGINT) AS total_tokens,
    CAST(get_json_object(event_json, '$.response_cost') AS DOUBLE) AS response_cost
  FROM your_catalog.your_namespace.raw_litellm_events
)
SELECT request_day, key_alias, status, count(*) AS requests,
       sum(total_tokens) AS total_tokens, sum(response_cost) AS response_cost
FROM events
GROUP BY request_day, key_alias, status;
```

Set the query session timezone explicitly. A missing alias can mean no virtual-key
context was supplied. Missing usage on a failed call is not a verified zero bill.

## Delivery and rollout

Kafka is sufficient as the initial LiteLLM export boundary for asynchronous
analytics. It separates proxy resource use from lake ingestion and supports
independent consumers and replay within topic retention. A direct table writer
would add Arrow/file serialization, catalog authentication, batching, commit
retries and file maintenance to proxy workers. Keep that integration separate.

The exporter leaves PostgreSQL spend writes enabled. Verify both destinations
before considering the existing `disable_spend_logs` option. Producer idempotence
does not make PostgreSQL, Kafka and Iceberg one atomic transaction. This
best-effort exporter can lose an event before Kafka acknowledges it. Strict audit
completeness requires a durable outbox or another durable capture layer.

Before production rollout, verify successful, failed and streamed calls with a
virtual key that has an alias. Compare consumed JSON usage and attribution with
API results and PostgreSQL records. Check bad credentials, CA/hostname failure,
unavailable brokers, buffer saturation and graceful shutdown. Confirm model
responses remain independent of export failures. Verify those records reach an
OSS Tables snapshot and can be queried. Measure response latency, CPU, memory,
producer drops, consumer lag and table commit delay at the intended worker count
and payload size. Local tests do not replace cloud and lake acceptance runs.
