# Managed Kafka services

The exporter uses standard Kafka transport and SASL settings. It has no
cloud-specific producer implementation or required cloud SDK. These recipes
map provider settings to the same `kafka_params` configuration.

| Service and authentication | `security_protocol` | `sasl_mechanism` | Credentials |
| --- | --- | --- | --- |
| Confluent Cloud Kafka API key | `SASL_SSL` | `PLAIN` | Kafka API key and secret |
| Amazon MSK with SCRAM enabled | `SASL_SSL` | `SCRAM-SHA-512` | Associated SCRAM username and password |
| Azure Event Hubs connection string | `SASL_SSL` | `PLAIN` | `$ConnectionString` and the SAS connection string |
| Aiven client certificate | `SSL` | unused | PEM CA, client certificate and private key |
| Alibaba Cloud Kafka | `SASL_SSL` | Console-selected `PLAIN` or SCRAM | Instance SASL credentials |

These are configuration recipes, not claims that each cloud instance has been
tested. Local tests cover the generic protocols. Verify your instance, endpoint,
authorization, certificates and network access before production use.

## Common setup

Install `litellm[proxy,kafka]`, create a topic, and grant permission to describe
and write it. The broker must support idempotent production when
`enable_idempotence` is true. Provision any extra permissions it requires.
Configure `litellm_settings.callbacks: ["kafka"]`, then choose one of the
`kafka_params` blocks below. Topic, message size, retention and partition settings
remain the responsibility of the Kafka administrator.

Use the exact bootstrap list from your provider. Every advertised broker must
be reachable from the proxy, including with private endpoints. Keep hostname
verification enabled, mount PEM files read-only and rotate them through your
normal deployment process. The [generic guide](README.md) documents buffer,
deadline and retry controls. No recipe changes PostgreSQL spend persistence.

## Confluent Cloud API keys

```yaml
kafka_params:
  bootstrap_servers: os.environ/KAFKA_BOOTSTRAP_SERVERS
  topic: litellm_events
  security_protocol: SASL_SSL
  sasl_mechanism: PLAIN
  sasl_username: os.environ/CONFLUENT_KAFKA_API_KEY
  sasl_password: os.environ/CONFLUENT_KAFKA_API_SECRET
  turn_off_message_logging: true
```

Use a Kafka cluster API key, with access to the destination topic. Configure a
CA bundle only if your environment requires one. See
[Confluent's client guide](https://docs.confluent.io/cloud/current/cp-component/clients-cloud-config.html).

## Amazon MSK with SASL/SCRAM

```yaml
kafka_params:
  bootstrap_servers: os.environ/MSK_SCRAM_BOOTSTRAP_SERVERS
  topic: litellm_events
  security_protocol: SASL_SSL
  sasl_mechanism: SCRAM-SHA-512
  sasl_username: os.environ/MSK_SCRAM_USERNAME
  sasl_password: os.environ/MSK_SCRAM_PASSWORD
  turn_off_message_logging: true
```

Enable SCRAM and associate the credentials with the cluster before connecting.
MSK's SCRAM support uses SHA-512, as documented in its
[SCRAM limitations](https://docs.aws.amazon.com/msk/latest/developerguide/msk-password-limitations.html).
Use its SCRAM bootstrap endpoints, rather than IAM endpoints. This configuration
does not implement IAM token signing, assume-role refresh or the `AWS_MSK_IAM`
mechanism. MSK offerings that require IAM need a separate credential integration.

## Azure Event Hubs Kafka endpoint with a SAS connection string

```yaml
kafka_params:
  bootstrap_servers: os.environ/EVENT_HUBS_KAFKA_BOOTSTRAP_SERVERS
  topic: litellm_events
  security_protocol: SASL_SSL
  sasl_mechanism: PLAIN
  sasl_username: "$ConnectionString"
  sasl_password: os.environ/EVENT_HUBS_CONNECTION_STRING
  request_timeout_ms: 60000
  delivery_timeout_ms: 120000
  turn_off_message_logging: true
```

The topic is the event hub name. The username is the literal `$ConnectionString`,
not an environment reference. Give the SAS identity send permission. The
delivery deadline covers the larger request timeout and batching delay. See
[Microsoft's Kafka connection example](https://learn.microsoft.com/en-us/azure/event-hubs/event-hubs-faq).
Entra ID OAuth authentication is a different mode and is not enabled by this
connection-string example.

## Client certificates, including Aiven

```yaml
kafka_params:
  bootstrap_servers: os.environ/KAFKA_BOOTSTRAP_SERVERS
  topic: litellm_events
  security_protocol: SSL
  ssl_ca_location: /etc/kafka/ca.pem
  ssl_certificate_location: /etc/kafka/service.cert
  ssl_key_location: /etc/kafka/service.key
  turn_off_message_logging: true
```

Mount the service user's certificate and key. For encrypted keys, configure
`ssl_key_password: os.environ/KAFKA_SSL_KEY_PASSWORD`. SASL credentials are not
needed for certificate-only authentication. Follow your provider's rotation
instructions, such as [Aiven's certificate guide](https://aiven.io/docs/platform/concepts/tls-ssl-certificates).

## Alibaba Cloud and downstream lake ingestion

The [Alibaba Cloud Kafka and OSS Tables recipe](oss-tables.md) uses these same
generic transport options and describes downstream table ingestion. Other cloud
lakehouses can consume the same standard JSON topic using their supported Kafka
ingestion service or an independently deployed consumer.

## Authentication boundaries

PLAIN, SCRAM-SHA-256, SCRAM-SHA-512 and client certificates are supported. OAuth
token refresh, AWS IAM signing and Kerberos are not implemented. Do not configure
these modes as a static SASL password or suggest that all authentication modes
of a named cloud service are covered. Cloud-specific credentials should be added
through a separately tested refresh mechanism without changing the event format
or blocking the request callback.
