# Trace queries

Use a dedicated ClickHouse reader account for `general_settings.clickhouse_url`. The SQL helper sends the supplied SQL to ClickHouse without parsing it. Its `readonly=1` request setting does not make an account with write or administrative grants safe

Pass `execute_admin_sql` a shared HTTP client obtained from `HttpClientPool` with `ClientVariant::NoRedirect` and the host's HTTP configuration. The helper applies a 15-second request timeout and validates the bounded JSON response, including errors returned with HTTP 200

For self-hosted ClickHouse, install [config/reader.xml](config/reader.xml) as `/etc/clickhouse-server/users.d/litellm-traces-reader.xml` and provide `LITELLM_TRACES_READER_PASSWORD` in the ClickHouse server environment. Restrict the configured network range to your proxy network. The configuration grants SELECT only on `default.otel_traces` and locks read-only mode, execution time, result rows and bytes, overflow behavior and memory limits. Adjust the database name if traces live elsewhere

Set `clickhouse_url` to `https://litellm_traces_reader:<percent-encoded-password>@<clickhouse-host>:8443/?database=default`. Keep ingestion and migration credentials separate. For managed ClickHouse, provision the equivalent SELECT grant and settings constraints through its access-management interface

Do not grant write, backup, named-collection management or grant-option privileges to the reader, directly or through roles. Do not make `readonly` changeable in read-only mode. ClickHouse documents exceptions to `readonly` in [query permissions](https://clickhouse.com/docs/concepts/features/configuration/settings/permissions-for-queries) and explains locked settings in [settings constraints](https://clickhouse.com/docs/concepts/features/configuration/settings/constraints-on-settings)

Run the reader permission, authentication and response-limit tests with `cargo test -p litellm-traces --test admin_sql -- --test-threads=2` from `litellm-rust`. They start disposable ClickHouse instances through Testcontainers and load the same reader configuration
