# ClickHouse storage

`litellm-storage-clickhouse` exports `Storage`, a shared writer connection and optional reader connection for one ClickHouse database. It also exports bounded HTTP read and insert execution

The crate has no trace tables, OTLP types, or named trace queries. `litellm-traces` supplies those rules and uses this storage for both trace rows and spend rows
