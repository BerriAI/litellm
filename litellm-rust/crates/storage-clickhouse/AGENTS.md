# ClickHouse storage

`litellm-storage-clickhouse` exports `Storage`, a writer and bounded reader derived from one ClickHouse URL and database. It also exports bounded HTTP read and insert execution

The crate has no trace tables, OTLP types, or named trace queries. `litellm-traces-clickhouse` supplies those rules and uses this storage for both trace rows and spend rows
