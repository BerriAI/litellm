## Redis settings

- `RedisSettings::resolve` follows `litellm/_redis.py` `_get_redis_client_logic` plus `_credential_provider_auth_kwargs` (redis-py 5.3.1). The `resolution_matches_python_redis` cases in `tests/redis.rs` were checked against that Python on 2026-10-29. Python and Rust pods share one Redis during the switch, so a change there is a change to a wire contract
- Precedence matches Python. The target is cluster, then URL, then Sentinel (`sentinel_nodes` plus `service_name`), then host. Auth is an explicit `credential_provider`, then GCP, then Azure, then AWS, and the first one wins with a warning
- Known divergences, kept on purpose
  - A malformed number (`port`, `db`, `socket_timeout`, ...) fails with `RedisError::Invalid`. Python's `_coerce_redis_kwargs_types` drops it silently and connects with the default
  - redis-py kwargs Rust can't honor are `RedisError::Unsupported` naming the setting: `ssl_ciphers`, `ssl_min_version`, `ssl_password`, `ssl_ca_data`, `ssl_ca_path`, OCSP options, `retry`, `retry_on_timeout`, `retry_on_error`, `protocol`, `cache`, `cache_config`, `unix_socket_path`, `connection_pool`, `event_dispatcher`, `redis_connect_func`. Python accepts them
  - `health_check_interval`, `socket_keepalive` and `max_connections` are accepted with a warning and ignored. Cluster clients still turn TCP keepalive on, like Python's async cluster default, but redis-rs has no periodic health check, so Python's 25s cluster default has no equivalent
