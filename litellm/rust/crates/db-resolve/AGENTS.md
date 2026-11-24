# Database settings to a pool endpoint

The crate has two layers, like `redis-resolve`

- `interpret` is the layer both hosts share: the final url a host connects with in, a `litellm_db::Endpoint` out, under the `Overlay` the host sets on top. It asks the server about TLS where it has to. The Python host calls it through `DatabaseEndpoint.resolve` in python-bridge, with the url Prisma connects with and the Rust share of the connection budget, so no endpoint rule lives in Python
- `resolve` is the gateway's own layer, one call like `redis-resolve`'s. It turns `general_settings` and the `DATABASE_*` variables into the writer and read replica locations and runs each through the same code as `interpret`. The Python host assembles its url in `litellm/proxy/db/db_url_settings.py` for Prisma, so only a Rust host calls this layer
- `resolve` takes the host's `SecretResolver`, the one type that reads the environment, the secret manager and OIDC. Python reads every `DATABASE_*` variable and both token auth toggles straight from `os.environ`, so they are read from `secrets.environment()`, which on the gateway is the process environment under the config's `environment_variables`. A name Python reads with `get_secret` goes through `secrets.get_secret`. Never read the environment by iterating it, because a secret manager cannot be listed
- An `Overlay` says, per limit, whether the host's value replaces what the url pins (`Rule::OverUrl`) or is only the default (`Rule::UnderUrl`). The gateway's writer puts the configured pool size, pool timeout and socket timeout over the url and its replica puts them under. The Python host puts only its connection share over the url

- The writer comes from `general_settings.database_url`, then `DATABASE_URL`, then `DATABASE_HOST`, `DATABASE_PORT`, `DATABASE_USER` or `DATABASE_USERNAME`, `DATABASE_PASSWORD`, `DATABASE_NAME` and `DATABASE_SCHEMA`. The read replica comes from `DATABASE_URL_READ_REPLICA`, then `DATABASE_HOST_READ_REPLICA` with each `*_READ_REPLICA` field falling back to the writer's variable. No writer means no database, which is `Ok(None)`
- No setting name is spelled twice. The `fields!` macro in `src/location.rs` declares `Fields` and asks for each field by name, `DATABASE_<FIELD>` for the writer and `DATABASE_<FIELD>_READ_REPLICA` for the replica, so a field is its variable and only the exact upper case name is read, like the proxy. An empty variable is unset. `Params` in `src/params.rs` is the url query, so a field is its parameter. The `general_settings` keys are typed fields of `litellm_config::GeneralSettings`. An error or an ignored setting takes its name from the field through `stringify!`
- An `os.environ/` reference in the config is already resolved by `config-load` before this crate sees it, so nothing here strips the prefix
- Precedence follows `proxy_cli.py`. On the writer the configured pool size, pool timeout and socket timeout replace what the URL pins. On the replica what the URL pins wins. `max_idle_connection_lifetime` on a URL wins on both
- TLS is read in both dialects. libpq's `verify-ca`, `verify-full` and `sslrootcert` mean the same as Prisma's `sslmode=require` with `sslaccept=strict` and `sslcert`. `sslmode=prefer` with strict verification asks the server once with a Postgres `SSLRequest` and pins `verify-full` or `disable`, and a server that cannot be reached counts as offering TLS
- An error names the setting it came from and never the URL, because the URL carries the password
- A setting the pool cannot carry is returned in `Database::ignored` for the host to log. Only a malformed value, a non-PostgreSQL scheme or a token auth toggle fails

## Differences between the two hosts on purpose

| Setting | Python host | Rust gateway |
| --- | --- | --- |
| `IAM_TOKEN_DB_AUTH`, `AZURE_POSTGRESQL_AUTH` | Mints a token and refreshes it | `Error::TokenAuth`, since nothing in Rust mints one yet. The gateway logs it and runs without a pool |
| `sslidentity`, `sslpassword` | Read in Python and passed as `Overlay::client_identity`, PEM | The gateway passes none, so they are ignored with a warning and no client certificate is sent |
| `sslrootcert` holding several roots | Pinned down to the server's root, since Prisma takes one certificate | Passed as it is, since rustls trusts every certificate in the file |
| Configured statement, lock and idle-in-transaction timeouts | Appended to the `options` text unless a regex finds the setting there | Added unless the parsed `options` already set it, so a quoted setting still wins and an unreadable `options` still gets the timeouts |
| `DATABASE_DISABLE_PREPARED_STATEMENTS` with `pgbouncer=false` on the URL | The URL wins | Statements stay unnamed |
| `database_extra_connection_params`, `DATABASE_MAX_IDLE_CONNECTION_LIFETIME`, `connect_timeout` | Read | Not read |
| A replica `connection_limit` that is not a positive integer | Left for Prisma to judge | `Error::Invalid` |
