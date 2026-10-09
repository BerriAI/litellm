# litellm-owned-params

The names LiteLLM consumes from a call, so a provider body never sees them. `is_owned` is the one check; `is_secret` reads the same rows for redaction; `compose_body` is the one function that merges `extra_body` into a provider body while dropping owned names

A setting is declared once, by the crate that consumes it, as a `litellm_auth_types::Setting` row: kwarg aliases in precedence order, env fallbacks in precedence order, secret flag. `connection.rs` and `route.rs` hold the rows the inference layer consumes; `auth-aws`, `auth-azure` and `auth-gcp` each export their own `settings::SETTINGS`; this crate only unions them. Never add a name here that another crate reads

`owned.rs` is generated from Python's `all_litellm_params` by `python -m scripts.generate_owned_kwargs`. It covers names Python consumes before a call reaches Rust (proxy metadata, callbacks, pricing keys). Do not edit it; `tests/unit/test_generate_owned_kwargs.py` fails when it is stale
