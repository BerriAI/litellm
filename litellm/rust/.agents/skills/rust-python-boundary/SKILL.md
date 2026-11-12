---
name: rust-python-boundary
description: Decide where Python-shaped input becomes typed Rust in litellm-rust. Use when reading config.yaml values, os.environ/ references, REDIS_* or other environment layering, get_secret fallbacks, or kwargs and objects handed over from Python, when coercing text to a bool, int or float the way Python does, or when adding a dependency on litellm-python-compat
metadata:
  internal: true
---

# Python boundary

Python-shaped input becomes typed Rust in exactly two crates. Every crate past them handles typed values, so no crate re-guesses how Python would have read a value

## The two boundaries

- `crates/config` reads config.yaml, which is Python-shaped: `os.environ/` references, scalars that Helm or the environment turned into text, raw kwargs maps. It also owns the environment layering and `get_secret` fallbacks Python applies while it loads that config. It has no Python runtime, so it reproduces Python value semantics with `litellm-python-compat`
- `crates/python-bridge` receives live Python objects, such as `RedisCache.redis_kwargs`. It reads them through PyO3 with the real protocol (`is_truthy()`, `str()`, `extract`), so it never imitates Python
- Both produce the same typed input, defined in the data crate the consumer already depends on (`redis-types` for Redis), so a downstream crate cannot tell which host it runs under

## Python value semantics

Reading a value the way Python itself reads it. This lives only at the two boundaries

- Truthiness and `x or fallback`, `str(x)`, `int(text)` and `float(text)`
- Each surface's true and false text: `_str_to_bool`, pydantic's lax bool, `get_secret`'s `true` and `false`
- `os.environ/` references and `REDIS_<KWARG>` style environment layering

A boundary knows which surface a value came from. `general_settings.coordination_redis` is a pydantic model and `cache_params` goes through `_str_to_bool`, so the boundary applies that surface's rule exactly instead of refusing the text two surfaces disagree on

Each typed field keeps where its value came from (the key, `REDIS_<KWARG>` or the fallback name), so an error or warning downstream still names it

## What stays downstream

A crate that stands in for a Python library keeps that library's semantics, because they are its compatibility contract and not value coercion. `redis-resolve` stands in for redis-py, so it keeps

- Which target wins, and the defaults redis-py applies
- How `parse_url` reads a url, including `int()` on `?db=` and redis-py's `to_bool` on the TLS query parameters
- Which kwargs redis-py accepts, and whether the Rust client reads, warns about or refuses each one

Its input is typed. A rule it applies reads typed fields, never a `serde_json::Value` it has to coerce first. A value it only carries back to Python untouched, like `passthrough`, may stay untyped

## litellm-python-compat

The rule is about where config and kwargs get coerced, not about who may depend on the crate

- Its data formats (`pickle`, `json.dumps`, `repr`) are fine anywhere Rust reads or writes state Python also reads or writes, such as cache values in a shared Redis
- Its coercions (`truthy`, `number`) belong at the two boundaries, and in a crate that stands in for a Python library when that library calls them itself

## Existing code

Some code still coerces Python-shaped values past the boundary, for example the `REDIS_*` merge in `gateway/src/redis/layers.rs` and the flag and number readers in `redis-resolve/src/options.rs`. Do not add more. When you change one, move the coercion to its boundary instead of extending it in place

## Tests

Pin a boundary's coercion with named cases next to the Python function it mirrors, in the Rust test and in its Python twin, like the Redis contract cases in `redis-resolve/tests/interpret.rs` and `tests/unit/test_redis.py`. A downstream crate's tests build its typed input directly and never feed it raw text to coerce
