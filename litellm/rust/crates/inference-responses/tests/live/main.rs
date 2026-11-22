//! Real provider calls, one module per provider. Each test skips when its provider's key is not
//! in the environment, so `cargo test` stays offline by default.

#[path = "../support/mod.rs"]
mod support;

mod openrouter;
