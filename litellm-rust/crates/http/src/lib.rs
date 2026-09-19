//! Rust counterpart of `litellm/llms/custom_httpx/http_handler.py`: the plain HTTP settings
//! LiteLLM exposes, their resolution into one typed client configuration, and a pool that
//! caches `reqwest::Client`s per resolved configuration.

mod config;
mod error;
mod pool;
mod settings;

pub use config::{HttpClientConfig, Verify};
pub use error::Error;
pub use pool::{ClientVariant, HttpClientPool};
pub use settings::{HttpSettings, SslVerify};
