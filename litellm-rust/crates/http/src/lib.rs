#![allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "this crate is the one place reqwest clients are built"
)]

mod client;
mod config;
mod error;
pub mod media;
pub mod outbound;
mod pool;
mod proxy;
pub mod request;
mod settings;
mod tls;
pub mod transport;

pub use client::Client;
pub use config::{ClientIdentity, HttpClientConfig, Resolution, Verify};
pub use error::{Error, TlsSource};
pub use pool::{ClientVariant, HttpClientPool};
pub use proxy::EnvironmentProxies;
pub use settings::{HttpSettings, HttpSettingsLayer, SslVerify, TcpKeepalive};
pub use tls::{KeyExchangeGroup, Tls12CipherSuite, Unsupported};
