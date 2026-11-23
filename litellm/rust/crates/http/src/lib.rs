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
mod phases;
mod pool;
mod proxy;
pub mod request;
pub mod requests_env;
mod settings;
mod tls;
pub mod websocket;

pub use client::Client;
pub use config::{ClientIdentity, HttpClientConfig, Resolution, Verify};
pub use error::{BodyCut, Error, TlsSource};
pub use phases::Phases;
pub use pool::{ClientVariant, HttpClientPool};
pub use proxy::EnvironmentProxies;
pub use settings::{HttpSettings, HttpSettingsLayer, SslVerify, TcpKeepalive};
pub use tls::{KeyExchangeGroup, Tls12CipherSuite, Unsupported};
