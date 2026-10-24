#![forbid(unsafe_code)]

mod proxy;
mod settings;

pub use proxy::EnvironmentProxies;
pub use settings::{HttpSettings, HttpSettingsLayer, SslVerify, TcpKeepalive};
