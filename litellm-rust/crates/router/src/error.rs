#[derive(Debug, PartialEq, thiserror::Error)]
pub enum Error {
    #[error(
        "router_settings.routing_strategy {0:?} is not supported by the Rust gateway, use simple-shuffle or least-busy"
    )]
    UnsupportedStrategy(String),
    #[error("{setting} must be a non-negative number of seconds, got {value:?}")]
    InvalidSeconds {
        setting: &'static str,
        value: String,
    },
}
