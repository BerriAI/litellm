#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Connect(#[from] litellm_redis_client::ConnectError),
    #[error("Redis setting credential_provider names a Python object, which this host cannot supply")]
    NoHostCredentialProvider,
}
