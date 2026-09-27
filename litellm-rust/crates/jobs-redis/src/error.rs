#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Redis lease command failed")]
    Redis(#[from] redis::RedisError),
}
