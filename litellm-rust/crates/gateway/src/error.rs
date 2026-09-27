use litellm_cost::PricingError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("use_redis_transaction_buffer is set but neither REDIS_URL nor REDIS_HOST is")]
    NoRedis,
    #[error("connecting to the spend Redis failed")]
    Redis(#[from] redis::RedisError),
    #[error("model {model} has invalid custom pricing: {error:?}")]
    Pricing { model: String, error: PricingError },
}
