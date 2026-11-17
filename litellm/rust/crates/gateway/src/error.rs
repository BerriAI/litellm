#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Http(#[from] litellm_http::Error),
    #[error(transparent)]
    Config(#[from] litellm_config_load::Error),
    #[error(transparent)]
    Redis(#[from] litellm_redis_resolve::Error),
    #[error("Redis setting {setting} is not plain data")]
    RedisNotPlainData { setting: String },
    #[error(transparent)]
    RedisConnect(#[from] litellm_redis_connect::Error),
    #[error(transparent)]
    ModelCatalog(#[from] litellm_model_catalog::Error),
}
