#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Http(#[from] litellm_http::Error),
    #[error(transparent)]
    Config(#[from] litellm_config_load::Error),
    #[error(transparent)]
    Redis(#[from] litellm_redis_resolve::Error),
    #[error(transparent)]
    RedisConnect(#[from] litellm_redis_connect::Error),
    #[error(transparent)]
    Database(#[from] litellm_db_resolve::Error),
    #[error(transparent)]
    DatabasePool(#[from] litellm_db::Error),
    #[error(transparent)]
    ModelCatalog(#[from] litellm_model_catalog::Error),
    #[error(transparent)]
    RouterConfig(#[from] litellm_router_core::ConfigError),
    #[error(transparent)]
    Router(#[from] litellm_router_core::Error),
    #[error(transparent)]
    Callback(#[from] crate::callbacks::Unregistrable),
}
