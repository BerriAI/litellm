use std::sync::Arc;

use litellm_cache::{CacheCodec, CacheScript, Error, ScriptCache};

use crate::{
    cache::{RedisCache, namespaced_key},
    connection::Connections,
};

#[derive(Clone, Debug, PartialEq)]
pub enum RedisArg {
    Bytes(Vec<u8>),
    Integer(i64),
    Float(f64),
}

impl From<&str> for RedisArg {
    fn from(value: &str) -> Self {
        Self::Bytes(value.as_bytes().to_vec())
    }
}

impl From<String> for RedisArg {
    fn from(value: String) -> Self {
        Self::Bytes(value.into_bytes())
    }
}

impl From<Vec<u8>> for RedisArg {
    fn from(value: Vec<u8>) -> Self {
        Self::Bytes(value)
    }
}

impl From<i64> for RedisArg {
    fn from(value: i64) -> Self {
        Self::Integer(value)
    }
}

impl From<f64> for RedisArg {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

impl redis::ToRedisArgs for RedisArg {
    fn write_redis_args<W>(&self, out: &mut W)
    where
        W: ?Sized + redis::RedisWrite,
    {
        match self {
            Self::Bytes(value) => value.write_redis_args(out),
            Self::Integer(value) => value.write_redis_args(out),
            Self::Float(value) => value.write_redis_args(out),
        }
    }
}

pub struct RedisScript<C> {
    connections: Arc<Connections<C>>,
    namespace: Option<String>,
    source: String,
}

impl<C> CacheScript for RedisScript<C>
where
    C: redis::ConnectionLike + Send + 'static,
{
    type Argument = RedisArg;
    type Output = redis::Value;

    async fn invoke(
        &self,
        keys: Vec<String>,
        arguments: Vec<Self::Argument>,
    ) -> Result<Self::Output, Error> {
        let keys = keys
            .into_iter()
            .map(|key| namespaced_key(self.namespace.as_deref(), &key))
            .collect::<Vec<_>>();
        let source = self.source.clone();
        Connections::run_blocking(Arc::clone(&self.connections), move |connection| {
            eval(connection, &source, keys, arguments)
        })
        .await
    }
}

impl<S, C> RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    pub async fn async_eval(
        &self,
        script: String,
        keys: Vec<String>,
        arguments: Vec<RedisArg>,
    ) -> Result<redis::Value, Error> {
        let keys = self.namespaced_keys(&keys);
        self.run(move |connection| eval(connection, &script, keys, arguments))
            .await
    }
}

fn eval(
    connection: &mut impl redis::ConnectionLike,
    script: &str,
    keys: Vec<String>,
    arguments: Vec<RedisArg>,
) -> Result<redis::Value, Error> {
    redis::cmd("EVAL")
        .arg(script)
        .arg(keys.len())
        .arg(keys)
        .arg(arguments)
        .query(connection)
        .map_err(|_| Error::Unavailable)
}

impl<S, C> ScriptCache for RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    type Script = RedisScript<C>;

    fn async_register_script(&self, source: String) -> Self::Script {
        RedisScript {
            connections: Arc::clone(&self.connections),
            namespace: self.namespace.clone(),
            source,
        }
    }
}
