mod cache;
mod topology;

pub mod connection {
    pub use crate::cache::{ConnectionRef, Connections, ttl_seconds};
}

pub use cache::{
    RedisArg, RedisCache, RedisLpopOperation, RedisLpopResult, RedisRpushOperation, RedisScript,
};
pub use topology::{RedisNode, RedisTopology};
