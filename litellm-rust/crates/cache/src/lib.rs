mod base_cache;
mod cache_type;
mod caching;
mod capabilities;
mod codec;
mod dual;
mod error;
pub mod semantic;

pub use base_cache::{
    BaseCache, BatchEntry, CacheConnectionResult, CacheConnectionStatus, CacheContext,
    ExactCacheContext, SemanticCacheContext,
};
pub use cache_type::CacheType;
pub use caching::{Cache, CacheBackend, get_cache, set_cache};
pub use capabilities::{
    BatchCache, BoundedCounterCache, BulkDeleteCache, CacheScript, ClaimCache, ClientInfoCache,
    ConnectionCache, CountReadCache, CounterCache, DeleteCache, DisconnectCache, FlushAllCache,
    FlushCache, IncrementOperation, Message, MessageStream, PingCache, PopOperation, PubSubCache,
    PushOperation, QueueCache, RefreshTtlCache, ScanCache, ScriptCache, SetCache, TtlCache,
    TtlPipelineCache,
};
pub use codec::{CacheCodec, JsonCodec};
pub use dual::{
    DEFAULT_DELETE_BATCH_SIZE, DualCache, ReadPolicy, RemoteFailurePolicy, WritePolicy,
};
pub use error::Error;
