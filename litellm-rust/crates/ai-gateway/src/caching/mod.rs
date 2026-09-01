//! Caching strategies for the AI gateway.
//!
//! This module provides various caching strategies to improve performance
//! and reduce latency for repeated or similar requests.

pub mod semantic_cache;

pub use semantic_cache::{CacheStats, SemanticCache, SemanticCacheConfig};
