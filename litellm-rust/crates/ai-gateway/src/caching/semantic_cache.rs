//! Semantic caching using embeddings for similarity-based cache lookups.
//!
//! Instead of exact match, uses semantic similarity to find cached responses
//! for similar requests. This improves cache hit rates for similar but not
//! identical requests.

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

/// A cached entry with its embedding and response.
#[derive(Clone)]
pub struct SemanticCacheEntry {
    /// The embedding vector for the request.
    pub embedding: Vec<f32>,
    /// The cached response.
    pub response: String,
    /// Timestamp when the entry was cached.
    pub cached_at: std::time::Instant,
    /// Number of times this entry has been accessed.
    pub access_count: u64,
    /// Time-to-live in seconds.
    pub ttl_secs: u64,
}

/// Semantic cache configuration.
#[derive(Debug, Clone)]
pub struct SemanticCacheConfig {
    /// Maximum number of entries in the cache.
    pub max_entries: usize,
    /// Similarity threshold for cache hits (0.0 to 1.0).
    pub similarity_threshold: f32,
    /// Default TTL for cache entries in seconds.
    pub default_ttl_secs: u64,
    /// Whether to enable cache warming.
    pub enable_warming: bool,
}

impl Default for SemanticCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10000,
            similarity_threshold: 0.85,
            default_ttl_secs: 3600,
            enable_warming: true,
        }
    }
}

/// Semantic cache for caching similar requests.
pub struct SemanticCache {
    /// The cache entries.
    entries: Arc<RwLock<HashMap<String, SemanticCacheEntry>>>,
    /// Configuration.
    config: SemanticCacheConfig,
    /// Cache statistics.
    stats: Arc<RwLock<CacheStats>>,
}

/// Cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total number of cache hits.
    pub hits: u64,
    /// Total number of cache misses.
    pub misses: u64,
    /// Total number of cache evictions.
    pub evictions: u64,
    /// Total number of cache inserts.
    pub inserts: u64,
}

impl SemanticCache {
    /// Create a new semantic cache.
    pub fn new(config: SemanticCacheConfig) -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            config,
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    /// Calculate cosine similarity between two vectors.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }

    /// Get a cached response for a request embedding.
    pub fn get(&self, embedding: &[f32]) -> Option<String> {
        let entries = self.entries.read();
        
        // Find the most similar entry above the threshold.
        let mut best_match = None;
        let mut best_similarity = self.config.similarity_threshold;

        for (key, entry) in entries.iter() {
            // Check if entry has expired.
            if entry.cached_at.elapsed().as_secs() > entry.ttl_secs {
                continue;
            }

            let similarity = Self::cosine_similarity(embedding, &entry.embedding);
            if similarity > best_similarity {
                best_similarity = similarity;
                best_match = Some(key.clone());
            }
        }

        if let Some(key) = best_match {
            let mut stats = self.stats.write();
            stats.hits += 1;
            
            // Update access count.
            if let Some(entry) = self.entries.write().get_mut(&key) {
                entry.access_count += 1;
            }
            
            entries.get(&key).map(|e| e.response.clone())
        } else {
            let mut stats = self.stats.write();
            stats.misses += 1;
            None
        }
    }

    /// Insert a new entry into the cache.
    pub fn insert(&self, key: String, embedding: Vec<f32>, response: String, ttl_secs: Option<u64>) {
        let mut entries = self.entries.write();
        let mut stats = self.stats.write();

        // Evict entries if cache is full.
        while entries.len() >= self.config.max_entries {
            // Evict the least recently accessed entry.
            if let Some(oldest_key) = entries
                .iter()
                .min_by_key(|(_, entry)| entry.access_count)
                .map(|(key, _)| key.clone())
            {
                entries.remove(&oldest_key);
                stats.evictions += 1;
            } else {
                break;
            }
        }

        let entry = SemanticCacheEntry {
            embedding,
            response,
            cached_at: std::time::Instant::now(),
            access_count: 0,
            ttl_secs: ttl_secs.unwrap_or(self.config.default_ttl_secs),
        };

        entries.insert(key, entry);
        stats.inserts += 1;
    }

    /// Remove expired entries from the cache.
    pub fn cleanup_expired(&self) {
        let mut entries = self.entries.write();
        let mut stats = self.stats.write();

        let expired_keys: Vec<String> = entries
            .iter()
            .filter(|(_, entry)| entry.cached_at.elapsed().as_secs() > entry.ttl_secs)
            .map(|(key, _)| key.clone())
            .collect();

        for key in expired_keys {
            entries.remove(&key);
            stats.evictions += 1;
        }
    }

    /// Get cache statistics.
    pub fn get_stats(&self) -> CacheStats {
        self.stats.read().clone()
    }

    /// Get the number of entries in the cache.
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Clear all entries from the cache.
    pub fn clear(&self) {
        self.entries.write().clear();
    }

    /// Warm the cache with common requests.
    pub fn warm(&self, requests: Vec<(String, Vec<f32>, String)>) {
        if !self.config.enable_warming {
            return;
        }

        for (key, embedding, response) in requests {
            self.insert(key, embedding, response, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(SemanticCache::cosine_similarity(&a, &b), 1.0);

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert_eq!(SemanticCache::cosine_similarity(&a, &b), 0.0);

        let a = vec![1.0, 1.0, 0.0];
        let b = vec![1.0, 1.0, 0.0];
        assert_eq!(SemanticCache::cosine_similarity(&a, &b), 1.0);
    }

    #[test]
    fn test_cache_insert_and_get() {
        let cache = SemanticCache::new(SemanticCacheConfig::default());
        let embedding = vec![1.0, 0.0, 0.0];
        
        cache.insert("key1".to_string(), embedding.clone(), "response1".to_string(), None);
        
        let result = cache.get(&embedding);
        assert_eq!(result, Some("response1".to_string()));
    }

    #[test]
    fn test_cache_miss() {
        let cache = SemanticCache::new(SemanticCacheConfig::default());
        let embedding = vec![1.0, 0.0, 0.0];
        
        let result = cache.get(&embedding);
        assert_eq!(result, None);
    }

    #[test]
    fn test_cache_eviction() {
        let config = SemanticCacheConfig {
            max_entries: 2,
            ..Default::default()
        };
        let cache = SemanticCache::new(config);
        
        cache.insert("key1".to_string(), vec![1.0, 0.0, 0.0], "response1".to_string(), None);
        cache.insert("key2".to_string(), vec![0.0, 1.0, 0.0], "response2".to_string(), None);
        cache.insert("key3".to_string(), vec![0.0, 0.0, 1.0], "response3".to_string(), None);
        
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_cache_stats() {
        let cache = SemanticCache::new(SemanticCacheConfig::default());
        let embedding = vec![1.0, 0.0, 0.0];
        
        cache.insert("key1".to_string(), embedding.clone(), "response1".to_string(), None);
        
        let _ = cache.get(&embedding); // Hit
        let _ = cache.get(&vec![0.0, 1.0, 0.0]); // Miss
        
        let stats = cache.get_stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.inserts, 1);
    }

    #[test]
    fn test_cache_warming() {
        let cache = SemanticCache::new(SemanticCacheConfig::default());
        
        let requests = vec![
            ("key1".to_string(), vec![1.0, 0.0, 0.0], "response1".to_string()),
            ("key2".to_string(), vec![0.0, 1.0, 0.0], "response2".to_string()),
        ];
        
        cache.warm(requests);
        
        assert_eq!(cache.len(), 2);
    }
}
