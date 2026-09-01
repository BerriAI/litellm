//! Performance benchmarks for the AI gateway.
//!
//! These benchmarks measure the performance of critical paths in the gateway,
//! including caching, authentication, and request handling.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use litellm_ai_gateway::auth::hash_token;
use litellm_ai_gateway::caching::semantic_cache::{SemanticCache, SemanticCacheConfig};
use std::sync::Arc;

/// Benchmark semantic cache operations.
fn benchmark_semantic_cache(c: &mut Criterion) {
    let cache = Arc::new(SemanticCache::new(SemanticCacheConfig::default()));

    // Benchmark cache insert
    c.bench_function("semantic_cache_insert", |b| {
        let mut i = 0;
        b.iter(|| {
            let embedding = vec![i as f32; 128];
            cache.insert(
                format!("key_{}", i),
                embedding,
                format!("response_{}", i),
                None,
            );
            i += 1;
        });
    });

    // Benchmark cache get (hit)
    c.bench_function("semantic_cache_get_hit", |b| {
        let embedding = vec![1.0; 128];
        cache.insert(
            "hit_key".to_string(),
            embedding.clone(),
            "hit_response".to_string(),
            None,
        );

        b.iter(|| {
            black_box(cache.get(black_box(&embedding)));
        });
    });

    // Benchmark cache get (miss)
    c.bench_function("semantic_cache_get_miss", |b| {
        let embedding = vec![999.0; 128];

        b.iter(|| {
            black_box(cache.get(black_box(&embedding)));
        });
    });

    // Benchmark cache cleanup
    c.bench_function("semantic_cache_cleanup", |b| {
        b.iter(|| {
            cache.cleanup_expired();
            black_box(());
        });
    });
}

/// Benchmark token hashing.
fn benchmark_token_hashing(c: &mut Criterion) {
    c.bench_function("hash_token", |b| {
        b.iter(|| {
            black_box(hash_token(black_box("sk-test-token-12345")));
        });
    });
}

/// Benchmark request validation.
fn benchmark_request_parsing(c: &mut Criterion) {
    c.bench_function("parse_json_request", |b| {
        let request_str = r#"{
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100
        }"#;

        b.iter(|| {
            let _ = black_box(serde_json::from_str::<serde_json::Value>(black_box(
                request_str,
            )));
        });
    });
}

criterion_group!(
    benches,
    benchmark_semantic_cache,
    benchmark_token_hashing,
    benchmark_request_parsing,
);

criterion_main!(benches);
