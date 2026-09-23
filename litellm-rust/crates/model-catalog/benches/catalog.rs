use criterion::{Criterion, criterion_group, criterion_main};
use litellm_model_catalog::{Catalog, Provenance};
use std::hint::black_box;

fn benchmarks(c: &mut Criterion) {
    let body = include_bytes!("../../../../model_prices_and_context_window.json");
    c.bench_function("parse_current_catalog", |b| {
        b.iter(|| Catalog::parse(black_box(body), Provenance::default()).unwrap())
    });
    let catalog = Catalog::parse(body, Provenance::default()).unwrap();
    let key = catalog
        .model_names()
        .next()
        .expect("catalog must have a benchmark key");
    c.bench_function("lookup_catalog_key", |b| {
        b.iter(|| black_box(&catalog).lookup(black_box(key)))
    });
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
