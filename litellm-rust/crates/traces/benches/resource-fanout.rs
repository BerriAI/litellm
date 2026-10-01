use std::{collections::BTreeMap, hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use litellm_traces::Shared;

fn fanout<T: Clone>(resource: &T, spans: usize) -> Vec<T> {
    (0..spans).map(|_| resource.clone()).collect()
}

fn resource_fanout(c: &mut Criterion) {
    let mut group = c.benchmark_group("resource_fanout");
    for (attribute_bytes, spans) in [(256, 1), (256, 64), (8192, 1024), (16384, 1024)] {
        let attributes = BTreeMap::from([
            ("service.name".to_owned(), "benchmark".to_owned()),
            ("payload".to_owned(), "x".repeat(attribute_bytes)),
        ]);
        let owned = Box::new(attributes.clone());
        let shared = Shared::new(attributes);
        let case = format!("{attribute_bytes}B_{spans}_spans");
        group.throughput(Throughput::Elements(spans as u64));
        group.bench_with_input(BenchmarkId::new("owned", &case), &owned, |b, resource| {
            b.iter(|| black_box(fanout(black_box(resource), spans)));
        });
        group.bench_with_input(BenchmarkId::new("shared", &case), &shared, |b, resource| {
            b.iter(|| black_box(fanout(black_box(resource), spans)));
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    targets = resource_fanout
}
criterion_main!(benches);
