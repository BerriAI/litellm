use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use litellm_cost::money::UnitPrice;
use litellm_cost::price::{
    Band, CacheReadMode, Channel, InputUnit, Meter, OutputUnit, Price, Prices, ServiceTier,
    TokenKind, When as PriceWhen,
};
use litellm_cost::{Usage, When, calculate};
use time::OffsetDateTime;

fn bench(c: &mut Criterion) {
    let input = Meter::Input(InputUnit::Token(TokenKind::Text));
    let output = Meter::Output(OutputUnit::Token(TokenKind::Text));
    let cache = Meter::CacheRead {
        kind: TokenKind::Text,
        mode: CacheReadMode::Implicit,
    };
    let prices = Prices::new(
        [
            (input.clone(), Band::Base, 0.000002),
            (output.clone(), Band::Base, 0.000008),
            (cache.clone(), Band::Base, 0.0000005),
            (input.clone(), Band::over_prompt_tokens(1000), 0.000004),
        ]
        .into_iter()
        .map(|(meter, band, rate)| {
            (
                Price::online(
                    meter,
                    PriceWhen {
                        band,
                        ..PriceWhen::default()
                    },
                ),
                UnitPrice::new(rate).unwrap(),
            )
        })
        .collect(),
        None,
    );
    let usage = Usage {
        quantities: [(input, 750), (output, 200), (cache, 250)].into(),
        prompt_tokens: 1000,
        ..Usage::default()
    };
    let when = When {
        tier: ServiceTier::Standard,
        channel: Channel::Online,
        at: OffsetDateTime::UNIX_EPOCH,
    };
    c.bench_function("catalog_calculation", |b| {
        b.iter(|| {
            black_box(calculate(
                black_box(&prices),
                black_box(&usage),
                black_box(&when),
            ))
        })
    });
    let above = Usage {
        prompt_tokens: 1001,
        ..usage
    };
    c.bench_function("catalog_threshold_calculation", |b| {
        b.iter(|| {
            black_box(calculate(
                black_box(&prices),
                black_box(&above),
                black_box(&when),
            ))
        })
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
