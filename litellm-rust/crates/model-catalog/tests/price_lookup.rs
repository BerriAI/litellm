use litellm_cost::money::UnitPrice;
use litellm_cost::price::{
    CacheReadMode, CacheTtl, InputUnit, Meter, OutputUnit, Prices, ServiceTier, TokenKind,
};
use litellm_model_catalog::wire::{EntryWire, Fields};
use rstest::rstest;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    models: Vec<ReferenceModel>,
}

#[derive(Deserialize)]
struct ReferenceModel {
    info: Fields,
    inclusive: bool,
    checks: Vec<(Option<String>, u64, [f64; 5])>,
}

const METERS: [Meter; 5] = [
    Meter::Input(InputUnit::Token(TokenKind::Text)),
    Meter::Output(OutputUnit::Token(TokenKind::Text)),
    Meter::CacheWrite {
        kind: TokenKind::Text,
        ttl: CacheTtl::Default,
    },
    Meter::CacheWrite {
        kind: TokenKind::Text,
        ttl: CacheTtl::OneHour,
    },
    Meter::CacheRead {
        kind: TokenKind::Text,
        mode: CacheReadMode::Implicit,
    },
];

fn service_tier(name: Option<&str>) -> ServiceTier {
    match name {
        None => ServiceTier::Standard,
        Some("priority") => ServiceTier::Priority,
        Some("flex") => ServiceTier::Flex,
        Some("ultrafast") => ServiceTier::Ultrafast,
        Some("balanced") => ServiceTier::Balanced,
        Some(other) => panic!("the reference uses an unknown service tier {other}"),
    }
}

fn mismatch(
    prices: &Prices,
    (tier_name, prompt, python): &(Option<String>, u64, [f64; 5]),
) -> Option<String> {
    let tier = service_tier(tier_name.as_deref());
    let band = prices.band(tier, *prompt);
    let rust = METERS
        .each_ref()
        .map(|meter| prices.rate(meter, tier, band).map_or(0.0, UnitPrice::get));
    (rust != *python)
        .then(|| format!("tier={tier_name:?} prompt={prompt}: rust {rust:?}, python {python:?}"))
}

#[rstest]
fn band_and_rate_match_python_token_base_cost() {
    let reference: Reference =
        serde_json::from_str(include_str!("price_lookup_reference.json")).unwrap();

    let mismatches: Vec<String> = reference
        .models
        .iter()
        .flat_map(|model| {
            let provider = if model.inclusive { "xai" } else { "openai" };
            let fields: Fields = model
                .info
                .clone()
                .into_iter()
                .chain([("litellm_provider".to_owned(), provider.into())])
                .collect();
            let prices = Prices::try_from(&EntryWire::read(&fields)).unwrap();
            model
                .checks
                .iter()
                .filter_map(move |check| mismatch(&prices, check))
                .map(move |found| format!("{:?} {found}", model.info))
        })
        .collect();

    assert!(reference.models.len() > 50);
    assert!(mismatches.is_empty(), "{mismatches:#?}");
}
