use crate::call::CallContext;
use crate::money::UnitPrice;
use crate::price::{
    Band, Billing, CacheTtl, InputUnit, Meter, OffPeak, OffPeakRates, OutputUnit, Price, Prices,
    ServiceTier, TokenKind, When,
};
use crate::schedule::{Schedule, Window};

pub(crate) const BASE: Band = Band::Base;
pub(crate) const AUDIO_INPUT: Meter = Meter::Input(InputUnit::Token(TokenKind::Audio));
pub(crate) const IMAGE_OUTPUT: Meter = Meter::Output(OutputUnit::Token(TokenKind::Image));
pub(crate) const CACHE_WRITE_HOUR: Meter = Meter::CacheWrite {
    kind: TokenKind::Text,
    ttl: CacheTtl::OneHour,
};

pub(crate) fn over(tokens: u64) -> Band {
    Band::over_prompt_tokens(tokens)
}

pub(crate) fn prices(rates: &[(Meter, ServiceTier, Band, f64)]) -> Prices {
    prices_with(rates, None, Billing::default())
}

pub(crate) fn prices_with(
    rates: &[(Meter, ServiceTier, Band, f64)],
    off_peak: Option<OffPeak>,
    billing: Billing,
) -> Prices {
    Prices::new(
        rates
            .iter()
            .map(|(meter, tier, band, rate)| {
                (
                    Price::online(
                        meter.clone(),
                        When {
                            tier: *tier,
                            band: *band,
                        },
                    ),
                    UnitPrice::new(*rate).unwrap(),
                )
            })
            .collect(),
        off_peak,
        billing,
    )
}

pub(crate) fn off_peak_all_day(rates: OffPeakRates) -> Option<OffPeak> {
    OffPeak::new(
        Schedule {
            windows: vec![Window {
                hours: vec!["00:00-00:00".parse().unwrap()],
                weekdays: None,
            }],
            weekday_timezone: None,
        },
        rates,
    )
}

pub(crate) fn call() -> CallContext {
    tier_call(ServiceTier::Standard)
}

pub(crate) fn tier_call(tier: ServiceTier) -> CallContext {
    CallContext {
        tier,
        channel: crate::price::Channel::Online,
        residency: None,
        fedramp_endpoint: false,
        non_global_endpoint: false,
        speed: None,
        inference_geo: None,
        at: "2026-08-26T07:00:00Z".parse().unwrap(),
        overlay: None,
    }
}

pub(crate) fn quoted(
    prices: &Prices,
    call: &CallContext,
    prompt_tokens: u64,
    meter: &Meter,
) -> Option<(f64, Option<Meter>)> {
    prices
        .quote(call, prompt_tokens)
        .rate(meter)
        .map(|rate| (rate.unit_price.get(), rate.fallback_from))
}
