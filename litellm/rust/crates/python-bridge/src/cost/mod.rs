//! The pricing entry points Python calls over the live table: `query` for token calls, budget holds, rate cards and
//! adjustments, `kinds`, `fees` and `session` for the other call kinds

mod fees;
mod kinds;
mod query;
mod session;

pub(crate) use fees::price_fee;
pub(crate) use kinds::{NativeUnpriced, price_many, price_units};
pub(crate) use query::{
    NativeRateCard, RoutePricing, adjust, adjustment_config, bound, price, price_tools, rates,
};
pub(crate) use session::price_session;

use std::time::Duration;

use litellm_cost::price::ServiceTier;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The service tier a call bills, from the tier it asked for and the tier the provider reports serving it at
#[pyfunction]
#[pyo3(signature = (requested, served))]
pub(crate) fn billable_service_tier(
    requested: Option<&str>,
    served: Option<&str>,
) -> Option<String> {
    litellm_inference_messages::billable_tier(requested, served)
}

fn service_tier_named(name: Option<&str>) -> ServiceTier {
    match name.map(str::to_lowercase).as_deref() {
        Some("flex") => ServiceTier::Flex,
        Some("balanced") => ServiceTier::Balanced,
        Some("priority" | "fast") => ServiceTier::Priority,
        Some("ultrafast") => ServiceTier::Ultrafast,
        _ => ServiceTier::Standard,
    }
}

/// A measured length the core can hold, which rejects a negative or non-finite one the usage reported
pub(super) fn seconds(length: f64) -> PyResult<Duration> {
    Duration::try_from_secs_f64(length)
        .map_err(|error| PyValueError::new_err(format!("{length} seconds: {error}")))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::none(None, ServiceTier::Standard)]
    #[case::flex(Some("flex"), ServiceTier::Flex)]
    #[case::upper(Some("PRIORITY"), ServiceTier::Priority)]
    #[case::fast_is_priority(Some("fast"), ServiceTier::Priority)]
    #[case::ultrafast(Some("ultrafast"), ServiceTier::Ultrafast)]
    #[case::balanced(Some("balanced"), ServiceTier::Balanced)]
    #[case::auto_is_standard(Some("auto"), ServiceTier::Standard)]
    #[case::unknown_is_standard(Some("scale"), ServiceTier::Standard)]
    fn a_service_tier_name_picks_the_tier_main_priced_it_at(
        #[case] name: Option<&str>,
        #[case] expected: ServiceTier,
    ) {
        assert_eq!(service_tier_named(name), expected);
    }
}
