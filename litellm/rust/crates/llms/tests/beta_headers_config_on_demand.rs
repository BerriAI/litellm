use std::sync::atomic::{AtomicUsize, Ordering};

use litellm_llms::anthropic::beta_headers::{
    BetaPolicy, accepted_or_unlisted, on_beta_headers_config_needed,
};
use litellm_llms_types::providers::anthropic::{BetaProvider, BetaSet};

static LOAD_REQUESTS: AtomicUsize = AtomicUsize::new(0);

fn count_load_request() {
    LOAD_REQUESTS.fetch_add(1, Ordering::SeqCst);
}

fn headers(beta: Option<&str>) -> Vec<(String, String)> {
    let plain = ("content-type".to_string(), "application/json".to_string());
    match beta {
        Some(beta) => vec![plain, ("anthropic-beta".to_string(), beta.to_string())],
        None => vec![plain],
    }
}

#[test]
fn only_a_filter_with_a_beta_asks_for_the_unloaded_map() {
    on_beta_headers_config_needed(count_load_request);

    let unfiltered = BetaPolicy::Filter(BetaProvider::AzureAi).apply(headers(None));
    let foundry_unfiltered = accepted_or_unlisted(BetaSet::default(), BetaProvider::AzureAi);
    assert_eq!(unfiltered, headers(None));
    assert!(foundry_unfiltered.is_empty());
    assert_eq!(LOAD_REQUESTS.load(Ordering::SeqCst), 0);

    let _ = BetaPolicy::Filter(BetaProvider::Bedrock).apply(headers(Some("effort-2025-11-24")));
    assert!(LOAD_REQUESTS.load(Ordering::SeqCst) >= 1);
}
