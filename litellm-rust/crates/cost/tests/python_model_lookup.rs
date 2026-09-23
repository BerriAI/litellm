use std::collections::HashMap;

use litellm_cost::catalog::{CatalogError, CostCatalog};
use litellm_cost::{PromptConvention, Rate, Rates, Request, ServiceTier, ThresholdPolicy, Usage};
use rstest::rstest;

fn rates(input: f64, output: f64) -> Rates {
    Rates {
        input: Rate::Value(input),
        output: Rate::Value(output),
        ..Rates::EMPTY
    }
}

fn catalog() -> CostCatalog {
    CostCatalog::new(HashMap::from([
        ("openai/model".to_owned(), rates(1e-6, 2e-6)),
        ("openai/openai/model".to_owned(), rates(9e-6, 9e-6)),
        ("bedrock_mantle/model".to_owned(), rates(3e-6, 4e-6)),
        (
            "bedrock_mantle/us-gov-west-1/model".to_owned(),
            rates(5e-6, 6e-6),
        ),
        ("bare-model".to_owned(), rates(7e-6, 8e-6)),
    ]))
}

fn request() -> Request {
    Request {
        usage: Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            cache_write_5m_tokens: None,
            cache_write_1h_tokens: None,
            prompt_convention: PromptConvention::IncludesCache,
        },
        service_tier: ServiceTier::Standard,
        threshold_policy: ThresholdPolicy::Exclusive,
        region_multiplier: None,
        billed_at_utc_minute: None,
    }
}

#[rstest]
#[case("openai/openai/model", Some("openai"), None, Some("openai/model"))]
#[case("model", Some("openai"), None, Some("openai/model"))]
#[case("openai/model", None, None, Some("openai/model"))]
#[case(
    "bedrock_mantle/model",
    Some("bedrock_mantle"),
    Some("us-gov-west-1"),
    Some("bedrock_mantle/us-gov-west-1/model")
)]
#[case(
    "bedrock_mantle/model",
    Some("bedrock_mantle"),
    Some("missing"),
    Some("bedrock_mantle/model")
)]
#[case(
    "bedrock_mantle/bare-model",
    Some("bedrock_mantle"),
    None,
    Some("bare-model")
)]
#[case("missing", Some("openai"), None, None)]
fn cost_per_token_resolves_model_key_in_python_order(
    #[case] model: &str,
    #[case] provider: Option<&str>,
    #[case] region: Option<&str>,
    #[case] expected: Option<&str>,
) {
    assert_eq!(
        catalog().select_model_key(model, provider, region),
        expected
    );
}

#[rstest]
fn cost_per_token_uses_the_selected_regional_prices() {
    let cost = catalog()
        .cost_per_token(
            "bedrock_mantle/model",
            Some("bedrock_mantle"),
            Some("us-gov-west-1"),
            &request(),
        )
        .unwrap();
    assert!((cost.input() - 100.0 * 5e-6).abs() < 1e-12);
    assert!((cost.output() - 50.0 * 6e-6).abs() < 1e-12);
}

#[rstest]
fn cost_per_token_reports_an_unmapped_model() {
    assert_eq!(
        catalog().cost_per_token("missing", Some("openai"), None, &request()),
        Err(CatalogError::ModelNotFound)
    );
}
