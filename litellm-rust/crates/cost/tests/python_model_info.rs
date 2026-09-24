#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/test_utils.py::test_get_model_info_* (get_model_info key resolution)
// expected values recorded from litellm/utils.py::_get_model_info_helper and
// litellm/litellm_core_utils/get_llm_provider_logic.py::get_llm_provider on this catalog

use std::collections::HashMap;

use litellm_cost::catalog::ModelInfoCatalog;
use litellm_cost::error::CostError;
use rstest::rstest;
use serde_json::{Value, json};

fn catalog() -> ModelInfoCatalog {
    let provider = |name: &str| json!({"litellm_provider": name});
    ModelInfoCatalog::new(HashMap::from([
        ("openai/gpt-zz".to_owned(), provider("openai")),
        ("gpt-zz".to_owned(), provider("openai")),
        ("gemini/gemini-zz".to_owned(), provider("gemini")),
        (
            "gemini-zz-pro".to_owned(),
            provider("vertex_ai-language-models"),
        ),
        ("claude-zz".to_owned(), provider("anthropic")),
        ("ft:gpt-yy".to_owned(), provider("openai")),
        (
            "vertex_ai/meta/llama-zz".to_owned(),
            provider("vertex_ai-llama_models"),
        ),
        (
            "vertex_ai/mistral-zz@latest".to_owned(),
            provider("vertex_ai-mistral_models"),
        ),
        (
            "anthropic.claude-zz-v1".to_owned(),
            provider("bedrock_converse"),
        ),
        (
            "bedrock_mantle/model-zz".to_owned(),
            provider("bedrock_mantle"),
        ),
        (
            "fireworks_ai/accounts/fireworks/models/fw-zz".to_owned(),
            provider("fireworks_ai"),
        ),
        ("gpt-4.1".to_owned(), provider("openai")),
        ("Mixed-Case-ZZ".to_owned(), provider("together_ai")),
        (
            "fallback_generalizations".to_owned(),
            json!({"rules": [
                {"name": "zz-route", "pattern": "^zzroute-", "model_info": {"litellm_provider": "anthropic"}},
                {"name": "zz-cap", "pattern": "zzcap-\\d+", "model_info": {"mode": "chat", "max_tokens": 10}},
                {"name": "zz-cap2", "pattern": "zzcap-9", "model_info": {"max_tokens": 20}},
                {
                    "name": "zz-legacy",
                    "pattern": "^legacy-zz",
                    "extends": "zz-cap",
                    "model_info": {"litellm_provider": "xai", "max_tokens": 30}
                }
            ]}),
        ),
    ]))
}

type Resolved = Result<(&'static str, Option<&'static str>, Option<u64>), CostError>;

#[rstest]
#[case::provider_prefixed_key_first("gpt-zz", Some("openai"), Ok(("openai/gpt-zz", Some("openai"), None)))]
#[case::bare_model_without_provider("gpt-zz", None, Ok(("gpt-zz", Some("openai"), None)))]
#[case::prefixed_model_without_provider("openai/gpt-zz", None, Ok(("openai/gpt-zz", Some("openai"), None)))]
#[case::case_insensitive_key("GPT-ZZ", Some("openai"), Ok(("openai/gpt-zz", Some("openai"), None)))]
#[case::dated_snapshot_suffix("gpt-zz-2025-01-01", Some("openai"), Ok(("openai/gpt-zz", Some("openai"), None)))]
#[case::dated_snapshot_without_provider("gpt-zz-2025-01-01", None, Ok(("gpt-zz", Some("openai"), None)))]
#[case::stable_version_suffix("gemini-zz-001", Some("gemini"), Ok(("gemini/gemini-zz", Some("gemini"), None)))]
#[case::vertex_family_entry(
    "gemini-zz-pro",
    Some("vertex_ai"),
    Ok(("gemini-zz-pro", Some("vertex_ai-language-models"), None))
)]
#[case::mismatched_provider_is_skipped(
    "gemini/gemini-zz",
    Some("vertex_ai"),
    Err(CostError::ModelNotFound)
)]
#[case::openai_finetune("ft:gpt-yy:org:suffix:id", Some("openai"), Ok(("ft:gpt-yy", Some("openai"), None)))]
#[case::vertex_llama_alias(
    "llama-zz",
    Some("vertex_ai"),
    Ok(("vertex_ai/meta/llama-zz", Some("vertex_ai-llama_models"), None))
)]
#[case::vertex_beta_latest_alias(
    "mistral-zz",
    Some("vertex_ai_beta"),
    Ok(("vertex_ai/mistral-zz@latest", Some("vertex_ai-mistral_models"), None))
)]
#[case::bedrock_cross_region_keeps_the_version(
    "us.anthropic.claude-zz-v1:0",
    Some("bedrock"),
    Err(CostError::ModelNotFound)
)]
#[case::bedrock_routing_prefix(
    "bedrock/converse/anthropic.claude-zz-v1",
    Some("bedrock"),
    Ok(("anthropic.claude-zz-v1", Some("bedrock_converse"), None))
)]
#[case::bedrock_throughput_suffix_keeps_the_version(
    "anthropic.claude-zz-v1:0:51k",
    Some("bedrock_converse"),
    Err(CostError::ModelNotFound)
)]
#[case::mantle_region_prefix(
    "bedrock_mantle/us-east-1/model-zz",
    Some("bedrock_mantle"),
    Ok(("bedrock_mantle/model-zz", Some("bedrock_mantle"), None))
)]
#[case::fireworks_resource_key(
    "fw-zz",
    Some("fireworks_ai"),
    Ok(("fireworks_ai/accounts/fireworks/models/fw-zz", Some("fireworks_ai"), None))
)]
#[case::azure_alias("azure/gpt-41", None, Ok(("gpt-4.1", Some("openai"), None)))]
#[case::lowercase_query_finds_mixed_case_key(
    "mixed-case-zz",
    Some("together_ai"),
    Ok(("Mixed-Case-ZZ", Some("together_ai"), None))
)]
#[case::huggingface_is_zero_rated("any-model", Some("huggingface"), Ok(("any-model", Some("huggingface"), None)))]
#[case::other_provider_entry("claude-zz", Some("openai"), Err(CostError::ModelNotFound))]
#[case::capability_union_later_rule_wins("zzcap-9", None, Ok(("zzcap-9", None, Some(20))))]
#[case::capability_backfills_the_provider("zzcap-9", Some("openai"), Ok(("openai/zzcap-9", Some("openai"), Some(20))))]
#[case::routing_rule_alone_has_no_capabilities("zzroute-x", None, Err(CostError::ModelNotFound))]
#[case::legacy_rule_routes_and_extends("legacy-zz-1", None, Ok(("legacy-zz-1", Some("xai"), Some(30))))]
#[case::unmapped("nothing-matches", None, Err(CostError::ModelNotFound))]
fn get_model_info_resolves_keys_like_python(
    #[case] model: &str,
    #[case] provider: Option<&str>,
    #[case] expected: Resolved,
) {
    let catalog = catalog();
    let actual = catalog.get_model_info(model, provider).map(|resolved| {
        (
            resolved.key.into_owned(),
            resolved
                .info
                .get("litellm_provider")
                .and_then(Value::as_str)
                .map(str::to_owned),
            resolved.info.get("max_tokens").and_then(Value::as_u64),
        )
    });
    assert_eq!(
        actual,
        expected.map(|(key, provider, max_tokens)| {
            (key.to_owned(), provider.map(str::to_owned), max_tokens)
        })
    );
}

#[rstest]
fn get_model_info_prices_huggingface_at_zero() {
    let catalog = catalog();
    let info = catalog
        .get_model_info("any-model", Some("huggingface"))
        .unwrap()
        .info;
    assert_eq!(info.get("input_cost_per_token"), Some(&json!(0)));
    assert_eq!(info.get("output_cost_per_token"), Some(&json!(0)));
}

#[rstest]
fn capability_rules_lose_to_an_exact_key_owned_by_another_provider() {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        ("zzcap-1".to_owned(), json!({"litellm_provider": "gemini"})),
        (
            "fallback_generalizations".to_owned(),
            json!({"rules": [{"name": "cap", "pattern": "zzcap", "model_info": {"mode": "chat"}}]}),
        ),
    ]));
    assert_eq!(
        catalog
            .get_model_info("zzcap-1", Some("openai"))
            .map(|info| info.key.into_owned()),
        Err(CostError::ModelNotFound)
    );
    assert_eq!(
        catalog
            .get_model_info("zzcap-2", Some("openai"))
            .map(|info| info.key.into_owned()),
        Ok("openai/zzcap-2".to_owned())
    );
}

#[rstest]
#[case::cataloged_bare("gpt-zz", Some("openai"))]
#[case::provider_prefix("gemini/anything", Some("gemini"))]
#[case::vertex_family("gemini-zz-pro", Some("vertex_ai"))]
#[case::bedrock_converse_family("anthropic.claude-zz-v1", Some("bedrock"))]
#[case::routing_rule("zzroute-x", Some("anthropic"))]
#[case::legacy_routing_rule("legacy-zz-1", Some("xai"))]
#[case::azure_prefix_stays_azure("azure/gpt-41", Some("azure"))]
#[case::json_registry_provider("gmi/any", Some("gmi"))]
#[case::static_cohere_embedding("embed-v4.0", Some("cohere"))]
#[case::static_openai_image("dall-e-3", Some("openai"))]
#[case::unroutable("GPT-ZZ", None)]
fn get_llm_provider_infers_like_python(#[case] model: &str, #[case] expected: Option<&str>) {
    assert_eq!(
        catalog()
            .get_llm_provider(model)
            .map(|resolved| resolved.custom_llm_provider)
            .as_deref(),
        expected
    );
}

fn usage() -> litellm_cost::responses_usage::ChatUsage {
    litellm_cost::usage_dispatch::get_usage_object(
        &json!({"usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}}),
    )
    .unwrap()
    .unwrap()
}

// expected values recorded from litellm.completion_cost for unmapped ids the shipped rules generalize
#[rstest]
#[case::routed_and_generalized("claude-newfam-9", None, Ok((0.0, 0.0)))]
#[case::generalized_under_explicit_provider("claude-newfam-9", Some("anthropic"), Ok((0.0, 0.0)))]
#[case::generalized_bedrock_id("anthropic.claude-newfam-9", Some("bedrock"), Ok((0.0, 0.0)))]
#[case::unroutable("totally-unknown", None, Err(CostError::MissingProvider))]
#[case::routable_but_unmapped("zzroute-x", None, Err(CostError::ModelNotFound))]
fn unmapped_models_price_through_capability_rules(
    #[case] model: &str,
    #[case] provider: Option<&str>,
    #[case] expected: Result<(f64, f64), CostError>,
) {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "fallback_generalizations".to_owned(),
        json!({"rules": [
            {"name": "route-bedrock", "pattern": "^(?:[a-z-]+\\.)?anthropic\\.claude-", "model_info": {"litellm_provider": "bedrock"}},
            {"name": "route", "pattern": "^claude-[a-z]+-\\d+$", "model_info": {"litellm_provider": "anthropic"}},
            {"name": "route-only", "pattern": "^zzroute-", "model_info": {"litellm_provider": "anthropic"}},
            {"name": "baseline", "pattern": "claude-[a-z]+-\\d+", "model_info": {"mode": "chat"}}
        ]}),
    )]));
    let usage = usage();
    assert_eq!(
        litellm_cost::cost_calculator::cost_per_token(
            &catalog,
            litellm_cost::catalog::ModelCostRequest {
                model,
                provider,
                region: None,
                usage: &usage,
                service_tier: None,
                data_residency: None,
                vertex_location: None,
                at: "2026-01-01T12:00Z".parse().unwrap(),
                response_time_ms: None,
            },
        ),
        expected
    );
}
