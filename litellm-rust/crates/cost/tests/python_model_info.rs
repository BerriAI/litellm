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
            "vertex_ai/jamba-zz@latest".to_owned(),
            provider("vertex_ai-ai21_models"),
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
        ("amazon.zz-model".to_owned(), provider("bedrock")),
        (
            "zz-guardrail".to_owned(),
            json!({"litellm_provider": "bedrock", "mode": "guardrail"}),
        ),
        ("Mixed-Case-ZZ".to_owned(), provider("together_ai")),
        (
            "fallback_generalizations".to_owned(),
            json!({"rules": [
                {"name": "zz-route", "pattern": "^zzroute-", "model_info": {"litellm_provider": "anthropic"}},
                {"name": "zz-cap", "pattern": "zzcap-\\d+", "model_info": {"mode": "chat", "max_tokens": 10}},
                {"name": "zz-cap2", "pattern": "zzcap-9", "model_info": {"max_tokens": 20}},
                {
                    "name": "zz-bad",
                    "pattern": "^zzbad-",
                    "model_info": {"mode": "chat"},
                    "fill_missing_for_providers": "openai"
                },
                {
                    "name": "zz-good",
                    "pattern": "^zzgood-",
                    "model_info": {"mode": "chat", "max_tokens": 5},
                    "fill_missing_for_providers": ["openai"]
                },
                {
                    "name": "zz-legacy",
                    "pattern": "^legacy-zz",
                    "extends": "zz-cap",
                    "model_info": {"litellm_provider": "xai", "max_tokens": 30}
                },
                {
                    "name": "zz-legacy-inherits",
                    "pattern": "^legacy-yy",
                    "extends": "zz-cap",
                    "model_info": {"litellm_provider": "xai"}
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
#[case::vertex_ai21_latest_alias(
    "jamba-zz",
    Some("vertex_ai"),
    Ok(("vertex_ai/jamba-zz@latest", Some("vertex_ai-ai21_models"), None))
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
#[case::mantle_keeps_a_non_region_segment(
    "bedrock_mantle/team/model-zz",
    Some("bedrock_mantle"),
    Err(CostError::ModelNotFound)
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
#[case::legacy_rule_inherits_parent_capabilities("legacy-yy-1", None, Ok(("legacy-yy-1", Some("xai"), Some(10))))]
#[case::malformed_fill_missing_skips_the_rule("zzbad-1", None, Err(CostError::ModelNotFound))]
#[case::listed_fill_missing_keeps_the_rule("zzgood-1", None, Ok(("zzgood-1", None, Some(5))))]
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
#[case::finetune_pricing_key_is_not_a_chat_model("ft:gpt-yy", None)]
#[case::bedrock_family("amazon.zz-model", Some("bedrock"))]
#[case::bedrock_guardrail_is_not_a_model("zz-guardrail", None)]
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

fn provider_list_catalog() -> ModelInfoCatalog {
    let providers = [
        "openai",
        "text-completion-openai",
        "cohere",
        "cohere_chat",
        "mistral",
        "anthropic",
        "empower",
        "openrouter",
        "vertex_ai-text-models",
        "vertex_ai-code-text-models",
        "vertex_ai-language-models",
        "vertex_ai-vision-models",
        "vertex_ai-chat-models",
        "vertex_ai-code-chat-models",
        "vertex_ai-embedding-models",
        "vertex_ai-llama_models",
        "vertex_ai-mistral_models",
        "vertex_ai-ai21_models",
        "vertex_ai-image-models",
        "vertex_ai-video-models",
        "nlp_cloud",
        "aleph_alpha",
        "bedrock",
        "bedrock_converse",
        "watsonx",
        "gradient_ai",
        "xai",
    ];
    let entries = providers
        .into_iter()
        .map(|provider| {
            (
                format!("zz-{provider}"),
                json!({"litellm_provider": provider}),
            )
        })
        .chain([
            (
                "vertex_ai/zz-vimg".to_owned(),
                json!({"litellm_provider": "vertex_ai-image-models"}),
            ),
            (
                "zz-ai21-chat".to_owned(),
                json!({"litellm_provider": "ai21", "mode": "chat"}),
            ),
            (
                "zz-ai21-text".to_owned(),
                json!({"litellm_provider": "ai21", "mode": "completion"}),
            ),
            (
                "claude-2".to_owned(),
                json!({"litellm_provider": "anthropic"}),
            ),
            (
                "zz-colon:1".to_owned(),
                json!({"litellm_provider": "bedrock"}),
            ),
            (
                format!("zz:{}", "a".repeat(61)),
                json!({"litellm_provider": "bedrock"}),
            ),
        ])
        .collect();
    ModelInfoCatalog::new(entries)
}

// expected values recorded from get_llm_provider after add_known_models() folds this catalog into the model lists
#[rstest]
#[case::claude_2("claude-2", Some("anthropic_text"))]
#[case::vertex_ai_zz_vimg("vertex_ai/zz-vimg", Some("vertex_ai"))]
#[case::zz_ai21_chat("zz-ai21-chat", Some("ai21_chat"))]
#[case::zz_ai21_text("zz-ai21-text", Some("ai21_chat"))]
#[case::zz_aleph_alpha("zz-aleph_alpha", Some("aleph_alpha"))]
#[case::zz_anthropic("zz-anthropic", Some("anthropic"))]
#[case::zz_bedrock("zz-bedrock", Some("bedrock"))]
#[case::zz_bedrock_converse("zz-bedrock_converse", Some("bedrock"))]
#[case::zz_cohere("zz-cohere", Some("cohere"))]
#[case::zz_cohere_chat("zz-cohere_chat", Some("cohere_chat"))]
#[case::zz_empower("zz-empower", Some("empower"))]
#[case::zz_gradient_ai("zz-gradient_ai", Some("gradient_ai"))]
#[case::zz_mistral("zz-mistral", None)]
#[case::zz_nlp_cloud("zz-nlp_cloud", Some("nlp_cloud"))]
#[case::zz_openai("zz-openai", Some("openai"))]
#[case::zz_openrouter("zz-openrouter", Some("openrouter"))]
#[case::zz_text_completion_openai("zz-text-completion-openai", Some("text-completion-openai"))]
#[case::zz_vertex_ai_ai21_models("zz-vertex_ai-ai21_models", None)]
#[case::zz_vertex_ai_chat_models("zz-vertex_ai-chat-models", Some("vertex_ai"))]
#[case::zz_vertex_ai_code_chat_models("zz-vertex_ai-code-chat-models", Some("vertex_ai"))]
#[case::zz_vertex_ai_code_text_models("zz-vertex_ai-code-text-models", Some("vertex_ai"))]
#[case::zz_vertex_ai_embedding_models("zz-vertex_ai-embedding-models", Some("vertex_ai"))]
#[case::zz_vertex_ai_image_models("zz-vertex_ai-image-models", Some("vertex_ai"))]
#[case::zz_vertex_ai_language_models("zz-vertex_ai-language-models", Some("vertex_ai"))]
#[case::zz_vertex_ai_llama_models("zz-vertex_ai-llama_models", None)]
#[case::zz_vertex_ai_mistral_models("zz-vertex_ai-mistral_models", None)]
#[case::zz_vertex_ai_text_models("zz-vertex_ai-text-models", Some("vertex_ai"))]
#[case::zz_vertex_ai_video_models("zz-vertex_ai-video-models", Some("vertex_ai"))]
#[case::zz_vertex_ai_vision_models("zz-vertex_ai-vision-models", Some("vertex_ai"))]
#[case::zz_watsonx("zz-watsonx", Some("watsonx"))]
#[case::zz_xai("zz-xai", None)]
#[case::zz_vimg("zz-vimg", Some("vertex_ai"))]
#[case::bytez_x("bytez/x", Some("bytez"))]
#[case::amazon_nova_x("amazon_nova-x", Some("amazon_nova"))]
#[case::sap_x("sap/x", Some("sap"))]
#[case::gpt_image_9("gpt-image-9", Some("openai"))]
#[case::sora_2("sora-2", Some("openai"))]
#[case::ft_gpt_4o_o_1("ft:gpt-4o:o::1", Some("openai"))]
#[case::ft_gpt_3_5_turbo_o_1("ft:gpt-3.5-turbo:o::1", Some("openai"))]
#[case::short_id_with_a_colon_is_not_a_replicate_candidate("zz-colon:1", Some("bedrock"))]
#[case::sixty_four_chars_is_not_a_replicate_candidate(&*format!("zz:{}", "a".repeat(61)), Some("bedrock"))]
#[case::cohere_prefix_without_a_chat_model("cohere/zz-other", Some("cohere"))]
#[case::cohere_prefix_with_a_chat_model("cohere/zz-cohere_chat", Some("cohere_chat"))]
#[case::chat_model_under_another_prefix("openai/zz-cohere_chat", Some("openai"))]
#[case::anthropic_prefix_without_a_text_model("anthropic/zz-other", Some("anthropic"))]
#[case::text_model_under_another_prefix("openai/claude-2", Some("openai"))]
#[case::replicate_64_char_version_id(&*format!("owner/m:{}b", "a".repeat(63)), Some("replicate"))]
#[case::long_id_that_is_not_a_replicate_version(&*format!("owner/m:{}", "a".repeat(70)), None)]
fn get_llm_provider_routes_every_model_list_like_python(
    #[case] model: &str,
    #[case] expected: Option<&str>,
) {
    assert_eq!(
        provider_list_catalog()
            .get_llm_provider(model)
            .map(|resolved| resolved.custom_llm_provider)
            .as_deref(),
        expected
    );
}

// expected value recorded from _get_model_info_helper: an authenticating provider named in the
// model is adopted as declared, so the doubled provider prefix becomes a candidate
#[rstest]
fn authenticating_provider_prefix_is_adopted_without_inference() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "github_copilot/github_copilot/zz".to_owned(),
        json!({"litellm_provider": "github_copilot"}),
    )]));
    assert_eq!(
        catalog
            .get_model_info("github_copilot/zz", None)
            .map(|info| info.key.into_owned()),
        Ok("github_copilot/github_copilot/zz".to_owned())
    );
}

// expected values recorded from litellm/llms/fireworks_ai/common_utils.py::resolve_fireworks_resource_name
#[rstest]
#[case::account_path("accounts/me/models/x", "accounts/me/models/x")]
#[case::foundry_id("FW-abc", "FW-abc")]
#[case::deployment_suffix("model#v2", "model#v2")]
#[case::router_path("routers/r", "accounts/fireworks/routers/r")]
#[case::model_path("models/m", "accounts/fireworks/models/m")]
#[case::fast_router("llama-fast", "accounts/fireworks/routers/llama-fast")]
#[case::provider_prefix("fireworks_ai/llama", "accounts/fireworks/models/llama")]
#[case::provider_prefixed_account("fireworks_ai/accounts/a/b", "accounts/a/b")]
#[case::bare_model("plain", "accounts/fireworks/models/plain")]
fn resolve_fireworks_resource_name_matches_python(#[case] model: &str, #[case] expected: &str) {
    assert_eq!(
        litellm_cost::model_info::resolve_fireworks_resource_name(model),
        expected
    );
}

// expected value recorded from _get_model_info_helper: the split name drops only the routing
// prefix, so a throughput-suffixed key still resolves after the base-model names miss
#[rstest]
fn bedrock_split_name_strips_only_the_routing_prefix() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "amazon.zz-model:0:51k".to_owned(),
        json!({"litellm_provider": "bedrock"}),
    )]));
    assert_eq!(
        catalog
            .get_model_info("invoke/amazon.zz-model:0:51k", Some("bedrock"))
            .map(|info| info.key.into_owned()),
        Ok("amazon.zz-model:0:51k".to_owned())
    );
}

// Python keeps the last of several keys that differ only in case, by insertion order. The Rust
// catalog is a HashMap with no insertion order, so it deterministically keeps the smallest key.
#[rstest]
fn case_insensitive_collisions_resolve_to_the_smallest_key() {
    let casings = [
        "zz-case", "Zz-case", "zZ-case", "ZZ-case", "zz-Case", "ZZ-CASE",
    ];
    for _ in 0..10 {
        let catalog = ModelInfoCatalog::new(
            casings
                .iter()
                .map(|key| ((*key).to_owned(), json!({})))
                .collect(),
        );
        assert_eq!(
            catalog
                .get_model_info("zz-CASE", None)
                .map(|info| info.key.into_owned()),
            Ok("ZZ-CASE".to_owned())
        );
    }
}
