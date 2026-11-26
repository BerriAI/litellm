use std::sync::LazyLock;

use crate::anthropic::common_utils::supports_effort_param;
use crate::base_llm::messages::context::{
    BedrockCapabilities, MessagesModelCapabilities, NativeOutputSupport, SupportedEffortTiers,
};
use crate::bedrock::common_utils::{
    bedrock_base_model, is_bedrock_application_inference_profile_arn, strip_bedrock_routing_prefix,
};
use crate::bedrock_mantle::messages::transformation::mantle_capability_lookup;
use fancy_regex::Regex;
use litellm_auth_aws::is_bedrock_region;
use litellm_core_utils::get_llm_provider_logic::get_custom_llm_provider;
use litellm_llms_types::formats::messages::EffortLevel;
use litellm_model_catalog::{Catalog, wire::Fields};
use litellm_python_compat::{json::from_json, truthy::truthy};
use serde::Deserialize;
use serde_json::Value;

const SAMPLING_RESTRICTED_NAMES: [&str; 9] = [
    "fable", "opus-4-7", "opus_4_7", "opus-4.7", "opus_4.7", "opus-4-8", "opus_4_8", "opus-4.8",
    "opus_4.8",
];
const SAMPLING_TOLERANT_PROVIDERS: [&str; 1] = ["openrouter"];
const BEDROCK: &str = "bedrock";
const NATIVE_OUTPUT: &str = "supports_native_structured_output";
const BEDROCK_NATIVE_OUTPUT: &str = "supports_bedrock_native_structured_output";
const NATIVE_OUTPUT_BY_DEFAULT: &str = "supports_bedrock_native_structured_output_by_default";
const OUTPUT_CONFIG: &str = "supports_output_config";
const PER_TURN_CONTROL: &str = "supports_per_turn_control";
const SERVER_SIDE_FALLBACK: &str = "supports_server_side_fallback";
const PROMPT_CACHE_TTL_PRICE: &str = "cache_creation_input_token_cost_above_1hr";
const BACKSTOP_PREFIXES: [&str; 4] = [
    "bedrock/converse/",
    "bedrock/invoke/",
    "bedrock/",
    "vertex_ai/",
];

static VERSION_SUFFIX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"-v\d+(?::\d+)?$").expect("valid regex"));
static PROFILE_MINOR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r":\d+$").expect("valid regex"));
static DATED_RELEASE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"-\d{8}$").expect("valid regex"));
static DOTTED_VERSION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(\d)\.(\d)").expect("valid regex"));
static TRAILING_NUMBER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"-\d+$").expect("valid regex"));
static TRAILING_DATE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"-\d{4}-\d{2}-\d{2}$").expect("valid regex"));

pub fn anthropic_model_capabilities(
    catalog: &Catalog,
    model: &str,
    custom_llm_provider: Option<&str>,
    base_model: Option<&str>,
    declared: &dyn Fn(&str) -> Option<bool>,
) -> MessagesModelCapabilities {
    let (resolved, provider) = get_custom_llm_provider(model, custom_llm_provider)
        .map_or((model, "anthropic"), |resolved| {
            (resolved.model, resolved.custom_llm_provider)
        });
    let requested = model;
    let (model, name) = match base_model.map(strip_bedrock_routing_prefix) {
        Some(base)
            if is_bedrock(provider) && is_bedrock_application_inference_profile_arn(resolved) =>
        {
            (base, base)
        }
        _ => (model, resolved),
    };
    let probe = Probe::new(catalog, model, name, provider, declared);
    let thinking_always_on = probe.supports("thinking_always_on");
    let capabilities = MessagesModelCapabilities {
        supports_reasoning: probe.supports("supports_reasoning"),
        supports_adaptive_thinking: probe.supports("supports_adaptive_thinking"),
        thinking_always_on,
        thinking_on_by_default: thinking_always_on || probe.supports("thinking_on_by_default"),
        supports_legacy_thinking: probe.supports("supports_legacy_thinking"),
        supports_output_config: probe.supports("supports_output_config"),
        supports_sampling_params: declared("supports_sampling_params").unwrap_or_else(|| {
            SAMPLING_TOLERANT_PROVIDERS.contains(&provider)
                || backstop(
                    catalog,
                    &backstop_candidates(name),
                    "supports_sampling_params",
                )
                .unwrap_or_else(|| {
                    let lowered = name.to_lowercase();
                    !SAMPLING_RESTRICTED_NAMES
                        .iter()
                        .any(|restricted| lowered.contains(restricted))
                })
        }),
        supports_speed: provider == "anthropic"
            && flag(exact(catalog, name), "supports_speed") == Some(true),
        supports_mid_conversation_system: probe.supports("supports_mid_conversation_system"),
        effort_tiers: SupportedEffortTiers {
            minimal: probe.supports("supports_minimal_reasoning_effort"),
            low: probe.supports("supports_low_reasoning_effort"),
            medium: probe.supports("supports_medium_reasoning_effort"),
            high: probe.supports("supports_high_reasoning_effort"),
            xhigh: probe.supports("supports_xhigh_reasoning_effort"),
            max: probe.supports("supports_max_reasoning_effort"),
        },
        bedrock: BedrockCapabilities::default(),
    };
    MessagesModelCapabilities {
        bedrock: if provider == BEDROCK {
            with_invoke_flags(
                bedrock_model_capabilities(catalog, requested, base_model, declared),
                &capabilities,
            )
        } else {
            BedrockCapabilities::default()
        },
        ..capabilities
    }
}

/// `bedrock` with the flags the Bedrock Invoke helpers read off `invoke`, the capabilities probed
/// under `bedrock` for the model the Invoke transform receives.
pub fn with_invoke_flags(
    bedrock: BedrockCapabilities,
    invoke: &MessagesModelCapabilities,
) -> BedrockCapabilities {
    BedrockCapabilities {
        supports_adaptive_thinking: invoke.supports_adaptive_thinking,
        supports_effort_param: supports_effort_param(invoke),
        ..bedrock
    }
}

/// Python's Mantle routes gate sampling and speed on the `requested` model, run the shared
/// Anthropic transform on `route` and the Bedrock Invoke helpers on `invoke`.
fn mantle_capabilities(
    requested: &MessagesModelCapabilities,
    route: MessagesModelCapabilities,
    invoke: &MessagesModelCapabilities,
    bedrock: BedrockCapabilities,
) -> MessagesModelCapabilities {
    MessagesModelCapabilities {
        supports_sampling_params: requested.supports_sampling_params,
        supports_speed: requested.supports_speed,
        bedrock: with_invoke_flags(bedrock, invoke),
        ..route
    }
}

/// `anthropic_model_capabilities`, except on the Mantle routes, which probe the keys
/// `mantle_capability_lookup` names.
pub fn messages_route_capabilities(
    catalog: &Catalog,
    model: &str,
    custom_llm_provider: Option<&str>,
    base_model: Option<&str>,
    declared: &dyn Fn(&str) -> Option<bool>,
) -> MessagesModelCapabilities {
    let requested =
        anthropic_model_capabilities(catalog, model, custom_llm_provider, base_model, declared);
    let Some(lookup) = get_custom_llm_provider(model, custom_llm_provider)
        .and_then(|resolved| mantle_capability_lookup(resolved.custom_llm_provider, model))
    else {
        return requested;
    };
    let invoke =
        anthropic_model_capabilities(catalog, &lookup.model, Some(BEDROCK), None, declared);
    mantle_capabilities(
        &requested,
        anthropic_model_capabilities(
            catalog,
            &lookup.model,
            Some(lookup.provider),
            None,
            declared,
        ),
        &invoke,
        invoke.bedrock,
    )
}

pub fn bedrock_model_capabilities(
    catalog: &Catalog,
    model: &str,
    base_model: Option<&str>,
    declared: &dyn Fn(&str) -> Option<bool>,
) -> BedrockCapabilities {
    let model = bedrock_model(model);
    let probe = Probe::new(catalog, model, bedrock_model(model), BEDROCK, &|_| None);
    let mapped = probe.factory(NATIVE_OUTPUT);
    let opted_out = declared(BEDROCK_NATIVE_OUTPUT).map_or_else(
        || mapped_native_output_flag(catalog, model) == Some(false),
        |opted_in| !opted_in,
    );
    BedrockCapabilities {
        output_config_effort_ceiling: probe
            .info
            .as_ref()
            .and_then(|info| info.get("bedrock_output_config_effort_ceiling"))
            .and_then(|ceiling| EffortLevel::deserialize(ceiling).ok()),
        supports_output_config: probe.factory(OUTPUT_CONFIG),
        supports_prompt_cache_ttl: [model, bedrock_base_model(model)].iter().any(|name| {
            catalog
                .get(name)
                .and_then(|fields| fields.get(PROMPT_CACHE_TTL_PRICE))
                .is_some_and(|price| !price.is_null())
        }),
        supports_tool_search: probe.supports("supports_tool_search"),
        supports_per_turn_control: declared(PER_TURN_CONTROL)
            .unwrap_or_else(|| probe.supports(PER_TURN_CONTROL)),
        supports_server_side_fallback: declared(SERVER_SIDE_FALLBACK).unwrap_or_else(|| {
            match base_model.map(strip_bedrock_routing_prefix) {
                Some(base) if is_bedrock_application_inference_profile_arn(model) => {
                    Probe::new(catalog, base, base, BEDROCK, &|_| None)
                        .supports(SERVER_SIDE_FALLBACK)
                }
                _ => probe.supports(SERVER_SIDE_FALLBACK),
            }
        }),
        native_output: NativeOutputSupport {
            enabled: declared(NATIVE_OUTPUT)
                .unwrap_or_else(|| declared(BEDROCK_NATIVE_OUTPUT) != Some(false) && mapped),
            opt_in: declared(BEDROCK_NATIVE_OUTPUT)
                .unwrap_or_else(|| probe.factory(BEDROCK_NATIVE_OUTPUT)),
            mapped,
            replaces_unforceable_tool: backstop(
                catalog,
                &probe.fallbacks,
                "supports_forced_tool_use",
            ) == Some(false)
                && !opted_out,
        },
        ..BedrockCapabilities::default()
    }
}

fn bedrock_model(model: &str) -> &str {
    get_custom_llm_provider(model, Some(BEDROCK)).map_or(model, |found| found.model)
}

fn mapped_native_output_flag(catalog: &Catalog, model: &str) -> Option<bool> {
    let routed = strip_bedrock_routing_prefix(model);
    [
        format!("{BEDROCK}/{routed}"),
        routed.to_owned(),
        without_region(routed).to_owned(),
    ]
    .iter()
    .find_map(|name| flag(catalog.get(name), BEDROCK_NATIVE_OUTPUT))
}

struct Probe<'a> {
    catalog: &'a Catalog,
    declared: &'a dyn Fn(&str) -> Option<bool>,
    fallbacks: Vec<String>,
    info: Option<Fields>,
    bare: Option<&'a Fields>,
}

impl<'a> Probe<'a> {
    fn new(
        catalog: &'a Catalog,
        model: &str,
        name: &str,
        provider: &str,
        declared: &'a dyn Fn(&str) -> Option<bool>,
    ) -> Self {
        Self {
            catalog,
            declared,
            fallbacks: backstop_candidates(model),
            info: catalog.model_info(
                &candidate_names(name, provider)
                    .each_ref()
                    .map(String::as_str),
                provider,
            ),
            bare: catalog.lookup(name).map(|found| found.entry.fields()),
        }
    }

    fn supports(&self, capability: &str) -> bool {
        if let Some(supported) = (self.declared)(capability) {
            return supported;
        }
        if let Some(supported) = flag(self.info.as_ref(), capability) {
            return supported;
        }
        self.bare_supports(capability)
            || backstop(self.catalog, &self.fallbacks, capability) == Some(true)
    }

    fn factory(&self, capability: &str) -> bool {
        match self.info.as_ref() {
            Some(info) => {
                model_info_flag(info, capability).unwrap_or_else(|| self.bare_supports(capability))
            }
            None => false,
        }
    }

    fn bare_supports(&self, capability: &str) -> bool {
        self.info.is_some() && flag(self.bare, capability) == Some(true)
    }
}

fn model_info_flag(info: &Fields, capability: &str) -> Option<bool> {
    match capability {
        NATIVE_OUTPUT
            if info
                .get(NATIVE_OUTPUT_BY_DEFAULT)
                .is_some_and(|value| truthy(&from_json(value.clone()))) =>
        {
            Some(info.get(BEDROCK_NATIVE_OUTPUT) != Some(&Value::Bool(false)))
        }
        OUTPUT_CONFIG => None,
        _ => flag(Some(info), capability),
    }
}

fn flag(fields: Option<&Fields>, capability: &str) -> Option<bool> {
    fields?.get(capability)?.as_bool()
}

fn exact<'a>(catalog: &'a Catalog, name: &str) -> Option<&'a Fields> {
    catalog
        .lookup(name)
        .filter(|found| found.matched_key == name)
        .map(|found| found.entry.fields())
}

fn is_bedrock(provider: &str) -> bool {
    matches!(provider, "bedrock" | "bedrock_converse")
}

fn candidate_names(model: &str, provider: &str) -> [String; 6] {
    let split = split_model(model, provider);
    let combined = format!("{provider}/{model}");
    let (region_free_combined, combined_stripped) = if provider == "bedrock_mantle" {
        (
            format!("{provider}/{split}"),
            format!("{provider}/{}", stripped_model(split, provider)),
        )
    } else {
        (
            combined.clone(),
            format!("{provider}/{}", stripped_model(model, provider)),
        )
    };
    [
        combined,
        model.to_owned(),
        region_free_combined,
        split.to_owned(),
        combined_stripped,
        stripped_model(model, provider),
    ]
}

fn split_model<'a>(model: &'a str, provider: &str) -> &'a str {
    if provider == "bedrock_mantle" {
        return without_region(model);
    }
    if !is_bedrock(provider) {
        return model;
    }
    let routed = strip_bedrock_routing_prefix(model);
    let region_free = without_region(routed);
    if !region_free.starts_with("arn:") {
        return routed;
    }
    region_free.rsplit('/').next().unwrap_or(region_free)
}

fn without_region(model: &str) -> &str {
    match model.split_once('/') {
        Some((region, rest)) if is_bedrock_region(region) => rest,
        _ => model,
    }
}

fn stripped_model(model: &str, provider: &str) -> String {
    match provider {
        provider if is_bedrock(provider) => bedrock_base_model(model).to_owned(),
        "vertex_ai" | "gemini" | "databricks" => TRAILING_NUMBER.replace(model, "").into_owned(),
        _ => TRAILING_DATE.replace(model, "").into_owned(),
    }
}

fn backstop(catalog: &Catalog, candidates: &[String], capability: &str) -> Option<bool> {
    candidates
        .iter()
        .find_map(|name| flag(exact(catalog, name), capability))
}

fn backstop_candidates(model: &str) -> Vec<String> {
    let base = bedrock_base_model(model);
    let primary: Vec<String> = std::iter::once(model)
        .chain(
            BACKSTOP_PREFIXES
                .iter()
                .filter_map(|prefix| model.strip_prefix(prefix)),
        )
        .map(str::to_owned)
        .chain([base.to_owned(), format!("bedrock/{base}")])
        .collect();
    let normalized: Vec<String> = primary
        .iter()
        .flat_map(|name| normalized_names(name))
        .collect();
    primary.into_iter().chain(normalized).collect()
}

fn normalized_names(name: &str) -> [String; 6] {
    let undotted = DOTTED_VERSION.replace_all(name, "$1-$2");
    let unversioned = VERSION_SUFFIX.replace(&undotted, "").into_owned();
    [
        VERSION_SUFFIX.replace(name, "").into_owned(),
        PROFILE_MINOR.replace(name, "").into_owned(),
        DATED_RELEASE.replace(name, "").into_owned(),
        undotted.into_owned(),
        DATED_RELEASE.replace(&unversioned, "").into_owned(),
        name.rfind('@')
            .filter(|at| *at > 0)
            .map_or(name, |at| &name[..at])
            .to_owned(),
    ]
}

#[cfg(test)]
mod tests {
    use litellm_model_catalog::{BUNDLED_REGISTRY, Catalog};
    use rstest::{fixture, rstest};
    use serde_json::Value;

    use super::{
        anthropic_model_capabilities, bedrock_model_capabilities, messages_route_capabilities,
    };
    use crate::base_llm::messages::context::{BedrockCapabilities, NativeOutputSupport};
    use litellm_llms_types::formats::messages::EffortLevel;

    const CATALOG: &str = r#"{
        "claude-test-5": {
            "litellm_provider": "anthropic",
            "mode": "chat",
            "supports_reasoning": true,
            "supports_speed": true,
            "supports_xhigh_reasoning_effort": true,
            "aliases": ["claude-test-5-latest"]
        },
        "claude-slow-5": {
            "litellm_provider": "anthropic",
            "mode": "chat",
            "supports_reasoning": false,
            "supports_adaptive_thinking": false
        },
        "claude-embed-5": {"litellm_provider": "anthropic", "mode": "embedding"},
        "claude-nosample-5": {"litellm_provider": "anthropic", "mode": "chat", "supports_sampling_params": false},
        "anthropic/claude-nosample-5": {"litellm_provider": "anthropic", "mode": "chat", "supports_sampling_params": false},
        "claude-default-think-5": {"litellm_provider": "anthropic", "mode": "chat", "thinking_on_by_default": true},
        "claude-always-think-5": {"litellm_provider": "anthropic", "mode": "chat", "thinking_always_on": true},
        "anthropic/claude-split-5": {"litellm_provider": "anthropic", "mode": "chat"},
        "claude-split-5": {"litellm_provider": "anthropic", "mode": "chat", "supports_reasoning": true},
        "anthropic.claude-test-5-v1:0": {
            "litellm_provider": "bedrock_converse",
            "mode": "chat",
            "supports_reasoning": true,
            "supports_speed": true
        },
        "bedrock_mantle/anthropic.claude-mantle-5": {
            "litellm_provider": "bedrock_mantle",
            "mode": "chat",
            "supports_reasoning": true,
            "supports_adaptive_thinking": true
        },
        "bedrock_mantle/anthropic.claude-mantle-split-5": {
            "litellm_provider": "bedrock_mantle",
            "mode": "chat",
            "supports_adaptive_thinking": false
        },
        "anthropic.claude-mantle-split-5": {
            "litellm_provider": "bedrock",
            "mode": "chat",
            "supports_adaptive_thinking": true,
            "supports_output_config": true
        },
        "vertex_ai/claude-test-5": {
            "litellm_provider": "vertex_ai-anthropic_models",
            "mode": "chat",
            "supports_reasoning": true,
            "supports_speed": true
        },
        "fallback_generalizations": {
            "rules": [
                {"name": "routing", "pattern": "claude", "model_info": {"litellm_provider": "anthropic", "supports_reasoning": true}},
                {"name": "derived", "pattern": "claude", "extends": "fill", "model_info": {"supports_reasoning": true}},
                {"name": "new-family", "pattern": "claude-(?=newfamily)", "model_info": {"supports_reasoning": true}},
                {
                    "name": "fill",
                    "pattern": "claude",
                    "model_info": {"supports_adaptive_thinking": true},
                    "fill_missing_for_providers": ["anthropic"]
                }
            ]
        }
    }"#;

    #[fixture]
    fn catalog() -> Catalog {
        Catalog::parse(CATALOG.as_bytes()).unwrap().value
    }

    #[rstest]
    #[case::row("anthropic/claude-test-5", None, (true, true, true, true))]
    #[case::alias("anthropic/claude-test-5-latest", None, (true, true, true, true))]
    #[case::bare_name("claude-test-5", None, (true, true, true, true))]
    #[case::configured_provider("claude-test-5", Some("anthropic"), (true, true, true, true))]
    #[case::other_casing("anthropic/Claude-Test-5", None, (true, true, false, true))]
    #[case::row_over_fill("anthropic/claude-slow-5", None, (false, false, false, true))]
    #[case::fill_skips_other_modes("anthropic/claude-embed-5", None, (false, false, false, true))]
    #[case::bare_row_under_prefixed_row("anthropic/claude-split-5", None, (true, true, false, true))]
    #[case::other_provider_row("azure_ai/claude-test-5", None, (false, false, false, true))]
    #[case::unknown_family("anthropic/claude-newfamily-6", None, (true, true, false, true))]
    #[case::unknown_model_skips_routing_and_extends_rules("anthropic/claude-other-7", None, (false, true, false, true))]
    #[case::cross_region_bedrock("bedrock/us.anthropic.claude-test-5-v1:0", None, (true, false, false, true))]
    #[case::region_path_bedrock("bedrock/us-east-1/anthropic.claude-test-5-v1:0", None, (true, false, false, true))]
    #[case::region_path_mantle("bedrock_mantle/us-east-1/anthropic.claude-mantle-5", None, (true, true, false, true))]
    #[case::converse_bedrock("bedrock/converse/anthropic.claude-test-5-v1:0", None, (true, false, false, true))]
    #[case::vertex_version("vertex_ai/claude-test-5@20261001", None, (true, true, false, true))]
    #[case::sampling_restricted_name("anthropic/claude-fable-test", None, (false, true, false, false))]
    fn reads_capabilities_from_the_catalog(
        catalog: Catalog,
        #[case] model: &str,
        #[case] custom_llm_provider: Option<&str>,
        #[case] expected: (bool, bool, bool, bool),
    ) {
        let capabilities =
            anthropic_model_capabilities(&catalog, model, custom_llm_provider, None, &|_| None);

        assert_eq!(
            (
                capabilities.supports_reasoning,
                capabilities.supports_adaptive_thinking,
                capabilities.supports_speed,
                capabilities.supports_sampling_params,
            ),
            expected
        );
    }

    #[rstest]
    #[case::bedrock_mantle("bedrock_mantle/anthropic.claude-mantle-split-5", None, (false, true, true))]
    #[case::bedrock_mantle_region(
        "bedrock_mantle/us-west-2/anthropic.claude-mantle-split-5",
        None,
        (false, true, true)
    )]
    #[case::bedrock_mantle_configured("anthropic.claude-mantle-split-5", Some("bedrock_mantle"), (false, true, true))]
    #[case::mantle_route("bedrock/mantle/anthropic.claude-mantle-split-5", None, (true, true, true))]
    fn mantle_routes_shape_on_the_route_row_and_run_invoke_helpers_on_the_bedrock_row(
        catalog: Catalog,
        #[case] model: &str,
        #[case] custom_llm_provider: Option<&str>,
        #[case] expected: (bool, bool, bool),
    ) {
        let capabilities =
            messages_route_capabilities(&catalog, model, custom_llm_provider, None, &|_| None);

        assert_eq!(
            (
                capabilities.supports_adaptive_thinking,
                capabilities.bedrock.supports_adaptive_thinking,
                capabilities.bedrock.supports_effort_param,
            ),
            expected
        );
    }

    #[rstest]
    #[case::anthropic("anthropic/claude-test-5", None)]
    #[case::bedrock_invoke("bedrock/us.anthropic.claude-test-5-v1:0", None)]
    #[case::bedrock_converse("bedrock/converse/anthropic.claude-test-5-v1:0", None)]
    #[case::bedrock_claude_platform("bedrock/claude_platform/mantle/claude-test-5", None)]
    #[case::bedrock_bare_split_row("anthropic.claude-mantle-split-5", Some("bedrock"))]
    #[case::vertex("vertex_ai/claude-test-5", None)]
    fn other_routes_keep_their_own_lookup(
        catalog: Catalog,
        #[case] model: &str,
        #[case] custom_llm_provider: Option<&str>,
    ) {
        assert_eq!(
            messages_route_capabilities(&catalog, model, custom_llm_provider, None, &|_| None),
            anthropic_model_capabilities(&catalog, model, custom_llm_provider, None, &|_| None)
        );
    }

    #[rstest]
    #[case::default_on("anthropic/claude-default-think-5", true)]
    #[case::always_on("anthropic/claude-always-think-5", true)]
    #[case::off("anthropic/claude-test-5", false)]
    fn thinking_on_by_default_follows_the_row(
        catalog: Catalog,
        #[case] model: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(
            anthropic_model_capabilities(&catalog, model, None, None, &|_| None)
                .thinking_on_by_default,
            expected
        );
    }

    #[rstest]
    fn effort_tiers_follow_the_row(catalog: Catalog) {
        let tiers =
            anthropic_model_capabilities(&catalog, "anthropic/claude-test-5", None, None, &|_| {
                None
            })
            .effort_tiers;

        assert_eq!((tiers.xhigh, tiers.low), (true, false));
    }

    #[rstest]
    #[case::declares_a_model_the_map_lacks(
        "anthropic/claude-other-7",
        &[("supports_reasoning", true)],
        (true, true, false, true)
    )]
    #[case::turns_off_a_filled_flag(
        "anthropic/claude-test-5",
        &[("supports_adaptive_thinking", false)],
        (true, false, true, true)
    )]
    #[case::turns_off_sampling("anthropic/claude-test-5", &[("supports_sampling_params", false)], (true, true, true, false))]
    #[case::speed_still_needs_its_row("anthropic/claude-slow-5", &[("supports_speed", true)], (false, false, false, true))]
    #[case::undeclared_keeps_the_row("anthropic/claude-test-5", &[], (true, true, true, true))]
    fn declared_flags_win_over_the_catalog(
        catalog: Catalog,
        #[case] model: &str,
        #[case] declared: &[(&str, bool)],
        #[case] expected: (bool, bool, bool, bool),
    ) {
        let lookup = |capability: &str| {
            declared
                .iter()
                .find(|(name, _)| *name == capability)
                .map(|(_, supported)| *supported)
        };
        let capabilities = anthropic_model_capabilities(&catalog, model, None, None, &lookup);

        assert_eq!(
            (
                capabilities.supports_reasoning,
                capabilities.supports_adaptive_thinking,
                capabilities.supports_speed,
                capabilities.supports_sampling_params,
            ),
            expected
        );
    }

    #[rstest]
    #[case::anthropic_row("anthropic/claude-nosample-5", &[], false)]
    #[case::anthropic_name("anthropic/claude-fable-test", &[], false)]
    #[case::openrouter_row("openrouter/anthropic/claude-nosample-5", &[], true)]
    #[case::openrouter_name("openrouter/anthropic/claude-fable-test", &[], true)]
    #[case::openrouter_declared_off(
        "openrouter/anthropic/claude-fable-test",
        &[("supports_sampling_params", false)],
        false
    )]
    fn sampling_params_are_gated_only_where_the_provider_rejects_them(
        catalog: Catalog,
        #[case] model: &str,
        #[case] declared: &[(&str, bool)],
        #[case] expected: bool,
    ) {
        let lookup = |capability: &str| {
            declared
                .iter()
                .find(|(name, _)| *name == capability)
                .map(|(_, supported)| *supported)
        };

        assert_eq!(
            anthropic_model_capabilities(&catalog, model, None, None, &lookup)
                .supports_sampling_params,
            expected
        );
    }

    const PROFILE_ARN: &str =
        "bedrock/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123";

    #[rstest]
    #[case::base_model_names_the_profile(PROFILE_ARN, Some("anthropic.claude-test-5-v1:0"), true)]
    #[case::prefixed_base_model(PROFILE_ARN, Some("bedrock/anthropic.claude-test-5-v1:0"), true)]
    #[case::profile_without_base_model(PROFILE_ARN, None, false)]
    #[case::named_model_keeps_its_own_row(
        "bedrock/anthropic.claude-test-5-v1:0",
        Some("anthropic.claude-slow-5"),
        true
    )]
    fn application_inference_profile_reads_its_base_model(
        catalog: Catalog,
        #[case] model: &str,
        #[case] base_model: Option<&str>,
        #[case] expected: bool,
    ) {
        assert_eq!(
            anthropic_model_capabilities(&catalog, model, None, base_model, &|_| None)
                .supports_reasoning,
            expected
        );
    }

    #[fixture]
    fn bundled() -> Catalog {
        Catalog::parse(BUNDLED_REGISTRY).unwrap().value
    }

    type Expected = (Option<EffortLevel>, [bool; 8]);

    fn projected(capabilities: BedrockCapabilities) -> Expected {
        let BedrockCapabilities {
            output_config_effort_ceiling,
            supports_output_config,
            supports_prompt_cache_ttl,
            supports_tool_search,
            supports_per_turn_control,
            native_output:
                NativeOutputSupport {
                    enabled,
                    opt_in,
                    mapped,
                    replaces_unforceable_tool,
                },
            ..
        } = capabilities;
        (
            output_config_effort_ceiling,
            [
                supports_output_config,
                supports_prompt_cache_ttl,
                supports_tool_search,
                supports_per_turn_control,
                enabled,
                opt_in,
                mapped,
                replaces_unforceable_tool,
            ],
        )
    }

    const XHIGH: Option<EffortLevel> = Some(EffortLevel::Xhigh);
    const PROFILE: &str = "bedrock/arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-opus-4-8";
    const FOUNDATION: &str = "bedrock/invoke/arn:aws:bedrock:us-east-1:123456789012:foundation-model/anthropic.claude-opus-4-8";
    const APPLICATION: &str =
        "bedrock/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123";

    #[rstest]
    #[case::us("bedrock/us.anthropic.claude-opus-4-8", (XHIGH, [true, true, true, true, true, true, true, false]))]
    #[case::eu("bedrock/eu.anthropic.claude-opus-4-8", (XHIGH, [true, true, true, true, true, true, true, false]))]
    #[case::global("bedrock/global.anthropic.claude-opus-4-8", (XHIGH, [true, true, true, true, true, true, true, false]))]
    #[case::invoke_prefix("bedrock/invoke/us.anthropic.claude-opus-4-8", (XHIGH, [false, true, true, true, true, true, true, false]))]
    #[case::region_path("bedrock/us-east-1/anthropic.claude-opus-4-8", (XHIGH, [false, true, true, true, false, true, false, false]))]
    #[case::inference_profile_arn(PROFILE, (XHIGH, [false, true, true, true, true, true, true, false]))]
    #[case::invoke_foundation_arn(FOUNDATION, (XHIGH, [false, true, true, true, false, true, false, false]))]
    #[case::unprefixed("anthropic.claude-opus-4-8", (XHIGH, [true, true, true, true, false, true, false, false]))]
    #[case::other_casing("bedrock/ANTHROPIC.CLAUDE-OPUS-4-8", (XHIGH, [true, false, true, true, false, true, false, false]))]
    #[case::sonnet_4_5("bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0", (None, [false, true, true, true, true, false, true, false]))]
    #[case::converse_prefix("bedrock/converse/anthropic.claude-sonnet-4-5-20250929-v1:0", (None, [false, true, true, true, true, false, true, false]))]
    #[case::throughput_suffix("bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0:1000k", (None, [false, true, true, true, true, false, true, false]))]
    #[case::claude_3_5("bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0", (None, [false; 8]))]
    #[case::opus_4_7_without_version("bedrock/global.anthropic.claude-opus-4-7", (XHIGH, [true, true, true, true, false, false, false, false]))]
    #[case::opus_4_7_invoke("bedrock/invoke/global.anthropic.claude-opus-4-7-v1:0", (None, [false, false, true, true, false, false, false, false]))]
    #[case::unforceable_tool("bedrock/anthropic.claude-fable-5-1", (XHIGH, [true, true, true, true, false, false, false, true]))]
    #[case::native_by_default("bedrock/us.anthropic.claude-mythos-5-2", (XHIGH, [true, true, true, true, true, false, true, false]))]
    #[case::opt_in_only("bedrock/anthropic.claude-sonnet-5", (XHIGH, [true, true, true, true, false, true, false, false]))]
    #[case::unmapped_claude_version("bedrock/anthropic.claude-opus-9-9-v1:0", (None, [false, false, true, true, false, false, false, false]))]
    #[case::unmapped_claude("bedrock/anthropic.claude-newthing", (None, [false; 8]))]
    #[case::application_profile(APPLICATION, (None, [false; 8]))]
    #[case::other_family("bedrock/amazon.nova-pro-v1:0", (None, [false; 8]))]
    #[case::unknown("bedrock/unknown-model", (None, [false; 8]))]
    fn bedrock_capabilities_match_python(
        bundled: Catalog,
        #[case] model: &str,
        #[case] expected: Expected,
    ) {
        assert_eq!(
            projected(bedrock_model_capabilities(&bundled, model, None, &|_| None)),
            expected
        );
    }

    #[rstest]
    #[case::native_on("bedrock/us.anthropic.claude-opus-4-8", &[("supports_native_structured_output", true)], (XHIGH, [true, true, true, true, true, true, true, false]))]
    #[case::native_off("bedrock/us.anthropic.claude-opus-4-8", &[("supports_native_structured_output", false)], (XHIGH, [true, true, true, true, false, true, true, false]))]
    #[case::bedrock_native_off("bedrock/us.anthropic.claude-opus-4-8", &[("supports_bedrock_native_structured_output", false)], (XHIGH, [true, true, true, true, false, false, true, false]))]
    #[case::per_turn_off("bedrock/us.anthropic.claude-opus-4-8", &[("supports_per_turn_control", false)], (XHIGH, [true, true, true, false, true, true, true, false]))]
    #[case::native_off_opt_in(
        "bedrock/us.anthropic.claude-opus-4-8",
        &[("supports_native_structured_output", false), ("supports_bedrock_native_structured_output", true)],
        (XHIGH, [true, true, true, true, false, true, true, false])
    )]
    #[case::unforceable_native_on("bedrock/anthropic.claude-fable-5-1", &[("supports_native_structured_output", true)], (XHIGH, [true, true, true, true, true, false, false, true]))]
    #[case::unforceable_opt_in("bedrock/anthropic.claude-fable-5-1", &[("supports_bedrock_native_structured_output", true)], (XHIGH, [true, true, true, true, false, true, false, true]))]
    #[case::unforceable_opted_out("bedrock/anthropic.claude-fable-5-1", &[("supports_bedrock_native_structured_output", false)], (XHIGH, [true, true, true, true, false, false, false, false]))]
    #[case::bedrock_native_on("bedrock/anthropic.claude-haiku-4-5-20251001-v1:0", &[("supports_bedrock_native_structured_output", true)], (None, [false, true, true, true, true, true, true, false]))]
    #[case::per_turn_off_haiku("bedrock/anthropic.claude-haiku-4-5-20251001-v1:0", &[("supports_per_turn_control", false)], (None, [false, true, true, false, true, false, true, false]))]
    fn deployment_model_info_overrides_the_catalog_as_python_does(
        bundled: Catalog,
        #[case] model: &str,
        #[case] declared: &[(&str, bool)],
        #[case] expected: Expected,
    ) {
        let lookup = |capability: &str| {
            declared
                .iter()
                .find(|(name, _)| *name == capability)
                .map(|(_, supported)| *supported)
        };

        assert_eq!(
            projected(bedrock_model_capabilities(&bundled, model, None, &lookup)),
            expected
        );
    }

    #[rstest]
    #[case::haiku_4_5("bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0", &[], true)]
    #[case::sonnet_4_5_invoke("bedrock/invoke/us.anthropic.claude-sonnet-4-5-20250929-v1:0", &[], true)]
    #[case::opus_4_8("bedrock/global.anthropic.claude-opus-4-8", &[], true)]
    #[case::opus_4_1("bedrock/us.anthropic.claude-opus-4-1-20250805-v1:0", &[], false)]
    #[case::opus_4_1_invoke("bedrock/invoke/anthropic.claude-opus-4-1-20250805-v1:0", &[], false)]
    #[case::declared_off(
        "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        &[("supports_server_side_fallback", false)],
        false
    )]
    #[case::declared_on(
        "bedrock/us.anthropic.claude-opus-4-1-20250805-v1:0",
        &[("supports_server_side_fallback", true)],
        true
    )]
    fn server_side_fallback_reads_the_deployment_flag_then_the_catalog(
        bundled: Catalog,
        #[case] model: &str,
        #[case] declared: &[(&str, bool)],
        #[case] expected: bool,
    ) {
        let lookup = |capability: &str| {
            declared
                .iter()
                .find(|(name, _)| *name == capability)
                .map(|(_, supported)| *supported)
        };

        assert_eq!(
            bedrock_model_capabilities(&bundled, model, None, &lookup)
                .supports_server_side_fallback,
            expected
        );
    }

    #[rstest]
    #[case::haiku_4_5_base(Some("bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"), true)]
    #[case::opus_4_1_base(Some("anthropic.claude-opus-4-1-20250805-v1:0"), false)]
    #[case::no_base(None, false)]
    fn server_side_fallback_reads_base_model_for_an_application_inference_profile(
        bundled: Catalog,
        #[case] base_model: Option<&str>,
        #[case] expected: bool,
    ) {
        let profile =
            "bedrock/arn:aws:bedrock:us-east-1:000000000000:application-inference-profile/abc123";

        assert_eq!(
            messages_route_capabilities(&bundled, profile, None, base_model, &|_| None)
                .bedrock
                .supports_server_side_fallback,
            expected
        );
    }

    #[rstest]
    #[case::bedrock_mantle("bedrock_mantle/anthropic.claude-haiku-4-5", true)]
    #[case::amazon_mantle("bedrock/mantle/anthropic.claude-haiku-4-5", true)]
    #[case::mantle_opus_4_1("bedrock_mantle/anthropic.claude-opus-4-1", false)]
    fn mantle_routes_resolve_server_side_fallback_for_the_invoke_helpers(
        bundled: Catalog,
        #[case] model: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(
            messages_route_capabilities(&bundled, model, None, None, &|_| None)
                .bedrock
                .supports_server_side_fallback,
            expected
        );
    }

    #[rstest]
    #[case::bedrock("bedrock/us.anthropic.claude-opus-4-8", None, true)]
    #[case::configured_bedrock("us.anthropic.claude-opus-4-8", Some("bedrock"), true)]
    #[case::anthropic("anthropic/claude-opus-4-8", None, false)]
    fn only_bedrock_deployments_carry_bedrock_capabilities(
        bundled: Catalog,
        #[case] model: &str,
        #[case] custom_llm_provider: Option<&str>,
        #[case] carried: bool,
    ) {
        let capabilities =
            anthropic_model_capabilities(&bundled, model, custom_llm_provider, None, &|_| None);

        assert_eq!(
            capabilities.bedrock != BedrockCapabilities::default(),
            carried
        );
    }

    #[rstest]
    fn every_bundled_anthropic_row_shapes_with_its_own_flags() {
        let catalog = Catalog::parse(BUNDLED_REGISTRY).unwrap().value;
        let mismatches: Vec<String> = catalog
            .served()
            .filter(|(name, fields)| {
                !name.contains('/')
                    && fields.get("litellm_provider").and_then(Value::as_str) == Some("anthropic")
            })
            .flat_map(|(name, fields)| {
                let capabilities = anthropic_model_capabilities(
                    &catalog,
                    &format!("anthropic/{name}"),
                    None,
                    None,
                    &|_| None,
                );
                [
                    ("supports_reasoning", capabilities.supports_reasoning),
                    (
                        "supports_adaptive_thinking",
                        capabilities.supports_adaptive_thinking,
                    ),
                    (
                        "supports_output_config",
                        capabilities.supports_output_config,
                    ),
                    (
                        "supports_sampling_params",
                        capabilities.supports_sampling_params,
                    ),
                    ("supports_speed", capabilities.supports_speed),
                    (
                        "supports_legacy_thinking",
                        capabilities.supports_legacy_thinking,
                    ),
                    ("thinking_always_on", capabilities.thinking_always_on),
                    (
                        "thinking_on_by_default",
                        capabilities.thinking_on_by_default,
                    ),
                    (
                        "supports_minimal_reasoning_effort",
                        capabilities.effort_tiers.minimal,
                    ),
                    (
                        "supports_low_reasoning_effort",
                        capabilities.effort_tiers.low,
                    ),
                    (
                        "supports_medium_reasoning_effort",
                        capabilities.effort_tiers.medium,
                    ),
                    (
                        "supports_high_reasoning_effort",
                        capabilities.effort_tiers.high,
                    ),
                    (
                        "supports_mid_conversation_system",
                        capabilities.supports_mid_conversation_system,
                    ),
                    (
                        "supports_xhigh_reasoning_effort",
                        capabilities.effort_tiers.xhigh,
                    ),
                    (
                        "supports_max_reasoning_effort",
                        capabilities.effort_tiers.max,
                    ),
                ]
                .into_iter()
                .filter(|(flag, shaped)| {
                    fields
                        .get(*flag)
                        .and_then(Value::as_bool)
                        .is_some_and(|row| row != *shaped)
                })
                .map(move |(flag, _)| format!("{name} {flag}"))
            })
            .collect();

        assert_eq!(mismatches, Vec::<String>::new());
    }
}
