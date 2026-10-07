use std::collections::BTreeMap;

use litellm_model_catalog::PricingCatalog;
use litellm_traces::{decode_otlp, estimate_cost};
use rstest::{fixture, rstest};
use serde::Deserialize;
use serde_json::{Map, Value};

#[derive(Deserialize)]
struct Reference {
    test: String,
    attributes: BTreeMap<String, String>,
    catalog: Map<String, Value>,
    expected: Option<f64>,
}

fn captured_reference(index: usize) -> Reference {
    let line = include_str!("fixtures/pricing/python_reference.jsonl")
        .lines()
        .nth(index)
        .unwrap();
    serde_json::from_str(line).unwrap()
}

fn reference_catalog(reference: &Reference) -> PricingCatalog {
    PricingCatalog::parse(&serde_json::to_vec(&reference.catalog).unwrap()).unwrap()
}

#[rstest]
#[case::standard_cache_usage_uses_inclusive_input_and_existing_duration_pricing_1(0)]
#[case::standard_cache_usage_uses_inclusive_input_and_existing_duration_pricing_2(1)]
#[case::standard_cache_usage_uses_inclusive_input_and_existing_duration_pricing_3(2)]
#[case::invalid_or_unsupported_usage_remains_unknown_gen_ai_usage_input_tokens_1(3)]
#[case::invalid_or_unsupported_usage_remains_unknown_gen_ai_usage_output_tokens_1_1(4)]
#[case::invalid_or_unsupported_usage_remains_unknown_gen_ai_usage_output_tokens_1_5_1(5)]
#[case::invalid_or_unsupported_usage_remains_unknown_gen_ai_usage_input_tokens_79_1(6)]
#[case::invalid_or_unsupported_usage_remains_unknown_gen_ai_usage_cache_read_input_tokens_bad_1(7)]
#[case::invalid_or_unsupported_usage_remains_unknown_anthropic_usage_cache_creation_ephemeral_1h_input_tokens_11_1(8)]
#[case::invalid_or_unsupported_usage_remains_unknown_gen_ai_response_model_unknown_trace_model_1(9)]
#[case::invalid_or_unsupported_usage_remains_unknown_gen_ai_usage_input_tokens_audio_2_1(10)]
#[case::missing_usage_is_unknown_but_explicit_zero_is_priced_1(11)]
#[case::missing_usage_is_unknown_but_explicit_zero_is_priced_2(12)]
#[case::served_tier_wins_over_requested_tier_and_invalid_span_does_not_lose_batch_1(13)]
#[case::served_tier_wins_over_requested_tier_and_invalid_span_does_not_lose_batch_2(14)]
#[case::served_tier_wins_over_requested_tier_and_invalid_span_does_not_lose_batch_3(15)]
#[case::served_tier_wins_over_requested_tier_and_invalid_span_does_not_lose_batch_4(16)]
#[case::untrusted_model_names_never_fetch_metadata_false_huggingface_1(17)]
#[case::untrusted_model_names_never_fetch_metadata_false_ollama_1(18)]
#[case::untrusted_model_names_never_fetch_metadata_false_ollama_chat_1(19)]
#[case::untrusted_model_names_never_fetch_metadata_false_lemonade_1(20)]
#[case::untrusted_model_names_never_fetch_metadata_false_deepseek_1(21)]
#[case::untrusted_model_names_never_fetch_metadata_true_huggingface_1(22)]
#[case::untrusted_model_names_never_fetch_metadata_true_ollama_1(23)]
#[case::untrusted_model_names_never_fetch_metadata_true_ollama_chat_1(24)]
#[case::untrusted_model_names_never_fetch_metadata_true_lemonade_1(25)]
#[case::untrusted_model_names_never_fetch_metadata_true_deepseek_1(26)]
#[case::static_catalog_requires_declared_rates_without_dynamic_lookup_none_huggingface_1(27)]
#[case::static_catalog_requires_declared_rates_without_dynamic_lookup_none_ollama_1(28)]
#[case::static_catalog_requires_declared_rates_without_dynamic_lookup_none_ollama_chat_1(29)]
#[case::static_catalog_requires_declared_rates_without_dynamic_lookup_none_lemonade_1(30)]
#[case::static_catalog_requires_declared_rates_without_dynamic_lookup_none_openai_1(31)]
#[case::static_catalog_requires_declared_rates_without_dynamic_lookup_0_0_huggingface_1(32)]
#[case::static_catalog_requires_declared_rates_without_dynamic_lookup_0_0_ollama_1(33)]
#[case::static_catalog_requires_declared_rates_without_dynamic_lookup_0_0_ollama_chat_1(34)]
#[case::static_catalog_requires_declared_rates_without_dynamic_lookup_0_0_lemonade_1(35)]
#[case::static_catalog_requires_declared_rates_without_dynamic_lookup_0_0_openai_1(36)]
#[case::static_catalog_requires_declared_rates_without_dynamic_lookup_0_5_huggingface_1(37)]
#[case::static_catalog_requires_declared_rates_without_dynamic_lookup_0_5_ollama_1(38)]
#[case::static_catalog_requires_declared_rates_without_dynamic_lookup_0_5_ollama_chat_1(39)]
#[case::static_catalog_requires_declared_rates_without_dynamic_lookup_0_5_lemonade_1(40)]
#[case::static_catalog_requires_declared_rates_without_dynamic_lookup_0_5_openai_1(41)]
#[case::estimates_use_selected_rates_instead_of_requiring_flat_prices_numeric_strings_1(42)]
#[case::estimates_use_selected_rates_instead_of_requiring_flat_prices_tier_only_at_boundary_1(43)]
#[case::estimates_use_selected_rates_instead_of_requiring_flat_prices_second_tier_1(44)]
#[case::estimates_use_selected_rates_instead_of_requiring_flat_prices_above_last_tier_1(45)]
#[case::estimates_use_selected_rates_instead_of_requiring_flat_prices_tier_uses_flat_output_1(46)]
#[case::estimates_use_selected_rates_instead_of_requiring_flat_prices_unpriced_tier_uses_flat_1(47)]
#[case::estimates_use_selected_rates_instead_of_requiring_flat_prices_missing_tier_output_1(48)]
#[case::estimates_use_selected_rates_instead_of_requiring_flat_prices_no_tier_for_zero_input_1(49)]
#[case::estimates_use_selected_rates_instead_of_requiring_flat_prices_empty_tiers_1(50)]
#[case::estimates_use_selected_rates_instead_of_requiring_flat_prices_service_tier_only_1(51)]
#[case::estimates_use_selected_rates_instead_of_requiring_flat_prices_service_tier_not_selected_1(
    52
)]
#[case::estimates_use_selected_rates_instead_of_requiring_flat_prices_threshold_only_1(53)]
#[case::estimates_use_selected_rates_instead_of_requiring_flat_prices_threshold_not_crossed_1(54)]
#[case::estimates_use_selected_rates_instead_of_requiring_flat_prices_active_off_peak_only_1(55)]
#[case::estimates_use_selected_rates_instead_of_requiring_flat_prices_inactive_off_peak_only_1(56)]
#[case::estimates_use_selected_rates_instead_of_requiring_flat_prices_non_token_input_1(57)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_flat_bad_1(58)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_flat_1_0_1(59)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_flat_nan_1(60)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_flat_inf_1(61)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_flat_true_1(62)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_flat_false_1(63)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_tier_bad_1(64)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_tier_1_0_1(65)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_tier_nan_1(66)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_tier_inf_1(67)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_tier_true_1(68)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_tier_false_1(69)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_priority_bad_1(70)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_priority_1_0_1(71)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_priority_nan_1(72)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_priority_inf_1(73)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_priority_true_1(74)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_priority_false_1(75)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_threshold_bad_1(76)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_threshold_1_0_1(77)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_threshold_nan_1(78)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_threshold_inf_1(79)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_threshold_true_1(80)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_threshold_false_1(81)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_off_peak_bad_1(82)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_off_peak_1_0_1(83)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_off_peak_nan_1(84)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_off_peak_inf_1(85)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_off_peak_true_1(86)]
#[case::invalid_selected_rates_never_become_free_or_offset_another_charge_off_peak_false_1(87)]
#[case::tier_cache_rates_preserve_defaults_explicit_zero_and_invalid_values_cache_rates0_120_1(88)]
#[case::tier_cache_rates_preserve_defaults_explicit_zero_and_invalid_values_cache_rates1_40_1(89)]
#[case::tier_cache_rates_preserve_defaults_explicit_zero_and_invalid_values_cache_rates2_none_1(90)]
#[case::cache_write_rate_validation_follows_used_duration_bad_2_0_20_140_false_1(91)]
#[case::cache_write_rate_validation_follows_used_duration_bad_2_0_20_140_true_1(92)]
#[case::cache_write_rate_validation_follows_used_duration_2_bad_20_0_140_false_1(93)]
#[case::cache_write_rate_validation_follows_used_duration_2_bad_20_0_140_true_1(94)]
#[case::cache_write_rate_validation_follows_used_duration_bad_2_20_0_none_false_1(97)]
#[case::cache_write_rate_validation_follows_used_duration_bad_2_20_0_none_true_1(98)]
#[case::cache_write_rate_validation_follows_used_duration_2_bad_0_20_none_false_1(99)]
#[case::cache_write_rate_validation_follows_used_duration_2_bad_0_20_none_true_1(100)]
#[case::cache_write_rate_validation_follows_used_duration_bad_2_none_none_none_false_1(101)]
#[case::cache_write_rate_validation_follows_used_duration_bad_2_none_none_none_true_1(102)]
#[case::unused_cache_rates_cannot_contaminate_uncached_cost_1(103)]
#[case::legacy_genai_usage_keeps_inclusive_cache_math_aliases0_1(104)]
#[case::legacy_genai_usage_keeps_inclusive_cache_math_aliases0_2(105)]
#[case::legacy_genai_usage_keeps_inclusive_cache_math_aliases0_3(106)]
#[case::legacy_genai_usage_keeps_inclusive_cache_math_aliases1_1(107)]
#[case::legacy_genai_usage_keeps_inclusive_cache_math_aliases1_2(108)]
#[case::legacy_genai_usage_keeps_inclusive_cache_math_aliases1_3(109)]
#[case::legacy_genai_usage_keeps_inclusive_cache_math_aliases2_1(110)]
#[case::legacy_genai_usage_keeps_inclusive_cache_math_aliases2_2(111)]
#[case::legacy_genai_usage_keeps_inclusive_cache_math_aliases2_3(112)]
#[case::invalid_or_conflicting_numeric_alias_cannot_hide_behind_current_key_1_gen_ai_usage_prompt_tokens_1(113)]
#[case::invalid_or_conflicting_numeric_alias_cannot_hide_behind_current_key_1_gen_ai_usage_completion_tokens_1(114)]
#[case::invalid_or_conflicting_numeric_alias_cannot_hide_behind_current_key_1_gen_ai_usage_cache_creation_input_tokens_1(115)]
#[case::invalid_or_conflicting_numeric_alias_cannot_hide_behind_current_key_bad_gen_ai_usage_prompt_tokens_1(116)]
#[case::invalid_or_conflicting_numeric_alias_cannot_hide_behind_current_key_bad_gen_ai_usage_completion_tokens_1(117)]
#[case::invalid_or_conflicting_numeric_alias_cannot_hide_behind_current_key_bad_gen_ai_usage_cache_creation_input_tokens_1(118)]
#[case::invalid_or_conflicting_numeric_alias_cannot_hide_behind_current_key_1_gen_ai_usage_prompt_tokens_2(119)]
#[case::invalid_or_conflicting_numeric_alias_cannot_hide_behind_current_key_1_gen_ai_usage_completion_tokens_2(120)]
#[case::invalid_or_conflicting_numeric_alias_cannot_hide_behind_current_key_1_gen_ai_usage_cache_creation_input_tokens_2(121)]
#[case::invalid_or_conflicting_numeric_alias_cannot_hide_behind_current_key_18446744073709551616_gen_ai_usage_prompt_tokens_1(122)]
#[case::invalid_or_conflicting_numeric_alias_cannot_hide_behind_current_key_18446744073709551616_gen_ai_usage_completion_tokens_1(123)]
#[case::invalid_or_conflicting_numeric_alias_cannot_hide_behind_current_key_18446744073709551616_gen_ai_usage_cache_creation_input_tokens_1(124)]
#[case::legacy_cache_write_never_adds_to_an_inclusive_input_total_1(125)]
#[case::tier_aliases_preserve_served_precedence_and_reject_same_fact_conflicts_tiers0_340_1(126)]
#[case::tier_aliases_preserve_served_precedence_and_reject_same_fact_conflicts_tiers1_340_1(127)]
#[case::tier_aliases_preserve_served_precedence_and_reject_same_fact_conflicts_tiers2_340_1(128)]
#[case::tier_aliases_preserve_served_precedence_and_reject_same_fact_conflicts_tiers3_120_1(129)]
#[case::tier_aliases_preserve_served_precedence_and_reject_same_fact_conflicts_tiers4_340_1(130)]
#[case::tier_aliases_preserve_served_precedence_and_reject_same_fact_conflicts_tiers5_none_1(131)]
#[case::tier_aliases_preserve_served_precedence_and_reject_same_fact_conflicts_tiers6_none_1(132)]
#[case::tier_aliases_preserve_served_precedence_and_reject_same_fact_conflicts_tiers7_none_1(133)]
#[case::canonical_provider_values_reuse_emitter_mapping_without_network_azure_1(135)]
#[case::canonical_provider_values_reuse_emitter_mapping_without_network_azure_2(136)]
#[case::canonical_provider_values_reuse_emitter_mapping_without_network_azure_ai_1(137)]
#[case::canonical_provider_values_reuse_emitter_mapping_without_network_azure_ai_2(138)]
#[case::canonical_provider_values_reuse_emitter_mapping_without_network_gemini_1(139)]
#[case::canonical_provider_values_reuse_emitter_mapping_without_network_gemini_2(140)]
#[case::canonical_provider_values_reuse_emitter_mapping_without_network_mistral_1(141)]
#[case::canonical_provider_values_reuse_emitter_mapping_without_network_mistral_2(142)]
#[case::canonical_provider_values_reuse_emitter_mapping_without_network_xai_1(143)]
#[case::canonical_provider_values_reuse_emitter_mapping_without_network_xai_2(144)]
#[case::canonical_provider_values_reuse_emitter_mapping_without_network_watsonx_1(145)]
#[case::canonical_provider_values_reuse_emitter_mapping_without_network_watsonx_2(146)]
#[case::canonical_provider_values_reuse_emitter_mapping_without_network_cohere_chat_1(147)]
#[case::canonical_provider_values_reuse_emitter_mapping_without_network_cohere_chat_2(148)]
#[case::canonical_provider_values_reuse_emitter_mapping_without_network_text_completion_openai_1(
    149
)]
#[case::canonical_provider_values_reuse_emitter_mapping_without_network_text_completion_openai_2(
    150
)]
#[case::canonical_provider_values_reuse_emitter_mapping_without_network_bedrock_1(151)]
#[case::canonical_provider_values_reuse_emitter_mapping_without_network_bedrock_2(152)]
#[case::canonical_provider_values_reuse_emitter_mapping_without_network_vertex_ai_1(153)]
#[case::canonical_provider_values_reuse_emitter_mapping_without_network_vertex_ai_2(154)]
#[case::provider_ambiguity_needs_a_unique_catalog_row_or_explicit_identifier_1(155)]
#[case::provider_ambiguity_needs_a_unique_catalog_row_or_explicit_identifier_2(156)]
#[case::provider_ambiguity_needs_a_unique_catalog_row_or_explicit_identifier_3(157)]
#[case::provider_ambiguity_needs_a_unique_catalog_row_or_explicit_identifier_4(158)]
#[case::reasoning_is_an_inclusive_output_subset_with_explicit_unknowns_4_5_10_132_1(159)]
#[case::reasoning_is_an_inclusive_output_subset_with_explicit_unknowns_4_0_10_112_1(160)]
#[case::reasoning_is_an_inclusive_output_subset_with_explicit_unknowns_4_none_10_120_1(161)]
#[case::reasoning_is_an_inclusive_output_subset_with_explicit_unknowns_none_5_10_none_1(162)]
#[case::reasoning_is_an_inclusive_output_subset_with_explicit_unknowns_none_2_10_120_1(163)]
#[case::reasoning_is_an_inclusive_output_subset_with_explicit_unknowns_none_bad_10_none_1(164)]
#[case::reasoning_is_an_inclusive_output_subset_with_explicit_unknowns_0_bad_10_120_1(165)]
#[case::reasoning_is_an_inclusive_output_subset_with_explicit_unknowns_none_bad_0_100_1(166)]
#[case::reasoning_is_an_inclusive_output_subset_with_explicit_unknowns_11_5_10_none_1(167)]
#[case::reasoning_is_an_inclusive_output_subset_with_explicit_unknowns_1_5_10_none_1(168)]
#[case::reasoning_is_an_inclusive_output_subset_with_explicit_unknowns_bad_5_10_none_1(169)]
#[case::reasoning_is_an_inclusive_output_subset_with_explicit_unknowns_18446744073709551616_5_10_none_1(170)]
#[case::reasoning_uses_strict_selected_tier_rates_flat_5_1(171)]
#[case::reasoning_uses_strict_selected_tier_rates_flat_bad_1(172)]
#[case::reasoning_uses_strict_selected_tier_rates_flat_1_1(173)]
#[case::reasoning_uses_strict_selected_tier_rates_flat_nan_1(174)]
#[case::reasoning_uses_strict_selected_tier_rates_flat_inf_1(175)]
#[case::reasoning_uses_strict_selected_tier_rates_flat_true_1(176)]
#[case::reasoning_uses_strict_selected_tier_rates_flat_false_1(177)]
#[case::reasoning_uses_strict_selected_tier_rates_tier_5_1(178)]
#[case::reasoning_uses_strict_selected_tier_rates_tier_bad_1(179)]
#[case::reasoning_uses_strict_selected_tier_rates_tier_1_1(180)]
#[case::reasoning_uses_strict_selected_tier_rates_tier_nan_1(181)]
#[case::reasoning_uses_strict_selected_tier_rates_tier_inf_1(182)]
#[case::reasoning_uses_strict_selected_tier_rates_tier_true_1(183)]
#[case::reasoning_uses_strict_selected_tier_rates_tier_false_1(184)]
#[case::reasoning_uses_strict_selected_tier_rates_priority_5_1(185)]
#[case::reasoning_uses_strict_selected_tier_rates_priority_bad_1(186)]
#[case::reasoning_uses_strict_selected_tier_rates_priority_1_1(187)]
#[case::reasoning_uses_strict_selected_tier_rates_priority_nan_1(188)]
#[case::reasoning_uses_strict_selected_tier_rates_priority_inf_1(189)]
#[case::reasoning_uses_strict_selected_tier_rates_priority_true_1(190)]
#[case::reasoning_uses_strict_selected_tier_rates_priority_false_1(191)]
#[case::reasoning_uses_strict_selected_tier_rates_off_peak_5_1(192)]
#[case::reasoning_uses_strict_selected_tier_rates_off_peak_bad_1(193)]
#[case::reasoning_uses_strict_selected_tier_rates_off_peak_1_1(194)]
#[case::reasoning_uses_strict_selected_tier_rates_off_peak_nan_1(195)]
#[case::reasoning_uses_strict_selected_tier_rates_off_peak_inf_1(196)]
#[case::reasoning_uses_strict_selected_tier_rates_off_peak_true_1(197)]
#[case::reasoning_uses_strict_selected_tier_rates_off_peak_false_1(198)]
#[case::selected_output_rate_can_make_a_missing_reasoning_split_irrelevant_1(199)]
#[case::off_peak_prices_follow_captured_time_including_nanosecond_boundaries_1(200)]
#[case::off_peak_prices_follow_captured_time_including_nanosecond_boundaries_2(201)]
#[case::off_peak_prices_follow_captured_time_including_nanosecond_boundaries_3(202)]
#[case::off_peak_prices_follow_captured_time_including_nanosecond_boundaries_4(203)]
#[case::off_peak_prices_follow_captured_time_including_nanosecond_boundaries_5(204)]
#[case::off_peak_prices_follow_captured_time_including_nanosecond_boundaries_6(205)]
#[case::invalid_supplied_call_time_remains_unknown_bad_1(206)]
#[case::invalid_supplied_call_time_remains_unknown_9223372036854775808_1(207)]
#[case::invalid_supplied_call_time_remains_unknown_9223372036854775809_1(208)]
fn preserved_python_estimates(#[case] index: usize) {
    let reference = captured_reference(index);
    let synthetic = (!reference.catalog.is_empty()).then(|| reference_catalog(&reference));
    let catalog = synthetic
        .as_ref()
        .unwrap_or_else(|| litellm_model_catalog::bundled_pricing_catalog().unwrap());
    let actual = estimate_cost(&reference.attributes, catalog);
    match (actual, reference.expected) {
        (Some(actual), Some(expected)) => assert!(
            (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
            "{}: expected {expected}, got {actual}",
            reference.test
        ),
        _ => assert_eq!(actual, reference.expected, "{}", reference.test),
    }
}

#[rstest]
#[case::invalid_flat_hourly_rate(95)]
#[case::invalid_tier_hourly_rate(96)]
#[case::exporter_missing_duration(134)]
fn incomplete_cache_duration_corrects_historical_estimates(#[case] index: usize) {
    let reference = captured_reference(index);
    assert!(reference.expected.is_some());
    assert_eq!(
        estimate_cost(&reference.attributes, &reference_catalog(&reference)),
        None
    );
}

#[rstest]
#[case::speech("gen_ai.output.type", "speech", false)]
#[case::audio("gen_ai.output.type", "audio", false)]
#[case::image("gen_ai.output.type", "image", false)]
#[case::text("gen_ai.output.type", "text", true)]
#[case::json("gen_ai.output.type", "json", true)]
#[case::canonical_audio("gen_ai.usage.input_tokens.audio", "2", false)]
#[case::openinference_input("llm.token_count.prompt_details.audio", "2", false)]
#[case::openinference_output("llm.token_count.completion_details.audio", "2", false)]
#[case::pydantic_audio("gen_ai.usage.details.audio_tokens", "2", false)]
#[case::invalid_audio("llm.token_count.prompt_details.audio", "invalid", false)]
#[case::zero_audio("llm.token_count.prompt_details.audio", "0", true)]
#[case::decimal_zero_audio("gen_ai.usage.details.audio_tokens", "0.0", true)]
#[case::pydantic_search("gen_ai.usage.details.web_search_requests", "1", false)]
#[case::search_without_catalog_price("gen_ai.usage.web_search_requests", "1", false)]
#[case::openinference_search("llm.token_count.prompt_details.web_search_requests", "1", false)]
#[case::anthropic_search("anthropic.usage.server_tool_use.web_search_requests", "1", false)]
#[case::server_side_search(
    "gen_ai.usage.server_side_tool_usage_details.web_search_calls",
    "1",
    false
)]
#[case::grounding("gen_ai.usage.details.google_maps_grounding_requests", "1", false)]
#[case::browser("gen_ai.usage.details.browser_open_requests", "1", false)]
#[case::invalid_search("gen_ai.usage.details.web_search_requests", "invalid", false)]
#[case::zero_search("gen_ai.usage.details.web_search_requests", "0", true)]
#[case::decimal_zero_search("gen_ai.usage.details.web_search_requests", "0.0", true)]
#[case::code_interpreter("gen_ai.usage.details.code_interpreter_sessions", "1", false)]
#[case::pydantic_code_execution(
    "gen_ai.usage.details.server_side_tools_code_execution",
    "1",
    false
)]
#[case::zero_code_execution("gen_ai.usage.details.server_side_tools_code_execution", "0", true)]
#[case::computer_use("gen_ai.usage.details.computer_use_input_tokens", "1", false)]
#[case::citation("gen_ai.usage.details.citation_tokens", "1", false)]
#[case::zero_code_interpreter("gen_ai.usage.details.code_interpreter_sessions", "0", true)]
#[case::client_tools("gen_ai.usage.tool_calls", "2", true)]
fn unsupported_usage_is_unknown_but_optional_capability_is_priced(
    #[case] key: &str,
    #[case] value: &str,
    #[case] priced: bool,
) {
    let catalog = PricingCatalog::parse(br#"{"text-model":{"mode":"chat","input_cost_per_token":1,"output_cost_per_token":2,"input_cost_per_audio_token":3,"supported_modalities":["text","audio"],"supports_web_search":true}}"#).unwrap();
    let attributes = BTreeMap::from([
        ("gen_ai.request.model".into(), "text-model".into()),
        ("gen_ai.usage.input_tokens".into(), "100".into()),
        ("gen_ai.usage.output_tokens".into(), "10".into()),
        (key.into(), value.into()),
    ]);
    assert_eq!(
        estimate_cost(&attributes, &catalog),
        priced.then_some(120.0)
    );
}

#[rstest]
#[case::response("gen_ai.response.model")]
#[case::request("gen_ai.request.model")]
#[case::openinference("llm.model_name")]
fn bundled_catalog_required_charges_remain_unknown(#[case] key: &str) {
    let catalog = litellm_model_catalog::bundled_pricing_catalog().unwrap();
    let rows: Value = serde_json::from_slice(include_bytes!(
        "../../../../model_prices_and_context_window.json"
    ))
    .unwrap();
    let required: BTreeMap<_, _> = rows
        .as_object()
        .unwrap()
        .iter()
        .filter(|(model, fields)| {
            model.as_str() != "sample_spec" && fields["supports_token_only_pricing"] == false
        })
        .map(|(model, _)| (model.clone(), None))
        .collect();
    let estimates: BTreeMap<_, _> = required
        .keys()
        .map(|model| {
            let attributes = BTreeMap::from([
                (key.into(), model.clone()),
                ("gen_ai.usage.input_tokens".into(), "100".into()),
                ("gen_ai.usage.output_tokens".into(), "10".into()),
            ]);
            (model.clone(), estimate_cost(&attributes, catalog))
        })
        .collect();
    assert_eq!(estimates, required);
}

#[rstest]
#[case::legacy(None, true)]
#[case::null(Some(serde_json::json!(null)), true)]
#[case::ordinary(Some(serde_json::json!(true)), true)]
#[case::required(Some(serde_json::json!(false)), false)]
#[case::invalid_string(Some(serde_json::json!("false")), false)]
#[case::invalid_number(Some(serde_json::json!(0)), false)]
fn catalog_contract_controls_estimates_for_any_model(
    #[case] required: Option<Value>,
    #[case] priced: bool,
    #[values("canonical", "alias", "case", "provider_prefix")] identity: &str,
    #[values(None, Some("0"))] searches: Option<&str>,
) {
    let mut fields = serde_json::json!({
        "mode":"chat", "litellm_provider":"openai", "aliases":["deployment"],
        "input_cost_per_token":1, "output_cost_per_token":2, "supports_web_search":true,
        "search_context_cost_per_query":{"search_context_size_low":3}
    });
    if let Some(required) = required {
        fields["supports_token_only_pricing"] = required;
    }
    let catalog = PricingCatalog::parse(
        &serde_json::to_vec(&serde_json::json!({"arbitrary-model": fields})).unwrap(),
    )
    .unwrap();
    let identity = match identity {
        "alias" => "deployment",
        "case" => "ARBITRARY-MODEL",
        "provider_prefix" => "openai/arbitrary-model",
        _ => "arbitrary-model",
    };
    let attributes = [
        ("gen_ai.request.model", Some("irrelevant-requested-alias")),
        ("gen_ai.response.model", Some(identity)),
        ("gen_ai.usage.input_tokens", Some("100")),
        ("gen_ai.usage.output_tokens", Some("10")),
        ("gen_ai.usage.details.web_search_requests", searches),
    ]
    .into_iter()
    .filter_map(|(key, value)| value.map(|value| (key.into(), value.into())))
    .collect();
    assert_eq!(
        estimate_cost(&attributes, &catalog),
        priced.then_some(120.0)
    );
}

#[rstest]
#[case::search_name("gpt-future-search", "openai")]
#[case::research_name("deep-research-future", "gemini")]
#[case::provider_name("sonar-future", "perplexity")]
fn model_identity_does_not_override_catalog_contract(#[case] model: &str, #[case] provider: &str) {
    let catalog = PricingCatalog::parse(
        &serde_json::to_vec(&serde_json::json!({model:{
            "litellm_provider":provider, "input_cost_per_token":1,"output_cost_per_token":2,
            "supports_token_only_pricing":true,"supports_web_search":true
        }}))
        .unwrap(),
    )
    .unwrap();
    let attributes = BTreeMap::from([
        ("gen_ai.response.model".into(), model.into()),
        ("gen_ai.usage.input_tokens".into(), "100".into()),
        ("gen_ai.usage.output_tokens".into(), "10".into()),
    ]);
    assert_eq!(estimate_cost(&attributes, &catalog), Some(120.0));
}

#[rstest]
#[case::additional_pricing(serde_json::json!({"supports_token_only_pricing":false}), None)]
#[case::request_fee(serde_json::json!({"input_cost_per_request":3}), None)]
#[case::ordinary_alias(serde_json::json!({"supports_token_only_pricing":true}), Some(120.0))]
fn requested_pricing_requirements_survive_served_model_selection(
    #[case] requested: Value,
    #[case] expected: Option<f64>,
    #[values("gen_ai.request.model", "llm.model_name")] key: &str,
) {
    let catalog = PricingCatalog::parse(
        &serde_json::to_vec(&serde_json::json!({
            "requested":requested,
            "served":{"input_cost_per_token":1,"output_cost_per_token":2}
        }))
        .unwrap(),
    )
    .unwrap();
    let attributes = BTreeMap::from([
        (key.into(), "requested".into()),
        ("gen_ai.response.model".into(), "served".into()),
        ("gen_ai.usage.input_tokens".into(), "100".into()),
        ("gen_ai.usage.output_tokens".into(), "10".into()),
    ]);
    assert_eq!(estimate_cost(&attributes, &catalog), expected);
}

#[rstest]
#[case::path("azure_ai/model_router/deployment", None, false)]
#[case::nested_path("azure_ai/model-router/team/deployment", None, false)]
#[case::custom_name("azure_ai/my-model-router-deployment", None, false)]
#[case::hint("prod-model_router", Some("azure_ai"), false)]
#[case::otel_hint("model-router/deployment", Some("azure.ai.inference"), false)]
#[case::case("AZURE_AI/MODEL_ROUTER/DEPLOYMENT", None, false)]
#[case::served_provider("azure_ai/model_router/deployment", Some("openai"), false)]
#[case::deployment_entry("azure_ai/model_router/known", None, false)]
#[case::deployment_alias("routed-alias", None, false)]
#[case::azure("azure/model-router/deployment", None, false)]
#[case::opaque("ordinary-deployment", Some("azure_ai"), true)]
#[case::different_provider("openai/model_router/deployment", Some("azure_ai"), true)]
#[case::ordinary_child("parent/deployment", None, true)]
fn routed_request_contract_survives_deployment_and_served_identity(
    #[case] requested: &str,
    #[case] provider: Option<&str>,
    #[case] priced: bool,
    #[values("gen_ai.request.model", "llm.model_name")] key: &str,
) {
    let catalog = PricingCatalog::parse(
        &serde_json::to_vec(&serde_json::json!({
            "azure_ai/model_router":{"supports_token_only_pricing":false},
            "azure/model-router":{"supports_token_only_pricing":false},
            "azure_ai/model_router/known":{
                "supports_token_only_pricing":true,"aliases":["routed-alias"]
            },
            "parent":{"supports_token_only_pricing":false},
            "served":{"input_cost_per_token":1,"output_cost_per_token":2}
        }))
        .unwrap(),
    )
    .unwrap();
    let attributes = [
        (key, Some(requested)),
        ("gen_ai.response.model", Some("served")),
        ("gen_ai.provider.name", provider),
        ("gen_ai.usage.input_tokens", Some("100")),
        ("gen_ai.usage.output_tokens", Some("10")),
    ]
    .into_iter()
    .filter_map(|(key, value)| value.map(|value| (key.into(), value.into())))
    .collect();
    assert_eq!(
        estimate_cost(&attributes, &catalog),
        priced.then_some(120.0)
    );
}

#[rstest]
#[case::absent(None, None)]
#[case::required(Some(false), None)]
#[case::complete(Some(true), Some(120.0))]
fn route_metadata_controls_estimates_even_for_a_priced_response_deployment(
    #[case] token_only: Option<bool>,
    #[case] expected: Option<f64>,
) {
    let mut rows = serde_json::json!({
        "azure_ai/model_router/deployment":{
            "input_cost_per_token":1,"output_cost_per_token":2,"supports_token_only_pricing":true
        }
    });
    if let Some(token_only) = token_only {
        rows["azure_ai/model_router"] =
            serde_json::json!({"supports_token_only_pricing":token_only});
    }
    let catalog = PricingCatalog::parse(&serde_json::to_vec(&rows).unwrap()).unwrap();
    let attributes = BTreeMap::from([
        (
            "gen_ai.response.model".into(),
            "azure_ai/model_router/deployment".into(),
        ),
        ("gen_ai.usage.input_tokens".into(), "100".into()),
        ("gen_ai.usage.output_tokens".into(), "10".into()),
    ]);
    assert_eq!(estimate_cost(&attributes, &catalog), expected);
}

#[rstest]
#[case::absent(None, true)]
#[case::null(Some(serde_json::json!(null)), true)]
#[case::free(Some(serde_json::json!(0)), true)]
#[case::string_free(Some(serde_json::json!("0")), true)]
#[case::paid(Some(serde_json::json!(3)), false)]
#[case::invalid(Some(serde_json::json!("invalid")), false)]
fn uncovered_request_fees_are_not_reported_as_token_totals(
    #[case] fee: Option<Value>,
    #[case] priced: bool,
    #[values("input_cost_per_request", "input_cost_per_query")] key: &str,
) {
    let mut fields = serde_json::json!({"input_cost_per_token":1,"output_cost_per_token":2});
    if let Some(fee) = fee {
        fields[key] = fee;
    }
    let catalog = PricingCatalog::parse(
        &serde_json::to_vec(&serde_json::json!({"text-model":fields})).unwrap(),
    )
    .unwrap();
    let attributes = BTreeMap::from([
        ("gen_ai.request.model".into(), "text-model".into()),
        ("gen_ai.usage.input_tokens".into(), "100".into()),
        ("gen_ai.usage.output_tokens".into(), "10".into()),
    ]);
    assert_eq!(
        estimate_cost(&attributes, &catalog),
        priced.then_some(120.0)
    );
}

#[rstest]
fn dedicated_nontext_model_never_uses_total_token_estimates(
    #[values(
        "audio_speech",
        "audio_transcription",
        "image_generation",
        "image_edit",
        "video_generation",
        "realtime",
        "ocr"
    )]
    mode: &str,
) {
    let catalog = PricingCatalog::parse(&serde_json::to_vec(&serde_json::json!({"nontext-model":{"mode":mode,"input_cost_per_token":1,"output_cost_per_token":2}})).unwrap()).unwrap();
    let attributes = BTreeMap::from([
        ("gen_ai.request.model".into(), "nontext-model".into()),
        ("gen_ai.usage.input_tokens".into(), "100".into()),
        ("gen_ai.usage.output_tokens".into(), "10".into()),
    ]);
    assert_eq!(estimate_cost(&attributes, &catalog), None);
}

#[rstest]
#[case::langchain(include_bytes!("fixtures/langchain_simple.json"), "1965258dc3bdbc38")]
#[case::openai_agents(include_bytes!("fixtures/openai_agents_simple.json"), "c638da940e89fc86")]
#[case::deepagents_cache_write(include_bytes!("fixtures/deepagents_simple.json"), "5cb5b0adf6820736")]
fn captured_openinference_usage_reaches_pricing(#[case] body: &[u8], #[case] id: &str) {
    let span = decode_otlp(body, Some("application/json"))
        .unwrap()
        .into_iter()
        .find(|span| span.span_id == id)
        .unwrap();
    let model = &span.attributes["llm.model_name"];
    let catalog = PricingCatalog::parse(
        &serde_json::to_vec(&serde_json::json!({model:{
            "litellm_provider":"openai", "input_cost_per_token":1, "output_cost_per_token":2,
            "cache_read_input_token_cost":0.1, "cache_creation_input_token_cost":3,
            "output_cost_per_reasoning_token":4
        }}))
        .unwrap(),
    )
    .unwrap();
    let read = span
        .attributes
        .get("llm.token_count.prompt_details.cache_read")
        .map_or(0.0, |value| value.parse::<f64>().unwrap());
    let write = span
        .attributes
        .get("llm.token_count.prompt_details.cache_write")
        .map_or(0.0, |value| value.parse::<f64>().unwrap());
    let reasoning = span.attributes["llm.token_count.completion_details.reasoning"]
        .parse::<f64>()
        .unwrap();
    let input = f64::from(span.normalized.input_tokens);
    let output = f64::from(span.normalized.output_tokens);
    let expected = (input - read - write)
        + read * 0.1
        + write * 3.0
        + (output - reasoning) * 2.0
        + reasoning * 4.0;
    assert_eq!(estimate_cost(&span.attributes, &catalog), Some(expected));
}

#[fixture]
fn openinference_attributes() -> BTreeMap<String, String> {
    BTreeMap::from([
        ("llm.model_name".into(), "text-model".into()),
        ("llm.token_count.prompt".into(), "100".into()),
        ("llm.token_count.completion".into(), "10".into()),
        (
            "llm.token_count.prompt_details.cache_read".into(),
            "20".into(),
        ),
        (
            "llm.token_count.prompt_details.cache_write".into(),
            "10".into(),
        ),
        (
            "llm.token_count.completion_details.reasoning".into(),
            "4".into(),
        ),
    ])
}

#[fixture]
fn openinference_catalog() -> PricingCatalog {
    PricingCatalog::parse(br#"{
        "text-model":{"litellm_provider":"moonshot","aliases":["short"],"input_cost_per_token":1,"output_cost_per_token":2,"cache_read_input_token_cost":0.1,"cache_creation_input_token_cost":3,"output_cost_per_reasoning_token":4},
        "other-model":{"litellm_provider":"moonshot","input_cost_per_token":2,"output_cost_per_token":4,"cache_read_input_token_cost":0.2,"cache_creation_input_token_cost":6,"output_cost_per_reasoning_token":8}
    }"#).unwrap()
}

#[rstest]
#[case::input("gen_ai.usage.input_tokens", "100")]
#[case::output("gen_ai.usage.output_tokens", "10")]
#[case::cache_read("gen_ai.usage.cache_read.input_tokens", "20")]
#[case::cache_write("gen_ai.usage.cache_write.input_tokens", "10")]
#[case::cache_creation("gen_ai.usage.cache_creation.input_tokens", "10")]
#[case::reasoning("gen_ai.usage.reasoning.output_tokens", "4")]
#[case::pydantic_reasoning("gen_ai.usage.details.reasoning_tokens", "4")]
#[case::mastra_reasoning("gen_ai.usage.reasoning_tokens", "4")]
fn numeric_aliases_must_agree_across_dialects(
    openinference_attributes: BTreeMap<String, String>,
    openinference_catalog: PricingCatalog,
    #[case] key: &str,
    #[case] recorded: &str,
    #[values("matching", "1", "invalid")] alias: &str,
) {
    let value = if alias == "matching" { recorded } else { alias };
    let attributes = openinference_attributes
        .into_iter()
        .chain([(key.into(), value.into())])
        .collect();
    assert_eq!(
        estimate_cost(&attributes, &openinference_catalog),
        (alias == "matching").then_some(130.0)
    );
}

#[rstest]
#[case::missing_input(None, Some("0"), None)]
#[case::missing_output(Some("0"), None, None)]
#[case::explicit_zero(Some("0"), Some("0"), Some(0.0))]
#[case::invalid_input(Some("bad"), Some("0"), None)]
#[case::negative_output(Some("0"), Some("-1"), None)]
fn openinference_missing_invalid_and_zero_usage_remain_distinct(
    openinference_catalog: PricingCatalog,
    #[case] input: Option<&str>,
    #[case] output: Option<&str>,
    #[case] expected: Option<f64>,
) {
    let attributes = [
        ("llm.model_name", Some("text-model")),
        ("llm.token_count.prompt", input),
        ("llm.token_count.completion", output),
    ]
    .into_iter()
    .filter_map(|(key, value)| value.map(|value| (key.into(), value.into())))
    .collect();
    assert_eq!(estimate_cost(&attributes, &openinference_catalog), expected);
}

#[rstest]
#[case::served_match("gen_ai.response.model", "text-model", Some(130.0))]
#[case::provider_prefix("gen_ai.response.model", "moonshot/text-model", Some(130.0))]
#[case::case_alias("gen_ai.response.model", "TEXT-MODEL", Some(130.0))]
#[case::declared_alias("gen_ai.response.model", "short", Some(130.0))]
#[case::distinct_returned_model("gen_ai.response.model", "other-model", Some(260.0))]
#[case::unknown_served("gen_ai.response.model", "unknown-model", None)]
#[case::empty_served("gen_ai.response.model", "", None)]
#[case::requested_alias("gen_ai.request.model", "requested-alias", Some(130.0))]
fn returned_model_wins_over_openinference_and_requested_aliases(
    openinference_attributes: BTreeMap<String, String>,
    openinference_catalog: PricingCatalog,
    #[case] key: &str,
    #[case] value: &str,
    #[case] expected: Option<f64>,
) {
    let attributes = openinference_attributes
        .into_iter()
        .chain([(key.into(), value.into())])
        .collect();
    assert_eq!(estimate_cost(&attributes, &openinference_catalog), expected);
}

#[rstest]
#[case::hosting_and_family(Some("azure"), None, Some(120.0))]
#[case::matching_hosting(Some("azure"), Some("azure"), Some(120.0))]
#[case::canonical_hosting_alias(Some("azure"), Some("azure.ai.openai"), Some(120.0))]
#[case::conflicting_hosting(Some("azure"), Some("openai"), None)]
#[case::system_fallback(None, None, Some(240.0))]
fn openinference_provider_and_api_family_are_distinct(
    #[case] provider: Option<&str>,
    #[case] genai_provider: Option<&str>,
    #[case] expected: Option<f64>,
) {
    let catalog = PricingCatalog::parse(br#"{
        "azure/text-model":{"litellm_provider":"azure","input_cost_per_token":1,"output_cost_per_token":2},
        "openai/text-model":{"litellm_provider":"openai","input_cost_per_token":2,"output_cost_per_token":4}
    }"#).unwrap();
    let attributes = [
        ("llm.model_name", Some("text-model")),
        ("llm.system", Some("openai")),
        ("llm.provider", provider),
        ("gen_ai.provider.name", genai_provider),
        ("llm.token_count.prompt", Some("100")),
        ("llm.token_count.completion", Some("10")),
    ]
    .into_iter()
    .filter_map(|(key, value)| value.map(|value| (key.into(), value.into())))
    .collect();
    assert_eq!(estimate_cost(&attributes, &catalog), expected);
}
