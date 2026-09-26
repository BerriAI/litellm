#![allow(clippy::disallowed_types)]

// mirrors: crates/cost/tests/generate_python_fixtures.py

use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::Rate;
use litellm_cost::anthropic_usage::transform_anthropic_usage_to_chat_usage;
use litellm_cost::base_rate_selection::uses_inclusive_token_thresholds;
use litellm_cost::batch::batch_cost_rates_from_model_info;
use litellm_cost::billed_token_rates::{BilledRatesRequest, calculate_billed_token_rates};
use litellm_cost::generic_cost::calculate_generic_cost_from_model_info_with_region;
use litellm_cost::off_peak::is_off_peak;
use litellm_cost::prompt_caching_savings::{
    PromptCachingSavingsRequest, calculate_prompt_caching_savings,
};
use litellm_cost::provider_cache::apply_provider_cache_read_default;
use litellm_cost::usage_dispatch::get_usage_object;
use rstest::rstest;
use serde_json::{Value, json};

fn fixtures() -> Value {
    serde_json::from_str(include_str!("python_fixtures.json")).unwrap()
}

fn instant(iso: &str) -> Timestamp {
    iso.parse().unwrap()
}

fn opt_str<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn chat_usage(payload: &Value) -> litellm_cost::responses_usage::ChatUsage {
    get_usage_object(&json!({"usage": payload}))
        .unwrap()
        .unwrap()
}

fn assert_close(actual: f64, expected: f64, id: &str) {
    assert!(
        (actual - expected).abs() <= expected.abs() * 1e-12,
        "{id}: rust={actual:?} python={expected:?}"
    );
}

#[rstest]
fn generic_token_costs_match_executed_python_fixtures() {
    for row in fixtures().get("generic").unwrap().as_array().unwrap() {
        let id = row.get("id").and_then(Value::as_str).unwrap();
        let usage = chat_usage(row.get("usage").unwrap());
        let provider = opt_str(row, "custom_llm_provider");
        let model_info =
            apply_provider_cache_read_default(row.get("model_info").unwrap(), provider);
        let at = opt_str(row, "current_time")
            .map_or_else(|| "2026-03-12T00:00:00Z".parse().unwrap(), instant);
        let (input_cost, output_cost) = calculate_generic_cost_from_model_info_with_region(
            &usage,
            &model_info,
            opt_str(row, "service_tier"),
            uses_inclusive_token_thresholds(provider),
            opt_str(row, "data_residency"),
            opt_str(row, "vertex_location"),
            at,
        );
        assert_close(
            input_cost,
            row.get("expected_input_cost").unwrap().as_f64().unwrap(),
            id,
        );
        assert_close(
            output_cost,
            row.get("expected_output_cost").unwrap().as_f64().unwrap(),
            id,
        );
    }
}

#[rstest]
fn billed_token_rates_match_executed_python_fixtures() {
    for row in fixtures().get("billed_rates").unwrap().as_array().unwrap() {
        let id = row.get("id").and_then(Value::as_str).unwrap();
        let mut payload = row.get("usage").unwrap().clone();
        if let Some(extra) = row.get("usage_extra").and_then(Value::as_object) {
            for (key, value) in extra {
                payload
                    .as_object_mut()
                    .unwrap()
                    .insert(key.clone(), value.clone());
            }
        }
        let usage = chat_usage(&payload);
        let provider = opt_str(row, "custom_llm_provider");
        let at = opt_str(row, "current_time")
            .map_or_else(|| "2026-03-12T00:00:00Z".parse().unwrap(), instant);
        let rates = calculate_billed_token_rates(BilledRatesRequest {
            model_info: Some(row.get("model_info").unwrap()),
            usage: &usage,
            provider,
            service_tier: opt_str(row, "service_tier"),
            data_residency: opt_str(row, "data_residency"),
            vertex_location: opt_str(row, "vertex_location"),
            at,
            custom_cost_per_token: None,
        })
        .unwrap_or_else(|| panic!("{id}: rates unresolved"));
        let expected = row.get("expected").unwrap();
        for field in [
            "input_cost_per_token",
            "output_cost_per_token",
            "cache_read_input_token_cost",
            "cache_read_input_audio_token_cost",
            "cache_creation_input_token_cost",
            "cache_creation_input_token_cost_above_1hr",
            "output_cost_per_reasoning_token",
        ] {
            let want = expected.get(field).unwrap().as_f64().unwrap();
            let got = match field {
                "input_cost_per_token" => rates.input_cost_per_token,
                "output_cost_per_token" => rates.output_cost_per_token,
                "cache_read_input_token_cost" => rates.cache_read_input_token_cost,
                "cache_read_input_audio_token_cost" => rates.cache_read_input_audio_token_cost,
                "cache_creation_input_token_cost" => rates.cache_creation_input_token_cost,
                "cache_creation_input_token_cost_above_1hr" => {
                    rates.cache_creation_input_token_cost_above_1hr
                }
                _ => rates.output_cost_per_reasoning_token,
            };
            assert_close(got, want, &format!("{id}/{field}"));
        }
    }
}

#[rstest]
fn batch_cost_rates_match_executed_python_fixtures() {
    for row in fixtures().get("batch").unwrap().as_array().unwrap() {
        let id = row.get("id").and_then(Value::as_str).unwrap();
        let model_info = row.get("model_info").unwrap();
        let provider = opt_str(row, "custom_llm_provider").unwrap();
        let prompt_tokens = row
            .get("usage")
            .unwrap()
            .get("prompt_tokens")
            .unwrap()
            .as_u64()
            .unwrap();
        let rates = batch_cost_rates_from_model_info(
            model_info,
            prompt_tokens,
            uses_inclusive_token_thresholds(Some(provider)),
        );
        let as_option = |rate: Rate| rate.value();
        let option_equals = |actual: Option<f64>, key: &str| match (actual, row.get(key)) {
            (Some(got), Some(want)) => {
                assert_close(got, want.as_f64().unwrap(), &format!("{id}/{key}"))
            }
            (None, None) | (None, Some(Value::Null)) => {}
            (got, want) => panic!("{id}/{key}: rust={got:?} python={want:?}"),
        };
        option_equals(as_option(rates.input), "expected_input");
        option_equals(as_option(rates.output), "expected_output");
        option_equals(as_option(rates.cache_read), "expected_cache_read");
        option_equals(as_option(rates.cache_creation), "expected_cache_creation");
    }
}

#[rstest]
fn off_peak_decisions_match_executed_python_fixtures() {
    for row in fixtures().get("off_peak").unwrap().as_array().unwrap() {
        let id = row.get("id").and_then(Value::as_str).unwrap();
        let block = row.get("off_peak").unwrap();
        let at = instant(row.get("current_time").and_then(Value::as_str).unwrap());
        let expected = row.get("expected_is_off_peak").unwrap().as_bool().unwrap();
        assert_eq!(is_off_peak(block, at), expected, "{id}");
    }
}

#[rstest]
fn responses_usage_transforms_match_executed_python_fixtures() {
    for row in fixtures()
        .get("responses_usage")
        .unwrap()
        .as_array()
        .unwrap()
    {
        let id = row.get("id").and_then(Value::as_str).unwrap();
        let usage = get_usage_object(&json!({"usage": row.get("raw").unwrap()}))
            .unwrap()
            .unwrap_or_else(|| panic!("{id}: usage unparsed"));
        assert_usage_matches(&usage, row.get("expected").unwrap(), id);
    }
}

#[rstest]
fn anthropic_usage_transforms_match_executed_python_fixtures() {
    for row in fixtures()
        .get("anthropic_usage")
        .unwrap()
        .as_array()
        .unwrap()
    {
        let id = row.get("id").and_then(Value::as_str).unwrap();
        let usage = transform_anthropic_usage_to_chat_usage(
            row.get("raw").unwrap(),
            None,
            row.get("response_has_thinking_block")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        )
        .unwrap();
        assert_usage_matches(&usage, row.get("expected").unwrap(), id);
    }
}

fn assert_usage_matches(
    usage: &litellm_cost::responses_usage::ChatUsage,
    expected: &Value,
    id: &str,
) {
    let count = |key: &str| expected.get(key).and_then(Value::as_u64).unwrap_or(0);
    assert_eq!(usage.prompt_tokens, count("prompt_tokens"), "{id}/prompt");
    assert_eq!(
        usage.completion_tokens,
        count("completion_tokens"),
        "{id}/completion"
    );
    assert_eq!(usage.total_tokens, count("total_tokens"), "{id}/total");
    let details = usage.prompt_tokens_details.clone().unwrap_or_default();
    let expected_details = expected.get("prompt_tokens_details");
    let detail = |key: &str| {
        expected_details
            .and_then(|d| d.get(key))
            .and_then(Value::as_u64)
    };
    assert_eq!(
        details.cached_tokens,
        count_of(expected_details, "cached_tokens"),
        "{id}/cached"
    );
    assert_eq!(
        details.web_search_requests,
        detail("web_search_requests"),
        "{id}/web_search_requests"
    );
    assert_eq!(
        details.cache_write_tokens,
        detail("cache_write_tokens"),
        "{id}/cache_write"
    );
    if let Some(want) = detail("cache_creation_tokens") {
        assert_eq!(
            details.cache_creation_tokens,
            Some(want),
            "{id}/cache_creation"
        );
    }
    if let (Some(want), Some(got)) = (
        expected_details
            .and_then(|d| d.get("cache_creation_token_details"))
            .and_then(Value::as_object),
        details.cache_creation_token_details.as_ref().map(|v| {
            json!({
                "ephemeral_5m_input_tokens": v.ephemeral_5m_input_tokens,
                "ephemeral_1h_input_tokens": v.ephemeral_1h_input_tokens,
            })
        }),
    ) {
        assert_eq!(
            got.get("ephemeral_5m_input_tokens").and_then(Value::as_u64),
            want.get("ephemeral_5m_input_tokens")
                .and_then(Value::as_u64),
            "{id}/ttl_5m"
        );
        assert_eq!(
            got.get("ephemeral_1h_input_tokens").and_then(Value::as_u64),
            want.get("ephemeral_1h_input_tokens")
                .and_then(Value::as_u64),
            "{id}/ttl_1h"
        );
    }
    let completion = usage.completion_tokens_details.clone();
    let expected_completion = expected.get("completion_tokens_details");
    assert_eq!(
        completion.as_ref().and_then(|d| d.reasoning_tokens),
        expected_completion
            .and_then(|d| d.get("reasoning_tokens"))
            .and_then(Value::as_u64),
        "{id}/reasoning"
    );
    assert_eq!(
        completion.as_ref().and_then(|d| d.text_tokens),
        expected_completion
            .and_then(|d| d.get("text_tokens"))
            .and_then(Value::as_u64),
        "{id}/text"
    );
    if let Some(server_tool_use) = expected.get("server_tool_use").and_then(Value::as_object) {
        for (key, value) in server_tool_use {
            assert_eq!(
                usage.extra.get("server_tool_use").and_then(|v| v.get(key)),
                Some(value),
                "{id}/server_tool_use/{key}"
            );
        }
    }
}

fn count_of(details: Option<&Value>, key: &str) -> u64 {
    details
        .and_then(|d| d.get(key))
        .and_then(Value::as_u64)
        .unwrap_or(0)
}

#[rstest]
fn prompt_caching_savings_match_executed_python_fixtures() {
    for row in fixtures()
        .get("caching_savings")
        .unwrap()
        .as_array()
        .unwrap()
    {
        let id = row.get("id").and_then(Value::as_str).unwrap();
        let usage = chat_usage(row.get("usage").unwrap());
        let savings = calculate_prompt_caching_savings(PromptCachingSavingsRequest {
            model_info: row.get("model_info").unwrap(),
            usage: &usage,
            provider: opt_str(row, "custom_llm_provider"),
            service_tier: opt_str(row, "service_tier"),
            data_residency: opt_str(row, "data_residency"),
            vertex_location: opt_str(row, "vertex_location"),
            at: "2026-03-12T00:00:00Z".parse().unwrap(),
        });
        assert_close(
            savings,
            row.get("expected_savings").unwrap().as_f64().unwrap(),
            id,
        );
    }
}

#[rstest]
fn fixture_ids_are_unique_per_surface() {
    let mut seen: HashMap<String, usize> = HashMap::new();
    let root = fixtures();
    for surface in [
        "generic",
        "batch",
        "billed_rates",
        "off_peak",
        "responses_usage",
        "caching_savings",
        "anthropic_usage",
    ] {
        for row in root.get(surface).unwrap().as_array().unwrap() {
            let id = row.get("id").and_then(Value::as_str).unwrap().to_owned();
            *seen.entry(format!("{surface}/{id}")).or_default() += 1;
        }
    }
    assert!(
        seen.values().all(|count| *count == 1),
        "duplicate fixture ids: {:?}",
        seen.iter().filter(|(_, c)| **c > 1).collect::<Vec<_>>()
    );
}
