// mirrors: crates/cost/tests/generate_python_reference.py

#![allow(clippy::disallowed_types)]

use jiff::Timestamp;
use litellm_cost::generic_cost::calculate_generic_cost_from_model_info_with_region;
use litellm_cost::responses_usage::ChatUsage;
use rstest::rstest;
use serde_json::Value;

fn chat_usage(prompt: u64, completion: u64, cache_read: u64, cache_write: u64) -> ChatUsage {
    ChatUsage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt + completion,
        prompt_tokens_details: (cache_read > 0 || cache_write > 0).then(|| {
            litellm_cost::responses_usage::PromptTokenDetails {
                cached_tokens: cache_read,
                cache_write_tokens: Some(cache_write),
                ..Default::default()
            }
        }),
        completion_tokens_details: None,
        cost: None,
        extra: Default::default(),
    }
}

fn model_info(fields: &[(&str, String)]) -> Value {
    let object = fields
        .iter()
        .map(|(key, value)| {
            (
                key.to_string(),
                serde_json::from_str::<Value>(value).expect("rate parses"),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    Value::Object(object)
}

#[rstest]
fn generic_path_matches_executed_python_reference_cases() {
    for row in include_str!("python_reference.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let fields: Vec<_> = row.split('\t').collect();
        let count = |index: usize| fields[index].parse::<u64>().unwrap();
        let number = |index: usize| fields[index].parse::<f64>().unwrap();
        let rate_field = |index: usize| {
            fields[index]
                .parse::<f64>()
                .ok()
                .map(|rate| rate.to_string())
        };
        let mut entries = vec![
            ("input_cost_per_token", number(5).to_string()),
            ("output_cost_per_token", number(6).to_string()),
        ];
        let threshold_keys: Vec<String>;
        if let Some(cache_read) = rate_field(7) {
            entries.push(("cache_read_input_token_cost", cache_read));
        }
        if let Some(cache_write) = rate_field(8) {
            entries.push(("cache_creation_input_token_cost", cache_write));
        }
        if !fields[9].is_empty() {
            let threshold = count(9);
            threshold_keys = vec![
                format!("input_cost_per_token_above_{threshold}_tokens"),
                format!("output_cost_per_token_above_{threshold}_tokens"),
            ];
            entries.push((threshold_keys[0].as_str(), number(10).to_string()));
            entries.push((threshold_keys[1].as_str(), number(11).to_string()));
        }
        let usage = chat_usage(count(1), count(2), count(3), count(4));
        let (input, output) = calculate_generic_cost_from_model_info_with_region(
            &usage,
            &model_info(&entries),
            None,
            false,
            None,
            None,
            "2026-01-01T12:00Z".parse::<Timestamp>().unwrap(),
        );
        let name = &fields[0];
        assert!((input - number(12)).abs() < 1e-9, "{name}: input {input}");
        assert!(
            (output - number(13)).abs() < 1e-9,
            "{name}: output {output}"
        );
    }
}
