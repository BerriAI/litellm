use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use litellm_model_catalog::{
    Catalog, FallbackGeneralizations, ModelInfo, Provenance, UtcHours, Weekday,
};
use rstest::{fixture, rstest};
use serde_json::{Map, Value};

#[fixture]
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn json_eq(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => left.as_f64() == right.as_f64(),
        (Value::Array(left), Value::Array(right)) => {
            left.len() == right.len() && left.iter().zip(right).all(|(a, b)| json_eq(a, b))
        }
        (Value::Object(left), Value::Object(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .all(|(key, value)| right.get(key).is_some_and(|other| json_eq(value, other)))
        }
        _ => left == right,
    }
}

fn keys(value: &Map<String, Value>) -> BTreeSet<String> {
    value.keys().cloned().collect()
}

fn symmetric_difference(left: &BTreeSet<String>, right: &BTreeSet<String>) -> BTreeSet<String> {
    left.symmetric_difference(right).cloned().collect()
}

fn validate_entry(model_name: &str, value: &Value) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{model_name} must be an object"))?;
    let info: ModelInfo =
        serde_json::from_value(value.clone()).map_err(|error| format!("{model_name}: {error}"))?;
    if info.litellm_provider.is_none() {
        return Err(format!("{model_name}.litellm_provider is required"));
    }
    validate_dates_and_windows(model_name, &info)?;
    let serialized = serde_json::to_value(info).map_err(|error| error.to_string())?;
    let mut expected = object.clone();
    expected.remove("aliases");
    if !json_eq(&Value::Object(expected.clone()), &serialized) {
        let actual = serialized
            .as_object()
            .expect("ModelInfo serializes as an object");
        return Err(format!(
            "{model_name} has an unknown field, null, or changed value: {:?}",
            symmetric_difference(&keys(&expected), &keys(actual))
        ));
    }
    check_prices(model_name, value)
}

fn validate_dates_and_windows(model_name: &str, info: &ModelInfo) -> Result<(), String> {
    if let Some(date) = &info.deprecation_date {
        let format = time::format_description::parse_borrowed::<2>("[year]-[month]-[day]").unwrap();
        time::Date::parse(date, &format)
            .map_err(|error| format!("{model_name}.deprecation_date: {error}"))?;
    }
    let Some(pricing) = &info.off_peak_pricing else {
        return Ok(());
    };
    if pricing.hours_utc.is_none() && pricing.windows.is_none() {
        return Err(format!(
            "{model_name}.off_peak_pricing needs hours or windows"
        ));
    }
    if let Some(hours) = &pricing.hours_utc {
        validate_hours(hours)?;
    }
    if let Some(windows) = &pricing.windows {
        if windows.is_empty() {
            return Err(format!("{model_name}.off_peak_pricing.windows is empty"));
        }
        for window in windows {
            validate_hours(&window.hours_utc)?;
            if let Some(days) = &window.weekdays
                && (days.is_empty() || days.iter().any(|day| !valid_weekday(day)))
            {
                return Err(format!("{model_name}.off_peak_pricing.weekdays is invalid"));
            }
        }
    }
    Ok(())
}

fn validate_hours(hours: &UtcHours) -> Result<(), String> {
    let values = match hours {
        UtcHours::Single(value) => std::slice::from_ref(value),
        UtcHours::Multiple(values) => values.as_slice(),
    };
    if values.is_empty() || values.iter().any(|value| !valid_utc_window(value)) {
        return Err("off_peak_pricing.hours_utc is invalid".into());
    }
    Ok(())
}

fn valid_utc_window(value: &str) -> bool {
    let Some((start, end)) = value.split_once('-') else {
        return false;
    };
    [start, end].into_iter().all(|clock| {
        let Some((hour, minute)) = clock.split_once(':') else {
            return false;
        };
        hour.len() == 2
            && minute.len() == 2
            && hour.parse::<u8>().is_ok_and(|hour| hour < 24)
            && minute.parse::<u8>().is_ok_and(|minute| minute < 60)
    })
}

fn valid_weekday(day: &Weekday) -> bool {
    match day {
        Weekday::Number(number) => (1..=7).contains(number),
        Weekday::Name(name) => matches!(
            name.to_ascii_lowercase().as_str(),
            "mon"
                | "monday"
                | "tue"
                | "tues"
                | "tuesday"
                | "wed"
                | "wednesday"
                | "thu"
                | "thur"
                | "thurs"
                | "thursday"
                | "fri"
                | "friday"
                | "sat"
                | "saturday"
                | "sun"
                | "sunday"
        ),
    }
}

fn check_prices(path: &str, value: &Value) -> Result<(), String> {
    let Some(object) = value.as_object() else {
        return Ok(());
    };
    for (key, field) in object {
        let field_path = format!("{path}.{key}");
        if (key.contains("cost")
            || path.ends_with(".guardrail_cost_per_unit")
            || path.ends_with(".search_context_cost_per_query"))
            && let Some(number) = field.as_f64()
            && number < 0.0
        {
            return Err(format!("{field_path} must be nonnegative"));
        }
        if key.contains("uplift_multiplier")
            && let Some(number) = field.as_f64()
            && number < 1.0
        {
            return Err(format!("{field_path} must be at least one"));
        }
        if matches!(key.as_str(), "range" | "max_results_range")
            && field.as_array().is_some_and(|values| {
                values
                    .iter()
                    .any(|value| value.as_f64().is_some_and(|n| n < 0.0))
            })
        {
            return Err(format!("{field_path} must be nonnegative"));
        }
        if !matches!(key.as_str(), "metadata" | "provider_specific_entry") {
            if let Some(items) = field.as_array() {
                for (index, item) in items.iter().enumerate() {
                    check_prices(&format!("{field_path}[{index}]"), item)?;
                }
            } else {
                check_prices(&field_path, field)?;
            }
        }
    }
    Ok(())
}

#[rstest]
#[case("model_prices_and_context_window.json")]
#[case("litellm/model_prices_and_context_window_backup.json")]
fn every_entry_round_trips_through_model_info(repo_root: PathBuf, #[case] filename: &str) {
    let body = std::fs::read(repo_root.join(filename)).unwrap();
    let document: IndexMap<String, Value> = serde_json::from_slice(&body).unwrap();
    for (model_name, value) in document {
        if matches!(
            model_name.as_str(),
            "sample_spec" | "fallback_generalizations"
        ) {
            continue;
        }
        validate_entry(&model_name, &value).unwrap();
    }
}

#[rstest]
fn fallback_generalizations_are_typed(repo_root: PathBuf) {
    let body = std::fs::read(repo_root.join("model_prices_and_context_window.json")).unwrap();
    let document: Map<String, Value> = serde_json::from_slice(&body).unwrap();
    let Some(raw_rules) = document.get("fallback_generalizations") else {
        return;
    };
    let _: FallbackGeneralizations = serde_json::from_value(raw_rules.clone()).unwrap();
    let catalog = Catalog::parse(&body, Provenance::default()).unwrap();
    assert!(
        catalog
            .fallback_rules()
            .is_some_and(|rules| !rules.is_empty())
    );
}

#[test]
fn registry_validation_rejects_missing_provider_unknown_fields_and_negative_prices() {
    for (entry, expected) in [
        (serde_json::json!({"mode": "chat"}), "litellm_provider"),
        (
            serde_json::json!({"litellm_provider": "test", "typo": true}),
            "unknown field",
        ),
        (
            serde_json::json!({"litellm_provider": "test", "input_cost_per_token": -1}),
            "nonnegative",
        ),
        (
            serde_json::json!({"litellm_provider": "test", "guardrail_cost_per_unit": {"unit": -1}}),
            "nonnegative",
        ),
        (
            serde_json::json!({"litellm_provider": "test", "mode": "invalid"}),
            "unknown variant",
        ),
        (
            serde_json::json!({"litellm_provider": "test", "deprecation_date": "2026-02-31"}),
            "deprecation_date",
        ),
        (
            serde_json::json!({"litellm_provider": "test", "off_peak_pricing": {"hours_utc": "25:00-01:00"}}),
            "hours_utc",
        ),
        (
            serde_json::json!({"litellm_provider": "test", "off_peak_pricing": {"windows": []}}),
            "windows is empty",
        ),
        (
            serde_json::json!({"litellm_provider": "test", "off_peak_pricing": {"windows": [{"hours_utc": "00:00-01:00", "weekdays": [0]}]}}),
            "weekdays is invalid",
        ),
    ] {
        assert!(
            validate_entry("test", &entry)
                .unwrap_err()
                .contains(expected)
        );
    }
}

#[test]
fn checked_in_catalog_and_backup_match() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let current = std::fs::read(root.join("model_prices_and_context_window.json")).unwrap();
    let backup =
        std::fs::read(root.join("litellm/model_prices_and_context_window_backup.json")).unwrap();
    assert_eq!(current, backup);
    let catalog = Catalog::parse(&current, Provenance::default()).unwrap();
    assert!(
        catalog.alias_issues().is_empty(),
        "invalid registry aliases"
    );
}
