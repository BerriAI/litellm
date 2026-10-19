use std::collections::BTreeSet;

use serde_json::{Map, Value};
use thiserror::Error;

use crate::{AliasIssue, Catalog, ModelInfo, UtcHours, Weekday};

/// A registry entry violates the checked-in catalog contract.
#[derive(Debug, Error)]
pub enum RegistryValidationError {
    #[error("{reason}")]
    Entry { model: String, reason: String },
    #[error("alias issue: {0:?}")]
    Alias(AliasIssue),
}

/// Validate one registry entry without restricting the tolerant catalog reader.
pub fn validate_model_entry(model: &str, value: &Value) -> Result<(), RegistryValidationError> {
    validate_entry_inner(model, value).map_err(|reason| RegistryValidationError::Entry {
        model: model.to_owned(),
        reason,
    })
}

/// Check every model and alias in a parsed catalog against registry rules.
pub fn validate_registry(catalog: &Catalog) -> Result<(), RegistryValidationError> {
    if let Some(issue) = catalog.alias_issues().first() {
        return Err(RegistryValidationError::Alias(issue.clone()));
    }
    catalog.model_names().try_for_each(|name| {
        let entry = catalog.lookup(name).expect("catalog name must resolve");
        validate_model_entry(name, &Value::Object(entry.entry.fields().clone()))
    })
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

fn validate_entry_inner(model_name: &str, value: &Value) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{model_name} must be an object"))?;
    if let Some(aliases) = object.get("aliases") {
        let names = aliases
            .as_array()
            .ok_or_else(|| format!("{model_name}.aliases must be an array"))?;
        if names.iter().any(|name| !name.is_string()) {
            return Err(format!("{model_name}.aliases must contain strings"));
        }
    }
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
        windows.iter().try_for_each(|window| {
            validate_hours(&window.hours_utc)?;
            if let Some(days) = &window.weekdays
                && (days.is_empty() || days.iter().any(|day| !valid_weekday(day)))
            {
                return Err(format!("{model_name}.off_peak_pricing.weekdays is invalid"));
            }
            Ok(())
        })?;
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
    object.iter().try_for_each(|(key, field)| {
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
        if matches!(key.as_str(), "metadata" | "provider_specific_entry") {
            return Ok(());
        }
        match field.as_array() {
            Some(items) => items.iter().enumerate().try_for_each(|(index, item)| {
                check_prices(&format!("{field_path}[{index}]"), item)
            }),
            None => check_prices(&field_path, field),
        }
    })
}
