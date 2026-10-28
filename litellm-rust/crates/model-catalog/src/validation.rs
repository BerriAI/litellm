use std::collections::BTreeSet;

use serde_json::{Map, Value};
use thiserror::Error;

use crate::{AliasIssue, Catalog, ModelInfo, UtcHours, Weekday};

#[derive(Debug, Error)]
pub enum RegistryValidationError {
    #[error("invalid registry entry {model:?}")]
    Entry {
        model: String,
        #[source]
        reason: EntryViolation,
    },
    #[error(transparent)]
    Alias(AliasIssue),
}

#[derive(Debug, Error)]
pub enum EntryViolation {
    #[error("entry must be an object")]
    NotObject,
    #[error("aliases must be an array")]
    AliasesNotArray,
    #[error("aliases must contain strings")]
    AliasNotString,
    #[error("entry does not match the model schema")]
    Schema(#[source] serde_json::Error),
    #[error("litellm_provider is required")]
    MissingProvider,
    #[error("invalid deprecation_date")]
    DeprecationDate(#[source] time::error::Parse),
    #[error("off_peak_pricing needs hours or windows")]
    OffPeakWithoutHours,
    #[error("off_peak_pricing.hours_utc is invalid")]
    InvalidHours,
    #[error("off_peak_pricing.windows is empty")]
    EmptyWindows,
    #[error("off_peak_pricing.weekdays is invalid")]
    InvalidWeekdays,
    #[error("typed entry could not be serialized")]
    Serialize(#[source] serde_json::Error),
    #[error("entry has an unknown field, null, or changed value: {}", fields.join(", "))]
    ChangedFields { fields: Vec<String> },
    #[error("{field} must be nonnegative")]
    Negative { field: String },
    #[error("{field} must be at least one")]
    UpliftBelowOne { field: String },
}

pub fn validate_model_entry(model: &str, value: &Value) -> Result<(), RegistryValidationError> {
    validate_entry_inner(value).map_err(|reason| RegistryValidationError::Entry {
        model: model.to_owned(),
        reason,
    })
}

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

fn symmetric_difference(left: &BTreeSet<String>, right: &BTreeSet<String>) -> Vec<String> {
    left.symmetric_difference(right).cloned().collect()
}

fn validate_entry_inner(value: &Value) -> Result<(), EntryViolation> {
    let object = value.as_object().ok_or(EntryViolation::NotObject)?;
    if let Some(aliases) = object.get("aliases") {
        let names = aliases.as_array().ok_or(EntryViolation::AliasesNotArray)?;
        if names.iter().any(|name| !name.is_string()) {
            return Err(EntryViolation::AliasNotString);
        }
    }
    let info: ModelInfo = serde_json::from_value(value.clone()).map_err(EntryViolation::Schema)?;
    if info.litellm_provider.is_none() {
        return Err(EntryViolation::MissingProvider);
    }
    validate_dates_and_windows(&info)?;
    let serialized = serde_json::to_value(info).map_err(EntryViolation::Serialize)?;
    let mut expected = object.clone();
    expected.remove("aliases");
    if !json_eq(&Value::Object(expected.clone()), &serialized) {
        let actual = serialized
            .as_object()
            .expect("ModelInfo serializes as an object");
        return Err(EntryViolation::ChangedFields {
            fields: symmetric_difference(&keys(&expected), &keys(actual)),
        });
    }
    check_prices("", value)
}

fn validate_dates_and_windows(info: &ModelInfo) -> Result<(), EntryViolation> {
    if let Some(date) = &info.deprecation_date {
        let format = time::format_description::parse_borrowed::<2>("[year]-[month]-[day]").unwrap();
        time::Date::parse(date, &format).map_err(EntryViolation::DeprecationDate)?;
    }
    let Some(pricing) = &info.off_peak_pricing else {
        return Ok(());
    };
    if pricing.hours_utc.is_none() && pricing.windows.is_none() {
        return Err(EntryViolation::OffPeakWithoutHours);
    }
    if let Some(hours) = &pricing.hours_utc {
        validate_hours(hours)?;
    }
    if let Some(windows) = &pricing.windows {
        if windows.is_empty() {
            return Err(EntryViolation::EmptyWindows);
        }
        windows.iter().try_for_each(|window| {
            validate_hours(&window.hours_utc)?;
            if let Some(days) = &window.weekdays
                && (days.is_empty() || days.iter().any(|day| !valid_weekday(day)))
            {
                return Err(EntryViolation::InvalidWeekdays);
            }
            Ok(())
        })?;
    }
    Ok(())
}

fn validate_hours(hours: &UtcHours) -> Result<(), EntryViolation> {
    let values = match hours {
        UtcHours::Single(value) => std::slice::from_ref(value),
        UtcHours::Multiple(values) => values.as_slice(),
    };
    if values.is_empty() || values.iter().any(|value| !valid_utc_window(value)) {
        return Err(EntryViolation::InvalidHours);
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

fn check_prices(path: &str, value: &Value) -> Result<(), EntryViolation> {
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
            return Err(EntryViolation::Negative {
                field: relative(&field_path),
            });
        }
        if key.contains("uplift_multiplier")
            && let Some(number) = field.as_f64()
            && number < 1.0
        {
            return Err(EntryViolation::UpliftBelowOne {
                field: relative(&field_path),
            });
        }
        if matches!(key.as_str(), "range" | "max_results_range")
            && field.as_array().is_some_and(|values| {
                values
                    .iter()
                    .any(|value| value.as_f64().is_some_and(|n| n < 0.0))
            })
        {
            return Err(EntryViolation::Negative {
                field: relative(&field_path),
            });
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

fn relative(field_path: &str) -> String {
    field_path.trim_start_matches('.').to_owned()
}
