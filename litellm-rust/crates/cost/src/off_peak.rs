use crate::wire::py_float_unless_bool;
use jiff::{Timestamp, tz::TimeZone};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenRates {
    pub input_rate: f64,
    pub output_rate: f64,
    pub cache_read_rate: f64,
    pub cache_creation_rate: f64,
    pub reasoning_rate: Option<f64>,
}

impl TokenRates {
    pub fn billed_reasoning_rate(self) -> f64 {
        self.reasoning_rate.unwrap_or(self.output_rate)
    }
}

fn window_minutes(value: &str) -> Option<(i16, i16)> {
    fn minute(value: &str) -> Option<i16> {
        let (hour, minute) = value.trim().split_once(':')?;
        let hour = hour.parse::<i16>().ok()?;
        let minute = minute.parse::<i16>().ok()?;
        ((0..24).contains(&hour) && (0..60).contains(&minute)).then_some(hour * 60 + minute)
    }
    let (start, end) = value.split_once('-')?;
    Some((minute(start)?, minute(end)?))
}

fn windows(value: &Value) -> Vec<&str> {
    match value {
        Value::String(value) => vec![value],
        Value::Array(values) => values.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    }
}

pub fn is_within_off_peak_window(hours_utc: &Value, at: Timestamp) -> bool {
    let utc = at.to_zoned(TimeZone::UTC);
    let minute = i16::from(utc.hour()) * 60 + i16::from(utc.minute());
    windows(hours_utc).into_iter().any(|window| {
        let Some((start, end)) = window_minutes(window) else {
            return false;
        };
        if start < end {
            start <= minute && minute < end
        } else {
            minute >= start || minute < end
        }
    })
}

fn weekday(value: &Value) -> Option<i8> {
    match value {
        Value::Number(number) => i8::try_from(number.as_i64()?)
            .ok()
            .filter(|day| (1..=7).contains(day)),
        Value::String(day) => match day.trim().to_ascii_lowercase().as_str() {
            "mon" | "monday" => Some(1),
            "tue" | "tues" | "tuesday" => Some(2),
            "wed" | "wednesday" => Some(3),
            "thu" | "thur" | "thurs" | "thursday" => Some(4),
            "fri" | "friday" => Some(5),
            "sat" | "saturday" => Some(6),
            "sun" | "sunday" => Some(7),
            _ => None,
        },
        _ => None,
    }
}

fn matches_weekdays(rule: &Value, at: Timestamp, timezone: &TimeZone) -> bool {
    let Some(days) = rule.get("weekdays").filter(|value| !value.is_null()) else {
        return true;
    };
    let Some(days) = days.as_array() else {
        return false;
    };
    let current = at
        .to_zoned(timezone.clone())
        .weekday()
        .to_monday_one_offset();
    days.iter().any(|day| weekday(day) == Some(current))
}

pub fn is_off_peak(off_peak: &Value, at: Timestamp) -> bool {
    let Some(block) = off_peak.as_object() else {
        return false;
    };
    if block
        .get("hours_utc")
        .is_some_and(|hours| is_within_off_peak_window(hours, at))
    {
        return true;
    }
    let Some(rules) = block.get("windows").and_then(Value::as_array) else {
        return false;
    };
    let timezone = block
        .get("weekday_timezone")
        .and_then(Value::as_str)
        .and_then(|name| TimeZone::get(name.trim()).ok())
        .unwrap_or(TimeZone::UTC);
    rules.iter().any(|rule| {
        rule.get("hours_utc")
            .is_some_and(|hours| is_within_off_peak_window(hours, at))
            && matches_weekdays(rule, at, &timezone)
    })
}

pub fn open_off_peak_block(model_info: &Value, at: Timestamp) -> Option<&Value> {
    model_info
        .get("off_peak_pricing")
        .filter(|block| is_off_peak(block, at))
}

pub fn parse_off_peak_rate(value: Option<&Value>) -> Option<f64> {
    value.and_then(py_float_unless_bool)
}

pub fn apply_off_peak_pricing(model_info: &Value, at: Timestamp, rates: TokenRates) -> TokenRates {
    let Some(off_peak) = open_off_peak_block(model_info, at) else {
        return rates;
    };
    let rate = |key, standard| parse_off_peak_rate(off_peak.get(key)).unwrap_or(standard);
    TokenRates {
        input_rate: rate("input_cost_per_token", rates.input_rate),
        output_rate: rate("output_cost_per_token", rates.output_rate),
        cache_read_rate: rate("cache_read_input_token_cost", rates.cache_read_rate),
        cache_creation_rate: rate("cache_creation_input_token_cost", rates.cache_creation_rate),
        reasoning_rate: parse_off_peak_rate(off_peak.get("output_cost_per_reasoning_token"))
            .or(rates.reasoning_rate),
    }
}
