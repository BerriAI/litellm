use chrono::{DateTime, Datelike, NaiveTime, Utc};
use chrono_tz::Tz;
use serde_json::{Map, Value};

fn hours(value: Option<&Value>, now: NaiveTime) -> bool {
    let contains = |window: &str| {
        let Some((start, end)) = window.split_once('-') else {
            return false;
        };
        let (Ok(start), Ok(end)) = (
            NaiveTime::parse_from_str(start.trim(), "%H:%M"),
            NaiveTime::parse_from_str(end.trim(), "%H:%M"),
        ) else {
            return false;
        };
        if start < end {
            start <= now && now < end
        } else {
            now >= start || now < end
        }
    };
    match value {
        Some(Value::String(window)) => contains(window),
        Some(Value::Array(windows)) => windows.iter().filter_map(Value::as_str).any(contains),
        _ => false,
    }
}

fn weekday(value: &Value) -> Option<u32> {
    match value {
        Value::Number(value) => value
            .as_u64()
            .filter(|day| (1..=7).contains(day))
            .map(|day| day as u32),
        Value::String(value) => match value.trim().to_lowercase().as_str() {
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

pub(super) fn active(block: &Map<String, Value>, start_ns: i64) -> bool {
    let utc = DateTime::<Utc>::from_timestamp_nanos(start_ns);
    if hours(block.get("hours_utc"), utc.time()) {
        return true;
    }
    let Some(windows) = block.get("windows").and_then(Value::as_array) else {
        return false;
    };
    let zone = block
        .get("weekday_timezone")
        .and_then(Value::as_str)
        .and_then(|name| name.trim().parse::<Tz>().ok())
        .unwrap_or(chrono_tz::UTC);
    let day = utc.with_timezone(&zone).weekday().number_from_monday();
    windows.iter().filter_map(Value::as_object).any(|rule| {
        let matches_day = match rule.get("weekdays") {
            None | Some(Value::Null) => true,
            Some(Value::Array(days)) => days
                .iter()
                .filter_map(weekday)
                .any(|allowed| allowed == day),
            _ => false,
        };
        matches_day && hours(rule.get("hours_utc"), utc.time())
    })
}
