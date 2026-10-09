use serde::{Deserialize, Deserializer, de::DeserializeOwned};

pub(super) fn deserialize<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: DeserializeOwned,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Number {
        Quoted(String),
        Unquoted(serde_json::Number),
    }
    match Number::deserialize(deserializer)? {
        Number::Quoted(value) => serde_json::from_str(&value),
        Number::Unquoted(value) => serde_json::from_value(serde_json::Value::Number(value)),
    }
    .map_err(serde::de::Error::custom)
}

pub(super) fn optional_finite<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<f64>, D::Error> {
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    let Some(value) = value else {
        return Ok(None);
    };
    let number: f64 = deserialize(value).map_err(serde::de::Error::custom)?;
    if number.is_finite() {
        Ok(Some(number))
    } else {
        Err(serde::de::Error::custom("expected finite spend"))
    }
}

pub(super) fn flag<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u8, D::Error> {
    match deserialize(deserializer)? {
        value @ 0..=1 => Ok(value),
        _ => Err(serde::de::Error::custom("expected 0 or 1")),
    }
}

pub(super) fn percent<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    let value: f64 = deserialize(deserializer)?;
    if value.is_finite() && (0.0..=100.0).contains(&value) {
        Ok(value)
    } else {
        Err(serde::de::Error::custom(
            "expected a finite percentage between 0 and 100",
        ))
    }
}

pub(super) fn boolean<'de, D: Deserializer<'de>>(deserializer: D) -> Result<bool, D::Error> {
    flag(deserializer).map(|value| value == 1)
}

#[cfg(test)]
mod tests {
    use crate::query::named::SpanErrorRow;
    use rstest::rstest;

    #[rstest]
    #[case::flag_zero(serde_json::json!(0), true)]
    #[case::flag_one(serde_json::json!("1"), true)]
    #[case::invalid_flag(serde_json::json!(2), false)]
    fn access_rejects_non_boolean_flags(#[case] value: serde_json::Value, #[case] valid: bool) {
        let parameters = serde_json::json!({"all_teams": value, "team": "team", "key_hash": ""});
        assert_eq!(
            serde_json::from_value::<crate::query::lens::LensAccessParams>(parameters).is_ok(),
            valid
        );
    }

    #[rstest]
    #[case::zero(serde_json::json!(0), true)]
    #[case::hundred(serde_json::json!("100"), true)]
    #[case::negative(serde_json::json!(-0.1), false)]
    #[case::too_large(serde_json::json!(100.1), false)]
    #[case::nan(serde_json::json!("NaN"), false)]
    fn sampling_rejects_invalid_percentages(#[case] value: serde_json::Value, #[case] valid: bool) {
        let parameters = serde_json::json!({
            "all_teams": 0, "team": "team", "key_hash": "", "source": "both", "start": 0, "end": 1,
            "agent_name": "", "service": "", "filter_keys": [], "filter_values": [], "selected_team": "",
            "execution_ids": [], "sample_cap": 0, "sample_percent": value, "preview": 0, "after": "",
            "limit": 10, "offset": 0
        });
        assert_eq!(
            serde_json::from_value::<crate::query::lens::LensSampleParams>(parameters).is_ok(),
            valid
        );
    }

    #[rstest]
    #[case::trace("traces", true)]
    #[case::request("requests", true)]
    #[case::both("both", false)]
    #[case::unknown("unknown", false)]
    fn content_rejects_unsupported_sources(#[case] source: &str, #[case] valid: bool) {
        let parameters = serde_json::json!({
            "all_teams": 0, "team": "team", "key_hash": "", "source": source, "id": "id",
            "record_team": "team", "start_time": "", "trace_ref": "", "cursor": "", "offset": 0
        });
        assert_eq!(
            serde_json::from_value::<crate::query::lens::LensContentParams>(parameters).is_ok(),
            valid
        );
    }

    #[rstest]
    #[case::quoted_max(serde_json::json!(u64::MAX.to_string()), Some(u64::MAX))]
    #[case::unquoted_max(serde_json::json!(u64::MAX), Some(u64::MAX))]
    #[case::overflow(serde_json::json!("18446744073709551616"), None)]
    #[case::negative(serde_json::json!(-1), None)]
    #[case::fraction(serde_json::json!(1.5), None)]
    fn numeric_rows_enforce_integer_range(
        #[case] value: serde_json::Value,
        #[case] expected: Option<u64>,
    ) {
        let row = serde_json::from_value::<SpanErrorRow>(serde_json::json!({
            "span_id": "span", "message": "error", "total_chars": value, "version": "hash"
        }));
        match expected {
            Some(value) => assert_eq!(row.unwrap().0.total_chars, value),
            None => assert!(row.is_err()),
        }
    }
}
