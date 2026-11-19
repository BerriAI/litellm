//! The contract fields the gateway can work out from a logged call.

use litellm_callbacks::{AttemptTiming, Field, LoggedKeyIdentity, MetadataKind, Moment, Scalar};
use litellm_python_compat::datetime::UtcDateTime;
use litellm_standard_logging::StandardLoggingPayload;
use serde::Serialize;
use serde_json::Value;

const MICROS_PER_SECOND: i64 = 1_000_000;

fn moment(epoch_seconds: f64) -> Moment {
    let micros = (epoch_seconds * MICROS_PER_SECOND as f64).round() as i64;
    let utc = UtcDateTime::from_timestamp(
        micros.div_euclid(MICROS_PER_SECOND),
        micros.rem_euclid(MICROS_PER_SECOND) as u32,
    );
    Moment {
        year: utc.year as i32,
        month: utc.month as u8,
        day: utc.day as u8,
        hour: utc.hour as u8,
        minute: utc.minute as u8,
        second: utc.second as u8,
        microsecond: utc.microsecond,
        utc_offset_seconds: None,
    }
}

fn record(value: impl Serialize) -> Option<Value> {
    serde_json::to_value(value).ok()
}

/// `field` of the call `payload` logs, and nothing for a field the gateway cannot work out yet.
pub(super) fn derived(field: Field, payload: &StandardLoggingPayload) -> Option<Value> {
    match field {
        Field::LogId => Some(Value::from(payload.id.clone())),
        Field::ColdStorageObjectKey => payload
            .metadata
            .cold_storage_object_key
            .clone()
            .map(Value::from),
        Field::AttemptTiming => record(AttemptTiming {
            start: Some(moment(payload.start_time)),
            end: Some(moment(payload.end_time)),
            ..AttemptTiming::default()
        }),
        Field::LoggedKeyIdentity => record(LoggedKeyIdentity {
            metadata_kind: Some(MetadataKind::Mapping),
            key_hash: payload.caller.key.hash.clone().map(Scalar::Text),
            key_alias: payload.caller.key.alias.clone().map(Scalar::Text),
            team_id: payload.caller.team.id.clone().map(Scalar::Text),
            team_alias: payload.caller.team.alias.clone().map(Scalar::Text),
            user_id: payload.caller.user.id.clone().map(Scalar::Text),
            ..LoggedKeyIdentity::default()
        }),
        Field::StandardLoggingPayload => record(payload),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use litellm_callbacks::{FieldSet, FieldValues, typed};
    use litellm_standard_logging::{Caller, Key, Team};
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::test_support::payload;

    fn read(field: Field, payload: &StandardLoggingPayload) -> FieldValues {
        FieldValues::new(FieldSet::of(&[field])).with(field, derived(field, payload).unwrap())
    }

    #[rstest]
    fn the_timing_splits_the_start_and_end_into_utc_dates_and_clocks() {
        let timing = read(Field::AttemptTiming, &payload())
            .get::<typed::AttemptTiming>()
            .unwrap()
            .unwrap();
        let start = Moment {
            year: 2025,
            month: 11,
            day: 13,
            hour: 9,
            minute: 5,
            second: 7,
            microsecond: 42,
            utc_offset_seconds: None,
        };
        assert_eq!(
            (timing.start, timing.end),
            (
                Some(start),
                Some(Moment {
                    second: 8,
                    microsecond: 0,
                    ..start
                })
            )
        );
    }

    #[rstest]
    fn the_key_identity_carries_the_aliases_a_callback_prefixes_with() {
        let logged = StandardLoggingPayload {
            caller: Caller {
                key: Key {
                    alias: Some("ci-key".to_owned()),
                    ..Key::default()
                },
                team: Team {
                    alias: Some("platform".to_owned()),
                    ..Team::default()
                },
                ..Caller::default()
            },
            ..payload()
        };
        let identity = read(Field::LoggedKeyIdentity, &logged)
            .get::<typed::LoggedKeyIdentity>()
            .unwrap()
            .unwrap();
        assert_eq!(
            (
                identity.metadata_kind,
                identity.key_alias,
                identity.team_alias
            ),
            (
                Some(MetadataKind::Mapping),
                Some(Scalar::Text("ci-key".to_owned())),
                Some(Scalar::Text("platform".to_owned()))
            )
        );
    }

    #[rstest]
    #[case::log_id(Field::LogId, Some(json!("msg_1")))]
    #[case::no_cold_storage_key(Field::ColdStorageObjectKey, None)]
    #[case::not_worked_out_yet(Field::Usage, None)]
    fn a_field_is_its_value_or_nothing(#[case] field: Field, #[case] expected: Option<Value>) {
        assert_eq!(derived(field, &payload()), expected);
    }
}
