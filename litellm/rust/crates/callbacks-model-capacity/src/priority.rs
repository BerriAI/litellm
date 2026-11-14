use serde_json::Value;

use crate::{
    error::CapacityError,
    model::{DEFAULT_CLASS, GroupInfo, Priority, Settings, TeamShare},
    pyfmt,
};

const MEMBER_SEPARATOR: char = '\u{1f}';

fn as_float(value: &Value) -> Result<f64, CapacityError> {
    match value {
        Value::Number(number) => number
            .as_f64()
            .ok_or_else(|| CapacityError::Config(format!("{number} is not a float"))),
        Value::Bool(flag) => Ok(f64::from(u8::from(*flag))),
        Value::String(text) => pyfmt::numeric_text(text)
            .and_then(|text| text.parse::<f64>().ok())
            .ok_or_else(|| {
                CapacityError::Config(format!(
                    "could not convert string to float: {}",
                    pyfmt::str_repr(text)
                ))
            }),
        other => Err(CapacityError::Config(format!(
            "float() argument must be a string or a real number, not {other}"
        ))),
    }
}

pub fn convert_priority_to_percent(
    value: &Value,
    group: Option<&GroupInfo>,
) -> Result<f64, CapacityError> {
    let Value::Object(entry) = value else {
        return match value {
            Value::Number(_) | Value::Bool(_) => as_float(value),
            _ => Err(CapacityError::Config(format!(
                "priority reservation value {value} is neither a number nor a mapping"
            ))),
        };
    };
    let kind = entry
        .get("type")
        .cloned()
        .unwrap_or_else(|| Value::String("percent".into()));
    let amount = entry.get("value").cloned().unwrap_or(Value::from(1.0));
    let rpm = group.and_then(|group| group.rpm).filter(|rpm| *rpm > 0);
    let tpm = group.and_then(|group| group.tpm).filter(|tpm| *tpm > 0);
    match (kind.as_str(), rpm, tpm) {
        (Some("rpm"), Some(rpm), _) => Ok(as_float(&amount)? / rpm as f64),
        (Some("tpm"), _, Some(tpm)) => Ok(as_float(&amount)? / tpm as f64),
        _ => as_float(&amount),
    }
}

pub type Weights = Vec<(Value, f64)>;

pub fn normalized_weights(
    settings: &Settings,
    group: &GroupInfo,
) -> Result<Weights, CapacityError> {
    let Some(reservation) = &settings.priority_reservation else {
        return Ok(Vec::new());
    };
    if let Some(index) = reservation.iter().position(|(_, value)| {
        !matches!(value, Value::Number(_) | Value::Bool(_) | Value::Object(_))
    }) {
        let left = if index == 0 { "int" } else { "float" };
        return Err(CapacityError::Type(format!(
            "unsupported operand type(s) for +: '{left}' and 'NoneType'"
        )));
    }
    let weights = reservation
        .iter()
        .map(|(name, value)| {
            Ok((
                name.clone(),
                convert_priority_to_percent(value, Some(group))?,
            ))
        })
        .collect::<Result<Vec<_>, CapacityError>>()?;
    let total = pyfmt::sum(weights.iter().map(|(_, weight)| *weight));
    if total > 1.0 {
        return Ok(weights
            .into_iter()
            .map(|(name, weight)| (name, weight / total))
            .collect());
    }
    Ok(weights)
}

pub fn weight_of(weights: &[(Value, f64)], name: &Value) -> Option<f64> {
    weights
        .iter()
        .find(|(candidate, _)| pyfmt::equals(candidate, name))
        .map(|(_, weight)| *weight)
}

pub struct Allocation {
    pub weight: f64,
    pub key: String,
    pub unlicensed: bool,
}

pub fn allocation(
    settings: &Settings,
    model: &str,
    priority: Option<&Priority>,
    weights: &[(Value, f64)],
    group: &GroupInfo,
) -> Result<Allocation, CapacityError> {
    match settings.reserves(priority)? {
        Some(explicit) => {
            let fallback = explicit_weight(settings, explicit, group)?;
            Ok(Allocation {
                weight: weight_of(weights, explicit).unwrap_or(fallback),
                key: format!("{model}:{}", pyfmt::value(Some(explicit))),
                unlicensed: !settings.licensed,
            })
        }
        None => Ok(Allocation {
            weight: settings.default_priority,
            key: format!("{model}:default_pool"),
            unlicensed: false,
        }),
    }
}

fn explicit_weight(
    settings: &Settings,
    name: &Value,
    group: &GroupInfo,
) -> Result<f64, CapacityError> {
    if !settings.licensed {
        return Ok(settings.default_priority);
    }
    let value = settings
        .priority_reservation
        .iter()
        .flatten()
        .find(|(candidate, _)| pyfmt::equals(candidate, name))
        .map(|(_, value)| value);
    match value {
        Some(value) => convert_priority_to_percent(value, Some(group)),
        None => Ok(settings.default_priority),
    }
}

pub fn priority_class(
    settings: &Settings,
    priority: Option<&Priority>,
) -> Result<Value, CapacityError> {
    Ok(settings
        .reserves(priority)?
        .cloned()
        .unwrap_or_else(|| Value::from(DEFAULT_CLASS)))
}

pub fn pool_member(priority_class: &Value) -> String {
    format!(
        "{MEMBER_SEPARATOR}pool{MEMBER_SEPARATOR}{}",
        pyfmt::value(Some(priority_class))
    )
}

pub fn team_member(team_id: &str) -> String {
    format!("team{MEMBER_SEPARATOR}{team_id}")
}

fn is_pool_member(member: &str) -> bool {
    member.starts_with(&format!("{MEMBER_SEPARATOR}pool{MEMBER_SEPARATOR}"))
}

pub fn counter_value(model: &str, team_id: &str) -> String {
    format!("{}:{model}:{team_id}", model.chars().count())
}

pub fn team_share(
    priority: &Value,
    members: &[(String, Value)],
    class_weights: &[(Value, f64)],
    default_weight: f64,
) -> TeamShare {
    let mut present: Vec<&Value> = Vec::new();
    for (_, class) in members {
        if !present.iter().any(|seen| pyfmt::equals(seen, class)) {
            present.push(class);
        }
    }
    let weights: Vec<(&Value, f64)> = present
        .iter()
        .map(|class| {
            (
                *class,
                weight_of(class_weights, class).unwrap_or(default_weight),
            )
        })
        .collect();
    let total = pyfmt::sum(weights.iter().map(|(_, weight)| *weight));
    let own = weights
        .iter()
        .find(|(class, _)| pyfmt::equals(class, priority))
        .map_or(0.0, |(_, weight)| *weight);
    let class_fraction = if total > 0.0 {
        own / total
    } else {
        1.0 / present.len() as f64
    };
    let teams_in_class = members
        .iter()
        .filter(|(member, class)| pyfmt::equals(class, priority) && !is_pool_member(member))
        .count() as i64;
    TeamShare {
        fraction: class_fraction / teams_in_class.max(1) as f64,
        active_teams_in_class: teams_in_class,
        active_classes: present.len() as i64,
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::model::TokenRateLimitType;

    fn settings(reservation: &[(&str, Value)]) -> Settings {
        Settings {
            priority_reservation: Some(
                reservation
                    .iter()
                    .map(|(name, value)| (Value::from(*name), value.clone()))
                    .collect(),
            ),
            default_priority: 0.25,
            saturation_threshold: 0.8,
            class_split: false,
            saturation_check_cache_ttl: json!(60),
            window_size: 60,
            rate_limit_type: TokenRateLimitType::Total,
            licensed: true,
        }
    }

    fn group(tpm: Option<i64>, rpm: Option<i64>) -> GroupInfo {
        GroupInfo {
            tpm,
            rpm,
            ..GroupInfo::default()
        }
    }

    #[rstest]
    #[case(json!(0.9), 0.9)]
    #[case(json!({"type": "percent", "value": 0.4}), 0.4)]
    #[case(json!({"type": "rpm", "value": 25}), 0.25)]
    #[case(json!({"type": "tpm", "value": 500}), 0.5)]
    #[case(json!({"value": 0.3}), 0.3)]
    #[case(json!({"type": "rpm"}), 0.01)]
    fn reservations_convert_to_a_fraction_of_the_group(
        #[case] value: Value,
        #[case] expected: f64,
    ) {
        let converted =
            convert_priority_to_percent(&value, Some(&group(Some(1000), Some(100)))).unwrap();
        assert!(
            (converted - expected).abs() < 1e-12,
            "{converted} != {expected}"
        );
    }

    #[rstest]
    fn an_absolute_reservation_without_that_limit_on_the_group_reads_as_a_percent() {
        let converted = convert_priority_to_percent(
            &json!({"type": "rpm", "value": 0.5}),
            Some(&group(Some(1000), None)),
        )
        .unwrap();
        assert_eq!(converted, 0.5);
    }

    #[rstest]
    #[case::first(&[("a", json!("x")), ("b", json!(0.5))], "'int' and 'NoneType'")]
    #[case::later(&[("a", json!(0.5)), ("b", json!([1]))], "'float' and 'NoneType'")]
    fn a_reservation_python_cannot_convert_fails_the_sum_like_python(
        #[case] reservation: &[(&str, Value)],
        #[case] operands: &str,
    ) {
        let error =
            normalized_weights(&settings(reservation), &group(Some(1000), None)).unwrap_err();
        assert!(error.is_type_error());
        assert_eq!(
            error.to_string(),
            format!("unsupported operand type(s) for +: {operands}")
        );
    }

    #[rstest]
    fn over_allocated_reservations_scale_down_to_the_whole_group() {
        let weights = normalized_weights(
            &settings(&[("a", json!(0.6)), ("b", json!(0.8))]),
            &group(Some(1000), None),
        )
        .unwrap();
        assert!((weight_of(&weights, &json!("a")).unwrap() - 0.6 / 1.4).abs() < 1e-12);
        assert!((weight_of(&weights, &json!("b")).unwrap() - 0.8 / 1.4).abs() < 1e-12);
    }

    #[rstest]
    fn reservations_within_the_group_keep_their_weights() {
        let weights = normalized_weights(
            &settings(&[("a", json!(0.6)), ("b", json!(0.4))]),
            &group(Some(1000), None),
        )
        .unwrap();
        assert_eq!(weights, vec![(json!("a"), 0.6), (json!("b"), 0.4)]);
    }

    #[rstest]
    fn keys_without_a_reserved_priority_share_the_default_pool() {
        let settings = settings(&[("high", json!(0.9))]);
        let weights = normalized_weights(&settings, &group(Some(1000), None)).unwrap();
        let unknown = allocation(
            &settings,
            "m",
            Some(&Priority {
                value: json!("gold"),
                hash_error: None,
            }),
            &weights,
            &group(Some(1000), None),
        )
        .unwrap();
        assert_eq!(
            (unknown.weight, unknown.key.as_str()),
            (0.25, "m:default_pool")
        );
        let high = allocation(
            &settings,
            "m",
            Some(&Priority {
                value: json!("high"),
                hash_error: None,
            }),
            &weights,
            &group(Some(1000), None),
        )
        .unwrap();
        assert_eq!((high.weight, high.key.as_str()), (0.9, "m:high"));
    }

    #[rstest]
    fn a_class_share_splits_among_its_active_teams_and_absent_classes_give_theirs_away() {
        let members = vec![
            ("team\u{1f}a".to_owned(), json!("high")),
            ("team\u{1f}b".to_owned(), json!("high")),
            ("team\u{1f}c".to_owned(), json!("low")),
        ];
        let weights = vec![
            (json!("high"), 0.6),
            (json!("low"), 0.2),
            (json!("idle"), 0.2),
        ];
        let share = team_share(&json!("high"), &members, &weights, 0.1);
        assert!((share.fraction - 0.375).abs() < 1e-12);
        assert_eq!((share.active_teams_in_class, share.active_classes), (2, 2));
    }

    #[rstest]
    fn a_pool_member_keeps_its_class_present_without_taking_a_slice() {
        let members = vec![
            ("team\u{1f}a".to_owned(), json!("high")),
            (pool_member(&json!("high")), json!("high")),
        ];
        let share = team_share(&json!("high"), &members, &[(json!("high"), 0.5)], 0.1);
        assert_eq!((share.fraction, share.active_teams_in_class), (1.0, 1));
    }

    #[rstest]
    #[case("chat", "prod:victim", "4:chat:prod:victim")]
    #[case("chat:prod", "victim", "9:chat:prod:victim")]
    #[case("é", "t", "1:é:t")]
    fn counter_values_cannot_collide_across_the_model_and_team_boundary(
        #[case] model: &str,
        #[case] team: &str,
        #[case] expected: &str,
    ) {
        assert_eq!(counter_value(model, team), expected);
    }
}
