//! `litellm_settings.priority_reservation` and `priority_reservation_settings`, which the proxy's
//! `dynamic_rate_limiter_v3` callback reads: how much of a model group each priority reserves, and
//! when and among whom that reservation is enforced.

use serde::Deserialize;

/// What one priority reserves of a model group: a share of it, or a number of its requests or
/// tokens per minute.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum PriorityReservation {
    Share(f64),
    Sized {
        #[serde(rename = "type", default)]
        unit: ReservationUnit,
        value: f64,
    },
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, strum::IntoStaticStr)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum ReservationUnit {
    #[default]
    Percent,
    Rpm,
    Tpm,
}

/// Who takes an equal slice of its priority class's share of a model.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ClassShareEntity {
    Team,
    Key,
    CustomerOrganization,
    CustomerMember,
}

/// Whose per-model limits are shares that borrow a model's idle capacity.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BorrowingEntity {
    Team,
    Key,
    CustomerOrganization,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct ModelLimitBorrowingConfig {
    pub borrow_idle_capacity: Box<[BorrowingEntity]>,
    pub max_borrow_multiple: Option<f64>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct PriorityClassShareConfig {
    pub split_equally_among_active: Box<[ClassShareEntity]>,
}

/// The proxy's `PriorityReservationSettings`, with its defaults.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default)]
pub struct PriorityReservationSettings {
    /// The share a caller without a reserved priority is counted under.
    pub default_priority: f64,
    /// How full a model group is, from 0 to 1, when each priority is held to its reservation.
    pub saturation_threshold: f64,
    /// Seconds a saturation counter read from Redis is kept in memory.
    pub saturation_check_cache_ttl: u64,
    pub model_limit_borrowing_config: ModelLimitBorrowingConfig,
    pub priority_class_share_config: PriorityClassShareConfig,
}

impl Default for PriorityReservationSettings {
    fn default() -> Self {
        Self {
            default_priority: 0.25,
            saturation_threshold: 0.5,
            saturation_check_cache_ttl: 60,
            model_limit_borrowing_config: ModelLimitBorrowingConfig::default(),
            priority_class_share_config: PriorityClassShareConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::Config;

    #[rstest]
    fn each_priority_keeps_the_order_and_the_shape_it_is_written_in() {
        let config = Config::from_yaml(
            "litellm_settings:\n  priority_reservation:\n    prod: 0.9\n    batch: {type: rpm, value: 10}\n    dev: {value: 1}\n",
        )
        .unwrap();

        assert_eq!(
            config.litellm_settings.priority_reservation,
            Some(vec![
                ("prod".to_owned(), PriorityReservation::Share(0.9)),
                (
                    "batch".to_owned(),
                    PriorityReservation::Sized {
                        unit: ReservationUnit::Rpm,
                        value: 10.0
                    }
                ),
                (
                    "dev".to_owned(),
                    PriorityReservation::Sized {
                        unit: ReservationUnit::Percent,
                        value: 1.0
                    }
                ),
            ])
        );
    }

    #[rstest]
    fn settings_left_out_take_the_proxys_defaults() {
        let config = Config::from_yaml(
            "litellm_settings:\n  priority_reservation_settings:\n    saturation_threshold: 0.8\n    priority_class_share_config:\n      split_equally_among_active: [key, customer_member]\n",
        )
        .unwrap();

        assert_eq!(
            config.litellm_settings.priority_reservation_settings,
            Some(PriorityReservationSettings {
                default_priority: 0.25,
                saturation_threshold: 0.8,
                saturation_check_cache_ttl: 60,
                model_limit_borrowing_config: ModelLimitBorrowingConfig {
                    borrow_idle_capacity: Box::new([]),
                    max_borrow_multiple: None,
                },
                priority_class_share_config: PriorityClassShareConfig {
                    split_equally_among_active: Box::new([
                        ClassShareEntity::Key,
                        ClassShareEntity::CustomerMember
                    ]),
                },
            })
        );
    }
}
