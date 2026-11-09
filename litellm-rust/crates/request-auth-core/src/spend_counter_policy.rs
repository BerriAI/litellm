use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
#[serde(tag = "kind")]
pub enum Facts {
    Spend {
        current: f64,
        verified: bool,
        fallback: f64,
        limit: Option<f64>,
        window: bool,
        fallback_authoritative: bool,
        strict_before: Option<bool>,
        authoritative_read: bool,
        authoritative: Option<f64>,
        strict_after: Option<bool>,
    },
    ThrottleKey {
        counter_key: String,
        token_present: bool,
        token: Option<String>,
    },
    Overflow {
        current: f64,
        limit: f64,
    },
    OverBudget {
        throttled: bool,
        current: f64,
        limit: f64,
        reservation: f64,
        strict: bool,
    },
    ReservationFailure {
        strict: bool,
    },
    ReservationGroups {
        currents: Vec<f64>,
        limits: Vec<f64>,
        reservation: f64,
    },
    Adjustment {
        current: Option<f64>,
        adjustment: f64,
    },
    ResizeConsistency {
        inconsistent: bool,
        reseed: bool,
    },
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    Value { spend: f64 },
    Repair { spend: f64 },
    ReadStrictBefore,
    ReadStrictAfter,
    ReadFloor,
    Unverifiable,
    ReadThrottle,
    Skip,
    Release,
    Reject { current: f64, estimate: Option<f64> },
    Resize { cost: f64 },
    CheckOverflow,
    Unavailable,
    Compatible,
    Inconsistent,
    RejectResize,
    ReservationGroups { groups: Vec<Vec<usize>> },
}

pub fn decide(facts: Facts) -> Decision {
    match facts {
        Facts::Adjustment {
            current,
            adjustment,
        } => {
            if current.is_some_and(|current| !(adjustment < 0.0 && current + adjustment < -1e-12)) {
                Decision::Compatible
            } else {
                Decision::Inconsistent
            }
        }
        Facts::ResizeConsistency {
            inconsistent,
            reseed,
        } => {
            if inconsistent && !reseed {
                Decision::RejectResize
            } else {
                Decision::Compatible
            }
        }
        Facts::Spend {
            current,
            verified,
            fallback,
            limit,
            window,
            fallback_authoritative,
            strict_before,
            authoritative_read,
            authoritative,
            strict_after,
        } => {
            if limit.is_none_or(|limit| current >= limit) {
                return Decision::Value { spend: current };
            }
            let obvious_floor = fallback > current || window;
            if !obvious_floor && strict_before.is_none() {
                return Decision::ReadStrictBefore;
            }
            if obvious_floor || strict_before == Some(true) {
                if !authoritative_read {
                    return Decision::ReadFloor;
                }
                if let Some(spend) = authoritative {
                    if spend > current {
                        return Decision::Repair { spend };
                    }
                } else if fallback > current {
                    return Decision::Value { spend: fallback };
                }
            }
            if !(verified || fallback_authoritative || authoritative.is_some()) {
                return match strict_after {
                    None => Decision::ReadStrictAfter,
                    Some(true) => Decision::Unverifiable,
                    Some(false) => Decision::Value { spend: current },
                };
            }
            Decision::Value { spend: current }
        }
        Facts::ThrottleKey {
            counter_key,
            token_present,
            token,
        } => {
            if token_present
                && counter_key == format!("spend:key:{}", token.as_deref().unwrap_or("None"))
            {
                Decision::ReadThrottle
            } else {
                Decision::Skip
            }
        }
        Facts::Overflow { current, limit } => {
            if current > limit {
                Decision::CheckOverflow
            } else {
                Decision::Skip
            }
        }
        Facts::OverBudget {
            throttled,
            current,
            limit,
            reservation,
            strict,
        } => {
            if throttled {
                return Decision::Release;
            }
            let remaining = limit - (current - reservation);
            if remaining <= 1e-12 {
                return Decision::Reject {
                    current,
                    estimate: None,
                };
            }
            if strict && current - limit > 1e-12 {
                return Decision::Reject {
                    current: current - reservation,
                    estimate: Some(reservation),
                };
            }
            Decision::Resize { cost: remaining }
        }
        Facts::ReservationFailure { strict } => {
            if strict {
                Decision::Unavailable
            } else {
                Decision::Skip
            }
        }
        Facts::ReservationGroups {
            currents,
            limits,
            reservation,
        } => {
            let indices: Vec<usize> = (0..currents.len()).collect();
            let groups = if currents.is_empty() {
                vec![]
            } else if currents
                .iter()
                .zip(&limits)
                .all(|(current, limit)| current + reservation <= *limit)
            {
                vec![indices]
            } else {
                indices.into_iter().map(|index| vec![index]).collect()
            };
            Decision::ReservationGroups { groups }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::missing(None, 1.0, Decision::Inconsistent)]
    #[case::positive(Some(1.0), 1.0, Decision::Compatible)]
    #[case::negative(Some(1.0), -2.0, Decision::Inconsistent)]
    #[case::zero(Some(1.0), -1.0, Decision::Compatible)]
    fn adjustment(
        #[case] current: Option<f64>,
        #[case] adjustment: f64,
        #[case] expected: Decision,
    ) {
        assert_eq!(
            decide(Facts::Adjustment {
                current,
                adjustment
            }),
            expected
        );
    }

    #[rstest]
    #[case::resize(true, false, Decision::RejectResize)]
    #[case::settlement(true, true, Decision::Compatible)]
    #[case::consistent(false, false, Decision::Compatible)]
    fn consistency(#[case] inconsistent: bool, #[case] reseed: bool, #[case] expected: Decision) {
        assert_eq!(
            decide(Facts::ResizeConsistency {
                inconsistent,
                reseed
            }),
            expected
        );
    }

    #[rstest]
    #[case::throttled(true, 12.0, true, Decision::Release)]
    #[case::empty(false, 12.0, false, Decision::Reject { current:12.0, estimate:None })]
    #[case::partial(false, 11.0, false, Decision::Resize { cost:1.0 })]
    #[case::strict(false, 11.0, true, Decision::Reject { current:9.0, estimate:Some(2.0) })]
    fn reservation_outcome(
        #[case] throttled: bool,
        #[case] current: f64,
        #[case] strict: bool,
        #[case] expected: Decision,
    ) {
        assert_eq!(
            decide(Facts::OverBudget {
                throttled,
                current,
                limit: 10.0,
                reservation: 2.0,
                strict
            }),
            expected
        );
    }

    #[rstest]
    #[case::unchecked(None, false, None, Decision::Value { spend:2.0 })]
    #[case::strict_read(Some(3.0), false, None, Decision::ReadStrictBefore)]
    #[case::window(Some(3.0), true, None, Decision::ReadFloor)]
    #[case::strict_floor(Some(3.0), false, Some(true), Decision::ReadFloor)]
    #[case::second_read(Some(3.0), false, Some(false), Decision::ReadStrictAfter)]
    fn spend_read_order(
        #[case] limit: Option<f64>,
        #[case] window: bool,
        #[case] strict_before: Option<bool>,
        #[case] expected: Decision,
    ) {
        assert_eq!(
            decide(Facts::Spend {
                current: 2.0,
                verified: false,
                fallback: 0.0,
                limit,
                window,
                fallback_authoritative: false,
                strict_before,
                authoritative_read: false,
                authoritative: None,
                strict_after: None
            }),
            expected
        );
    }
}
