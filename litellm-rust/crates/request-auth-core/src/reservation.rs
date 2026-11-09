use serde::{Deserialize, Serialize};
use std::cmp::Ordering::Greater;

#[derive(Deserialize)]
#[serde(tag = "kind")]
pub enum Facts {
    Gate {
        token_present: bool,
        llm_route: Option<bool>,
        unbilled: Option<bool>,
        model_present: Option<bool>,
        counter_count: Option<usize>,
        cost_read: bool,
        cost: Option<f64>,
    },
    Sources {
        key_present: bool,
        team_present: bool,
        team_id: Option<String>,
        user_present: bool,
        user_id: Option<String>,
        apply_user_budget_to_team_keys: bool,
    },
    Counter {
        scope: Scope,
        limit: Option<f64>,
    },
    Window {
        duration_present: bool,
        limit: Option<f64>,
        start_present: Option<bool>,
    },
    Identity {
        scope: IdentityScope,
        primary: Option<String>,
        fallback: Option<String>,
    },
    MemberGate {
        team_present: bool,
        team_id: Option<String>,
        user_present: bool,
        user_id: Option<String>,
    },
    MemberCap {
        member_limit: Option<f64>,
        default_read: bool,
    },
}

#[derive(Clone, Copy, Deserialize, strum::VariantArray)]
pub enum IdentityScope {
    EndUser,
    Organization,
    Project,
}

#[derive(Clone, Copy, Deserialize, strum::VariantArray)]
pub enum Scope {
    Key,
    Team,
    User,
    EndUser,
    Tag,
    AccessGroup,
    Member,
    Organization,
    Project,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    ReadRoute,
    ReadUnbilled,
    ReadModel,
    ReadCounters,
    ReadEstimate,
    Reserve,
    Skip,
    ReadStart,
    WarnWindow,
    AppendWindow,
    ReadMember,
    ReadDefault,
    ReadCap,
    Identity { id: Option<String> },
    Sources { key: bool, team: bool, user: bool },
    Counter { include: bool },
}

pub fn decide(facts: Facts) -> Decision {
    match facts {
        Facts::Identity {
            scope,
            primary,
            fallback,
        } => Decision::Identity {
            id: match scope {
                IdentityScope::EndUser => primary.filter(|id| !id.is_empty()).or(fallback),
                IdentityScope::Organization => primary.or(fallback),
                IdentityScope::Project => primary,
            },
        },
        Facts::MemberGate {
            team_present,
            team_id,
            user_present,
            user_id,
        } => {
            if team_present && team_id.is_some() && user_present && user_id.is_some() {
                Decision::ReadMember
            } else {
                Decision::Skip
            }
        }
        Facts::MemberCap {
            member_limit,
            default_read,
        } => {
            if member_limit.is_none() && !default_read {
                Decision::ReadDefault
            } else {
                Decision::ReadCap
            }
        }
        Facts::Gate {
            token_present,
            llm_route,
            unbilled,
            model_present,
            counter_count,
            cost_read,
            cost,
        } => {
            if !token_present {
                return Decision::Skip;
            }
            match llm_route {
                None => return Decision::ReadRoute,
                Some(false) => return Decision::Skip,
                Some(true) => {}
            }
            match unbilled {
                None => return Decision::ReadUnbilled,
                Some(true) => return Decision::Skip,
                Some(false) => {}
            }
            match model_present {
                None => return Decision::ReadModel,
                Some(false) => return Decision::Skip,
                Some(true) => {}
            }
            match counter_count {
                None => return Decision::ReadCounters,
                Some(0) => return Decision::Skip,
                Some(_) => {}
            }
            if !cost_read {
                return Decision::ReadEstimate;
            }
            match cost {
                None => Decision::Skip,
                Some(value) if value <= 0.0 => Decision::Skip,
                Some(_) => Decision::Reserve,
            }
        }
        Facts::Sources {
            key_present,
            team_present,
            team_id,
            user_present,
            user_id,
            apply_user_budget_to_team_keys,
        } => {
            let team = team_present && team_id.is_some();
            Decision::Sources {
                key: key_present,
                team,
                user: (!team || apply_user_budget_to_team_keys)
                    && user_present
                    && user_id.is_some(),
            }
        }
        Facts::Counter { scope, limit } => Decision::Counter {
            include: limit.is_some_and(|limit| match scope {
                Scope::Key | Scope::Team | Scope::User => limit.partial_cmp(&0.0) == Some(Greater),
                Scope::Project => limit.is_finite() && limit > 0.0,
                _ => matches!(limit.partial_cmp(&0.0), None | Some(Greater)),
            }),
        },
        Facts::Window {
            duration_present,
            limit,
            start_present,
        } => {
            if !duration_present || limit.is_none_or(|limit| limit <= 0.0) {
                return Decision::Skip;
            }
            match start_present {
                None => Decision::ReadStart,
                Some(false) => Decision::WarnWindow,
                Some(true) => Decision::AppendWindow,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::customer_empty(IdentityScope::EndUser, Some(""), Some("fallback"), Some("fallback"))]
    #[case::org_empty(IdentityScope::Organization, Some(""), Some("fallback"), Some(""))]
    #[case::org_fallback(IdentityScope::Organization, None, Some("fallback"), Some("fallback"))]
    #[case::project_no_fallback(IdentityScope::Project, None, Some("fallback"), None)]
    fn identities(
        #[case] scope: IdentityScope,
        #[case] primary: Option<&str>,
        #[case] fallback: Option<&str>,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(
            decide(Facts::Identity {
                scope,
                primary: primary.map(str::to_owned),
                fallback: fallback.map(str::to_owned)
            }),
            Decision::Identity {
                id: expected.map(str::to_owned)
            }
        );
    }

    #[rstest]
    #[case::member_cap(Some(0.0), false, Decision::ReadCap)]
    #[case::default_needed(None, false, Decision::ReadDefault)]
    #[case::default_read(None, true, Decision::ReadCap)]
    fn member_cap(
        #[case] member_limit: Option<f64>,
        #[case] default_read: bool,
        #[case] expected: Decision,
    ) {
        assert_eq!(
            decide(Facts::MemberCap {
                member_limit,
                default_read
            }),
            expected
        );
    }

    #[rstest]
    #[case::all_present(true, Some(""), true, Some(""), Decision::ReadMember)]
    #[case::missing_team(true, None, true, Some("user"), Decision::Skip)]
    #[case::missing_user(true, Some("team"), false, Some("user"), Decision::Skip)]
    fn member_gate(
        #[case] team_present: bool,
        #[case] team_id: Option<&str>,
        #[case] user_present: bool,
        #[case] user_id: Option<&str>,
        #[case] expected: Decision,
    ) {
        assert_eq!(
            decide(Facts::MemberGate {
                team_present,
                team_id: team_id.map(str::to_owned),
                user_present,
                user_id: user_id.map(str::to_owned)
            }),
            expected
        );
    }

    #[rstest]
    #[case::key_nan(Scope::Key, f64::NAN, false)]
    #[case::customer_nan(Scope::EndUser, f64::NAN, true)]
    #[case::project_infinite(Scope::Project, f64::INFINITY, false)]
    #[case::team_infinite(Scope::Team, f64::INFINITY, true)]
    #[case::zero(Scope::Organization, 0.0, false)]
    #[case::positive(Scope::User, 10.0, true)]
    fn limits(#[case] scope: Scope, #[case] limit: f64, #[case] include: bool) {
        assert_eq!(
            decide(Facts::Counter {
                scope,
                limit: Some(limit)
            }),
            Decision::Counter { include }
        );
    }

    #[rstest]
    #[case::no_duration(false, None, Decision::Skip)]
    #[case::unread(true, None, Decision::ReadStart)]
    #[case::invalid(true, Some(false), Decision::WarnWindow)]
    #[case::valid(true, Some(true), Decision::AppendWindow)]
    fn windows(
        #[case] duration_present: bool,
        #[case] start_present: Option<bool>,
        #[case] expected: Decision,
    ) {
        assert_eq!(
            decide(Facts::Window {
                duration_present,
                limit: Some(1.0),
                start_present
            }),
            expected
        );
    }
}
