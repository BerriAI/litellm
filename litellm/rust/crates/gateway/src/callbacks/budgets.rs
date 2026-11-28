//! The budget counters a caller holds through its key and what it belongs to, as the proxy's
//! `_get_budget_counters` plans them for a reservation: one lifetime counter for each of the key,
//! team, user, end user, team member, organization and project that has a budget, in that order.
//! A counter starts from the spend its row holds. Unlike the reservation, a budget of zero or less
//! is planned too, since the proxy's read-time budget checks refuse a caller at it.
//!
//! Accepted difference, while the gateway declines every caller that holds a budget other than its
//! key's own: no `budget_limits` window is planned, no counter is planned for a team the key only
//! copies.

use jiff::Timestamp;
use litellm_callbacks_budgets::PlannedCounter;
use litellm_gateway_auth::Authenticated;
use litellm_gateway_auth_core::{
    Affiliation, Caller, Credential, Named, Team,
    record::{Budget, MembershipRecord, Record},
};
use litellm_redis_store::spend_counters::SpendCounter;

use crate::host::NativeRows;

/// `general_settings.apply_user_budget_to_team_keys`: a key in a team is held to its owner's
/// budget too. `admit` sends a config that sets it to Python, so the gateway plans with it off.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Settings {
    pub(crate) apply_user_budget_to_team_keys: bool,
}

/// The counter `entity_type` `entity_id` is held to, when it has a budget.
fn planned(
    counter: SpendCounter,
    entity_type: &str,
    entity_id: String,
    budget: Budget,
) -> Option<PlannedCounter> {
    let max_budget = budget.max?;
    Some(PlannedCounter {
        counter_key: counter.key(),
        max_budget,
        fallback_spend: budget.spend,
        entity_type: entity_type.to_owned(),
        entity_id,
        spend_log_entity_id: None,
    })
}

fn rollover_credit(
    base: f64,
    ceiling: Option<f64>,
    duration: Option<&str>,
    credit: Option<f64>,
) -> f64 {
    ceiling
        .filter(|ceiling| {
            ceiling.is_finite()
                && base > 0.0
                && base < *ceiling
                && duration.is_some_and(|duration| !duration.is_empty())
        })
        .map_or(0.0, |ceiling| {
            credit.unwrap_or(0.0).max(0.0).min(ceiling - base)
        })
}

fn member_budget(
    membership: Option<&MembershipRecord>,
    default_max: Option<f64>,
    now: Timestamp,
) -> Budget {
    let active_increase = membership
        .map(|member| &member.budget_extras)
        .filter(|extras| {
            extras
                .temp_budget_expiry
                .is_some_and(|expires| expires > now)
        })
        .and_then(|extras| extras.temp_budget_increase)
        .unwrap_or(0.0);
    let max = membership
        .and_then(|member| {
            let base = member.budget.max?;
            let credit = rollover_credit(
                base,
                member.budget_extras.rollover_max_budget,
                member.budget_duration.as_deref(),
                member.rollover_credit,
            );
            Some(base + active_increase + credit)
        })
        .or_else(|| {
            default_max
                .filter(|max| *max > 0.0)
                .map(|max| max + active_increase)
        });
    Budget {
        max,
        spend: membership.map_or(0.0, |member| member.budget.spend),
    }
}

/// The counters `identity` reserves against. The master key belongs to nothing, so it holds only
/// the budget of the end user its request names, as Python's does.
pub(crate) fn counters(
    identity: &Authenticated<NativeRows>,
    settings: Settings,
    now: Timestamp,
) -> Vec<PlannedCounter> {
    let Caller {
        credential,
        affiliation:
            Affiliation {
                holder,
                org,
                project,
            },
        end_user,
        customer_hierarchy: _,
    } = &**identity;
    let key = match credential {
        Credential::MasterKey => None,
        Credential::Key { key } => Some(key),
    };
    let token = identity.hash().map(ToString::to_string);
    let team = match holder.team() {
        Some(Team::Row(team)) => Some(team),
        Some(Team::Unstored(_) | Team::KeyCopy(_)) | None => None,
    };
    let is_team_key = holder.team().is_some();
    let user = holder
        .user()
        .and_then(Named::stored)
        .filter(|_| !is_team_key || settings.apply_user_budget_to_team_keys);
    let member = team
        .zip(holder.user().and_then(Named::stored))
        .and_then(|(team, user)| {
            planned(
                SpendCounter::TeamMember {
                    user_id: user.user_id.clone(),
                    team_id: team.team_id.clone(),
                },
                "TeamMember",
                format!("{}:{}", user.user_id, team.team_id),
                member_budget(holder.membership(), team.member_max_budget, now),
            )
        });
    [
        key.zip(token).and_then(|(key, token)| {
            planned(
                SpendCounter::Key {
                    token: token.clone(),
                    window: None,
                },
                "Key",
                token,
                key.budget(),
            )
        }),
        team.and_then(|team| {
            planned(
                SpendCounter::Team {
                    team_id: team.team_id.clone(),
                    window: None,
                },
                "Team",
                team.team_id.clone(),
                team.budget,
            )
        }),
        user.and_then(|user| {
            planned(
                SpendCounter::User(user.user_id.clone()),
                "User",
                user.user_id.clone(),
                Budget {
                    max: user.budget.max.map(|base| {
                        base + rollover_credit(
                            base,
                            user.rollover_max_budget,
                            user.budget_duration.as_deref(),
                            user.rollover_credit,
                        )
                    }),
                    spend: user.budget.spend,
                },
            )
        }),
        end_user.end_user_id.as_ref().and_then(|end_user_id| {
            planned(
                SpendCounter::EndUser(end_user_id.clone()),
                "EndUser",
                end_user_id.clone(),
                end_user.budget,
            )
        }),
        member,
        org.as_ref().and_then(Named::stored).and_then(|org| {
            planned(
                SpendCounter::Org(org.id().to_owned()),
                "Organization",
                org.id().to_owned(),
                org.budget,
            )
        }),
        project
            .as_ref()
            .and_then(Named::stored)
            .and_then(|project| {
                planned(
                    SpendCounter::Project(project.id().to_owned()),
                    "Project",
                    project.id().to_owned(),
                    Budget {
                        max: project.budget.max.filter(|max| max.is_finite()),
                        ..project.budget
                    },
                )
            }),
    ]
    .into_iter()
    .flatten()
    .collect()
}

#[cfg(test)]
mod tests {
    use litellm_gateway_auth_core::{
        Holder, TeamCopy,
        record::{
            EndUserRecord, MembershipRecord, OrgRecord, ProjectRecord, TeamRecord, UserRecord,
        },
    };
    use rstest::rstest;

    use super::*;
    use crate::{host::Key, test_support::master};

    fn user(max_budget: Option<f64>) -> UserRecord {
        UserRecord {
            user_id: "u1".to_owned(),
            budget: Budget {
                max: max_budget,
                spend: 1.0,
            },
            ..UserRecord::default()
        }
    }

    fn team(member_max_budget: Option<f64>) -> TeamRecord {
        TeamRecord {
            team_id: "t1".to_owned(),
            budget: Budget {
                max: Some(20.0),
                spend: 2.0,
            },
            member_max_budget,
            ..TeamRecord::default()
        }
    }

    fn membership(max_budget: Option<f64>) -> MembershipRecord {
        MembershipRecord {
            user_id: "u1".to_owned(),
            team_id: "t1".to_owned(),
            budget: Budget {
                max: max_budget,
                spend: 3.0,
            },
            ..MembershipRecord::default()
        }
    }

    fn caller(holder: Holder<NativeRows>, end_user: EndUserRecord) -> Authenticated<NativeRows> {
        Authenticated::from(Caller {
            credential: Credential::Key {
                key: Key::stored("hash", None).with_budget(Budget {
                    max: Some(10.0),
                    spend: 0.5,
                }),
            },
            affiliation: Affiliation {
                holder,
                org: Some(Named::Stored(OrgRecord {
                    organization_id: "o1".to_owned(),
                    budget: Budget {
                        max: Some(50.0),
                        spend: 5.0,
                    },
                    ..OrgRecord::default()
                })),
                project: Some(Named::Stored(ProjectRecord {
                    project_id: "p1".to_owned(),
                    budget: Budget {
                        max: Some(60.0),
                        spend: 6.0,
                    },
                    ..ProjectRecord::default()
                })),
            },
            end_user,
            customer_hierarchy: None,
        })
    }

    fn end_user() -> EndUserRecord {
        EndUserRecord {
            end_user_id: Some("e1".to_owned()),
            found: true,
            budget: Budget {
                max: Some(40.0),
                spend: 4.0,
            },
            ..EndUserRecord::default()
        }
    }

    fn planned(
        identity: &Authenticated<NativeRows>,
        settings: Settings,
    ) -> Vec<(String, String, String, f64, f64)> {
        counters(identity, settings, Timestamp::UNIX_EPOCH)
            .into_iter()
            .map(|counter| {
                (
                    counter.counter_key,
                    counter.entity_type,
                    counter.entity_id,
                    counter.max_budget,
                    counter.fallback_spend,
                )
            })
            .collect()
    }

    fn row(
        key: &str,
        kind: &str,
        id: &str,
        budget: f64,
        spend: f64,
    ) -> (String, String, String, f64, f64) {
        (
            key.to_owned(),
            kind.to_owned(),
            id.to_owned(),
            budget,
            spend,
        )
    }

    #[rstest]
    fn the_master_key_reserves_against_the_end_user_its_request_names() {
        assert_eq!(
            planned(
                &Authenticated::from(Caller::master_key(end_user())),
                Settings::default()
            ),
            [row("spend:end_user:e1", "EndUser", "e1", 40.0, 4.0)]
        );
    }

    #[rstest]
    fn a_member_key_reserves_against_every_budget_it_belongs_to_in_pythons_order() {
        let member = Holder::Member {
            user: Named::Stored(user(Some(30.0))),
            team: Team::Row(team(None)),
            membership: Some(membership(Some(70.0))),
        };

        assert_eq!(
            planned(
                &caller(member, end_user()),
                Settings {
                    apply_user_budget_to_team_keys: true
                }
            ),
            [
                row("spend:key:hash", "Key", "hash", 10.0, 0.5),
                row("spend:team:t1", "Team", "t1", 20.0, 2.0),
                row("spend:user:u1", "User", "u1", 30.0, 1.0),
                row("spend:end_user:e1", "EndUser", "e1", 40.0, 4.0),
                row("spend:team_member:u1:t1", "TeamMember", "u1:t1", 70.0, 3.0),
                row("spend:org:o1", "Organization", "o1", 50.0, 5.0),
                row("spend:project:p1", "Project", "p1", 60.0, 6.0),
            ]
        );
    }

    #[rstest]
    #[case::a_team_key(Holder::Member {
        user: Named::Stored(user(Some(30.0))),
        team: Team::Row(team(None)),
        membership: None,
    }, false)]
    #[case::a_key_copy_of_a_team(Holder::Member {
        user: Named::Stored(user(Some(30.0))),
        team: Team::KeyCopy(TeamCopy { team_id: "t1".to_owned(), object_permission: None }),
        membership: None,
    }, false)]
    #[case::a_personal_key(Holder::User(Named::Stored(user(Some(30.0)))), true)]
    fn a_team_key_skips_its_owners_budget_unless_told_to_apply_it(
        #[case] holder: Holder<NativeRows>,
        #[case] holds_the_user: bool,
    ) {
        let kinds = planned(
            &caller(holder, EndUserRecord::default()),
            Settings::default(),
        )
        .into_iter()
        .map(|(_, kind, ..)| kind)
        .collect::<Vec<_>>();

        assert_eq!(
            kinds.contains(&"User".to_owned()),
            holds_the_user,
            "{kinds:?}"
        );
    }

    #[rstest]
    #[case::its_own_budget(Some(70.0), Some(80.0), Some((70.0, 3.0)))]
    #[case::the_teams_default(None, Some(80.0), Some((80.0, 3.0)))]
    #[case::explicit_zero_wins(Some(0.0), Some(80.0), Some((0.0, 3.0)))]
    #[case::zero_default_does_not_hold(None, Some(0.0), None)]
    #[case::negative_default_does_not_hold(None, Some(-1.0), None)]
    #[case::neither(None, None, None)]
    fn a_member_is_held_to_its_own_budget_or_its_teams_default(
        #[case] own: Option<f64>,
        #[case] default: Option<f64>,
        #[case] expected: Option<(f64, f64)>,
    ) {
        let member = Holder::Member {
            user: Named::Stored(user(None)),
            team: Team::Row(team(default)),
            membership: Some(membership(own)),
        };

        let held = counters(
            &caller(member, EndUserRecord::default()),
            Settings::default(),
            Timestamp::UNIX_EPOCH,
        )
        .into_iter()
        .find(|counter| counter.entity_type == "TeamMember")
        .map(|counter| (counter.max_budget, counter.fallback_spend));

        assert_eq!(held, expected);
    }

    #[rstest]
    #[case::active_own(Some(70.0), Some("1d"), Timestamp::MAX, 120.0)]
    #[case::expired_at_the_boundary(Some(70.0), Some("1d"), Timestamp::UNIX_EPOCH, 100.0)]
    #[case::no_duration_no_rollover(Some(70.0), None, Timestamp::MAX, 90.0)]
    #[case::inherited_only_gets_temp_increase(None, Some("1d"), Timestamp::MAX, 100.0)]
    fn the_member_counter_applies_only_active_increases_and_capped_rollover(
        #[case] own: Option<f64>,
        #[case] duration: Option<&str>,
        #[case] expiry: Timestamp,
        #[case] expected: f64,
    ) {
        let mut membership = membership(own);
        membership.budget_duration = duration.map(str::to_owned);
        membership.rollover_credit = Some(60.0);
        membership.budget_extras.rollover_max_budget = Some(100.0);
        membership.budget_extras.temp_budget_increase = Some(20.0);
        membership.budget_extras.temp_budget_expiry = Some(expiry);
        let member = Holder::Member {
            user: Named::Stored(user(None)),
            team: Team::Row(team(Some(80.0))),
            membership: Some(membership),
        };
        let held = counters(
            &caller(member, EndUserRecord::default()),
            Settings::default(),
            Timestamp::UNIX_EPOCH,
        )
        .into_iter()
        .find(|counter| counter.entity_type == "TeamMember");
        assert_eq!(
            held.map(|counter| (counter.max_budget, counter.fallback_spend)),
            Some((expected, 3.0))
        );
    }

    #[rstest]
    #[case::no_membership_row(Holder::Member {
        user: Named::Stored(user(None)),
        team: Team::Row(team(Some(80.0))),
        membership: None,
    }, Some((80.0, 0.0)))]
    #[case::an_owner_with_no_row(Holder::Member {
        user: Named::Unstored("u1".to_owned()),
        team: Team::Row(team(Some(80.0))),
        membership: None,
    }, None)]
    fn a_members_counter_needs_the_owners_row_and_starts_from_zero_without_a_membership(
        #[case] holder: Holder<NativeRows>,
        #[case] expected: Option<(f64, f64)>,
    ) {
        let held = counters(
            &caller(holder, EndUserRecord::default()),
            Settings::default(),
            Timestamp::UNIX_EPOCH,
        )
        .into_iter()
        .find(|counter| counter.entity_type == "TeamMember")
        .map(|counter| (counter.max_budget, counter.fallback_spend));

        assert_eq!(held, expected);
    }

    #[rstest]
    #[case::zero(Some(0.0), &["Key", "User", "Organization", "Project"])]
    #[case::negative(Some(-1.0), &["Key", "User", "Organization", "Project"])]
    #[case::unset(None, &["Key", "Organization", "Project"])]
    fn only_an_unset_budget_plans_no_counter(
        #[case] max_budget: Option<f64>,
        #[case] expected: &[&str],
    ) {
        let identity = caller(
            Holder::User(Named::Stored(user(max_budget))),
            EndUserRecord::default(),
        );

        let kinds = planned(&identity, Settings::default())
            .into_iter()
            .map(|(_, kind, ..)| kind)
            .collect::<Vec<_>>();

        assert_eq!(kinds, expected);
    }

    #[rstest]
    #[case::capped(Some(10.0), Some(15.0), Some("1d"), Some(10.0), Some(15.0))]
    #[case::negative(Some(10.0), Some(15.0), Some("1d"), Some(-2.0), Some(10.0))]
    #[case::unbounded(Some(10.0), Some(f64::INFINITY), Some("1d"), Some(10.0), Some(10.0))]
    #[case::no_duration(Some(10.0), Some(15.0), None, Some(10.0), Some(10.0))]
    #[case::no_budget(None, Some(15.0), Some("1d"), Some(10.0), None)]
    #[case::zero_budget(Some(0.0), Some(15.0), Some("1d"), Some(10.0), Some(0.0))]
    fn owner_rollover_keeps_the_same_counter_and_raises_only_an_eligible_cap(
        #[case] base: Option<f64>,
        #[case] ceiling: Option<f64>,
        #[case] duration: Option<&str>,
        #[case] credit: Option<f64>,
        #[case] expected: Option<f64>,
    ) {
        let mut owner = user(base);
        owner.rollover_max_budget = ceiling;
        owner.budget_duration = duration.map(str::to_owned);
        owner.rollover_credit = credit;
        let identity = caller(Holder::User(Named::Stored(owner)), EndUserRecord::default());
        let owned = planned(&identity, Settings::default())
            .into_iter()
            .find(|(_, kind, ..)| kind == "User");
        assert_eq!(
            owned,
            expected.map(|cap| row("spend:user:u1", "User", "u1", cap, 1.0))
        );
    }

    #[rstest]
    fn the_master_key_reserves_against_nothing() {
        assert_eq!(planned(&master(), Settings::default()), []);
    }
}
