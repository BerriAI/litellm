use serde::{Deserialize, Serialize};

use crate::{
    Error,
    common_budgets::{self, BudgetFacts, MemberLimitFacts},
};

#[derive(Clone, Default, Deserialize)]
pub struct Facts {
    pub user_present: bool,
    pub team_id_present: bool,
    pub check_db_only: bool,
    pub token_org: Option<String>,
    pub team_org: Option<String>,
    pub source_ready: bool,
    pub team: Option<BudgetFacts>,
    pub team_alerted: bool,
    pub window_count: Option<usize>,
    pub windows: Vec<BudgetFacts>,
    pub membership_loaded: bool,
    pub member_limit: Option<MemberLimitFacts>,
    pub default_id: Option<String>,
    pub default_loaded: bool,
    pub member: Option<BudgetFacts>,
    pub member_alerted: bool,
    pub org_loaded: bool,
    pub org_failed: bool,
    pub org: Option<BudgetFacts>,
    pub org_alerted: bool,
    pub io_failed: bool,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    Complete { allowed: bool, log_failure: bool },
    Source { org_id: Option<String> },
    ReadTeam,
    AlertTeam,
    ReadWindows,
    ReadWindow { index: usize },
    ReadMembership,
    ReadMemberLimit,
    ReadDefault { id: String },
    ReadMember { limit: Option<f64> },
    AlertMember,
    ReadOrg { id: String },
    AlertOrg { id: String },
}

pub fn decide(facts: &Facts) -> Result<Decision, Error> {
    let complete = |allowed, log_failure| {
        Ok(Decision::Complete {
            allowed,
            log_failure,
        })
    };
    let org_id = facts
        .token_org
        .as_ref()
        .filter(|id| !id.is_empty())
        .or(facts.team_org.as_ref());
    if !facts.source_ready {
        return Ok(Decision::Source {
            org_id: org_id.cloned(),
        });
    }
    if facts.io_failed {
        return complete(false, false);
    }
    let Some(team) = &facts.team else {
        return Ok(Decision::ReadTeam);
    };
    let team = common_budgets::check(team);
    if team.read_spend {
        return Err(Error::InvalidAnswer);
    }
    if team.alert.is_some() && !facts.team_alerted {
        return Ok(Decision::AlertTeam);
    }
    if team.denied {
        return complete(false, false);
    }
    let Some(count) = facts.window_count else {
        return Ok(Decision::ReadWindows);
    };
    for index in 0..count {
        let Some(window) = facts.windows.get(index) else {
            return Ok(Decision::ReadWindow { index });
        };
        let budget = common_budgets::check(window);
        if budget.read_spend {
            return Err(Error::InvalidAnswer);
        }
        if budget.denied {
            return complete(false, false);
        }
    }
    if facts.check_db_only && !facts.user_present {
        return complete(false, false);
    }
    if facts.user_present
        && (facts.team_id_present || facts.check_db_only)
        && !facts.membership_loaded
    {
        return Ok(Decision::ReadMembership);
    }
    if facts.user_present && facts.team_id_present {
        let Some(limit) = &facts.member_limit else {
            return Ok(Decision::ReadMemberLimit);
        };
        if limit.member_limit.is_none()
            && !facts.default_loaded
            && let Some(id) = &facts.default_id
        {
            return Ok(Decision::ReadDefault { id: id.clone() });
        }
        let limit = match common_budgets::member_limit(limit) {
            Ok(limit) => limit,
            Err(Error::IncompatibleDateTimes) => return complete(false, true),
            Err(error) => return Err(error),
        };
        let Some(member) = &facts.member else {
            return Ok(Decision::ReadMember { limit });
        };
        let member = common_budgets::check(member);
        if member.read_spend {
            return Err(Error::InvalidAnswer);
        }
        if member.alert.is_some() && !facts.member_alerted {
            return Ok(Decision::AlertMember);
        }
        if member.denied {
            return complete(false, false);
        }
    }
    if let Some(id) = org_id {
        if !facts.org_loaded {
            return Ok(Decision::ReadOrg { id: id.clone() });
        }
        if !facts.org_failed
            && let Some(org) = &facts.org
        {
            let org = common_budgets::check(org);
            if org.read_spend {
                return Err(Error::InvalidAnswer);
            }
            if org.alert.is_some() && !facts.org_alerted {
                return Ok(Decision::AlertOrg { id: id.clone() });
            }
            if org.denied {
                return complete(false, false);
            }
        }
    }
    complete(true, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::token(Some("token"), Some("team"), Some("token"))]
    #[case::empty_token(Some(""), Some("team"), Some("team"))]
    #[case::empty_team(None, Some(""), Some(""))]
    #[case::absent(None, None, None)]
    fn source_org(
        #[case] token: Option<&str>,
        #[case] team: Option<&str>,
        #[case] selected: Option<&str>,
    ) {
        let facts = Facts {
            token_org: token.map(str::to_owned),
            team_org: team.map(str::to_owned),
            ..Facts::default()
        };
        assert_eq!(
            decide(&facts).unwrap(),
            Decision::Source {
                org_id: selected.map(str::to_owned)
            }
        );
    }
}
