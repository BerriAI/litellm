use std::cmp::Ordering::{Equal, Greater, Less};

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, strum::VariantArray)]
pub enum Scope {
    Team,
    Key,
    TeamWindow,
    KeyWindow,
    Project,
    Organization,
    Tag,
    AccessGroup,
    User,
    TeamMember,
    EndUser,
    Proxy,
    KeySoft,
    TeamSoft,
    ProjectSoft,
}

#[derive(Clone, Debug, Deserialize)]
pub struct BudgetFacts {
    pub scope: Scope,
    pub limit: Option<f64>,
    pub spend: Option<f64>,
    pub token_present: bool,
    pub team_key: bool,
    pub apply_user_budget_to_team_keys: bool,
    pub info_route: bool,
    pub llm_route: bool,
    pub model_list_route: bool,
    pub skip: bool,
    pub alert_recipients: Vec<String>,
    pub throttle_percentage: Option<f64>,
    pub throttle_opt_in: bool,
    pub tpm_limit: Option<i64>,
    pub rpm_limit: Option<i64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct BudgetDecision {
    pub read_spend: bool,
    pub denied: bool,
    pub alert: Option<&'static str>,
    pub throttle_percentage: Option<f64>,
}

pub fn check(facts: &BudgetFacts) -> BudgetDecision {
    use Scope::*;
    let Some(limit) = facts.limit else {
        return BudgetDecision::default();
    };
    if matches!(facts.scope, User) && facts.team_key && !facts.apply_user_budget_to_team_keys
        || matches!(facts.scope, EndUser) && facts.info_route
        || matches!(facts.scope, Project) && !limit.is_finite()
        || matches!(facts.scope, AccessGroup) && limit <= 0.0
        || matches!(facts.scope, Proxy)
            && (limit.partial_cmp(&0.0) != Some(Greater)
                || facts.skip
                || !facts.llm_route
                || facts.model_list_route)
    {
        return BudgetDecision::default();
    }
    let soft = matches!(facts.scope, KeySoft | TeamSoft | ProjectSoft);
    let Some(spend) = facts.spend else {
        return BudgetDecision {
            read_spend: !soft && facts.scope != Proxy,
            ..BudgetDecision::default()
        };
    };
    if soft {
        let alert = spend >= limit
            && match facts.scope {
                KeySoft => limit != 0.0,
                TeamSoft => facts.token_present && !facts.alert_recipients.is_empty(),
                ProjectSoft => facts.token_present,
                _ => false,
            };
        return BudgetDecision {
            alert: alert.then_some("soft_budget"),
            ..BudgetDecision::default()
        };
    }
    let exceeded = match facts.scope {
        Team | Proxy => limit.is_finite() && spend > limit,
        EndUser => spend > limit,
        Tag => !matches!(spend.partial_cmp(&limit), Some(Less | Equal)),
        AccessGroup => spend.partial_cmp(&limit) != Some(Less),
        _ => limit.is_finite() && spend >= limit,
    };
    let alert = match facts.scope {
        Key => Some("token_budget"),
        User => Some("user_budget"),
        Team if exceeded && facts.token_present => Some("team_budget"),
        Project if exceeded && facts.token_present => Some("project_budget"),
        Organization if exceeded => Some("organization_budget"),
        TeamMember if limit.is_finite() => Some("team_member_threshold"),
        _ => None,
    };
    let throttle_percentage = if exceeded
        && facts.scope == Key
        && facts.throttle_opt_in
        && (facts.tpm_limit.is_some() || facts.rpm_limit.is_some())
    {
        facts
            .throttle_percentage
            .filter(|pct| *pct > 0.0 && *pct <= 1.0)
    } else {
        None
    };
    BudgetDecision {
        read_spend: false,
        denied: exceeded && throttle_percentage.is_none(),
        alert,
        throttle_percentage,
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct AlertFacts {
    pub spend: Option<f64>,
    pub max_budget: Option<f64>,
    pub configured: bool,
    pub minimum_percent: Option<f64>,
    pub default_ratio: f64,
    pub team_member: bool,
}

pub fn alert(facts: &AlertFacts) -> bool {
    let (Some(spend), Some(limit)) = (facts.spend, facts.max_budget) else {
        return false;
    };
    if facts.team_member {
        if !facts.configured || spend <= 0.0 {
            return false;
        }
    } else if spend.partial_cmp(&0.0) != Some(Greater) {
        return false;
    }
    if facts.configured {
        return facts
            .minimum_percent
            .is_some_and(|pct| spend.partial_cmp(&(limit * (pct / 100.0))) != Some(Less));
    }
    spend >= limit * facts.default_ratio && spend < limit
}

#[derive(Clone, Debug, Deserialize)]
pub struct MemberLimitFacts {
    pub member_limit: Option<f64>,
    pub default_limit: Option<f64>,
    pub temporary_increase: Option<f64>,
    pub expires_at: Option<f64>,
    pub now: f64,
    #[serde(default)]
    pub now_naive: bool,
}

pub fn member_limit(facts: &MemberLimitFacts) -> Result<Option<f64>, crate::Error> {
    let base = facts.member_limit.or_else(|| {
        facts
            .default_limit
            .filter(|limit| !matches!(limit.partial_cmp(&0.0), Some(Less | Equal)))
    });
    let Some(base) = base else {
        return Ok(None);
    };
    let increase = match (facts.temporary_increase, facts.expires_at) {
        (Some(_), Some(_)) if facts.now_naive => return Err(crate::Error::IncompatibleDateTimes),
        (Some(increase), Some(expiry)) if expiry > facts.now => increase,
        _ => 0.0,
    };
    Ok(Some(base + increase))
}

#[derive(Clone, Debug, Deserialize)]
pub struct WindowFacts {
    pub duration_seconds: Option<f64>,
    pub reset_at: Option<f64>,
    pub now: f64,
}

pub fn window_start(facts: &WindowFacts) -> Option<f64> {
    facts
        .duration_seconds
        .map(|seconds| facts.reset_at.unwrap_or(facts.now) - seconds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::team_equality(Scope::Team, 1.0, 1.0, false)]
    #[case::key_equality(Scope::Key, 1.0, 1.0, true)]
    #[case::tag_nan(Scope::Tag, f64::NAN, 1.0, true)]
    #[case::group_nan_spend(Scope::AccessGroup, 1.0, f64::NAN, true)]
    #[case::key_nan(Scope::Key, f64::NAN, 1.0, false)]
    #[case::end_user_infinity(Scope::EndUser, 1.0, f64::INFINITY, true)]
    fn boundary(
        #[case] scope: Scope,
        #[case] limit: f64,
        #[case] spend: f64,
        #[case] denied: bool,
    ) {
        let facts = BudgetFacts {
            scope,
            limit: Some(limit),
            spend: Some(spend),
            token_present: false,
            team_key: false,
            apply_user_budget_to_team_keys: false,
            info_route: false,
            llm_route: true,
            model_list_route: false,
            skip: false,
            alert_recipients: vec![],
            throttle_percentage: None,
            throttle_opt_in: false,
            tpm_limit: None,
            rpm_limit: None,
        };
        assert_eq!(check(&facts).denied, denied);
    }
}
