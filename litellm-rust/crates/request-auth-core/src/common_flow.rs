use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct Facts {
    pub model_present: bool,
    pub skip_budgets: bool,
    pub team_present: bool,
    pub user_present: bool,
    pub token_present: bool,
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub customer_present: bool,
    pub customer_models: Vec<String>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct Plan {
    pub membership: bool,
    pub team_models: bool,
    pub member_models: bool,
    pub user_models: bool,
    pub customer_models: bool,
    pub agent_trace: bool,
}

pub fn plan(facts: Facts) -> Plan {
    Plan {
        membership: facts.token_present
            && (facts.model_present || !facts.skip_budgets)
            && facts.team_present
            && facts.user_id.is_some(),
        team_models: facts.model_present && facts.team_present,
        member_models: facts.model_present
            && facts.team_present
            && facts.token_present
            && facts.user_id.is_some_and(|id| !id.is_empty()),
        user_models: facts.model_present && !facts.team_present && facts.user_present,
        customer_models: facts.customer_present && !facts.customer_models.is_empty(),
        agent_trace: facts.token_present && facts.agent_id.is_some_and(|id| !id.is_empty()),
    }
}

#[derive(Deserialize)]
pub struct BudgetFacts {
    pub token_present: bool,
    pub group_count: usize,
    pub customer_present: bool,
    pub customer_budget_present: bool,
}

#[derive(Debug, PartialEq, Eq, Serialize, strum::VariantArray)]
pub enum BudgetScope {
    Team,
    TeamWindows,
    KeyWindows,
    TeamSoft,
    Organization,
    Tags,
    Groups,
    User,
    Member,
    Customer,
}

pub fn budgets(facts: BudgetFacts) -> Vec<BudgetScope> {
    use BudgetScope as S;
    [
        Some(S::Team),
        Some(S::TeamWindows),
        facts.token_present.then_some(S::KeyWindows),
        Some(S::TeamSoft),
        Some(S::Organization),
        Some(S::Tags),
        (facts.group_count > 0).then_some(S::Groups),
        Some(S::User),
        Some(S::Member),
        (facts.customer_present && facts.customer_budget_present).then_some(S::Customer),
    ]
    .into_iter()
    .flatten()
    .collect()
}

#[derive(Deserialize)]
pub struct CustomFacts {
    pub end_user_id: Option<String>,
    pub end_user_present: bool,
    pub common_checks: bool,
    pub model_present: bool,
    pub router_present: bool,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct CustomPlan {
    pub read_end_user: bool,
    pub check_end_user: bool,
    pub check_model_access: bool,
    pub read_pricing: bool,
}

pub fn custom(facts: CustomFacts) -> CustomPlan {
    CustomPlan {
        read_end_user: facts.end_user_id.is_some(),
        check_end_user: facts.end_user_present && !facts.common_checks,
        check_model_access: facts.common_checks,
        read_pricing: facts.model_present && facts.router_present,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::absent(None, false, false)]
    #[case::empty(Some(""), true, false)]
    #[case::user(Some("user"), true, true)]
    fn member_identity(#[case] user: Option<&str>, #[case] membership: bool, #[case] models: bool) {
        let actual = plan(Facts {
            model_present: true,
            skip_budgets: false,
            team_present: true,
            user_present: false,
            token_present: true,
            user_id: user.map(str::to_owned),
            agent_id: None,
            customer_present: false,
            customer_models: vec![],
        });
        assert_eq!(actual.membership, membership);
        assert_eq!(actual.member_models, models);
    }

    #[rstest]
    #[case::empty(false, 0, false, false, vec![BudgetScope::Team, BudgetScope::TeamWindows, BudgetScope::TeamSoft, BudgetScope::Organization, BudgetScope::Tags, BudgetScope::User, BudgetScope::Member])]
    #[case::all(true, 1, true, true, vec![BudgetScope::Team, BudgetScope::TeamWindows, BudgetScope::KeyWindows, BudgetScope::TeamSoft, BudgetScope::Organization, BudgetScope::Tags, BudgetScope::Groups, BudgetScope::User, BudgetScope::Member, BudgetScope::Customer])]
    fn budget_order(
        #[case] token_present: bool,
        #[case] group_count: usize,
        #[case] customer_present: bool,
        #[case] customer_budget_present: bool,
        #[case] expected: Vec<BudgetScope>,
    ) {
        assert_eq!(
            budgets(BudgetFacts {
                token_present,
                group_count,
                customer_present,
                customer_budget_present
            }),
            expected
        );
    }
}
