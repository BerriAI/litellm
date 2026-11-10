use litellm_gateway_auth_core::{BudgetExemption, skips_model_budgets};
use rstest::rstest;

#[rstest]
#[case(false, false, false, false)]
#[case(true, false, false, true)]
#[case(false, true, false, true)]
#[case(false, false, true, true)]
fn model_budgets_are_skipped_for_budget_free_routes_mcp_discovery_and_free_models(
    #[case] route_skips_budgets: bool,
    #[case] mcp_discovery: bool,
    #[case] model_is_free: bool,
    #[case] skips: bool,
) {
    let exemption = BudgetExemption {
        route_skips_budgets,
        mcp_discovery,
        model_is_free,
    };
    assert_eq!(skips_model_budgets(&exemption), skips);
}
