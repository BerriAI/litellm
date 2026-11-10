use crate::answers::BudgetExemption;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WithinModelBudgets {
    _checked: (),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BudgetsSkipped {
    _skipped: (),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelBudgetProof {
    Within(WithinModelBudgets),
    Skipped(BudgetsSkipped),
}

pub fn skips_model_budgets(exemption: &BudgetExemption) -> bool {
    exemption.route_skips_budgets || exemption.mcp_discovery || exemption.model_is_free
}
