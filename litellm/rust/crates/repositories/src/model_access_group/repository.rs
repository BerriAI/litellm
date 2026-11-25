//! The model access group repository: which groups have a budget row, and those rows' budgets.

use litellm_db::Call;
use litellm_python_compat::{json, truthy::truthy};

use super::queries::{self, ModelAccessGroupBudgetRow};
use crate::{Database, RawJson, Side};

const GROUP_NAMES: Call = Call {
    call_type: "find_many",
    table: Some("LiteLLM_ModelAccessGroupBudgetTable"),
    operation: Some("find_many"),
};

const GROUP_BUDGETS: Call = Call {
    call_type: "get_model_access_group_budgets_batch",
    table: Some("LiteLLM_ModelAccessGroupBudgetTable"),
    operation: Some("find_many"),
};

/// Reads `LiteLLM_ModelAccessGroupBudgetTable`.
pub struct ModelAccessGroupRepository {
    database: Database,
}

/// A model access group's spend and the budget it is held to.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelAccessGroupBudget {
    pub access_group_name: String,
    pub spend: f64,
    pub max_budget: Option<f64>,
    /// The group's budget row sets concurrent budget windows, which only customer hierarchies support.
    pub has_budget_windows: bool,
}

impl ModelAccessGroupRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// Up to `limit` names of groups that have a budget row.
    pub async fn names(&self, limit: i64) -> Result<Vec<String>, litellm_db::Error> {
        self.database
            .run(GROUP_NAMES, || async {
                let mut session = self.database.session(Side::Read).await?;
                queries::model_access_group_names(session.executor(), limit).await
            })
            .await
    }

    /// The budgets of whichever of `names` have a budget row.
    pub async fn budgets(
        &self,
        names: &[String],
    ) -> Result<Vec<ModelAccessGroupBudget>, litellm_db::Error> {
        if names.is_empty() {
            return Ok(Vec::new());
        }
        let rows = self
            .database
            .run(GROUP_BUDGETS, || async {
                let mut session = self.database.session(Side::Read).await?;
                queries::find_model_access_group_budgets(session.executor(), names).await
            })
            .await?;
        Ok(rows.into_iter().map(ModelAccessGroupBudget::from).collect())
    }
}

impl From<ModelAccessGroupBudgetRow> for ModelAccessGroupBudget {
    fn from(row: ModelAccessGroupBudgetRow) -> Self {
        let has_budget_row = row.budget_id.is_some();
        Self {
            access_group_name: row.access_group_name,
            spend: row.spend,
            max_budget: row.max_budget,
            has_budget_windows: has_budget_row
                && row.budget_limits.as_ref().is_some_and(sets_windows),
        }
    }
}

fn sets_windows(budget_limits: &RawJson) -> bool {
    json::loads(budget_limits.get()).is_ok_and(|value| truthy(&value))
}
