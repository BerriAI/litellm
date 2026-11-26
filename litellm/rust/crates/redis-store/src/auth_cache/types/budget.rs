use litellm_python_compat::{Value, datetime::IsoDateTime};

use super::{
    bag::{Bag, Ints},
    record::{ExcludedIfNone, Field, fields, record},
};

const SERVER_MANAGED: [&str; 2] = ["budget_reset_at", "created_at"];

record! {
    /// `LiteLLM_BudgetTable.model_dump(mode="json")`, nested under `litellm_budget_table` in
    /// [`ProjectEntry`](super::ProjectEntry), and the first fields of a [`BudgetFullEntry`].
    pub struct BudgetEntry {
        budget_id: Option<String>,
        soft_budget: Option<f64>,
        max_budget: Option<f64>,
        rollover_max_budget: ExcludedIfNone<f64>,
        max_parallel_requests: Option<i64>,
        tpm_limit: Option<i64>,
        rpm_limit: Option<i64>,
        tpd_limit: Option<i64>,
        model_max_budget: Option<Bag>,
        model_tpm_limit: Option<Ints>,
        model_rpm_limit: Option<Ints>,
        budget_duration: Option<String>,
        allowed_models: Option<Vec<String>>,
        temp_budget_increase: Option<f64>,
        temp_budget_expiry: Option<IsoDateTime>,
    }
}

/// `LiteLLM_BudgetTableFull.model_dump(mode="json")`, nested under `litellm_budget_table` in
/// [`TeamMembershipEntry`](super::TeamMembershipEntry): the [`BudgetEntry`] with the row's reset
/// and creation times after it. Pydantic picks this branch of the membership's
/// `LiteLLM_BudgetTableFull | LiteLLM_BudgetTable` for any budget with a `created_at`, which no
/// budget row lacks, so an entry without one does not decode.
#[derive(Clone, Debug, PartialEq)]
pub struct BudgetFullEntry {
    pub budget: BudgetEntry,
    pub budget_reset_at: Option<IsoDateTime>,
    pub created_at: IsoDateTime,
}

impl Field for BudgetFullEntry {
    fn read(value: &Value) -> Option<Self> {
        let Value::Dict(entries) = value else {
            return None;
        };
        let named = fields(entries);
        let budget_reset_at = named
            .get("budget_reset_at")
            .map_or(Some(None), |at| Field::read(at))?;
        let created_at = IsoDateTime::read(named.get("created_at")?)?;
        let inherited = entries
            .iter()
            .filter(|(name, _)| {
                !matches!(name, Value::Str(name) if SERVER_MANAGED.contains(&name.as_str()))
            })
            .cloned()
            .collect();
        Some(Self {
            budget: BudgetEntry::read(&Value::Dict(inherited))?,
            budget_reset_at,
            created_at,
        })
    }

    fn write(self) -> Value {
        let server_managed = [
            ("budget_reset_at", self.budget_reset_at.write()),
            ("created_at", self.created_at.write()),
        ]
        .map(|(name, value)| (Value::Str(name.to_owned()), value));
        Value::Dict(
            self.budget
                .into_entries()
                .into_iter()
                .chain(server_managed)
                .collect(),
        )
    }
}
