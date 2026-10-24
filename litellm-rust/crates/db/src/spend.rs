use sqlx::PgExecutor;

use crate::Error;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EntitySpend {
    pub ids: Vec<String>,
    pub costs: Vec<f64>,
}

impl EntitySpend {
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

impl FromIterator<(String, f64)> for EntitySpend {
    fn from_iter<I: IntoIterator<Item = (String, f64)>>(entries: I) -> Self {
        let mut entries: Vec<_> = entries.into_iter().collect();
        entries.sort_by(|left, right| left.0.cmp(&right.0));
        let (ids, costs) = entries.into_iter().unzip();
        Self { ids, costs }
    }
}

pub async fn add_key_spend(
    executor: impl PgExecutor<'_>,
    spend: &EntitySpend,
) -> Result<(), Error> {
    if spend.is_empty() {
        return Ok(());
    }
    sqlx::query_file!("queries/spend/add_key_spend.sql", &spend.ids, &spend.costs)
        .execute(executor)
        .await?;
    Ok(())
}

pub async fn add_user_spend(
    executor: impl PgExecutor<'_>,
    spend: &EntitySpend,
) -> Result<(), Error> {
    if spend.is_empty() {
        return Ok(());
    }
    sqlx::query_file!("queries/spend/add_user_spend.sql", &spend.ids, &spend.costs)
        .execute(executor)
        .await?;
    Ok(())
}
