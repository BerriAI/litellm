//! Tag rows and their budgets in the shape the auth cache holds.

use std::sync::Arc;

use litellm_redis_store::auth_cache::TagEntry;
use litellm_repositories::{Tag, TagRepository};

use crate::{
    Error,
    budget::budget_entry,
    column::{nullable, time},
    error::within,
};

/// Reads the tag registry and the cache entries for named tags.
pub struct TagDatabase {
    tags: Arc<TagRepository>,
}

impl TagDatabase {
    pub fn new(tags: Arc<TagRepository>) -> Self {
        Self { tags }
    }

    /// Up to `limit` names for the registry. Its caller decides when the registry overflows.
    pub async fn names(&self, limit: i64) -> Result<Vec<String>, Error> {
        Ok(self.tags.names(limit).await?)
    }

    /// Cache entries for the named tags that exist, in no particular order.
    pub async fn tags(&self, names: &[String]) -> Result<Vec<TagEntry>, Error> {
        self.tags
            .find_many_with_budgets(names)
            .await?
            .into_iter()
            .map(tag_entry)
            .collect()
    }
}

fn tag_entry(tag: Tag) -> Result<TagEntry, Error> {
    let Tag {
        tag_name,
        description,
        models,
        model_info,
        spend,
        budget_id,
        created_at,
        created_by,
        updated_at,
        litellm_budget_table,
    } = tag;
    Ok(TagEntry {
        tag_name,
        description,
        models,
        model_info: nullable("model_info", model_info.as_ref())?,
        spend,
        budget_id,
        litellm_budget_table: litellm_budget_table
            .map(|budget| budget_entry(budget.0))
            .transpose()
            .map_err(within("litellm_budget_table"))?,
        created_at: time("created_at", Some(created_at))?,
        created_by,
        updated_at: time("updated_at", Some(updated_at))?,
        unread: Vec::new(),
    })
}
