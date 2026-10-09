use crate::InvalidScope;

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Clone, Debug)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum QueryScope {
    #[cfg_attr(feature = "schema", schemars(title = "AllQueryScope"))]
    All,
    #[cfg_attr(feature = "schema", schemars(title = "OwnedQueryScope"))]
    Owned {
        user_id: String,
        team_ids: Vec<String>,
    },
}

impl QueryScope {
    pub fn validate(&self) -> Result<(), InvalidScope> {
        match self {
            Self::All => Ok(()),
            Self::Owned { user_id, team_ids }
                if (!user_id.is_empty() || !team_ids.is_empty())
                    && team_ids.iter().all(|team| !team.is_empty()) =>
            {
                Ok(())
            }
            _ => Err(InvalidScope),
        }
    }
}
