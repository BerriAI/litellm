use serde::{Deserialize, Serialize};
use strum::VariantArray;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct IdFacts {
    pub value: Option<String>,
}

pub fn normalize_id(facts: &IdFacts) -> Option<String> {
    facts.value.clone().filter(|value| !value.is_empty())
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DefaultFacts {
    pub prisma_present: bool,
    pub explicit_id: Option<String>,
    pub global_read: bool,
    pub global_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DefaultDecision {
    pub read_global: bool,
    pub budget_id: Option<String>,
}

pub fn default_id(facts: &DefaultFacts) -> DefaultDecision {
    if facts.explicit_id.is_none() && !facts.global_read {
        return DefaultDecision {
            read_global: true,
            budget_id: None,
        };
    }
    if !facts.prisma_present {
        return DefaultDecision {
            read_global: false,
            budget_id: None,
        };
    }
    DefaultDecision {
        read_global: false,
        budget_id: facts
            .explicit_id
            .clone()
            .or_else(|| facts.global_id.clone()),
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ApplyFacts {
    pub own_budget_id: Option<String>,
    pub own_budget_present: bool,
    pub key_id: Option<String>,
    pub global_read: bool,
    pub global_id: Option<String>,
    pub default_read: bool,
    pub default_present: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize, VariantArray)]
pub enum ApplyDecision {
    Keep,
    ReadGlobal,
    ReadDefault,
    Copy,
}

pub fn apply_default(facts: &ApplyFacts) -> ApplyDecision {
    if facts.own_budget_id.is_some() && facts.own_budget_present {
        return ApplyDecision::Keep;
    }
    if facts.key_id.is_none() {
        if !facts.global_read {
            return ApplyDecision::ReadGlobal;
        }
        if facts.global_id.is_none() {
            return ApplyDecision::Keep;
        }
    }
    if !facts.default_read {
        return ApplyDecision::ReadDefault;
    }
    if facts.default_present {
        ApplyDecision::Copy
    } else {
        ApplyDecision::Keep
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RegistryFacts {
    pub end_user_id: String,
    pub global_id: Option<String>,
    pub key_id: Option<String>,
    pub validate_id: bool,
    pub token_max_present: bool,
    pub registry_read: bool,
    pub registry: Option<Vec<String>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize, VariantArray)]
pub enum RegistryDecision {
    Read,
    Restricted,
    Unrestricted,
}

pub fn registry(facts: &RegistryFacts) -> RegistryDecision {
    if facts.global_id.is_some()
        || facts.key_id.is_some()
        || facts.validate_id
        || facts.token_max_present
    {
        return RegistryDecision::Restricted;
    }
    if !facts.registry_read {
        return RegistryDecision::Read;
    }
    if facts
        .registry
        .as_ref()
        .is_some_and(|ids| !ids.contains(&facts.end_user_id))
    {
        RegistryDecision::Unrestricted
    } else {
        RegistryDecision::Restricted
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize, VariantArray)]
pub enum Field {
    EndUserId,
    Tpm,
    Rpm,
    Tpd,
    MaxBudget,
    Models,
    Region,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Fields {
    pub tpm: bool,
    pub rpm: bool,
    pub tpd: bool,
    pub max_budget: bool,
    pub models: bool,
    pub region: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize, VariantArray)]
pub enum AssignmentMode {
    Params,
    Token,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AssignmentFacts {
    pub mode: AssignmentMode,
    pub fields: Fields,
}

pub fn assignments(facts: &AssignmentFacts) -> Vec<Field> {
    let fields = &facts.fields;
    match facts.mode {
        AssignmentMode::Params => [
            (Field::Tpm, fields.tpm),
            (Field::Rpm, fields.rpm),
            (Field::Tpd, fields.tpd),
            (Field::MaxBudget, fields.max_budget),
            (Field::Models, fields.models),
        ]
        .into_iter()
        .filter_map(|(field, present)| present.then_some(field))
        .collect(),
        AssignmentMode::Token => [
            (Field::EndUserId, true),
            (Field::Tpm, fields.tpm),
            (Field::Rpm, fields.rpm),
            (Field::Tpd, fields.tpd),
            (Field::Region, fields.region),
            (Field::Models, fields.models),
        ]
        .into_iter()
        .filter_map(|(field, present)| present.then_some(field))
        .collect(),
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct LookupFacts {
    pub own_budget_present: bool,
    pub key_id: Option<String>,
    pub end_user_read: bool,
    pub end_user_present: bool,
    pub global_read: bool,
    pub global_id: Option<String>,
    pub default_read: bool,
    pub default_present: bool,
    pub token_max_present: bool,
    pub applied: bool,
    pub read_failed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize, VariantArray)]
pub enum LookupAction {
    ReadEndUser,
    ReadGlobal,
    ReadDefault,
    ApplyOwn,
    ApplyDefault,
    Done,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LookupDecision {
    pub action: LookupAction,
    pub set_max_budget: bool,
    pub project_budget: bool,
}

pub fn lookup(facts: &LookupFacts) -> LookupDecision {
    let action = if facts.read_failed || facts.applied {
        LookupAction::Done
    } else if !facts.end_user_read {
        LookupAction::ReadEndUser
    } else if facts.end_user_present {
        LookupAction::ApplyOwn
    } else if facts.key_id.is_none() && !facts.global_read {
        LookupAction::ReadGlobal
    } else if facts.key_id.is_none() && facts.global_id.is_none() {
        LookupAction::Done
    } else if !facts.default_read {
        LookupAction::ReadDefault
    } else if facts.default_present {
        LookupAction::ApplyDefault
    } else {
        LookupAction::Done
    };
    LookupDecision {
        project_budget: action == LookupAction::ApplyDefault
            || (action == LookupAction::ApplyOwn && facts.own_budget_present),
        set_max_budget: facts.applied
            && !facts.read_failed
            && !facts.end_user_present
            && facts.default_present
            && !facts.token_max_present,
        action,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize, VariantArray)]
pub enum ValidationMode {
    #[default]
    Resolve,
    Exists,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ValidationFacts {
    pub mode: ValidationMode,
    pub end_user_id: Option<String>,
    pub enabled: bool,
    pub prisma_present: bool,
    pub global_id: Option<String>,
    pub key_id: Option<String>,
    pub cache_read: bool,
    pub cached: Option<String>,
    pub end_user_read: bool,
    pub end_user_present: bool,
    pub user_read: bool,
    pub user_present: bool,
    pub cache_written: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize, VariantArray)]
pub enum ValidationRead {
    Cache,
    EndUser,
    User,
    WriteCache,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ValidationDecision {
    pub read: Option<ValidationRead>,
    pub end_user_id: Option<String>,
    pub valid: bool,
    pub cache_value: Option<String>,
    pub cache_ttl: u64,
    pub user_email: Option<String>,
}

fn validation_result(facts: &ValidationFacts, valid: bool) -> ValidationDecision {
    let retain = valid
        || facts.key_id.is_some()
        || facts.global_id.as_ref().is_some_and(|id| !id.is_empty());
    ValidationDecision {
        end_user_id: facts.end_user_id.clone().filter(|_| retain),
        valid,
        ..ValidationDecision::default()
    }
}

pub fn validation(facts: &ValidationFacts) -> ValidationDecision {
    let Some(id) = &facts.end_user_id else {
        return ValidationDecision::default();
    };
    if facts.mode == ValidationMode::Resolve {
        if !facts.enabled || !facts.prisma_present {
            return ValidationDecision {
                end_user_id: Some(id.clone()),
                ..ValidationDecision::default()
            };
        }
        if !facts.cache_read {
            return ValidationDecision {
                read: Some(ValidationRead::Cache),
                ..ValidationDecision::default()
            };
        }
        match facts.cached.as_deref() {
            Some("valid") => return validation_result(facts, true),
            Some("invalid") => return validation_result(facts, false),
            _ => {}
        }
    }
    if !facts.end_user_read {
        return ValidationDecision {
            read: Some(ValidationRead::EndUser),
            ..ValidationDecision::default()
        };
    }
    if !facts.end_user_present && !facts.user_read {
        return ValidationDecision {
            read: Some(ValidationRead::User),
            user_email: id.contains('@').then(|| id.clone()),
            ..ValidationDecision::default()
        };
    }
    let valid = facts.end_user_present || facts.user_present;
    if facts.mode == ValidationMode::Resolve && !facts.cache_written {
        return ValidationDecision {
            read: Some(ValidationRead::WriteCache),
            valid,
            cache_value: Some(if valid { "valid" } else { "invalid" }.to_owned()),
            cache_ttl: if valid { 300 } else { 60 },
            ..ValidationDecision::default()
        };
    }
    validation_result(facts, valid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::absent(None, None)]
    #[case::empty(Some(""), None)]
    #[case::space(Some(" "), Some(" "))]
    #[case::id(Some("budget"), Some("budget"))]
    fn normalizes_metadata_id(#[case] value: Option<&str>, #[case] expected: Option<&str>) {
        assert_eq!(
            normalize_id(&IdFacts {
                value: value.map(str::to_owned)
            })
            .as_deref(),
            expected
        );
    }

    #[rstest]
    #[case::explicit(Some("key"), Some("global"), Some("key"))]
    #[case::empty_explicit(Some(""), Some("global"), Some(""))]
    #[case::global(None, Some("global"), Some("global"))]
    #[case::absent(None, None, None)]
    fn selects_default(
        #[case] explicit: Option<&str>,
        #[case] global: Option<&str>,
        #[case] expected: Option<&str>,
    ) {
        let facts = DefaultFacts {
            prisma_present: true,
            explicit_id: explicit.map(str::to_owned),
            global_read: true,
            global_id: global.map(str::to_owned),
        };
        assert_eq!(default_id(&facts).budget_id.as_deref(), expected);
    }

    #[rstest]
    #[case::none(None, None, None)]
    #[case::empty_global(Some(""), None, None)]
    #[case::global(Some("budget"), None, Some("missing"))]
    #[case::key(None, Some(""), Some("missing"))]
    fn retains_unknown_ids(
        #[case] global: Option<&str>,
        #[case] key: Option<&str>,
        #[case] expected: Option<&str>,
    ) {
        let facts = ValidationFacts {
            end_user_id: Some("missing".to_owned()),
            enabled: true,
            prisma_present: true,
            global_id: global.map(str::to_owned),
            key_id: key.map(str::to_owned),
            cache_read: true,
            cached: Some("invalid".to_owned()),
            ..ValidationFacts::default()
        };
        assert_eq!(validation(&facts).end_user_id.as_deref(), expected);
    }

    #[rstest]
    #[case::missing(AssignmentMode::Params, vec![])]
    #[case::token_identity(AssignmentMode::Token, vec![Field::EndUserId])]
    fn absent_fields(#[case] mode: AssignmentMode, #[case] expected: Vec<Field>) {
        assert_eq!(
            assignments(&AssignmentFacts {
                mode,
                fields: Fields::default()
            }),
            expected
        );
    }
}
