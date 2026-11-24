//! Python handles on the Rust repositories. Each wraps one repository built over a `NativeDatabase`'s pools and
//! hands its answers to Python as plain values or JSON text for `model_validate_json`.

mod access_group;
mod agent_identity;
mod audit_log;
mod budget;
mod budget_window_spend;
mod end_user;
mod jwt_key_mapping;
mod key;
mod model_access_group;
mod object_permission;
mod organization;
mod project;
mod spend_log;
mod tag;
mod team;
mod team_membership;
mod user;

pub(crate) use access_group::RustAccessGroupRepository;
pub(crate) use agent_identity::RustAgentIdentityRepository;
pub(crate) use audit_log::RustAuditLogRepository;
pub(crate) use budget::RustBudgetRepository;
pub(crate) use budget_window_spend::RustBudgetWindowSpendRepository;
pub(crate) use end_user::RustEndUserRepository;
pub(crate) use jwt_key_mapping::RustJwtKeyMappingRepository;
pub(crate) use key::RustKeyRepository;
pub(crate) use model_access_group::RustModelAccessGroupRepository;
pub(crate) use object_permission::RustObjectPermissionRepository;
pub(crate) use organization::RustOrganizationRepository;
pub(crate) use project::RustProjectRepository;
pub(crate) use spend_log::RustSpendLogRepository;
pub(crate) use tag::RustTagRepository;
pub(crate) use team::RustTeamRepository;
pub(crate) use team_membership::RustTeamMembershipRepository;
pub(crate) use user::RustUserRepository;

fn json<T: serde::Serialize>(value: Option<&T>) -> pyo3::PyResult<Option<String>> {
    value.map(litellm_host_python::json_dumps).transpose()
}

/// `(created, row)`: whether this call wrote the row, then the row now stored as JSON text, `None` when the one
/// that beat it no longer answers.
fn inserted_json<T: serde::Serialize>(
    inserted: litellm_repositories::Inserted<T>,
) -> pyo3::PyResult<(bool, Option<String>)> {
    match inserted {
        litellm_repositories::Inserted::Created(row) => Ok((true, json(Some(&row))?)),
        litellm_repositories::Inserted::Conflict(row) => Ok((false, json(row.as_ref())?)),
    }
}

fn json_list<T: serde::Serialize>(values: &[T]) -> pyo3::PyResult<Vec<String>> {
    values.iter().map(litellm_host_python::json_dumps).collect()
}

fn side(use_writer: bool) -> litellm_repositories::Side {
    if use_writer {
        litellm_repositories::Side::Writer
    } else {
        litellm_repositories::Side::Read
    }
}
