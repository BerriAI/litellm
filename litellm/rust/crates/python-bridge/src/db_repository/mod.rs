//! Python handles on the Rust auth repositories. Each wraps one repository built over a `NativeDatabase`'s pools and
//! hands its answers to Python as plain values or JSON text for `model_validate_json`.

mod budget;
mod end_user;
mod jwt_key_mapping;
mod key;
mod model_access_group;
mod organization;
mod team;
mod team_membership;
mod user;

pub(crate) use budget::RustBudgetRepository;
pub(crate) use end_user::RustEndUserRepository;
pub(crate) use jwt_key_mapping::RustJwtKeyMappingRepository;
pub(crate) use key::RustKeyRepository;
pub(crate) use model_access_group::RustModelAccessGroupRepository;
pub(crate) use organization::RustOrganizationRepository;
pub(crate) use team::RustTeamRepository;
pub(crate) use team_membership::RustTeamMembershipRepository;
pub(crate) use user::RustUserRepository;

fn json<T: serde::Serialize>(value: Option<&T>) -> pyo3::PyResult<Option<String>> {
    value.map(litellm_host_python::json_dumps).transpose()
}

fn json_list<T: serde::Serialize>(values: &[T]) -> pyo3::PyResult<Vec<String>> {
    values.iter().map(litellm_host_python::json_dumps).collect()
}
