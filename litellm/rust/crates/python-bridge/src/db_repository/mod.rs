//! Python handles on the Rust repositories. Each wraps one repository built over a `NativeDatabase`'s pools and
//! hands its answers to Python as plain values or JSON text for `model_validate_json`.

mod access_group;
mod app;
mod app_customer_member;
mod app_customer_organization;
mod budget;
mod customer_member;
mod customer_organization;
mod customer_team;
mod end_user;
mod jwt_key_mapping;
mod key;
mod model_access_group;
mod object_permission;
mod organization;
mod project;
mod tag;
mod team;
mod team_membership;
mod user;

pub(crate) use access_group::RustAccessGroupRepository;
pub(crate) use app::RustAppRepository;
pub(crate) use app_customer_member::RustAppCustomerMemberRepository;
pub(crate) use app_customer_organization::RustAppCustomerOrganizationRepository;
pub(crate) use budget::RustBudgetRepository;
pub(crate) use customer_member::RustCustomerMemberRepository;
pub(crate) use customer_organization::RustCustomerOrganizationRepository;
pub(crate) use customer_team::RustCustomerTeamRepository;
pub(crate) use end_user::RustEndUserRepository;
pub(crate) use jwt_key_mapping::RustJwtKeyMappingRepository;
pub(crate) use key::RustKeyRepository;
pub(crate) use model_access_group::RustModelAccessGroupRepository;
pub(crate) use object_permission::RustObjectPermissionRepository;
pub(crate) use organization::RustOrganizationRepository;
pub(crate) use project::RustProjectRepository;
pub(crate) use tag::RustTagRepository;
pub(crate) use team::RustTeamRepository;
pub(crate) use team_membership::RustTeamMembershipRepository;
pub(crate) use user::RustUserRepository;

fn json<T: serde::Serialize>(value: Option<&T>) -> pyo3::PyResult<Option<String>> {
    value.map(litellm_host_python::json_dumps).transpose()
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
