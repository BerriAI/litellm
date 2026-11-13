use litellm_gateway_auth_core::{
    PublicDenial, Token,
    plan::{self, Plan},
};
use pyo3::prelude::*;

use super::settings::AuthSettings;

fn presented_token(token: &Bound<'_, PyAny>) -> PyResult<Option<Token>> {
    Ok(token.extract::<Option<String>>()?.map(Token::new))
}

/// The request facts Python passes in, read from a frozen dataclass (`core.AuthRequest`).
#[derive(FromPyObject)]
pub(crate) struct RequestFacts {
    #[pyo3(attribute)]
    route: String,
    #[pyo3(attribute, from_py_with = presented_token)]
    token: Option<Token>,
    #[pyo3(attribute)]
    is_llm_api_route: bool,
    #[pyo3(attribute)]
    is_info_route: bool,
    #[pyo3(attribute)]
    pass_through_open: bool,
    #[pyo3(attribute)]
    from_custom_auth: bool,
}

impl RequestFacts {
    fn request(&self) -> plan::Request<'_> {
        plan::Request {
            route: &self.route,
            token: self.token.as_ref(),
            is_llm_api_route: self.is_llm_api_route,
            is_info_route: self.is_info_route,
            pass_through_open: self.pass_through_open,
            from_custom_auth: self.from_custom_auth,
        }
    }
}

/// Which path authenticates the request, for the host's dispatch.
#[pyclass(
    frozen,
    eq,
    skip_from_py_object,
    name = "RustPlan",
    module = "litellm.rust_bridge._native"
)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlanView {
    Public,
    OAuth2Introspection,
    OAuth2Proxy,
    JwtWithOperatorCode,
    Jwt,
    PassThroughOpen,
    Key,
}

impl From<Plan> for PlanView {
    fn from(plan: Plan) -> Self {
        match plan {
            Plan::Public => Self::Public,
            Plan::OAuth2Introspection => Self::OAuth2Introspection,
            Plan::OAuth2Proxy => Self::OAuth2Proxy,
            Plan::JwtWithOperatorCode => Self::JwtWithOperatorCode,
            Plan::Jwt => Self::Jwt,
            Plan::PassThroughOpen => Self::PassThroughOpen,
            Plan::Key => Self::Key,
        }
    }
}

/// The path to run, or the license error for a path the proxy may not run.
#[derive(IntoPyObject)]
enum Planned {
    Plan(PlanView),
    Denial(DenialView),
}

/// A denial as the exception the proxy raises for it; `denial.denial_error` only constructs it.
#[pyclass(frozen, name = "RustDenial", module = "litellm.rust_bridge._native")]
pub(crate) enum DenialView {
    Proxy {
        status: u16,
        message: String,
        error_type: String,
        param: Option<String>,
    },
}

impl From<PublicDenial> for DenialView {
    fn from(denial: PublicDenial) -> Self {
        match denial {
            PublicDenial::Proxy {
                status,
                message,
                error_type,
                param,
            } => Self::Proxy {
                status,
                message,
                error_type: <&'static str>::from(error_type).to_owned(),
                param,
            },
        }
    }
}

/// The auth core's calls, for the Python host.
#[pyclass(frozen, name = "RustAuth", module = "litellm.rust_bridge._native")]
pub(crate) struct AuthCall;

#[pymethods]
impl AuthCall {
    #[staticmethod]
    fn plan(request: RequestFacts, settings: &AuthSettings) -> Planned {
        match plan::plan(&request.request(), &settings.0) {
            Ok(plan) => Planned::Plan(plan.into()),
            Err(premium) => Planned::Denial(premium.public().into()),
        }
    }

    #[staticmethod]
    fn oauth2_introspection_applies(request: RequestFacts, settings: &AuthSettings) -> bool {
        plan::oauth2_introspection_applies(&request.request(), &settings.0)
    }
}
