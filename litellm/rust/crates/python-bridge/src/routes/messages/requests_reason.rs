use std::net::IpAddr;

use litellm_auth_azure::{CertificateRejection, RequestsReason};
use pyo3::prelude::*;

use super::workload_identity::LOOKUP_FAILURES;

const MODULE: &str = "litellm.rust_bridge.host.routes.messages";

/// The text `requests` gives for `reason`, a failed token request to `host:port`, built by the
/// host from the exceptions urllib3, `http.client` and `ssl` raise for it. `None` for a proxy
/// name lookup failure glibc words otherwise.
pub(super) fn requests_reason_text(
    py: Python<'_>,
    host: &str,
    port: u16,
    reason: &RequestsReason,
) -> PyResult<Option<String>> {
    let module = py.import(MODULE)?;
    let text = match reason {
        RequestsReason::ProxyUnreachable { host, port, errno } => module
            .getattr("requests_proxy_unreachable")?
            .call1((host, port, errno))?,
        RequestsReason::ProxyUnresolved { host, port, lookup } => {
            let Some((code, lookup)) = LOOKUP_FAILURES
                .iter()
                .find(|(_, failure)| failure == lookup)
            else {
                return Ok(None);
            };
            module
                .getattr("requests_proxy_unresolved")?
                .call1((host, port, code, lookup))?
        }
        RequestsReason::ProxyAuthenticationRequired => module
            .getattr("requests_proxy_authentication_required")?
            .call0()?,
        RequestsReason::ProxyDisconnected => {
            module.getattr("requests_proxy_disconnected")?.call0()?
        }
        RequestsReason::ConnectionAborted { errno } => module
            .getattr("requests_connection_aborted")?
            .call1((errno,))?,
        RequestsReason::TlsEof => module.getattr("requests_tls_eof")?.call0()?,
        RequestsReason::CertificateRejected(rejection) => module
            .getattr("requests_certificate_rejected")?
            .call1((verify_reason(host, rejection),))?,
        RequestsReason::ReadTimeout => module
            .getattr("requests_read_timeout")?
            .call1((host, port))?,
    };
    text.extract().map(Some)
}

/// OpenSSL's verify reason, and CPython's for the name `ssl` checks itself.
fn verify_reason(host: &str, rejection: &CertificateRejection) -> String {
    match rejection {
        CertificateRejection::NameMismatch => match host.parse::<IpAddr>() {
            Ok(_) => format!("IP address mismatch, certificate is not valid for '{host}'."),
            Err(_) => format!("Hostname mismatch, certificate is not valid for '{host}'."),
        },
        CertificateRejection::Expired => "certificate has expired".to_owned(),
        CertificateRejection::NotYetValid => "certificate is not yet valid".to_owned(),
    }
}
