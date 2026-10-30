use super::{Expected, Kwargs, RedisError, bool_value, parsed, text_value};

#[derive(Clone, Debug, PartialEq)]
pub struct RedisTls {
    pub ca_certs: Option<String>,
    pub certfile: Option<String>,
    pub keyfile: Option<String>,
    pub verify_certificate: bool,
    pub check_hostname: bool,
}

pub(super) fn tls_options(merged: &Kwargs, gcp_ca_certs: Option<String>) -> Result<RedisTls, RedisError> {
    let verify_certificate = parsed(merged, "ssl_cert_reqs", Expected::CertRequirement, |text| match text {
        "required" | "optional" => Some(true),
        "none" => Some(false),
        _ => None,
    })?
    .unwrap_or(true);
    Ok(RedisTls {
        ca_certs: gcp_ca_certs.or_else(|| text_value(merged, "ssl_ca_certs")),
        certfile: text_value(merged, "ssl_certfile"),
        keyfile: text_value(merged, "ssl_keyfile"),
        verify_certificate,
        check_hostname: bool_value(merged, "ssl_check_hostname"),
    })
}
