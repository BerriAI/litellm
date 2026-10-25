use crate::Spelled;

/// Read by the SSL context builder, `get_ssl_configuration`.
#[macro_rules_attribute::apply(crate::config_section)]
pub struct TlsSettings {
    pub ssl_verify: Option<Spelled<bool>>,
    pub ssl_security_level: Option<String>,
    pub ssl_ecdh_curve: Option<String>,
}

/// Read by the sync and async HTTP clients, which send it as the client certificate.
#[macro_rules_attribute::apply(crate::config_section)]
pub struct ClientCertificateSettings {
    pub ssl_certificate: Option<String>,
}

/// Read by the httpx and aiohttp transports when they bind the local socket.
#[macro_rules_attribute::apply(crate::config_section)]
pub struct SocketSettings {
    pub force_ipv4: Option<bool>,
}

/// Read by the httpx client.
#[macro_rules_attribute::apply(crate::config_section)]
pub struct Http2Settings {
    pub http2: Option<bool>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct AiohttpSettings {
    pub aiohttp_trust_env: Option<bool>,
    pub disable_aiohttp_trust_env: Option<bool>,
    pub disable_aiohttp_transport: Option<bool>,
}
