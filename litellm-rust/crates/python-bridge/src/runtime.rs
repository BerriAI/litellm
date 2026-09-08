use std::sync::{Arc, OnceLock};

use litellm_core::providers::auth::NativeAuthorizationServices;

pub fn authorization_services() -> &'static Arc<NativeAuthorizationServices> {
    static SERVICES: OnceLock<Arc<NativeAuthorizationServices>> = OnceLock::new();
    SERVICES.get_or_init(|| Arc::new(NativeAuthorizationServices::new()))
}
