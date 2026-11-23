//! What several inline test modules build the same way.

use litellm_gateway_auth::Authenticated;
use litellm_gateway_auth_core::Caller;

use crate::host::NativeRows;

/// The caller the master key authenticates.
pub(crate) fn master() -> Authenticated<NativeRows> {
    Authenticated::from(Caller::MasterKey)
}
