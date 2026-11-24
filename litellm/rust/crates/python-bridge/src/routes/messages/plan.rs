use litellm_inference_messages::{MessagesCall, MessagesHostServices};

use super::transport::Selected;
use crate::cache::Selection;

/// A decoded Messages call and what its host serves: the configured cache, the client that
/// sends it, and whether safeguards and pre-request hooks run.
pub(super) struct MessagesPlan {
    pub(super) call: MessagesCall,
    pub(super) cache: Selection,
    pub(super) transport: Selected,
    pub(super) safeguards: bool,
    pub(super) pre_request_hooks: bool,
}

impl MessagesPlan {
    pub(super) fn host_services(&self) -> MessagesHostServices {
        MessagesHostServices {
            transport: self.transport.hosted,
            pre_request_hooks: self.pre_request_hooks,
            safeguards: self.safeguards,
            answer: true,
            cache_namespace: self.cache.namespace.clone(),
        }
    }
}
