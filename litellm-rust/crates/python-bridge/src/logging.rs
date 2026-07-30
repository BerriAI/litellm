use std::sync::Arc;

use litellm_core::logging::{LogSink, console::hook as console_hook};

pub fn hook(enabled: bool) -> Option<Arc<dyn LogSink>> {
    console_hook(enabled)
}
