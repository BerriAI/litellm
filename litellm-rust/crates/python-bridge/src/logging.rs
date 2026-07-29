use std::sync::Arc;

use litellm_ai_gateway::integrations::logging::console::hook as console_hook;
use litellm_core::logging::LogSink;

pub fn hook(enabled: bool) -> Option<Arc<dyn LogSink>> {
    console_hook(enabled)
}
