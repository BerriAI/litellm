//! The process-wide sink, for threads no call scoped. It logs warnings and errors without request correlation.

use litellm_tracing::{Level, Logger, Metadata, Record, Sink};
use serde_json::Map;
use tokio::runtime::Handle;

use super::{
    background::{Background, Delivery},
    host::{self, Line},
    sink::litellm_target,
};

const UNCORRELATED: (&str, &str) = ("", "");
const CAPACITY: usize = 1024;

struct ToPython;

impl Delivery for ToPython {
    fn record(&self, record: &Record) {
        host::emit(&record.into(), UNCORRELATED);
    }

    fn lost(&self, count: u64) {
        host::emit(
            &Line {
                level: host::level(&Level::WARN),
                message: &format!(
                    "{count} native log lines were dropped because the Python logger fell behind"
                ),
                file: file!(),
                line: line!(),
                target: module_path!(),
                fields: &Map::new(),
            },
            UNCORRELATED,
        );
    }
}

/// The sink of a thread no call scoped, such as a detached task on the native runtime. Python rechecks the level, and
/// warnings are the floor so a disabled debug line is never queued.
struct Unscoped {
    background: Background<ToPython>,
}

impl Unscoped {
    fn accepts(level: &Level, target: &str) -> bool {
        *level <= Level::WARN && litellm_target(target)
    }
}

/// The native runtime, when this thread is one of a runtime's. Asking for it claims nothing new: the process already
/// entered native execution.
fn native_runtime() -> Option<Handle> {
    (Handle::try_current().is_ok() && litellm_host_python::runtime_started())
        .then(litellm_host_python::runtime_handle)
        .and_then(Result::ok)
}

impl Sink for Unscoped {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        Self::accepts(metadata.level(), metadata.target())
    }

    fn emit(&self, record: &Record) {
        match native_runtime() {
            Some(runtime) => self.background.push(&runtime, record.clone()),
            None => host::emit(&record.into(), UNCORRELATED),
        }
    }
}

pub(crate) fn unscoped() -> Logger {
    Logger::new(Unscoped {
        background: Background::new(ToPython, CAPACITY),
    })
}

#[cfg(test)]
mod tests {
    use litellm_tracing::Level;

    use super::Unscoped;

    #[rstest::rstest]
    #[case::error(Level::ERROR, "litellm_redis_store::cache_invalidation", true)]
    #[case::warning(Level::WARN, "_native::redis::events", true)]
    #[case::info(Level::INFO, "litellm_redis_store::cache_invalidation", false)]
    #[case::debug(Level::DEBUG, "litellm_redis_store::cache_invalidation", false)]
    #[case::dependency(Level::WARN, "hyper::proto::h1", false)]
    fn a_thread_no_call_scoped_logs_only_litellm_warnings_and_errors(
        #[case] level: Level,
        #[case] target: &str,
        #[case] accepted: bool,
    ) {
        assert_eq!(Unscoped::accepts(&level, target), accepted);
    }
}
