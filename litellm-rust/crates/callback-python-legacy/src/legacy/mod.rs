//! Legacy Python callback contract: string-named callbacks, `CustomLogger` instances,
//! target interning and the logged-once markers that keep the old sync/async paths from
//! double-reporting. The whole module expires with the callback contract it mirrors.

pub mod dispatch;
pub mod execute;
pub mod markers;
pub mod targets;
pub mod vocabulary;
