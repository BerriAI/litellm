mod error;
mod groups;
mod managed;
pub mod passthrough;
pub(crate) mod table;
mod template;

pub use error::MissingRouteGroup;
pub use groups::{RouteAllowlist, RouteGroups, matches_wildcard};
pub use managed::ManagedRoute;
pub use table::RouteGroup;
pub(crate) use template::RouteSet;
