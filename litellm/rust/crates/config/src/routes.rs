use litellm_python_compat::lenient::Lenient;

/// A route list from config or key metadata, such as `public_routes` or
/// `allowed_passthrough_routes`. A value that is not a list of strings is `Malformed`, and each
/// reader decides what that grants: an allow list grants nothing, a deny list denies every route.
pub type RouteList = Lenient<Box<[String]>>;
