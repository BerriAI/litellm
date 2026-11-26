use litellm_python_compat::lenient::Lenient;
use serde::{Deserialize, Deserializer};

/// A route list from config or key metadata, such as `public_routes` or
/// `allowed_passthrough_routes`. A value that is not a list of strings is `Malformed`, and each
/// reader decides what that grants: an allow list grants nothing, a deny list denies every route.
pub type RouteList = Lenient<Box<[String]>>;

/// A list whose key decides by being there, as `"allowed_routes" in general_settings` does. A key
/// written with no value is `Malformed`, never the same as a key left out.
pub(crate) fn where_the_key_is_written<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<RouteList>, D::Error> {
    RouteList::deserialize(deserializer).map(Some)
}

/// A list of client addresses, such as `allowed_ips`, read the way a [`RouteList`] is: a value
/// that is not a list of strings is `Malformed`, and an allowlist then admits no address.
pub type AddressList = Lenient<Box<[String]>>;
