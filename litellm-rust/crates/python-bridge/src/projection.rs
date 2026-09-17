/// The handoff out of a route's projection: the native input core takes ownership of,
/// beside the Python state the host keeps for the rest of the call.
pub(crate) struct Projection<Native, Retained> {
    pub(crate) native: Native,
    pub(crate) retained: Retained,
}
