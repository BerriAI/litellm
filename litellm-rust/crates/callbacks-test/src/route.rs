use std::marker::PhantomData;

use litellm_callbacks::route::Route;

/// A route whose ops are strings and whose error type is the test's own, so a test crate
/// can implement its own traits (a router's `AttemptError`, say) on it.
pub struct TestRoute<E>(PhantomData<E>);

impl<E: Clone + Send + Sync + 'static> Route for TestRoute<E> {
    type Response = String;
    type Error = E;
    type Op = String;
    type OpResult = String;
}
