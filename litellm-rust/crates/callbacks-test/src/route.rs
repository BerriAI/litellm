use std::marker::PhantomData;

use litellm_callbacks::route::Route;

/// A route whose ops and chunks are strings and whose error type is the test's own, so a
/// test crate can implement its own traits (`Classified`, say) on it.
pub struct TestRoute<E>(PhantomData<E>);

impl<E: Clone + Send + Sync + 'static> Route for TestRoute<E> {
    type Response = String;
    type Error = E;
    type Op = String;
    type OpResult = String;
    type Chunk = String;

    /// A cached response replays as one chunk.
    fn replay(response: &String) -> Vec<String> {
        vec![response.clone()]
    }
}
