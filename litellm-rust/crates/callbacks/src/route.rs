use std::marker::PhantomData;

/// One public call surface: what a completed call produces, how it fails, and the
/// route-specific operations only its host can perform (request projection, file reads,
/// token acquisition).
pub trait Route: Send + Sync + 'static {
    type Response: Send + 'static;
    type Error: Clone + Send + Sync + 'static;
    type Op: Send + 'static;
    type OpResult: Send + 'static;
}

/// The route a layer presents when it has ops of its own: the layer's ops beside the
/// inner route's, one error type, and the layer's completion value. A host answers the
/// outer half from whatever the layer needs (a routing table, a cache) and delegates the
/// inner half to the route's own host.
pub struct Layered<Outer, Inner>(PhantomData<fn() -> (Outer, Inner)>);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayeredOp<O, I> {
    Outer(O),
    Inner(I),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayeredResult<O, I> {
    Outer(O),
    Inner(I),
}

impl<Outer: Route, Inner: Route> Route for Layered<Outer, Inner> {
    type Response = Outer::Response;
    type Error = Inner::Error;
    type Op = LayeredOp<Outer::Op, Inner::Op>;
    type OpResult = LayeredResult<Outer::OpResult, Inner::OpResult>;
}
