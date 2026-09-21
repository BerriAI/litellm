use std::marker::PhantomData;

use litellm_host::{
    machine::{HostChannel, RouteMachine},
    route::Route,
};
use litellm_llms::base_llm::base_model_iterator::{
    ApiDialect, DeliveryMode, ServerStreamingContract, StreamError, StreamOutcome, StreamPolicy,
};

pub trait StreamingCall: Send + Sync + 'static {
    type Contract: ServerStreamingContract<Event: Send + 'static>;
    type Request: Send + 'static;
    type Response: Send + 'static;
}

pub struct StreamingRoute<C: StreamingCall>(PhantomData<C>);

pub struct StreamHead {
    pub dialect: ApiDialect,
    pub delivery: DeliveryMode,
}

pub enum StreamOp<C: StreamingCall> {
    ProjectRequest,
    LoadCachedResponse {
        key: String,
    },
    PostStreamEvent {
        event: <C::Contract as ServerStreamingContract>::Event,
    },
    StoreResponse {
        key: String,
        response: C::Response,
    },
    AccountOutcome {
        outcome: StreamOutcome<C::Response>,
    },
    PostStreamingHooks,
}

pub enum StreamOpResult<C: StreamingCall> {
    Request {
        request: C::Request,
        policy: StreamPolicy,
    },
    CachedResponse(Option<C::Response>),
    StoredResponse(C::Response),
    AccountedOutcome(StreamOutcome<C::Response>),
    Event(<C::Contract as ServerStreamingContract>::Event),
    Acknowledged,
}

impl<C: StreamingCall> Route for StreamingRoute<C> {
    type Response = StreamOutcome<C::Response>;
    type Error = StreamError;
    type Op = StreamOp<C>;
    type OpResult = StreamOpResult<C>;
    type Chunk = <C::Contract as ServerStreamingContract>::Event;
    type StreamHead = StreamHead;
}

pub fn streaming_machine<C: StreamingCall>() -> RouteMachine<StreamingRoute<C>> {
    todo!(
        "Project request, preserve acceptance gates, select native/bridged/buffered source, then drive host delivery and terminal accounting"
    )
}

pub async fn deliver_stream<C, S>(
    _host: &HostChannel<StreamingRoute<C>>,
    _source: S,
    _head: StreamHead,
    _policy: StreamPolicy,
) -> Result<StreamOutcome<C::Response>, StreamError>
where
    C: StreamingCall,
    S: futures_util::Stream<
            Item = Result<<C::Contract as ServerStreamingContract>::Event, StreamError>,
        > + Send,
{
    todo!(
        "Honor host demand, bound buffers, validate the terminator, retain late/failed usage, and emit cache/hooks/accounting once; detachment is cancellation"
    )
}
