use futures_util::TryStreamExt;
use litellm_cost::receipt::Receipt;
use litellm_host::interceptors::{
    Cost, ExecutionFacts, Interceptors, RawResponse, RequestContext, WireRequest,
};
use litellm_inference_messages::{MessagesCallResponse, PricedCall, Pricing, PricingIdentity};
use litellm_inference_testing::no_secrets;
use rstest::rstest;

use super::*;

/// Prices a call at its prompt tokens, and keeps what it was asked
#[derive(Default)]
struct PromptPricing(Mutex<Vec<PricedCall>>);

impl Pricing for PromptPricing {
    fn price(&self, call: &PricedCall) -> Option<Receipt> {
        self.0.lock().unwrap().push(call.clone());
        Some(Receipt::of_total(call.usage.prompt_tokens as f64))
    }
}

#[derive(Default)]
struct Facts(Mutex<Vec<ExecutionFacts>>);

impl Interceptors<Error> for Facts {
    async fn result_ready(&self, facts: ExecutionFacts) -> Result<(), Error> {
        self.0.lock().unwrap().push(facts);
        Ok(())
    }

    async fn before_provider_request(
        &self,
        wire: WireRequest,
        _: RequestContext,
    ) -> Result<WireRequest, Error> {
        Ok(wire)
    }

    async fn after_provider_response(&self, _: RawResponse) -> Result<(), Error> {
        Ok(())
    }
}

const STREAM: &str = "event: message_start\ndata: {\"type\": \"message_start\", \"message\": {\"id\": \"msg_1\", \
\"type\": \"message\", \"role\": \"assistant\", \"model\": \"claude-sonnet-4-5\", \"content\": [], \"usage\": \
{\"input_tokens\": 30, \"cache_read_input_tokens\": 12, \"output_tokens\": 1}}}\n\nevent: content_block_start\ndata: \
{\"type\": \"content_block_start\", \"index\": 0, \"content_block\": {\"type\": \"text\", \"text\": \"\"}}\n\n\
event: content_block_delta\ndata: {\"type\": \"content_block_delta\", \"index\": 0, \"delta\": {\"type\": \
\"text_delta\", \"text\": \"hi\"}}\n\nevent: content_block_stop\ndata: {\"type\": \"content_block_stop\", \"index\": \
0}\n\nevent: message_delta\ndata: {\"type\": \"message_delta\", \"delta\": {\"stop_reason\": \"end_turn\"}, \"usage\": \
{\"output_tokens\": 9}}\n\nevent: message_stop\ndata: {\"type\": \"message_stop\"}\n\n";

fn priced(call: MessagesCall, base: String, identity: Option<PricingIdentity>) -> MessagesCall {
    MessagesCall {
        api_key: Some("sk-ant".into()),
        api_base: Some(base),
        shaping: MessagesShaping {
            settings: MessagesSettings {
                pricing: identity,
                ..MessagesSettings::default()
            },
            ..MessagesShaping::default()
        },
        ..call
    }
}

fn identity() -> PricingIdentity {
    PricingIdentity {
        deployment_id: Some("deployment-1".into()),
        ..PricingIdentity::default()
    }
}

#[rstest]
#[case::priced_with_an_identity(Some(identity()))]
#[case::left_to_the_host_without_one(None)]
#[tokio::test]
async fn a_message_is_priced_before_the_host_sees_the_result(
    call: MessagesCall,
    #[case] identity: Option<PricingIdentity>,
) {
    let upstream = upstream([message_response()]).await;
    let pricing = Arc::new(PromptPricing::default());
    let facts = Facts::default();

    messages_route(no_secrets())
        .with_pricing(pricing.clone())
        .execute(priced(call, upstream.uri(), identity.clone()), &facts, None)
        .await
        .unwrap();

    let [ready]: [ExecutionFacts; 1] = facts.0.into_inner().unwrap().try_into().unwrap();
    let asked = pricing.0.lock().unwrap().clone();
    match identity {
        None => {
            assert_eq!(ready.cost, Cost::Deferred);
            assert!(asked.is_empty());
        }
        Some(identity) => {
            let Cost::Priced(priced) = ready.cost else {
                panic!("a message the route read is priced, got {:?}", ready.cost);
            };
            assert_eq!(priced.receipt.total, 1.0);
            let [call]: [PricedCall; 1] = asked.try_into().unwrap();
            assert_eq!(call.identity, identity);
            assert_eq!(
                (call.provider.as_str(), call.request_model.as_str()),
                ("anthropic", MODEL)
            );
            assert_eq!(call.usage.output.text, 2);
        }
    }
}

#[rstest]
#[tokio::test]
async fn a_stream_meter_settles_once_the_caller_read_the_stream(call: MessagesCall) {
    let upstream =
        upstream([ResponseTemplate::new(200).set_body_raw(STREAM, "text/event-stream")]).await;
    let pricing = Arc::new(PromptPricing::default());
    let facts = Facts::default();

    let response = messages_route(no_secrets())
        .with_pricing(pricing.clone())
        .execute(
            with_fields(
                priced(call, upstream.uri(), Some(identity())),
                json!({"stream": true}),
            ),
            &facts,
            None,
        )
        .await
        .unwrap();

    let [ready]: [ExecutionFacts; 1] = facts.0.into_inner().unwrap().try_into().unwrap();
    let Cost::Metered(meter) = ready.cost else {
        panic!("a stream is metered, got {:?}", ready.cost);
    };
    assert_eq!(meter.settled(), None);
    let MessagesCallResponse::Stream { chunks, .. } = response else {
        panic!("a streamed call returns a stream");
    };
    let delivered: Vec<bytes::Bytes> = chunks.try_collect().await.unwrap();
    assert_eq!(delivered.concat(), STREAM.as_bytes());
    assert_eq!(meter.settled().unwrap().receipt.total, 42.0);
    let [call]: [PricedCall; 1] = pricing.0.lock().unwrap().clone().try_into().unwrap();
    assert_eq!(call.usage.output.text, 9);
    assert_eq!(call.response_model.as_deref(), Some(MODEL));
}
