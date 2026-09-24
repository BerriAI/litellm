// mirrors: test_litellm/llms/bedrock/test_bedrock_common_utils.py::test_get_bedrock_base_model
// expected values recorded from litellm/llms/bedrock/common_utils.py

use litellm_cost::bedrock_common_utils::{get_bedrock_base_model, strip_bedrock_routing_prefix};
use rstest::rstest;

#[rstest]
#[case(
    "bedrock/converse/us.anthropic.claude-x-v1:0",
    "us.anthropic.claude-x-v1:0"
)]
#[case("converse/bedrock/m", "bedrock/m")]
#[case("openai/bedrock/m", "bedrock/m")]
#[case("nova-2/arn:aws:x", "arn:aws:x")]
#[case("bedrock/nova/arn:aws:y", "arn:aws:y")]
#[case("invoke/mantle/nova/m", "m")]
#[case("m", "m")]
fn strip_bedrock_routing_prefix_strips_each_prefix_once_in_order(
    #[case] model: &str,
    #[case] expected: &str,
) {
    assert_eq!(strip_bedrock_routing_prefix(model), expected);
}

#[rstest]
#[case("us.meta.llama3-v1:0", "meta.llama3-v1:0")]
#[case("bedrock/converse/model", "model")]
#[case("bedrock/nova-2/arn:aws:bedrock:x", "amazon.nova-2-custom")]
#[case("bedrock/nova/arn:aws:x", "amazon.nova-custom")]
#[case("converse/nova-2/x", "amazon.nova-2-custom")]
#[case("invoke/nova/x", "x")]
#[case("anthropic.claude-v2:0:51k", "anthropic.claude-v2:0")]
#[case("us.anthropic.claude-opus-v1[1m]", "anthropic.claude-opus-v1")]
#[case(
    "arn:aws:bedrock:us-east-1:1:foundation-model/anthropic.claude-v2",
    "anthropic.claude-v2"
)]
#[case("us-east-1/amazon.titan", "amazon.titan")]
#[case("ap-south-9/amazon.titan", "ap-south-9/amazon.titan")]
#[case("global.x", "x")]
#[case("mars.x", "mars.x")]
fn get_bedrock_base_model_matches_python(#[case] model: &str, #[case] expected: &str) {
    assert_eq!(get_bedrock_base_model(model), expected);
}
