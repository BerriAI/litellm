#![allow(clippy::disallowed_types)]

// mirrors: unit/a2a_protocol/test_cost_calculator.py::test_asend_message_uses_input_output_cost_per_token
// mirrors: test_litellm/proxy/_experimental/mcp_server/test_mcp_cost_calculator.py::TestMCPCostCalculator::test_calculate_mcp_tool_call_cost_with_tool_specific_cost

use litellm_cost::a2a_cost::calculate_a2a_cost;
use litellm_cost::mcp_cost::calculate_mcp_tool_call_cost;
use rstest::rstest;
use serde_json::json;

#[rstest]
fn a2a_cost_uses_response_then_query_then_token_rates() {
    let details = json!({
        "response_cost": 0.7,
        "litellm_params": {"cost_per_query": 0.4, "input_cost_per_token": 0.01},
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}
    });
    assert_eq!(calculate_a2a_cost(Some(&details)), Ok(0.7));

    let query = json!({
        "litellm_params": {"cost_per_query": 0.4, "input_cost_per_token": 0.01},
        "usage": {"prompt_tokens": 10}
    });
    assert_eq!(calculate_a2a_cost(Some(&query)), Ok(0.4));

    let tokens = json!({
        "litellm_params": {"input_cost_per_token": 0.01, "output_cost_per_token": 0.02},
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}
    });
    assert_eq!(calculate_a2a_cost(Some(&tokens)), Ok(0.2));
}

#[rstest]
fn a2a_cost_returns_zero_without_usage_or_configuration() {
    let configured_without_usage = json!({"litellm_params": {"input_cost_per_token": 0.01}});
    assert_eq!(calculate_a2a_cost(Some(&configured_without_usage)), Ok(0.0));
    assert_eq!(calculate_a2a_cost(Some(&json!({}))), Ok(0.0));
    assert_eq!(calculate_a2a_cost(None), Ok(0.0));
    assert_eq!(
        calculate_a2a_cost(Some(&json!({
            "litellm_params": {"input_cost_per_token": null, "output_cost_per_token": 0.02},
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }))),
        Ok(0.1)
    );
}

#[rstest]
fn mcp_cost_uses_response_then_tool_then_default_price() {
    let metadata = json!({
        "mcp_tool_call_metadata": {
            "name": "search",
            "mcp_server_cost_info": {
                "default_cost_per_query": 0.01,
                "tool_name_to_cost_per_query": {"search": 0.05}
            }
        }
    });
    assert_eq!(calculate_mcp_tool_call_cost(Some(&metadata)), 0.05);
    assert_eq!(
        calculate_mcp_tool_call_cost(Some(
            &json!({"response_cost": 0.2, "mcp_tool_call_metadata": metadata["mcp_tool_call_metadata"]})
        )),
        0.2
    );
    assert_eq!(
        calculate_mcp_tool_call_cost(Some(
            &json!({"mcp_tool_call_metadata": {"name": "other", "mcp_server_cost_info": metadata["mcp_tool_call_metadata"]["mcp_server_cost_info"]}})
        )),
        0.01
    );
    assert_eq!(calculate_mcp_tool_call_cost(None), 0.0);
}
