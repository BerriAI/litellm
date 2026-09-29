# Tests

Nothing on the other side of the call: `tests/unit`. A proxy we start with an upstream we script:
`tests/integration`. Someone else's service with real credentials: `tests/e2e`. Two fit, split it

## What good looks like

Red when the claim in the name is broken. Prove it: mutate the behaviour, red; restore, green. Put the
mutation in the PR body

```python
def test_custom_price_is_reported_and_charged(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        response = gateway.request("POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "price control"}]})
        assert response.status_code == 200, response.text
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(20 * 0.001 + 20 * 0.002)
```

Rates in the test, expected computed by hand, one call, `response.text` in the assert

Assert the whole value. Iterating `expected_body.items()` (`test_responses_api_request_body.py`) cannot
see an extra key; that is the shape of `stream_options.include_usage` (#19777, #28553)

The linter catches no-assert, mock-echo and credential skips. It cannot see an assert
behind an `if` (a poll that ends in `pytest.fail` is fine), `except Exception` around the call
(`test_router.py`: `except Exception as e: print(f"FAILED TEST")`), or blanket `--reruns`

## Where it goes

What the assertion depends on goes in the test; everything else in conftest. A rate in a fixture three
directories up makes a failed assertion unreadable. Extend the file that already covers the behaviour

## Writing it so a human can read it

Name says what broke: `test_send_batched_with_valid_data` says nothing. Build, one call, assert, on one
screen. Helpers named for what they return, `_pii_prompt(marker, email)`, not `_setup()`. Context in the
assert message, not a comment
