# tests/unit

In-process. No network, clock or subprocess

## What good looks like

```python
def test_send_result_same_version_is_identity_passthrough():
    rpc = _rpc(V03_MESSAGE)
    out = normalize_jsonrpc_response(rpc, "0.3", method="message/send")
    assert out is rpc
```

`is`, because `==` passes on a copy. Many inputs: parametrize
(`test_an_incomplete_reservation_accrues_nothing`, fifteen cases, fifteen results)

No doubles on our own code

```python
with patch.object(streamer, "_group_by_date") as mock_group, patch.object(streamer, "_send_daily_batch") as mock_send:
    mock_group.return_value = {"2025-01-19": pl.DataFrame({"test": ["data1"]}), "2025-01-20": pl.DataFrame({"test": ["data2"]})}
    streamer.send_batched(pl.DataFrame({"test": ["data"]}), "replace_hourly")
    assert mock_send.call_count == 2
```

Green if `send_batched` drops every row. pydantic doubles in 12 of 203 files, fastapi 11 of 594;
`tests/test_litellm` 59 percent. Exception: the count is the behaviour
(`test_dual_cache_async_batch_get_cache_coalesces_concurrent_redis_reads`, fifty readers, `call_count == 1`)

## Where it goes

`tests/unit/<path>` mirrors `litellm/<path>`, so a changed file selects its tests by path, not a mapping
file. New unit tests go here

## Writing it so a human can read it

A class only when tests share an arrange
