# Headroom tool-results evidence, cc331b30

Candidate `cc331b300c0ecc370640549afac9b8561f29b733` was tested against baseline `f43f2d8c1598e640daa7345d1d46925fe61b9783`

The `<scratch>/run-cc33/res` captures contain 10 final cc331b30 cases: K1_chat, K1_resp, N3, N4, N5, N6, C2, C3, N7 and N7c. They also contain two fresh base cases, N7 and N7c. Base for all other cases is the retained run812 baseline. The two copied K1 request bodies have SHA-256 values matching both run812 base requests and retained 572e5f60 final request copies

## Results

Token columns are provider input, cache_read and cache_create. Row lengths are raw tool-output text characters and JSON-serialized provider-row bytes, kept separate. The changed row is the fresh tool result unless noted

| Case | Base source | HTTP base/cc33 | Base input | Base read | Base create | cc33 input | cc33 read | cc33 create | Headroom called base/cc33 | Saved base/cc33 | Changed row | Raw text chars before -> after | Serialized row bytes before -> after | cc33 answer excerpt |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| K1_chat | run812 | 200/200 | 30,742 | 0 | 0 | 18,777 | 0 | 0 | N/Y | 0/9,983 | 2 | 53,841 -> 27,122 | 62,741 -> 27,625 | `/srv/app/pkg6/mod_263.lock`, 77,777 bytes |
| K1_resp | run812 | 200/200 | 22,918 | 0 | 22,915 | 12,935 | 12,932 | 0 | N/Y | 0/10,044 | 2 | 53,841 -> 27,122 | 62,714 -> 27,598 | `/srv/app/pkg6/mod_263.lock`, 77777 bytes |
| N3 | run812 | 200/200 | 30,787 | 0 | 0 | 18,419 | 0 | 0 | N/Y | 0/9,985 | 2 | 53,929 -> 26,808 | 62,849 -> 27,330 | `/srv/app/pkg6/mod_263.lock`, 77,777 bytes |
| N4 | run812 | 200/200 | 2 | 30,746 | 0 | 2 | 0 | 18,781 | N/Y | 0/9,983 | 2 | 53,841 -> 27,122 | 62,829 -> 27,713 | `/srv/app/pkg6/mod_263.lock`, 77,777 bytes |
| N5 | run812 | 200/200 | 31,796 | 0 | 0 | 31,796 | 0 | 0 | N/N | 0/0 | none | 53,921 -> 53,921 | 63,214 -> 63,214 | `/srv/app/pkg6/mod_263.lock`, 77,777 bytes |
| N6 | run812 | 200/200 | 62,562 | 0 | 0 | 50,597 | 0 | 0 | N/Y | 0/9,982 | 2 | 107,758 -> 81,039 | 125,923 -> 90,807 | No text answer, `stop_reason=max_tokens`, 300 thinking tokens |
| C2 | run812 | 200/200 | 30,748 | 0 | 0 | 18,783 | 0 | 0 | N/Y | 0/9,983 | 2 | 53,841 -> 27,122 | 62,741 -> 27,625 | `/srv/app/pkg6/mod_263.lock`, 77,777 bytes |
| C3 | run812 | 200/200 | 30,748 | 0 | 0 | 18,783 | 0 | 0 | N/Y | 0/9,983 | 2 | 53,841 -> 27,122 | 62,742 -> 27,626 | `/srv/app/pkg6/mod_263.lock`, 77,777 bytes |
| N7 | fresh base in run-cc33 | 200/200 | 30,748 | 0 | 0 | 18,783 | 0 | 0 | N/Y | 0/9,983 | 2 | 53,841 -> 27,122 | 62,742 -> 27,626 | `/srv/app/pkg6/mod_263.lock`, 77,777 bytes |
| N7c | fresh base in run-cc33 | 200/200 | 30,742 | 0 | 0 | 18,777 | 0 | 0 | N/Y | 0/9,983 | 2 | 53,841 -> 27,122 | 62,741 -> 27,625 | `/srv/app/pkg6/mod_263.lock`, 77,777 bytes |

Full per-case fields and answer excerpts are in `analysis.tsv`. The byte-level prefix comparison is in `byte-check.tsv`

## Control checks

N5 made no Headroom call and saved zero tokens. The client and upstream `tool_result` are byte-identical at 63,151 bytes. No retrieval or CCR text was injected upstream

N6 sent only `toolu_n6_0002` to Headroom. The mixed `toolu_n5_0001` result was byte-identical from client to upstream. The safe result shrank from 53,837 to 27,118 raw text characters and Headroom saved 9,982 tokens. The 200 response contained no text answer because it used all 300 output tokens for thinking and ended at `max_tokens`

C2 compressed the fresh result and preserved its call ID `call_c2_0001` and name `list_dir`. The base and cc33 provider prefixes match byte-for-byte: row 0 is 170 bytes on both, row 1 is 210 bytes on both. The assistant row's text block has `cache_control={"type":"ephemeral"}`; the `tool_use` block itself has no `cache_control`

C3 compressed the fresh result and preserved tool ID `toolu_c3_0001` and name `list_dir`. The cached assistant row is byte-identical at 211 bytes and the `tool_use` block retains `cache_control={"type":"ephemeral"}`. Row 0's user text is unchanged, but its representation changes from a string to a text block, 142 to 170 serialized bytes, so the whole prefix is not byte-identical

N3, N4 and N7 have the same row 0 representation change, with unchanged rendered user text and 142 to 170 serialized bytes. Their preceding assistant rows are byte-identical at 171 bytes, with `tool_use` IDs `toolu_n3_0001`, `toolu_n3_0001` and `toolu_n7_0001`, respectively, and name `list_dir`. Those tool-use blocks have no `cache_control`. N3 has `is_error=true` and no `cache_control` on its result. N4 has `is_error=false` and `cache_control` on its result text block

N7 used raw paired ID `toolu.n7:0001`, and N7c used `call.n7:0001`. Base accepted both requests with HTTP 200. On both base and cc33, the upstream IDs were normalized to `toolu_n7_0001` and `call_n7_0001`; each result ID matched its preceding tool-use ID, and both tool names remained `list_dir`. Both paths returned the canary file `/srv/app/pkg6/mod_263.lock`, 77,777 bytes

## Handler assertion and comparisons

`test_native_multipart_tool_result_is_excluded_before_compression` passed on cc331b30 with `8 passed in 2.49s`. With only the cc33 test file overlaid on 5d113958, it produced the expected `4 failed, 4 passed in 1.27s`. A failure showed `{'tool_use_id': 'call.unsafe:1'} != {'tool_use_id': 'call_unsafe_1'}`. Outputs remain at `<scratch>/run-cc33/logs/test-cc331b30.log` and `<scratch>/run-cc33/logs/test-5d113958.log`

For the eight cases shared with 5d113958, status, provider input/cache usage, Headroom savings and provider request rows match. Answer wording varies. The 5d captures and its analysis are preserved in `5d113958/`. K1_chat, K1_resp, N3 and N4 also match the comparable 572e5f60 evidence. N5, N6, C2, C3, N7 and N7c have no comparable 572e5f60 captures

The original mixed text plus `search_result` N5/N6 payloads were rejected with HTTP 400 on base and basefo because a tool result containing `search_result` blocks must contain only that block type. Those captures are in `rejected-mixed-text-search_result-400/` and are handler-only negative evidence, not passing control results

## Reproduction

These commands replay paid requests against final port 4034. Set `HR_MASTER_KEY` in the local shell first. `<scratch>` is the local verification directory

```bash
curl -sS -H "Authorization: Bearer $HR_MASTER_KEY" -H "Content-Type: application/json" --data-binary @<scratch>/run-cc33/res/final.K1_chat.client_req.json http://127.0.0.1:4034/v1/chat/completions
curl -sS -H "Authorization: Bearer $HR_MASTER_KEY" -H "Content-Type: application/json" --data-binary @<scratch>/run-cc33/res/final.N4_msgs_iserror_false_block.client_req.json http://127.0.0.1:4034/v1/messages
curl -sS -H "Authorization: Bearer $HR_MASTER_KEY" -H "Content-Type: application/json" --data-binary @<scratch>/run-cc33/res/final.K1_resp.client_req.json http://127.0.0.1:4034/v1/responses
```

The K1 request files were copied from run812 base and SHA-256 verified against the retained 572e5f60 final copies. K1_chat SHA-256 is `b151e5bca25328e705e8079b09fd92cb717f3496abfcf246817708449d2646e2`; K1_resp is `6cb9ea68a4a77bb8c90d71313e0802cb70de5030cf379dc3efc24de921ac9b0f`. Copy labels are beside each request in `res/`

## Runtime and scope

The `[final_cc331b30]` provenance and health output, together with the run812 base provenance, are in `provenance/run812-pids.txt`. Headroom reported healthy and ready at version 0.41.0. Final and finalfo reported healthy; listener PIDs were 274586 on 4034 and 274587 on 4035. The recorder remained listening on 8791

Paid cases took 46.39 seconds. Handler tests took 48.23 seconds combined. Setup and analysis time are excluded. No product files or tests were edited, and no product branch or pull request was touched
