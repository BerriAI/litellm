# Headroom fresh tool result compression, live evidence

Baseline f43f2d8c1598e640daa7345d1d46925fe61b9783, head 572e5f603f06e3cfc88a4155324055896a0ec86c (product files byte-identical to 93a5f048762dfd991b4de7053607f584fdd0fe53, test-only formatting diff). headroom-ai[proxy] 0.41.0, real Anthropic `claude-haiku-5-5` and OpenAI `gpt-6.1-sol`. Each case ran once per path. Source freshness (SHA, clean porcelain, `litellm.__file__`, PIDs, start times, key presence booleans) is in `run/pids`. The first base section there (PIDs 222987, 225491) was started without provider keys and its captures were discarded. Valid base PIDs are 231799/231800, head 231130/231131

Canary for all K1 style cases is `/srv/app/pkg6/mod_263.lock`, 77777 bytes, and every answer below returned it

| Case | Endpoint | Base status, prompt tokens | Head status, prompt tokens | Head behaviour |
|---|---|---|---|---|
| K1 | Chat | 200, 30742 | 200, 18777 | tool row 53841 -> 27122 chars, 9983 tokens saved |
| K1_resp | Responses | 200, 22918 | 200, 12935 | string `function_call_output.output` stays a string, 53841 -> 27122 |
| R2 single input_text | Responses | 200, 22918 | 200, 12935 | one block list kept, text 53841 -> 27122 |
| R3 multipart | Responses | 200, 22926 | 200, 22926 | unchanged, not sent to Headroom |
| L1 exact 41765 char fixture | Chat | 200, ANSWER: REQ00417 | 200, ANSWER: REQ00417 | not sent to Headroom, byte-identical |
| M1 multipart | Chat | 200, 30758 | 200, 30758 | unchanged |
| N1 cache_control on tool_result | Messages | 200, 30748 | 200, 30748 | protected on both |
| N2 multipart tool_result | Messages | 200, 30764 | 200, 30764 | unchanged |
| N3 is_error true | Messages | 200, 30787 | 200, 18419 | is_error true and tool_use_id kept, text 62849 -> 27330 |
| N4 is_error false, text block with cache_control | Messages | 200, 30748 | 200, 18783 | is_error false and inner cache_control kept, text 53841 -> 27122 |
| O1 older user log | Chat | 200, 31055 | 200, 19090 | older user row 42268 -> about 577 on both, latest instruction unchanged, head also compresses tool |
| C1 cached prefix | Chat | 200, 61481 | 200, 49516 | cached older row byte-identical, only fresh tool compressed |
| B1 x-headroom-bypass | Chat | 200, 30742 | 200, 30742 | no Headroom call |
| S1 stream | Chat | 200 SSE [DONE] | 200 SSE [DONE] | upstream tool row 62741 -> 27625 |
| S2 stream | Responses | 200, 22918 | 200, 12935 | response.completed |
| MT1-3 | Responses | 22895/35829/48762 | 12913/25849/38784 | prefix rows byte-stable across turns |
| 5 Headroom faults | Chat | 200 (no Headroom call) | fail_closed 502, fail_open 200 | |

O1 first returned 400 on both paths for a harness reason (gpt-6.1-sol rejects function tools with reasoning_effort on Chat), so it was rerun on claude-haiku-5-5. The 400 captures are under `run/o1-400`

Disclosed behaviour. When the Anthropic path writes compressed content back, string user content in other rows becomes a single text block with identical text (N3/N4 row 0). The inner text block `cache_control` policy follows the pre-existing `_tool_result_part` handling, a `cache_control` on the tool_result block itself protects the row. Headroom 0.41.0 smart_crusher returns a JSON-quoted table, the same on the base older-tool path. ccr_retrieval was not exercised live and the recorder buffers SSE, so streaming proves delivery, not incremental timing

Layout. `run/res/<path>.<case>.client_req.json` holds the exact client bodies and `client_resp.json` the responses. `run/tap/<path>.<case>.{hr,anthropic,openai}.json` holds the exact Headroom and provider bodies (no headers recorded). `REPRO.md` has the setup and curl commands, `fixtures.py` generates the payloads
