> Hand-off copy of `~/repos/notes/rust-migration/router-poc-behavior-spec.md` (Python hot-path rules the Rust backend must reproduce). Line cites use `PR:` for `litellm/router_backends/python_router.py`. Delete this file before opening a PR

# Router POC behavior spec (async hot path)

Written 2026-10-06 from branch `litellm_rust_router_poc` of `wt-one`. Companion to [[router-poc]], [[router-structure]] and [[router-rust-mapping]]. Scope is what a Rust backend must reproduce to make the same decisions as `PythonRouter` on the async path (`acompletion`, `aresponses`, `aanthropic_messages`, `factory_function` generic path) for the simplest config: no aliases, no wildcards, no routing groups, no tags, no team models, no pre-call checks, `simple-shuffle`. Branches that only fire for richer configs are named once so the porter knows where the config-check must decline

Line numbers are for this branch and will drift. Every rule cites the line that implements it

## File abbreviations

| Abbrev | Path |
|---|---|
| PR | `litellm/router_backends/python_router.py` |
| FEH | `litellm/router_utils/fallback_event_handlers.py` |
| CH | `litellm/router_utils/cooldown_handlers.py` |
| CC | `litellm/router_utils/cooldown_cache.py` |
| TDM | `litellm/router_utils/router_callbacks/track_deployment_metrics.py` |
| HDR | `litellm/router_utils/add_retry_fallback_headers.py` |
| RP | `litellm/router_utils/get_retry_from_policy.py` |
| SS | `litellm/router_strategy/simple_shuffle.py` |
| HE | `litellm/router_utils/handle_error.py` |
| CU | `litellm/router_utils/common_utils.py` |
| BU | `litellm/router_utils/batch_utils.py` |
| CIU | `litellm/router_utils/client_initalization_utils.py` |
| U | `litellm/utils.py` |
| LL | `litellm/litellm_core_utils/litellm_logging.py` |
| IMC | `litellm/caching/in_memory_cache.py` |
| DC | `litellm/caching/dual_cache.py` |
| SH | `litellm/litellm_core_utils/streaming_handler.py` |
| RSI | `litellm/responses/streaming_iterator.py` |
| C | `litellm/constants.py` |
| T | `litellm/types/router.py` |

## 0. Call stack at a glance

```
acompletion                         PR:2882
  async_function_with_fallbacks     PR:7852   (fallback layer, one per hop)
    async_function_with_retries     PR:7987   (retry layer, one per hop)
      make_call                     PR:8185   (per attempt: call, inline usage, response headers)
        _acompletion                PR:3775   (per attempt: select deployment, call litellm.acompletion)
          async_get_available_deployment PR:13688 -> async_get_healthy_deployments PR:13401 -> simple_shuffle SS:43
    on exception: async_function_with_fallbacks_common_utils PR:7582 -> run_async_fallback FEH:658
                  -> recursive async_function_with_fallbacks for each fallback group

factory_function(call_type)         PR:6906  -> async_wrapper PR:7146
  aresponses          -> _aresponses_with_streaming_fallbacks PR:5520 -> _ageneric_api_call_with_fallbacks(attempt_function=..._responses_attempt)
  anthropic_messages  -> _dispatch_generic_call_type PR:6094 -> _aanthropic_messages_with_streaming_fallbacks PR:6049
  generic (e.g. aimage_edit, agenerate_content, aocr, avideo_*) -> _dispatch_generic_call_type -> _ageneric_api_call_with_fallbacks PR:5375
  _ageneric_api_call_with_fallbacks -> async_function_with_fallbacks (same layers as above)
    per attempt original_function = _ageneric_api_call_with_fallbacks_helper PR:5437 (or the *_attempt wrapper)
```

Bound in `__init__`: `self.aresponses = factory_function(litellm.aresponses, "aresponses")` PR:2027, `self.aanthropic_messages = factory_function(litellm.anthropic_messages, "anthropic_messages")` PR:2021

## 1. Entry and dispatch

### 1.1 `acompletion` (PR:2882)

| Step | Rule | Cite |
|---|---|---|
| 1 | Writes `kwargs["model"]=model`, `kwargs["messages"]`, `kwargs["stream"]=stream` (default False), `kwargs["original_function"]=self._acompletion` | PR:2890-2893 |
| 2 | `_update_kwargs_before_fallbacks(model, kwargs)` with bucket `"metadata"`: `setdefault("litellm_trace_id", uuid4())`; `kwargs.setdefault("metadata", {}).update({"model_group": model, "model_group_alias": model if alias-map hit else None})`. Does NOT set num_retries | PR:3940-3962 |
| 3 | `priority` kwarg or `self.default_priority` int -> `schedule_acompletion` (scheduler, out of scope). Prompt-management model -> `_prompt_management_factory` (out of scope) | PR:2896-2907 |
| 4 | Else `await async_function_with_fallbacks(**kwargs)` | PR:2909 |
| 5 | Success: fire-and-forget service success hook (`call_type="acompletion"`). Failure: fire-and-forget `send_llm_exception_alert`, re-raise unchanged | PR:2910-2934 |

### 1.2 `_ageneric_api_call_with_fallbacks` (PR:5375)

| Rule | Cite |
|---|---|
| Writes `kwargs["model"]`, `kwargs["original_generic_function"]=original_function` (the litellm handler, e.g. `litellm.aresponses`), `kwargs["original_function"]=attempt_function or self._ageneric_api_call_with_fallbacks_helper` | PR:5385-5387 |
| When `attempt_function` given (aresponses, anthropic_messages): snapshot per-request controls (`fallbacks`, `context_window_fallbacks`, `content_policy_fallbacks`, `num_retries`, `model_group_retry_policy` if present) into `kwargs["_mid_stream_fallback_controls"]` | PR:5388-5390, FEH:307-336 |
| `_update_kwargs_before_fallbacks(..., metadata_variable_name="litellm_metadata")` so the bucket is `litellm_metadata` | PR:5391 |
| Then `async_function_with_fallbacks(**kwargs)`; on error, alert task and re-raise | PR:5395-5406 |

`factory_function` quirks: for `aresponses` and `anthropic_messages` the wrapper's own `custom_llm_provider` and `client` params are swallowed and NOT forwarded (PR:7146-7149, 7178-7182, 7193-7224). `afile_*` and realtime forward them. `allm_passthrough_route` sets `passthrough_on_no_deployment=True` (PR:7244-7250)

### 1.3 Metadata bucket name

| Function | Rule | Cite |
|---|---|---|
| `_get_router_metadata_variable_name(function_name)` | `"litellm_metadata"` if function_name contains any substring of `{"batch","generic_api_call","_acreate_batch","file","_ageneric_api_call_with_fallbacks"}`, else `"metadata"`. Substring match, so any name containing `file` qualifies | BU:153-173 |
| `get_metadata_variable_name_from_kwargs(kwargs)` | `"litellm_metadata"` if that key is present in kwargs, else `"metadata"` (key presence, not name) | core_helpers.py:297-311 |

Chat: `original_function=_acompletion` -> `metadata`. Generic: `_ageneric_api_call_with_fallbacks_helper`, `..._responses_attempt`, `..._anthropic_messages_attempt` all contain `_ageneric_api_call_with_fallbacks` -> `litellm_metadata`

### 1.4 kwargs keys read or written by the router (async path)

| Key | Who | Effect | Cite |
|---|---|---|---|
| `model` | all layers | model group; overwritten by fallback hop (`kwargs["model"]=mg`) | PR:2890, FEH:751-754 |
| `stream` | `_acompletion`, timeouts | chooses stream timeout; stream wrapper | PR:4251 |
| `original_function` | popped by retries | per-attempt callable | PR:7988 |
| `original_generic_function` | generic helper | real litellm handler | PR:5386 |
| `fallbacks` | `.get` in fallbacks layer (kept for hops), `.pop` in retries layer | default `self.fallbacks`; explicit `None` disables router fallbacks | PR:7883, 7990 |
| `context_window_fallbacks`, `content_policy_fallbacks` | same as above | default router attrs | PR:7884-7885, 7992-7993 |
| `model_group_retry_policy` | popped in retries | per-request override of router dict | PR:7995 |
| `num_retries` | popped in retries | request-level retries | PR:7997 |
| `disable_fallbacks` | popped in fallbacks layer, mirrored into bucket `_disable_fallbacks` | PR:7881-7882, FEH:440-456 |
| `max_fallbacks`, `fallback_depth` | set into `input_kwargs` if absent (router value, 0) | PR:7619-7622 |
| `attempted_targets` | `AttemptedFallbackTargets` shared by reference across hops | FEH:180-194, 706-713 |
| `include_fallback_errors` | adds `x-litellm-fallback-errors` header | PR:7880, FEH:771-775 |
| `mock_timeout` | popped and re-passed to retries layer | PR:7887, 7898 |
| `mock_testing_fallbacks`, `mock_testing_context_fallbacks`, `mock_testing_content_policy_fallbacks` | popped (string values via `str_to_bool`); raise before retries layer | PR:7922-7967, T:1174-1185 |
| `mock_testing_rate_limit_error` | popped in retries layer; raises `RateLimitError` inside retry try | PR:8199-8223 |
| `specific_deployment` | popped in `_acompletion` / generic helper (attempt-local copy, so it survives retries) | PR:3796, 5453 |
| `_target_order`, `_excluded_deployment_ids`, `_retry_skipped_deployment_ids` | popped in `async_get_healthy_deployments` | PR:13531-13555 |
| `priority` | scheduler gate | PR:2896 |
| `litellm_logging_obj` | passed through; read by fallback cooldown trigger and router-reject logging | FEH:791-792, PR:13849-13860 |
| `passthrough_on_no_deployment` | generic helper; on selection failure call handler directly | PR:5444-5459 |
| `_mid_stream_fallback_controls` | carrier for streaming hops | FEH:307-360 |
| `_context_compaction_state` | written when the handler surface supports compaction | PR:7858-7861 |
| `model_info`, `timeout` | written per attempt by `_update_kwargs_with_deployment` | PR:4169, 4197 |

kwargs dict identity: each layer gets a fresh top-level dict (`**kwargs`), but the metadata bucket dict is shared by reference across attempts within a hop, so per-attempt writes into it (`model_info`, `deployment`, `api_base`, retry counters) are visible to the retries layer and `log_retry`. A fallback hop replaces the bucket with a shallow copy (FEH:757-763)

## 2. Candidate resolution and selection

### 2.1 Per attempt in `_acompletion` (PR:3775)

| Step | Rule | Cite |
|---|---|---|
| 1 | Snapshot `input_kwargs_for_streaming_fallback = kwargs.copy()` with `model` = group (used only by stream fallback) | PR:3789-3790 |
| 2 | `deployment = await async_get_available_deployment(model, messages, specific_deployment=kwargs.pop("specific_deployment", None), request_kwargs=kwargs)` | PR:3793-3798 |
| 3 | `_track_deployment_metrics(deployment)` -> `_update_usage(id)`: in-memory key = bare deployment id, value request count; first write ttl 60, later writes keep ttl. Write-only in the hot path | PR:3820, 15102-15113, 8746-8763 |
| 4 | Copy `litellm_params`, pop `silent_model` (mirror traffic, out of scope) | PR:3824-3840 |
| 5 | `_update_kwargs_with_deployment` (section 8) | PR:3843 |
| 6 | `self.total_calls[litellm_params["model"]] += 1` (in-process counter keyed by provider model name) | PR:3851 |
| 7 | `input_kwargs = {**litellm_params, "messages", "caching": self.cache_responses, "client": cached async client or None, **kwargs}`; pop `silent_model`, `include_fallback_errors`. kwargs win over deployment params | PR:3853-3861 |
| 8 | Concurrency slot: `MaxParallelRequestsLimit` if deployment has `max_parallel_requests`, else `rpm`, else `int(tpm/1000*6)` (min 1), else router `default_max_parallel_requests`. Exhausted -> raises `RateLimitError` immediately (no waiting) | PR:3866-3876, CIU:16-78, U:5138-5178 |
| 9 | `async_routing_strategy_pre_call_checks` (no-op for simple-shuffle unless CustomLogger callbacks implement it) then `litellm.acompletion(**compacted_input)` | PR:3872-3879 |
| 10 | Non-stream `ModelResponse` with `finish_reason=="content_filter"` (or no choices, see quirks) and a content-policy fallback available -> raise `ContentPolicyViolationError("Response output was blocked.")` | PR:3882-3891, 8872-8884 |
| 11 | Lazy stream: `fetch_stream()` before returning | PR:3893-3898 |
| 12 | `success_calls[...] += 1`; streaming -> wrap with `_acompletion_streaming_iterator` (slot handed over) | PR:3900-3915 |
| 13 | On exception: `fail_calls[model_name] += 1`; stamp `exception.num_retries` from deployment `litellm_params.num_retries` (int-coerced) if exception has none; stamp `exception.failed_deployment_id = (kwargs.model_info or deployment.model_info).id` if not already set | PR:3916-3938, 3964-4011 |

Generic helper `_ageneric_api_call_with_fallbacks_helper` (PR:5437) is the same shape with these differences: `messages=kwargs.get("messages")`, `input=kwargs.get("input")`; `function_name="_ageneric_api_call_with_fallbacks"` (litellm_metadata bucket, passthrough timeout resolution PR:4171-4195); `response_kwargs = {**data, "caching", **kwargs, "model": model_name, ...}`; `custom_llm_provider` set from deployment; no `_track_deployment_metrics`, no silent experiment, no content-filter check (only Anthropic safeguard refusal, PR:5488-5499); on failure `fail_calls[model]` keyed by the group, not the provider model (PR:5513-5514)

### 2.2 `_async_get_available_deployment` (PR:13710)

| Step | Rule | Cite |
|---|---|---|
| 1 | Strategy not in {usage-based-routing-v2, simple-shuffle, cost-based-routing, latency-based-routing, least-busy} -> sync `get_available_deployment` | PR:13718-13731 |
| 2 | `async_pre_routing_hook`: always clears `routing_decision` and `_autorouter_baseline_route` from both buckets; with no pre-routing strategy also clears `SESSION_DEPLOYMENT_AFFINITY_TTL_METADATA_KEY` and `CONSUMED_REQUEST_TAGS_METADATA_KEY`, returns None | PR:14267, 14285-14294, 14531-14574 |
| 3 | No tier pin -> `_restore_client_ceiling_no_tier_pins` (no-op unless a previous pass stamped one) | PR:13751-13752 |
| 4 | `_get_routing_context` -> `("simple-shuffle", None)` for the default group | PR:1859-1900 |
| 5 | `healthy = async_get_healthy_deployments(...)` | PR:13768 |
| 6 | dict result (specific deployment or id lookup) -> returned directly | PR:13777-13782 |
| 7 | `simple_shuffle(resolve_model_alias, healthy, model, request_kwargs)` | PR:13795-13803 |
| 8 | Any exception: if `request_kwargs["litellm_logging_obj"]` exists, schedule `logging_obj.dispatch_failure_handlers(prefer_async_handlers=True)` as a task, then re-raise | PR:13849-13862 |

### 2.3 `_common_checks_available_deployment` (PR:13143), resolution precedence

| Order | Condition | Result | Cite |
|---|---|---|---|
| 1 | `specific_deployment is True` | every deployment whose `litellm_params.model == model` (list) | PR:13164-13172 |
| 2 | `model not in self.model_names and has_model_id(model)` | that one deployment as dict, model = its `litellm_params.model`; id in index but missing -> `ValueError` | PR:13173-13191 |
| 3 | alias map hit | model := alias target | PR:13193-13195 |
| 4 | routing group | group members | PR:13197 |
| 5 | model not in `model_names` | team, pattern (wildcard), team pattern, `default_deployment` | PR:13199, 13049-13094 |
| 6 | default | `_get_all_deployments(model_name=model)` via `model_name_to_deployment_indices` (config order), filtered by `should_include_deployment` (team rules) | PR:13232, 12030-12101, 12007-12028 |
| 7 | empty and pre-filter list empty | retry as litellm model name (`_get_deployment_by_litellm_model`) | PR:13246-13268 |
| 8 | still empty, router has a `"*"` fallback entry | use the first model of `fallbacks[*]["*"]` as the model group | PR:13281-13300 |
| 9 | still empty | `litellm.BadRequestError("You passed in model={model}. There are no healthy deployments for this model", model, llm_provider="")`, status 400 | PR:13302-13307, T:821 |
| 10 | `litellm.model_alias_map` hit | rename returned model (deployments unchanged) | PR:13309-13312 |

Every arm also runs `_drop_strategy_markers` (auto_router markers; error if only markers) and `_filter_reserved_deployments` (team reservation windows; error if reserved) PR:13316-13346

### 2.4 `async_get_healthy_deployments` filter pipeline (PR:13401), in order

| # | Filter | Simple-config behavior | Cite |
|---|---|---|---|
| 1 | `filter_team_based_models` | drops deployments with `model_info.team_id` != request team id | PR:13430, CU:143-207 |
| 2 | `filter_web_search_deployments` | ACTIVE: if any `tools[].type` in {`web_search`,`web_search_preview`}, keep only deployments where `model_info.supports_web_search` is absent or truthy | PR:13438, CU:230-257 |
| 3 | dict result | `model_info.blocked is True` -> `ServiceUnavailableError("Model '{model}' is currently paused and cannot accept requests.")`, else return dict (skips cooldown!) | PR:13446-13453 |
| 4 | health-check filter | no-op unless `enable_health_check_routing` | PR:13456, 15003-15046 |
| 5 | cooldown | ids from `_async_get_cooldown_deployments` (all router ids, active cooldown entries) removed | PR:13462-13478, CH:489-509 |
| 6 | cooldown bypass | only if result empty and `enable_health_check_routing` and `allowed_fails_policy` set | PR:13482-13486 |
| 7 | blocked | drop `model_info.blocked is True` | PR:13488, 14975-14990 |
| 8 | `async_callback_filter_deployments` | each `CustomLogger` in `litellm.callbacks` may narrow | PR:13490, 9039-9080 |
| 9 | pre-call checks | only if `enable_pre_call_checks` | PR:13498 |
| 10 | tag filter | no-op unless tag filtering enabled | PR:13516 |
| 11 | routing plugins | no-op unless configured | PR:13524, 14063-14090 |
| 12 | order | keep min `order` (litellm_params.order, else model_info.order); `_target_order` popped selects exact order | PR:13531-13534, U:5181-5210 |
| 13 | weighted-failover exclusion | `_excluded_deployment_ids` popped, removed | PR:13539-13543 |
| 14 | retry skip | `_retry_skipped_deployment_ids` popped and removed, but if that empties the list keep the unfiltered list | PR:13548-13556 |
| 15 | empty | raise `async_raise_no_deployment_exception` | PR:13558-13564 |

### 2.5 No deployment available (HE:71-97, T:1048-1074)

`RouterRateLimitError(ValueError)` (no `status_code` attribute):
- `cooldown_time = cooldown_cache.get_min_cooldown(get_model_ids(model_name=model))`: min `cooldown_time` over cooldown entries present in cache for that group's ids (expiry not checked), and `or default_cooldown_time` so a min of 0 or no entry gives the router default (CC:213-231)
- `cooldown_list` = ids of ALL active cooldowns router-wide, not just this group (HE:82-90, CH:512-524)
- `all_deployments_in_cooldown = bool(model_ids) and set(model_ids) <= set(cooldown_list)`; `type` = `"all_deployments_in_cooldown"` or `"rate_limit_error"`
- message: `"No deployments available for selected model, Try again in {cooldown_time} seconds.{reason} Passed model={model}. pre-call-checks={enable_pre_call_checks}, cooldown_list={cooldown_list}"` where reason is `" All deployments for selected model are in cooldown."` or empty

Retry-after header for 429 to client is computed by the proxy from `cooldown_time`, not here

## 3. `simple_shuffle` exactly (SS:43-71)

1. `resolved_model = resolve_model_alias(model) or model` (SS:49)
2. Weight sets tried in order (SS:50-56): (a) request-scoped `_router_weights[resolved_model][deployment_id]` from `request_kwargs["_router_weights"]` (validated; missing id -> 0.0; skipped entirely when no entry for the model), (b) `litellm_params.weight`, (c) `litellm_params.rpm`, (d) `litellm_params.tpm`. Only `litellm_params` is read, never top-level `deployment.rpm` or `model_info`
3. Metric value: None -> 0.0; int/float (bool counts as int) -> float; anything else -> `TypeError("Deployment {metric} must be numeric")` (SS:17-24)
4. For each set: `largest = max(weights, default=0.0)`; skip if `largest <= 0`; normalize `w / largest`; skip if `sum <= 0`; else `random.choices(healthy, weights=normalized)[0]` (one `random.random()` draw, cumulative-weight bisect) (SS:57-65)
5. No set qualifies -> `random.choice(healthy)` (SS:66)

Parity note: the RNG is Python's Mersenne Twister. A port can only match distributions, not individual picks, unless tests seed both sides identically

## 4. Retries: `async_function_with_retries` (PR:7987)

### 4.1 Setup

| Rule | Cite |
|---|---|
| Pops `original_function`, `fallbacks`, `context_window_fallbacks`, `content_policy_fallbacks`, `model_group_retry_policy` (default router attrs) | PR:7988-7995 |
| `request_num_retries = kwargs.pop("num_retries", None)`; working `num_retries = request ?? self.num_retries ?? 0` | PR:7997-8002 |
| Metadata bucket `_metadata = kwargs.get("litellm_metadata", kwargs.get("metadata")) or {}`; if it has a str `model_group`, write `model_group_size = len(get_model_list(model_group))` | PR:8005-8009 |
| Write `_metadata["attempted_retries"]=0`, `_metadata["max_retries"]=num_retries` | PR:8015-8016 |
| `_handle_mock_testing_rate_limit_error`, then attempt 0: `make_call`; success -> `x-litellm-attempted-retries: 0` header only (no max-retries header) | PR:8018-8022 |

### 4.2 On failure of attempt 0

| Step | Rule | Cite |
|---|---|---|
| 1 | Guardrail intervention -> re-raise immediately | PR:8024-8025 |
| 2 | Deployment override: if no request num_retries and `e.num_retries` is an int (stamped from deployment, or mock RateLimitError), `num_retries = e.num_retries` | PR:8028-8035 |
| 3 | `(healthy, all) = _async_get_healthy_deployments(kwargs["model"])` (section 4.4) | PR:8039-8045 |
| 4 | Retry policy: if `request_num_retries != 0` and (`self.retry_policy` or `model_group_retry_policy` not None), compute `_get_num_retries_from_retry_policy(e, model_group, mgrp, retry_policy)`; non-None result replaces `num_retries` and sets `_retry_policy_applies=True` | PR:8050-8064 |
| 5 | If no policy applies: `should_retry_this_error(...)` raises the error when not retryable (4.3) | PR:8068-8076 |
| 6 | `_metadata["max_retries"] = num_retries` | PR:8078 |
| 7 | `num_retries > 0`: `log_retry`, compute retry-skip ids; else re-raise | PR:8081-8091 |
| 8 | Sleep `_time_to_sleep_before_retry(e, remaining=num_retries, num_retries=num_retries, healthy, all)` | PR:8094-8102 |
| 9 | Loop `for current_attempt in range(num_retries)`: write `attempted_retries=current_attempt+1`, `max_retries=num_retries`; `make_call`; success -> headers `x-litellm-attempted-retries=current_attempt+1`, `x-litellm-max-retries=num_retries` | PR:8104-8121 |
| 10 | Loop failure: `original_exception = e` (latest), `log_retry`, `remaining = num_retries - current_attempt - 1`, refresh healthy list only (all list stays from step 3), re-check `should_retry_this_error` unless policy applies (raise the new error if not retryable), update skip ids, sleep `_time_to_sleep_before_retry(remaining)` (also after the final attempt) | PR:8123-8171 |
| 11 | Exhausted: if `type(e)` is exactly in `LITELLM_EXCEPTION_TYPES`, set `e.max_retries=num_retries`, `e.num_retries=current_attempt+1`; raise latest exception | PR:8173-8183 |

Every attempt re-runs deployment selection inside `_acompletion`, so cooldowns written by the previous failure's callback (section 6, runs synchronously before the exception surfaces) already exclude the failed deployment when it was cooled down

Retry skip (PR:7970-7984): if the exception has `retry_skip_deployment_id` or `failed_deployment_id` and an int `status_code` that `_should_retry` rejects (non-408/409/429/5xx), add that id to `kwargs["_retry_skipped_deployment_ids"]` (sorted tuple). Only applied on the retry layer's own kwargs copy

### 4.3 `should_retry_this_error` (PR:8225-8287), in order; "raise" means not retried

| # | Condition | Outcome |
|---|---|---|
| 1 | `ContextWindowExceededError` and `context_window_fallbacks is not None` | raise |
| 2 | `ContentPolicyViolationError` and `content_policy_fallbacks is not None` | raise |
| 3 | `status_code` not None and `not _should_retry(status)` and status not in (401, 403) | raise |
| 4 | `litellm.NotFoundError` | raise |
| 5 | `openai.RateLimitError` and healthy <= 0 and `regular_fallbacks` non-empty | raise |
| 6 | `openai.AuthenticationError` or `PermissionDeniedError` and `len(all) <= 1` | raise |
| 7 | healthy <= 0 | raise |
| 8 | otherwise | retry |

`_should_retry(status)` (U:7134-7160): True for 408, 409, 429, >= 500; False otherwise. Consequences: plain 400 BadRequest, 404, 422 never retry; 401/403 retry only with more than one deployment in the group; a context-window error with no context_window_fallbacks still hits rule 3 (status 400) and is not retried. Exceptions without `status_code` (e.g. `RouterRateLimitError`) skip rule 3 and usually die at rule 7 because every deployment is cooling

When a retry policy applies, `should_retry_this_error` is skipped entirely: a policy granting `BadRequestErrorRetries` or `DefaultRetries` retries 400s and context-window errors too

### 4.4 `_async_get_healthy_deployments` (PR:8925-8953)

Calls `_common_checks_available_deployment(model)` with no request kwargs (no team, no access groups), swallowing any exception (then all = []). Dict result -> returns `([], dict)`. Else healthy = all minus active cooldown ids minus blocked. Note `all` here is the full group, not filtered by web search, tags or order

### 4.5 Retry policy resolution (RP:37-66)

1. Policy = `model_group_retry_policy[model_group]` if present, else `retry_policy`; dict -> `RetryPolicy(**dict)` (RP:37-49)
2. If `e.status_code == 404` and `NotFoundErrorRetries` set -> that (RP:32-34)
3. Walk `type(e).__mro__` most specific first over {AuthenticationError, Timeout, RateLimitError, ContentPolicyViolationError, BadRequestError, ServiceUnavailableError, InternalServerError}; first non-None field wins (RP:19-29, 60-65)
4. Else `DefaultRetries` (may be None -> policy does not apply)

Router `model_group_retry_policy` defaults to `{}` not None (PR:880-960 signature, PR:1314), so the `is not None` guard at PR:8050 is always true; it resolves to None harmlessly when nothing is configured

### 4.6 Backoff `_time_to_sleep_before_retry` (PR:8320-8363) and `_calculate_retry_after` (U:7197-7220)

| Case | Sleep |
|---|---|
| `len(all) != 1` and `len(healthy) > 0` | 0 (instant retry, ignores Retry-After) |
| otherwise | `_calculate_retry_after(remaining, max=num_retries, headers, min_timeout=self.retry_after)` |

Headers used: `e.response.headers`, overridden by `e.litellm_response_headers` when that attribute exists (PR:8343-8346)

`_calculate_retry_after`: `ra = int(headers["retry-after"])`, or HTTP-date parse minus now, else -1 (U:7163-7194). `jitter = JITTER * random.random()`. If `0 < ra <= 60` -> `ra + jitter`. Else `INITIAL_RETRY_DELAY * 2**(max - remaining)`, then `max(., min_timeout)`, then `min(., MAX_RETRY_DELAY)`, plus jitter. First sleep uses exponent 0 (0.5s), after loop attempt i exponent i+1

### 4.7 `make_call` (PR:8185-8197)

Call, await if awaitable, `increment_deployment_usage_for_response` (section 7), `set_response_headers` (section 8). Runs once per successful attempt only

## 5. Fallbacks

### 5.1 `async_function_with_fallbacks` (PR:7852-7920)

| Step | Rule | Cite |
|---|---|---|
| 1 | Compaction state init if handler surface supports it | PR:7858-7861 |
| 2 | Clear `pre_routing_selected_model` from both buckets | PR:7863, FEH:270-283 |
| 3 | First hop only (`attempted_targets` not an `AttemptedFallbackTargets`): pop `attempted_fallbacks`, `original_model_group` from the sibling bucket; set `attempted_fallbacks=0`, `original_model_group=model` in the function's bucket | PR:7864-7879 |
| 4 | `include_fallback_errors = kwargs.get(...) is True`; pop `disable_fallbacks`; record bucket flag | PR:7880-7882 |
| 5 | `fallbacks`, `context_window_fallbacks`, `content_policy_fallbacks` via `.get` (kept in kwargs) | PR:7883-7885 |
| 6 | Pop `mock_timeout`; run mock fallback triggers (raise before any attempt) | PR:7887-7896 |
| 7 | `async_function_with_retries`; success -> `x-litellm-attempted-fallbacks: 0` | PR:7898-7908 |
| 8 | Any exception -> `async_function_with_fallbacks_common_utils` | PR:7909-7920 |

### 5.2 `async_function_with_fallbacks_common_utils` (PR:7582-7849)

Guard: `disable_fallbacks is True`, or `kwargs["model"] is None`, or guardrail intervention -> raise `e` (PR:7610-7611)

`input_kwargs = {"litellm_router": self, "original_exception": e, **kwargs}` plus `max_fallbacks` (router) and `fallback_depth=0` when absent (PR:7613-7624)

`lookup_groups = dedupe(pre_routing_selection, bucket.model_group, model_group, bucket.original_model_group)` (FEH:469-484). Simple config: just the group

Decision order:

| # | Branch | Condition | Action | Cite |
|---|---|---|---|---|
| 1 | Order fallback | >1 distinct `order` values in the group and error is not CW/CP | chain = `[{"model": group, "_target_order": o} for o > current]` + external chain; `run_async_fallback`; NOT inside the catch, so its error propagates as is | PR:7636-7681 |
| 2 | Weighted failover | `enable_weighted_failover` | out of scope | PR:7683-7694 |
| 3 | Client-side list | `fallbacks` is all strings, or all dicts where some dict has a `LiteLLMParamsTypedDict` key with a non-list value | `run_async_fallback(fallbacks)` | PR:7699-7713, FEH:858-880 |
| 4 | Context window | `ContextWindowExceededError` and `context_window_fallbacks is not None` | resolve chain for lookup groups; none -> raise original (does NOT try generic); else run. If cw list is None, append debug text (when exposed) and fall through to generic | PR:7716-7747 |
| 5 | Content policy | same shape with `content_policy_fallbacks` | PR:7749-7779 |
| 6 | Generic | `fallbacks is not None` | `get_fallback_model_group_for_lookup_groups`; none and `"*"` present -> `fallbacks[idx]["*"]`; none at all -> append "no fallback group" text (top-level hop only) and raise original; else `run_async_fallback` | PR:7781-7821 |
| 7 | Nothing configured | falls out of the try | append outcome text and raise original | PR:7841-7849 |

Inside the try (branches 3 to 6), any exception including the last fallback's error is caught (PR:7823-7839); the router logs it plus the cooldown list and then raises the ORIGINAL primary exception, with `"\n\nLiteLLM: model group '{g}' failed with the error above. Fallback to {targets} also failed: {detail}"` appended to `.message` when `litellm.expose_router_debug_in_errors` (default True) and this is the top-level hop (`fallback_depth` absent or 0). Detail is redacted and truncated to 2000 chars (CU:64-107, C:13)

Chain lookup (FEH:498-544): per lookup group, scan the list: exact key match wins and stops; a key equal to the provider-prefixed group or a key that equals the group with a `provider/` prefix stripped is a candidate (last one wins); a bare string item sets the chain to `[item]`; `"*"` remembers its index. Priority: exact, stripped/prefixed, `"*"`. Across lookup groups the first specific chain wins, `"*"` only after all miss

### 5.3 `run_async_fallback` (FEH:658-799)

| Step | Rule | Cite |
|---|---|---|
| 1 | `fallback_depth >= max_fallbacks` -> raise original | FEH:694-695 |
| 2 | `same_model_group_only` for file/batch/fine-tune ids (not chat) | FEH:701-703 |
| 3 | `attempted = kwargs.attempted_targets or new`; record failed group (`pre_routing_selection or original_model_group`) | FEH:707-713 |
| 4 | Dict targets validated against server-owned WIF params | FEH:719-721 |
| 5 | For each target in order: skip the failed group; skip unauthorized (`fallback_access_check`) or over budget (`fallback_budget_check`), both None in plain SDK; skip if `fallback_attempt_key` (string, or `{"model": x}` -> `x`, else sha256 of sorted JSON) already attempted, else record | FEH:723-746 |
| 6 | Per target: `log_retry(kwargs, original_exception)`; pop `_target_order`; str -> `kwargs["model"]=mg`, dict -> `kwargs.update(mg)`; `depth += 1` | FEH:748-755 |
| 7 | New bucket = shallow copy; keep first `original_model_group`; set `model_group=kwargs["model"]`, `attempted_fallbacks=depth`; set `fallback_depth`, `max_fallbacks`, `attempted_targets` | FEH:757-768 |
| 8 | `await router.async_function_with_fallbacks(**kwargs)` (full retries + nested fallbacks for that group) | FEH:769 |
| 9 | Success: `x-litellm-attempted-fallbacks = depth` (this level's depth), optional `x-litellm-fallback-errors` JSON list `[{message,type,param,code}]` starting with the original error; `log_success_fallback_event` on every CustomLogger | FEH:771-782 |
| 10 | Failure: record error info, `log_failure_fallback_event`; if a shared logging obj already logged an async failure, run `_trigger_cooldown_for_failed_deployment` on `e.failed_deployment_id`; continue | FEH:783-798 |
| 11 | All failed -> raise last error (which common_utils then swaps for the original, see 5.2) | FEH:799 |

`max_fallbacks` resolution: ctor arg, else `litellm.max_fallbacks`, else `ROUTER_MAX_FALLBACKS` (PR:1190-1195). Depth counts hops across nesting, so a chain of length N at depth 0 can try at most `max_fallbacks` hops total

`default_fallbacks` (ctor or `litellm.default_fallbacks`) are appended to `self.fallbacks` as `{"*": [...]}` at init (PR:1215-1220)

## 6. Cooldowns on failure

### 6.1 When it runs

`litellm.acompletion`'s async wrapper calls `logging_obj.failure_handler` synchronously and then awaits `async_failure_handler`, BEFORE the exception reaches the router (U:2210-2220, comment "DO NOT MAKE THREADED - router retry fallback relies on this"). Registered in `__init__` (PR:1272-1291):

| List | Callback | Runs for async requests |
|---|---|---|
| `litellm._async_success_callback` | `deployment_callback_on_success` | yes, in a background task |
| `litellm.success_callback` | `sync_deployment_callback_on_success` | no: sync callables are skipped for async requests (LL:3183-3195) and the name is filtered as internal (LL:4102-4116) |
| `litellm._async_failure_callback` | `async_deployment_callback_on_failure` | yes, awaited inline |
| `litellm.failure_callback` | `deployment_callback_on_failure` | yes, sync, inline (LL:3809-3819) |

Each logging object runs each failure handler once (`has_logged_sync_failure` / `has_logged_async_failure`, LL:2252-2284). When the caller passes `litellm_logging_obj` (the proxy does), every attempt of the request reuses it, so only the FIRST failure of the whole request runs these callbacks. `run_async_fallback` patches only fallback hops (6.5). Plain SDK calls get a fresh logging object per attempt, so every failure runs them

### 6.2 `deployment_callback_on_failure` (PR:8552-8651)

| Step | Rule | Cite |
|---|---|---|
| 1 | Advisor orchestration failure -> False | PR:8572-8577 |
| 2 | Background cost poll 404 -> False | PR:8583-8588, CH:69-75 |
| 3 | `exception_status = getattr(e, "status_code", "")` | PR:8590 |
| 4 | Caller-set timeout 408 (`client_side_timeout` and elapsed >= timeout) -> False | PR:8592-8597, CH:657-674 |
| 5 | Cooldown time: `model_info.cooldown_time` else `litellm_params.cooldown_time` if >= 0; else Retry-After header if >= 0 (no 60s cap; headers from `e.headers`, `e.response.headers`, `e.litellm_response_headers` in that order); else `self.cooldown_time` | PR:8599-8624, CH:92-102 |
| 6 | No `model_info.id` -> False | PR:8626-8629 |
| 7 | `increment_deployment_failures_for_current_minute(id)` | PR:8630-8633 |
| 8 | `_set_cooldown_deployments(...)`, `requested_model_group` from metadata | PR:8634-8641 |

### 6.3 Decision tree `_set_cooldown_deployments` (CH:429-486)

`_should_run_cooldown_logic` (CH:273-329), any False stops:
1. id None or `get_model_group(id)` None (unknown deployment, e.g. client-side-credential dynamic id) -> False
2. `time_to_cooldown` within 1e-9 of 0 -> False (so `Retry-After: 0` or `cooldown_time: 0` disables cooldown)
3. `router.disable_cooldowns` -> False
4. `_is_cooldown_required(status, str(e))` False and no deployment-level `allowed_fails_policy` entry for this exception type -> False
5. id in `provider_default_deployment_ids` -> False

`_is_cooldown_required` (CH:220-270): `"APIConnectionError"` in `str(e)` -> False; status `""` -> False; str -> int; 400-499: True only for 429, 401, 408, 404; everything else (including < 400 and >= 500) -> True; any exception -> True

`_should_cooldown_deployment` (CH:332-426):

| # | Condition | Result |
|---|---|---|
| 0 | `is_single_deployment_model_group = len(get_model_group(id)) == 1 and not (routing_group_has_alternatives or team_model_has_alternatives)` | |
| 1 | deployment `model_info.allowed_fails_policy` or `model_info.allowed_fails` set | deployment policy path (6.4) |
| 2 | router `allowed_fails_policy` None and `_is_allowed_fails_set_on_router` False | base case below |
| 2a | status 429 and not single | True |
| 2b | `percent_fails == 1.0` and total >= `SINGLE_DEPLOYMENT_TRAFFIC_FAILURE_THRESHOLD` | True |
| 2c | `percent_fails > DEFAULT_FAILURE_THRESHOLD_PERCENT` and total >= `DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS` and not single | True |
| 2d | `_should_retry(status)` is False (so 401, 404, and any other non-retryable status that passed `_is_cooldown_required`) | True, even for single-deployment groups |
| 2e | else | False |
| 3 | router policy or non-default allowed_fails | `should_cooldown_based_on_allowed_fails_policy` |

`percent_fails = fails / (successes + fails)` from in-memory counters, read after this failure was already counted (CH:380-400). `_is_allowed_fails_set_on_router` is True only when `router.allowed_fails` is not None and differs from `litellm.allowed_fails` (CH:628-642)

`should_cooldown_based_on_allowed_fails_policy` (CH:563-611): `allowed = override ?? policy-by-type ?? router.allowed_fails`; `ttl = override ?? (router.cooldown_time or DEFAULT_COOLDOWN_TIME_SECONDS)`; key `deployment:{id}:allowed_fails[:{suffix}]`; `DualCache.increment_cache` (memory, then Redis if configured; Redis error -> local count) with that ttl; return `updated > allowed`. Router policy type order: Auth, Timeout, RateLimit, ContentPolicy, BadRequest, InternalServer, ServiceUnavailable, BadGateway, NotFound (PR:15123-15180)

### 6.4 Deployment-level policy (CH:105-191)

`dep_policy` must be a dict; matched in order ContentPolicy, BadRequest, Auth, Timeout, RateLimit, InternalServer, ServiceUnavailable, BadGateway, NotFound (CH:79-90). If no policy match and `dep_allowed_fails` set and single-deployment group -> False. Else `allowed_fails_override = policy value ?? dep_allowed_fails`, suffix = exception class name (policy) or `"generic"`, cooldown ttl override from deployment `cooldown_time`

### 6.5 Cooldown write (CC:98-132, CH:473-486) and fallback-path trigger (FEH:46-154)

| Item | Value |
|---|---|
| Key | `deployment:{model_id}:cooldown` (CC:134-136) |
| Value | `{"exception_received": masked str(e) (first 50 chars kept, rest `*`), "status_code": str(status_int), "timestamp": time.time(), "cooldown_time": float}` (CC:78-96) |
| TTL | `cooldown_time` or router default (CC:113-118) |
| Store | separate `DualCache` (own InMemoryCache, router Redis attached lazily, Redis batch read interval `DEFAULT_COOLDOWN_REDIS_READ_INTERVAL_SECONDS`) (CC:42-76) |
| Read | batch get all router ids; active iff `timestamp + cooldown_time - now > 0`; in-memory TTL corrected down to min(remaining, 60) when a Redis-promoted entry carries the 600s default (CC:138-191) |
| Event | `router_cooldown_event_callback` task (CH:480-486) |

`_trigger_cooldown_for_failed_deployment` (fallback hops only, when the shared logging obj already logged): skips advisor failures, generic-call 404s, caller-timeout 408s; uses `e.failed_deployment_id` only; same cooldown-time precedence; increments fails; calls `_set_cooldown_deployments` without `requested_model_group` (FEH:46-154)

Failure-side usage (`async_deployment_callback_on_failure`, PR:8653-8684): `RPM` key `global_router:{id}:{deployment}:rpm:{HH-MM UTC}` += 1 via `DualCache.async_increment_cache` (memory and Redis), ttl 60. Runs regardless of whether tpm/rpm limits are configured

## 7. Success-path bookkeeping

| What | Where | Inline or callback | Keys, TTL | Cite |
|---|---|---|---|---|
| Per-attempt pick counter | `_update_usage` | inline, at selection | in-memory key = bare id, ttl 60 on first write | PR:8746-8763 |
| `total_calls` / `success_calls` / `fail_calls` | `_acompletion`, generic helper | inline | process-local defaultdicts | PR:3851, 3900, 3933 |
| TPM/RPM usage | `increment_deployment_usage_for_response` from `make_call` | inline, after a successful attempt | `global_router:{id}:{deployment}:tpm:{HH-MM}` += total tokens, `...:rpm:...` += 1, ttl 60, pipeline memory then Redis; Redis post-values written back to memory as `int` ttl 60. Skipped when deployment and model_info set no tpm/rpm and no itpm/otpm; skipped when tokens <= 0 and rpm <= 0. Stamps `_litellm_router_usage_counted_tokens` into the bucket | PR:8419-8513 |
| Successes counter | `deployment_callback_on_success` | async success callback (background task) | in-memory `{id}:successes` += 1, ttl 60 | PR:8367-8417, TDM:45-59 |
| Remaining tokens | same | same | adds `max(0, total_tokens - counted)` with rpm += 0 when inline already counted, else full tokens and rpm 1 | PR:8402-8413 |
| Fails counter | `deployment_callback_on_failure` | sync failure callback, inline | in-memory `{id}:fails` += 1, ttl 60 | TDM:62-75 |

Success callback skips `_aresponses_websocket`, `_arealtime`, batch retrieve, missing metadata, and ids not in the router (PR:8379-8392). Streaming: the inline increment sees the wrapper (0 tokens, rpm 1); the callback adds tokens once the complete response is assembled

Window semantics: `InMemoryCache.set_cache` sets a TTL only when the key has none or it expired (IMC:150-176), and `increment_cache` is get+set (IMC:230-236). So "per minute" counters are fixed 60s windows starting at the first increment, not calendar minutes. The TPM/RPM keys embed the UTC `%H-%M` minute, so they are calendar-minute buckets

## 8. Metadata and response fields written

Per attempt, `_update_kwargs_with_deployment` (PR:4095-4199):

| Write | Value |
|---|---|
| merge tools | deployment `tools` + request `tools`; `tool_choice` from deployment if absent (PR:4078-4093) |
| bucket `deployment` | `litellm_params.model` |
| bucket `model_info` | copy of deployment `model_info` (client-side credentials: new dynamic id, `original_model_id`) |
| bucket `api_base` | `litellm_params.api_base` |
| bucket `deployment_model_name` | deployment `model_name` |
| reservation refund / io-token kwargs | PR:4142-4143 |
| bucket `_routing_request_tags` | setdefault tuple of request tags |
| bucket `tags` | request tags + deployment tags (dedup, order kept) + `"Credential: {litellm_credential_name}"` |
| `kwargs["model_info"]` | same model_info |
| `kwargs["timeout"]` | chat: stream -> `stream_timeout` kwarg, deployment, router `stream_timeout`, router `request_timeout`, default params; then non-stream chain `timeout` kwarg, `request_timeout` kwarg, deployment `timeout`, deployment `request_timeout`, router `request_timeout`, router `timeout` (`= ctor timeout or litellm.request_timeout`), default params (PR:4225-4257) |
| default params | `setdefault` each non-None `default_litellm_params` key (`timeout`, `max_retries=0`, ...) and merge its `metadata` (`caching_groups`) into the bucket (PR:4019-4037, 1238-1241) |

Retry layer: bucket `model_group_size`, `attempted_retries`, `max_retries` (PR:8005-8016, 8106-8107). `log_retry` (PR:8700-8744): bucket chosen by key presence (`litellm_metadata` if in kwargs); `previous_models` = last 3 earlier records + new `{model_group, deployment_id (from bucket model_info), exception_type, exception_string, attempted_retries}` (cap `RETRY_BREADCRUMB_LIMIT=4`, PR:824); `request_retry_count += 1`. Fallback layer: `attempted_fallbacks`, `original_model_group`, `model_group` per hop, `_disable_fallbacks`

Response (`set_response_headers`, PR:11756-11784, HDR):
- bare async iterators are wrapped in `HiddenParamsAsyncIteratorWrapper` so they can carry `_hidden_params` (HDR:83-115)
- `_hidden_params["model_id"] = bucket.model_info.id` only if not already set (HDR:134-144)
- `_hidden_params.additional_headers["x-litellm-model-group"] = kwargs["model"]` of the hop that answered
- quality and complexity router headers (no-op in simple config)
- if group `tpm`/`rpm` limits exist: `x-ratelimit-remaining-tokens`, `x-ratelimit-limit-tokens`, `x-ratelimit-remaining-requests`, `x-ratelimit-limit-requests`, only when not already present (PR:11720-11754, HDR:179-185)
- `x-litellm-attempted-retries` (+ `x-litellm-max-retries` after a retry) and `x-litellm-attempted-fallbacks` (+ optional `x-litellm-fallback-errors`) (HDR:284-327)

`_record_routing_decision` (PR:14531-14574) only writes `routing_decision` / `_autorouter_baseline_route` for pre-routing strategies; in the simple config it clears both every attempt (PR:14267)

## 9. Streaming (summary)

| Surface | Commit rule (no more failover after) | Pre-commit failure | Cite |
|---|---|---|---|
| Chat (`_acompletion_streaming_iterator`) | `CustomStreamWrapper` raises plain mapped error for 4xx except 429 (never failover). Otherwise `MidStreamFallbackError(is_pre_first_chunk=not sent_first_chunk, generated_content=response_uptil_now)`. Router re-raises the original if `not is_pre_first_chunk` and (generated_content or any chunk delta has content, tool_calls, function_call, reasoning_content, thinking_blocks, reasoning_items, audio, images, annotations) | straight to `async_function_with_fallbacks_common_utils` with the pre-selection kwargs snapshot: NO same-group retry; trigger is the `MidStreamFallbackError` (a `ServiceUnavailableError`), so only generic fallbacks apply. Fallback usage is combined with partial usage; non-streaming fallback yields `None` | PR:2996-3226, SH:2263-2333, PR:468-485 |
| Responses (`_aresponses_streaming_iterator`) | iterator raises a plain error once a non-lifecycle event was yielded (`_output_started`) for transport drops / missing terminal event; error events become `MidStreamFallbackError` when 429, >= 500, no status, or content policy | lifecycle events `response.created/in_progress/queued` (max 200) are held and replayed only if no fallback lands; fallback even after content for error events, with a continuation input built from generated text; no same-group retry; content-policy trigger is unwrapped | PR:3335-3605, RSI:281-288, 640-670, 954-992 |
| Anthropic messages (`_aanthropic_messages_streaming_iterator`) | commit on first `content_block_delta` or 200 buffered frames, or immediately if neither retry nor fallback is possible; pings forwarded live | buffered lifecycle frames; SSE `event: error` with 429/5xx (not gateway verdict) or a safeguard refusal (with CP fallback) raise `MidStreamFallbackError`; FIRST a same-group retry loop (`_aanthropic_messages_retry_same_group`, budget = committed, else policy, else request/deployment/router num_retries; `should_retry_this_error` unless policy), THEN the fallback chain | PR:5562-6046, 566-680 |

## 10. Globals and constants read

| Name | Default | Read at | Cite |
|---|---|---|---|
| `litellm.num_retries` | None | ctor fallback for `router.num_retries` | `__init__.py:542`, PR:1183-1188 |
| `openai.DEFAULT_MAX_RETRIES` | 2 (openai SDK) | final ctor fallback for `router.num_retries` | PR:1188 |
| `litellm.max_fallbacks` | None | ctor | `__init__.py:543`, PR:1190-1195 |
| `ROUTER_MAX_FALLBACKS` | 5 (env) | ctor | C:12 |
| `litellm.fallbacks`, `default_fallbacks`, `context_window_fallbacks`, `content_policy_fallbacks` | None | ctor, only when ctor arg is falsy (`[]` counts as unset) | `__init__.py:544-547`, PR:1206-1226 |
| `litellm.allowed_fails` | 3 | ctor default and `_is_allowed_fails_set_on_router` | `__init__.py:548`, PR:1163-1166, CH:640 |
| `DEFAULT_COOLDOWN_TIME_SECONDS` | 5 (env) | `router.cooldown_time = arg or 5` | C:99, PR:1167 |
| `DEFAULT_COOLDOWN_REDIS_READ_INTERVAL_SECONDS` | 1.0 (env) | cooldown store | C:100-102 |
| `SINGLE_DEPLOYMENT_TRAFFIC_FAILURE_THRESHOLD` | 1000 (env) | base case 2b | C:153-155 |
| `DEFAULT_FAILURE_THRESHOLD_PERCENT` | 0.5 (env) | base case 2c (strict `>`) | C:93-95 |
| `DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS` | 5 (env) | base case 2c | C:156-158 |
| `INITIAL_RETRY_DELAY` | 0.5 (env) | backoff | C:500 |
| `MAX_RETRY_DELAY` | 8.0 (env) | backoff cap | C:501 |
| `JITTER` | 0.75 (env) | backoff jitter | C:502 |
| `ROUTER_FALLBACK_ERROR_DETAIL_MAX_CHARS` | 2000 | error text | C:13 |
| `RETRY_BREADCRUMB_LIMIT` | 4 | `previous_models` | PR:824 |
| `RoutingArgs.ttl` | 60 | TPM/RPM keys | PR:804-805 |
| `MAX_HELD_PRE_OUTPUT_RESPONSES_EVENTS` | 200 | responses stream | PR:698 |
| `MAX_BUFFERED_PRE_CONTENT_ANTHROPIC_CHUNKS` | 200 | anthropic stream | PR:566 |
| `litellm.request_timeout` | `REQUEST_TIMEOUT` env or 6000.0 | `router.timeout` fallback | C:573, 581 |
| `litellm.model_alias_map` | `{}` | renames returned model | `__init__.py:406`, PR:13309 |
| `litellm.expose_router_debug_in_errors` | True | appends debug text to messages | `__init__.py:226` |
| `litellm.LITELLM_EXCEPTION_TYPES` | list | exact-type gate for stamping retries | exceptions.py:970-991 |
| `litellm.callbacks` | | filter / pre-call hooks | PR:9061, 8955-8969 |
| `litellm.num_retries_per_request` | None | NOT enforced on the async path (only the sync `client` wrapper and `rust_bridge/preflight.py`) | U:1688, 1748 |
| Router ctor defaults | `retry_after=0`, `cooldown_time=None`, `allowed_fails=None`, `disable_cooldowns=None`, `routing_strategy="simple-shuffle"`, `model_group_retry_policy={}`, `fallbacks=[]`, `cache_responses=False`, `enable_weighted_failover=False` | | PR:880-945 |
| Router `default_litellm_params` | `timeout` = ctor timeout, `max_retries` = 0, `metadata.caching_groups` | per attempt | PR:1238-1241 |
| Router in-memory cache | `InMemoryCache()`: max 200 items, default ttl 600s, evicts earliest-expiring when full | all in-memory counters and clients | PR:1085-1087 (DualCache ctor), IMC:32-46, 106-139 |

## 11. Buggy or inconsistent behaviors (replicate for parity, flag upstream)

1. Final error after failed fallbacks is the PRIMARY group's exception (with fallback detail appended to its message), not the last fallback's, except on the order-based branch which raises the fallback error directly (PR:7823-7849 vs 7676-7681)
2. `x-litellm-attempted-fallbacks` is overwritten by each enclosing `run_async_fallback` with its own depth, so a success two hops deep reports 1 (FEH:771-775)
3. With a shared logging object (proxy), only the first failure of a request runs the cooldown and failure-RPM callbacks; later same-group retry failures never count toward cooldown. Fallback hops compensate via `_trigger_cooldown_for_failed_deployment`, retries do not (LL:2252-2284, FEH:791-797)
4. Retry loop sleeps once more after the last failed attempt before raising (PR:8164-8171)
5. `_async_get_healthy_deployments` returns `([], dict)` when the model is a deployment id, so rule 7 of `should_retry_this_error` always blocks retries for id-addressed calls (PR:8939-8941). Sync `_get_healthy_deployments` returns a bare `[]` there instead of a tuple (PR:8911-8912)
6. Retry policy bypasses `should_retry_this_error` completely; `BadRequestErrorRetries`/`DefaultRetries` makes 400s and context-window errors retry (PR:8050-8076, RP:60-66)
7. `_should_raise_content_policy_error` falls through to "raise" when `response.choices` is empty (PR:8880-8884). It also reads `content_policy_fallbacks` from `_acompletion`'s kwargs, where the retry layer already popped the request override, so only the router-level list is consulted (PR:7993, 8781)
8. `MAX_RETRY_DELAY` clamps after `min_timeout`, so `router.retry_after` above 8s is silently capped (U:7212-7216). Retry-After over 60s is ignored for retries but used uncapped for cooldown time (U:7206, PR:8611-8622)
9. Different header precedence for retry sleep (`litellm_response_headers` over `response.headers`, ignores `e.headers`) vs cooldown (`e.headers`, `response.headers`, `litellm_response_headers`) (PR:8343-8346, exception_mapping_utils.py:193-210)
10. `router.cooldown_time=0` is impossible at router level (`0 or 5`), but deployment `cooldown_time: 0` or `Retry-After: 0` disables cooldown (PR:1167, CH:300-302)
11. Setting `allowed_fails` equal to `litellm.allowed_fails` (3) is treated as unset, falling back to the percent rules (CH:628-642)
12. Router-level allowed-fails counter TTL is `cooldown_time` (5s default), not a minute; window is fixed from the first failure (CH:600-611, IMC:150-176, redis_cache.py:903-935). The sync Redis INCR runs on the event loop inside a sync callback
13. 401 and 404 cool a single-deployment group down immediately (base case 2d), while 429 on a single-deployment group needs 1000 requests at 100% failure (CH:402-416)
14. `RouterRateLimitError.cooldown_list` lists cooldowns from every group; `get_min_cooldown` ignores entry expiry and treats a 0 min as "use default" (HE:82-96, CC:213-231)
15. `get_allowed_fails_from_policy` checks Auth first, while the deployment-level table checks ContentPolicy first; harmless today but two orders for one concept (PR:15123-15180, CH:79-90)
16. `fail_calls` is keyed by provider model in `_acompletion` but by model group in the generic helper (PR:3933, 5514)
17. `default_fallbacks` are appended into whatever list `self.fallbacks` is, which can be the caller's list or the global `litellm.fallbacks` (mutation of shared state) (PR:1215-1220)
18. `log_retry` in `run_async_fallback` records the original exception for every hop and the deployment id currently in the bucket (the last attempt's), so breadcrumbs repeat (FEH:748)
19. `async_get_available_deployment` failure schedules `dispatch_failure_handlers` on the shared logging object, whose `model_info` may still describe a previous attempt's deployment (PR:13849-13860). Needs a test to confirm whether this double-counts fails
20. Chat streaming never retries in-group before fallback and gives up after content; Responses falls back even after content (continuation input); Anthropic retries in-group first. Three policies for one concept (section 9)
21. `num_retries_per_request` is not enforced for async calls (U:1676-1690 vs async wrapper U:1975-2020)
22. The router `InMemoryCache` holds at most 200 keys (counters, `_update_usage` keys, max-parallel limiters, clients). With many deployments the eviction of a `MaxParallelRequestsLimit` resets its in-flight count when re-created (CIU:55-78, IMC:136-139)
23. `_update_usage` writes a per-id counter that nothing on the hot path reads (PR:8746-8763, 15102-15113)
24. `_get_router_metadata_variable_name` substring-matches `"file"`, so any handler name containing it switches buckets (BU:161-171)
25. Exhausted-retry stamping uses `type(e) in LITELLM_EXCEPTION_TYPES`, so subclasses (e.g. `MidStreamFallbackError`) never get `max_retries`/`num_retries` (PR:8173)
