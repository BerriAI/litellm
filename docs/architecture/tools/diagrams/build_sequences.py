import os

from svglib import Seq

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "diagrams")


def out(name):
    return os.path.join(OUT, name + ".svg")


def startup():
    ps = [
        ("op", "Operator", "", "neutral"),
        ("cli", "litellm CLI", "proxy_cli.py:1086", "proxy"),
        ("ext", "proxy-extras", "migrate deploy", "store"),
        ("uv", "uvicorn", "", "neutral"),
        ("ls", "lifespan", "proxy_server.py:1334", "proxy"),
        ("pc", "ProxyConfig", "load_config :6471", "proxy"),
        ("db", "Postgres", "", "store"),
        ("sch", "APScheduler", ":10889", "proxy"),
    ]
    st = [
        ("m", "op", "cli", "litellm --config config.yaml"),
        ("m", "cli", "cli", "import proxy_server (every module global is built here)"),
        ("m", "cli", "cli", "WORKER_CONFIG env = json(args)"),
        ("m", "cli", "pc", "get_config, parse 1 (only to find database_url)"),
        ("m", "cli", "ext", "setup_database"),
        ("m", "ext", "db", "prisma migrate deploy, 220 migrations"),
        ("m", "cli", "uv", "run proxy_server:app"),
        ("m", "uv", "ls", "startup"),
        ("m", "ls", "ls", "worker hooks, license, master key boot check"),
        ("m", "ls", "pc", "load_config, parse 2"),
        ("m", "pc", "pc", "resolve os.environ/ refs, setattr(litellm, k, v), build Router, guardrails, MCP"),
        ("m", "ls", "db", "setup_prisma_client, connect, health check"),
        ("n", "ls", "pc", "load_config ran before Prisma connected, so DB config is not loaded yet"),
        ("m", "ls", "ls", "cost_tracking() -> _ProxyDBLogger, ProxyLogging.startup_event"),
        ("m", "ls", "sch", "schedule jobs (only with a DB)"),
        ("r", "ls", "uv", "ready, start serving"),
        ("m", "sch", "pc", "add_deployment every 30s (store_model_in_db)"),
        ("m", "pc", "db", "DB models, credentials, settings, guardrails"),
    ]
    Seq(ps, st, "Proxy startup", col_min=120).render(out("04-startup-sequence"))


def hooks():
    ps = [
        ("ep", "endpoint", "", "proxy"),
        ("pl", "ProxyLogging", "proxy/utils.py", "proxy"),
        ("ug", "UnifiedLLMGuardrails", "", "proxy"),
        ("rt", "Router", "", "sdk"),
        ("sdk", "SDK + Logging", "", "sdk"),
        ("cb", "callbacks", "litellm.callbacks", "sdk"),
    ]
    st = [
        ("m", "ep", "pl", "pre_call_hook (:2449)"),
        ("m", "pl", "pl", "policy pipelines, sequential then parallel guardrails"),
        ("m", "pl", "ug", "guardrails that define apply_guardrail"),
        ("m", "pl", "cb", "async_pre_call_hook (rate limiters, others)"),
        ("par", [
            ("LLM call", [
                ("m", "ep", "rt", "route_request"),
                ("m", "rt", "cb", "router hooks: pre_routing, filter_deployments"),
                ("m", "rt", "sdk", "acompletion"),
                ("m", "sdk", "cb", "log_pre_api_call"),
            ]),
            ("during call", [
                ("m", "ep", "pl", "during_call_hook (:3022)"),
                ("m", "pl", "cb", "async_moderation_hook"),
            ]),
        ]),
        ("m", "sdk", "cb", "async_log_success_event (incl. _ProxyDBLogger)"),
        ("m", "ep", "pl", "post_call_success_hook (:3599)"),
        ("m", "pl", "ug", "output guardrails"),
        ("m", "ep", "pl", "post_call_response_headers_hook (:3823)"),
    ]
    Seq(ps, st, "Where hooks fire in a proxy request", col_min=150).render(out("05-hook-sequence"))


def lazy_bypass():
    ps = [
        ("c", "Client", "", "neutral"),
        ("gw", "gateway pod", "gateway/main.py", "proxy"),
        ("mw", "LazyFeatureMiddleware", "_lazy_features.py", "proxy"),
        ("rt", "route table", "app.router.routes", "proxy"),
    ]
    st = [
        ("n", "gw", "rt", "startup: lifespan trims the route table to the gateway allowlist"),
        ("m", "c", "gw", "GET /guardrails/list"),
        ("m", "gw", "mw", "path matches lazy feature 'guardrails'"),
        ("m", "mw", "rt", "import module, register its routes (no allowlist check)"),
        ("r", "rt", "c", "200, a control-plane route served by the data-plane pod"),
    ]
    Seq(ps, st, "Lazy routes bypass the componentized trim", col_min=170).render(out("07-lazy-bypass"))


def flow_a():
    ps = [
        ("c", "Client", "", "neutral"),
        ("ep", "chat_completion", "proxy_server.py:12325", "proxy"),
        ("au", "user_api_key_auth", "auth/...:4184", "proxy"),
        ("pr", "RequestProcessing", "common_request_processing", "proxy"),
        ("pl", "ProxyLogging", "proxy/utils.py", "proxy"),
        ("rt", "Router", "router.py", "sdk"),
        ("m", "SDK", "dispatch -> main.py", "sdk"),
        ("p", "Provider", "", "ext"),
        ("lg", "Logging", "+ callbacks", "sdk"),
        ("db", "spend writer", "_ProxyDBLogger", "proxy"),
    ]
    st = [
        ("m", "c", "ep", "POST /v1/chat/completions"),
        ("m", "ep", "au", "Depends"),
        ("m", "au", "au", "IdentityStore.resolve(hash(key)): cache, then DB"),
        ("m", "au", "au", "validate key, common_checks, reserve budget (:3908)"),
        ("m", "ep", "pr", "base_process_llm_request(acompletion)"),
        ("m", "pr", "pr", "add_litellm_data_to_request, function_setup -> Logging obj"),
        ("m", "pr", "pl", "pre_call_hook: guardrails, rate limiters"),
        ("par", [
            ("route + call", [
                ("m", "pr", "rt", "route_request -> llm_router.acompletion"),
                ("m", "rt", "rt", "fallbacks -> retries -> make_call; pick deployment (cooldowns, RPM/TPM, strategy)"),
                ("m", "rt", "m", "litellm.acompletion(deployment params)"),
                ("m", "m", "m", "PYTHON_ONLY -> main.acompletion; run_in_executor(completion)"),
                ("m", "m", "m", "get_llm_provider, provider config, if/elif dispatch, transform_request"),
                ("m", "m", "p", "HTTP"),
                ("r", "p", "m", "response"),
                ("m", "m", "m", "transform_response, response_cost_calculator -> _hidden_params"),
                ("m", "m", "lg", "schedule async_success_handler (GLOBAL_LOGGING_WORKER)"),
            ]),
            ("during call", [
                ("m", "pr", "pl", "during_call_hook (moderation)"),
            ]),
        ]),
        ("m", "lg", "db", "async_log_success_event -> in-memory queues"),
        ("m", "pr", "pl", "post_call_success_hook"),
        ("r", "pr", "c", "response + x-litellm-response-cost, x-litellm-model-id, ..."),
        ("n", "db", "db", "update_spend job flushes to Postgres every 10s"),
    ]
    Seq(ps, st, "Flow A: chat completion through the proxy", col_min=112).render(out("flow-a-chat"))


def flow_b():
    ps = [
        ("c", "Claude Code", "", "neutral"),
        ("lz", "LazyFeature", "Middleware", "proxy"),
        ("ep", "anthropic_response", "endpoints.py:128", "proxy"),
        ("rt", "Router", "aanthropic_messages", "sdk"),
        ("ds", "messages/", "dispatch.py", "sdk"),
        ("h", "messages handler", "handler.py:469", "sdk"),
        ("rs", "aresponses", "flow C", "sdk"),
        ("ac", "acompletion", "flow A stage 5", "sdk"),
        ("ru", "_native", "Rust", "rust"),
    ]
    st = [
        ("m", "c", "lz", "POST /v1/messages (first call mounts the router)"),
        ("m", "lz", "ep", "route"),
        ("m", "ep", "ep", "stages 2-3 as in flow A"),
        ("m", "ep", "rt", "route_request"),
        ("m", "rt", "ds", "litellm.anthropic_messages(deployment)"),
        ("alt", [
            ("LITELLM_RUST on and provider is anthropic, deepseek or vertex_ai", [
                ("m", "ds", "ru", "native Messages"),
            ]),
            ("otherwise", [
                ("m", "ds", "h", "Python handler (@client)"),
                ("alt", [
                    ("native Messages config: Anthropic, Bedrock, Vertex Claude ...", [
                        ("m", "h", "h", "base_llm_http_handler.anthropic_messages_handler"),
                    ]),
                    ("provider is openai", [
                        ("m", "h", "rs", "responses_adapters/handler.py:242"),
                    ]),
                    ("any other provider", [
                        ("m", "h", "ac", "pass_through/adapters/handler.py:371"),
                    ]),
                ]),
            ]),
        ]),
    ]
    Seq(ps, st, "Flow B: Anthropic Messages through the proxy", col_min=110).render(out("flow-b-messages"))


def flow_c():
    ps = [
        ("c", "Client", "", "neutral"),
        ("ep", "responses_api", "endpoints.py:216", "proxy"),
        ("sec", "ResponsesIDSecurity", "hooks/", "proxy"),
        ("rt", "Router", "aresponses + affinity", "sdk"),
        ("rm", "responses/main.py", ":666", "sdk"),
        ("br", "chat bridge", "completion_transformation", "sdk"),
        ("sl", "SessionHandler", "session_handler.py:52", "sdk"),
        ("db", "Postgres", "LiteLLM_SpendLogs", "store"),
    ]
    st = [
        ("m", "c", "ep", "POST /v1/responses (previous_response_id?)"),
        ("m", "ep", "sec", "pre_call: decrypt id, check owner"),
        ("m", "ep", "rt", "route_request"),
        ("m", "rt", "rt", "pin deployment from the model id inside the response id"),
        ("m", "rt", "rm", "litellm.aresponses (PYTHON_ONLY)"),
        ("alt", [
            ("provider has a native Responses config", [
                ("m", "rm", "rm", "decode id to provider id, base_llm_http_handler.response_api_handler"),
            ]),
            ("no native Responses API", [
                ("m", "rm", "br", "transform to a chat request"),
                ("m", "br", "sl", "history for previous_response_id"),
                ("m", "sl", "db", "raw SQL on session_id (cold storage fallback)"),
                ("m", "br", "br", "litellm.acompletion(_skip_responses_api_bridge=True)"),
                ("r", "br", "rm", "chat response transformed to Responses"),
            ]),
        ]),
        ("r", "rm", "ep", "response, id re-encoded with model id"),
        ("m", "ep", "sec", "post_call: encrypt outgoing id"),
        ("r", "ep", "c", "response"),
    ]
    Seq(ps, st, "Flow C: Responses API through the proxy", col_min=118).render(out("flow-c-responses"))


def flow_d():
    ps = [
        ("c", "Client", "", "neutral"),
        ("ep", "anthropic_proxy_route", "llm_passthrough_endpoints:910", "proxy"),
        ("au", "user_api_key_auth", "", "proxy"),
        ("pt", "pass_through_request", "pass_through_endpoints:1089", "proxy"),
        ("pl", "ProxyLogging", "", "proxy"),
        ("p", "Provider", "", "ext"),
        ("sh", "PassThroughLogging", "success_handler.py:418", "proxy"),
        ("lg", "Logging", "-> _ProxyDBLogger", "sdk"),
    ]
    st = [
        ("m", "c", "ep", "POST /anthropic/v1/messages"),
        ("m", "ep", "au", "Depends"),
        ("m", "ep", "ep", "credentials from passthrough_endpoint_router (env / config), not the Router"),
        ("m", "ep", "pt", "create_pass_through_route(...)"),
        ("m", "pt", "pt", "new Logging(...) (:1234), no @client"),
        ("m", "pt", "pl", "pre_call_hook (guardrails, rate limits)"),
        ("m", "pt", "p", "raw httpx send"),
        ("r", "p", "pt", "response"),
        ("m", "pt", "pl", "post_call_success_hook (non-streaming)"),
        ("m", "pt", "sh", "enqueue on GLOBAL_LOGGING_WORKER"),
        ("m", "sh", "sh", "dispatch by URL; AnthropicConfig.transform_response; completion_cost"),
        ("m", "sh", "lg", "dispatch_success_handlers"),
        ("r", "pt", "c", "provider's raw response"),
    ]
    Seq(ps, st, "Flow D: provider pass-through", col_min=120).render(out("flow-d-passthrough"))


def flow_e():
    ps = [
        ("c", "MCP client", "", "neutral"),
        ("mt", "/mcp mount", "server.py:2183", "proxy"),
        ("au", "MCPRequestHandler", "user_api_key_auth_mcp", "proxy"),
        ("op", "GatewayOperations", "operations.py:3273", "proxy"),
        ("mg", "server manager", "mcp_server_manager", "proxy"),
        ("pl", "ProxyLogging", "", "proxy"),
        ("mc", "MCPClient", "experimental_mcp_client", "sdk"),
        ("u", "Upstream MCP", "", "ext"),
        ("lg", "Logging", "-> _ProxyDBLogger", "sdk"),
    ]
    st = [
        ("m", "c", "mt", "tools/call (streamable HTTP)"),
        ("m", "mt", "au", "extract auth: x-litellm key or Bearer, OAuth2, BYOK"),
        ("m", "au", "au", "user_api_key_auth, same as flow A"),
        ("m", "mt", "mt", "client allowlist, toolset scope, session manager"),
        ("m", "mt", "op", "mcp_server_tool_call"),
        ("m", "op", "op", "add_litellm_data_to_request, billing auth"),
        ("m", "op", "mg", "allowed servers (key, team, org, end user), is_tool_allowed"),
        ("m", "mg", "pl", "pre_call_tool_check -> pre_call_hook (guardrails)"),
        ("m", "mg", "mg", "rate limits"),
        ("m", "mg", "mc", "call_tool"),
        ("m", "mc", "u", "JSON-RPC"),
        ("r", "u", "mc", "result"),
        ("m", "op", "lg", "async_success_handler, MCPCostCalculator"),
        ("r", "op", "c", "result"),
    ]
    Seq(ps, st, "Flow E: MCP tool call", col_min=112).render(out("flow-e-mcp"))


def flow_f():
    ps = [
        ("app", "Application", "", "neutral"),
        ("init", "import litellm", "__init__.py", "sdk"),
        ("ds", "dispatch", "chat_completions/", "sdk"),
        ("w", "@client wrapper", "utils.py:1712", "sdk"),
        ("ch", "LLMCachingHandler", "", "sdk"),
        ("m", "main.completion", "", "sdk"),
        ("p", "Provider", "", "ext"),
        ("lg", "success_handler", "thread pool", "sdk"),
    ]
    st = [
        ("m", "app", "init", "import: globals, model_cost map, lazy registry, dispatch rebinds"),
        ("m", "app", "ds", "litellm.completion(model, messages)"),
        ("m", "ds", "w", "PYTHON_ONLY -> main.completion"),
        ("m", "w", "w", "function_setup: merge callbacks into litellm.* lists, create Logging"),
        ("m", "w", "ch", "sync_get_cache (only if litellm.cache)"),
        ("m", "w", "m", "completion"),
        ("m", "m", "p", "provider call (flow A stage 5)"),
        ("r", "p", "m", "response"),
        ("m", "w", "w", "cost into _hidden_params, sync_set_cache"),
        ("m", "w", "lg", "executor.submit(success_handler) (utils.py:1929)"),
        ("r", "w", "app", "ModelResponse"),
    ]
    Seq(ps, st, "Flow F: direct SDK call", col_min=120).render(out("flow-f-sdk"))


def flow_h():
    ps = [
        ("b", "Browser", "create_key_button", "ui"),
        ("n", "keyCreateCall", "networking.tsx:922", "ui"),
        ("ep", "generate_key_fn", "key_mgmt...:2153", "proxy"),
        ("hl", "key generation helper", ":1254", "proxy"),
        ("gk", "generate_key_helper_fn", ":4837", "proxy"),
        ("pc", "PrismaClient", "insert_data", "proxy"),
        ("db", "Postgres", "VerificationToken", "store"),
        ("hk", "event hooks", "KeyManagementEventHooks", "proxy"),
    ]
    st = [
        ("m", "b", "n", "submit form"),
        ("m", "n", "ep", "POST /key/generate (Bearer UI session)"),
        ("m", "ep", "ep", "auth, org admin, budget sanity, custom_key_generate, route perms, team and project limits"),
        ("m", "ep", "hl", "upperbound params, budget ceiling, custom policy, MCP / vector store checks, unique alias"),
        ("m", "hl", "gk", "build key"),
        ("m", "gk", "gk", "token = sk- + token_urlsafe"),
        ("m", "gk", "pc", "insert_data(table_name=key)"),
        ("m", "pc", "pc", "hash_token (SHA-256)"),
        ("m", "pc", "db", "VerificationTokenRepository.table.upsert"),
        ("m", "hl", "hk", "email, audit log, secret manager"),
        ("r", "ep", "b", "plaintext key, shown once"),
        ("div", "first use of the key"),
        ("n", "ep", "db", "user_api_key_auth -> IdentityStore._resolve_key: cache miss, combined view from Postgres, cache it"),
    ]
    Seq(ps, st, "Flow H: create a virtual key", col_min=120).render(out("flow-h-key"))


def flow_i():
    ps = [
        ("b", "Browser", "", "ui"),
        ("ep", "add_new_model", "model_mgmt...:2486", "proxy"),
        ("en", "encrypt_value_helper", "encrypt_decrypt_utils", "proxy"),
        ("repo", "ModelRepository", ".table + config sync", "proxy"),
        ("db", "Postgres", "ProxyModelTable", "store"),
        ("rp", "Redis pub/sub", "config_change", "store"),
        ("a", "this pod", "add_deployment", "proxy"),
        ("o", "other pods", "", "proxy"),
    ]
    st = [
        ("m", "b", "ep", "POST /model/new"),
        ("m", "ep", "ep", "auth, credential checks, rpm/tpm, PTU; 500 unless store_model_in_db"),
        ("m", "ep", "en", "encrypt every litellm_params string (LITELLM_SALT_KEY, else master key)"),
        ("m", "ep", "repo", "create"),
        ("m", "repo", "db", "insert"),
        ("m", "repo", "rp", "publish change"),
        ("m", "ep", "a", "add_deployment under MODEL_RECONCILE_LOCK"),
        ("m", "a", "db", "models, credentials"),
        ("m", "a", "a", "_update_llm_router: delete missing, upsert new"),
        ("m", "rp", "o", "ConfigSyncSubscriber (debounced) -> add_deployment"),
        ("n", "o", "o", "without Redis: wait for the 30s poll"),
    ]
    Seq(ps, st, "Flow I: add a model", col_min=118).render(out("flow-i-model"))


def flow_j():
    ps = [
        ("au", "auth (request N)", "", "proxy"),
        ("lg", "Logging", "", "sdk"),
        ("dl", "_ProxyDBLogger", "track_cost_callback", "proxy"),
        ("sc", "spend counters", "Redis spend:key:*", "store"),
        ("w", "DBSpendUpdateWriter", ":501", "proxy"),
        ("q", "in-memory queues", "", "proxy"),
        ("j", "update_spend job", "utils.py:7711", "proxy"),
        ("db", "Postgres", "", "store"),
        ("au2", "auth (request N+1)", "", "proxy"),
    ]
    st = [
        ("m", "au", "au", "reserve budget"),
        ("m", "lg", "lg", "response_cost_calculator"),
        ("m", "lg", "dl", "async_log_success_event"),
        ("alt", [
            ("LITELLM_COLLECTOR_ENABLED", [
                ("m", "dl", "dl", "publish to the collector sidecar"),
            ]),
            ("in process", [
                ("m", "dl", "sc", "increment counters, reconcile reservation"),
                ("m", "dl", "w", "update_database"),
                ("m", "w", "q", "spend log, entity increments, daily rows"),
            ]),
        ]),
        ("m", "j", "q", "drain (every 10s)"),
        ("opt", [
            ("use_redis_transaction_buffer", [
                ("m", "j", "j", "push to Redis, lock holder drains to DB"),
            ]),
        ]),
        ("m", "j", "db", "update_many token/user/team/org, upsert daily tables, insert SpendLogs"),
        ("m", "au2", "sc", "get_current_spend (Redis first)"),
        ("m", "au2", "au2", "max budget checks: key, team, org, project, global"),
    ]
    Seq(ps, st, "Flow J: spend and budgets", col_min=112).render(out("flow-j-spend"))


def flow_k():
    ps = [
        ("y", "config.yaml", "guardrails:", "neutral"),
        ("ui", "UI", "POST /guardrails", "ui"),
        ("gr", "guardrail registry", "guardrail_registry.py:727", "proxy"),
        ("cb", "litellm.callbacks", "", "sdk"),
        ("pl", "ProxyLogging", "per request", "proxy"),
        ("ug", "UnifiedLLMGuardrails", "", "proxy"),
        ("tr", "translation handler", "llms/*/guardrail_translation", "sdk"),
        ("v", "vendor guardrail", "CustomGuardrail", "ext"),
    ]
    st = [
        ("m", "y", "gr", "init_guardrails_v2 at startup"),
        ("m", "ui", "gr", "this pod now, others only via add_deployment"),
        ("m", "gr", "gr", "initializer for the type, else a custom class"),
        ("m", "gr", "cb", "add CustomGuardrail"),
        ("div", "per request"),
        ("m", "pl", "v", "should_run_guardrail (default_on, key/team lists, event_hook)"),
        ("alt", [
            ("class defines apply_guardrail", [
                ("m", "pl", "ug", "execute"),
                ("m", "ug", "tr", "pick handler by call type"),
                ("m", "tr", "v", "apply_guardrail(texts)"),
                ("m", "tr", "tr", "write the result back into request / response"),
            ]),
            ("native lifecycle hooks", [
                ("m", "pl", "v", "async_pre_call_hook / async_moderation_hook / async_post_call_success_hook"),
            ]),
        ]),
    ]
    Seq(ps, st, "Flow K: guardrails", col_min=120).render(out("flow-k-guardrails"))


def flow_l():
    ps = [
        ("y", "litellm_settings", "callbacks: [langfuse]", "neutral"),
        ("icp", "initialize_callbacks", "callback_utils.py:134", "proxy"),
        ("pl", "ProxyLogging", "_init_litellm_callbacks", "proxy"),
        ("bld", "build_custom_logger", "litellm_logging.py:4580", "sdk"),
        ("l", "callback lists", "litellm.*", "sdk"),
        ("lg", "Logging", "per request", "sdk"),
        ("v", "LangfusePromptManagement", "", "ext"),
    ]
    st = [
        ("m", "y", "icp", "known names stay strings, dotted paths -> get_instance_fn"),
        ("m", "icp", "l", "append"),
        ("m", "pl", "l", "add PROXY_HOOKS"),
        ("m", "pl", "bld", "replace each string with an instance (if/elif on name)"),
        ("m", "bld", "v", "construct, or reuse from _in_memory_loggers"),
        ("m", "pl", "l", "fan out into success / failure / async lists"),
        ("div", "per request"),
        ("m", "lg", "l", "read lists + per-request dynamic callbacks"),
        ("m", "lg", "v", "async_log_success_event(StandardLoggingPayload)"),
    ]
    Seq(ps, st, "Flow L: observability callback", col_min=130).render(out("flow-l-callbacks"))


def flow_g():
    startup()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (startup, hooks, lazy_bypass, flow_a, flow_b, flow_c, flow_d, flow_e, flow_f, flow_h, flow_i, flow_j, flow_k, flow_l):
        f()
