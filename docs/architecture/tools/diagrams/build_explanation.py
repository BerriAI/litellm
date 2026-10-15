import math
import os

from svglib import RED, SUB, Svg, badge, box, connect, cylinder, group, key, label_at, legend, path, text_width

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "diagrams")


def out(name):
    return os.path.join(OUT, name + ".svg")


def system_overview():
    s = Svg(1230, 530, "litellm system overview")
    s.text(20, 26, "Clients", 12, SUB, 700, "start")
    clients = [
        ("Browser", "Admin UI"),
        ("lite CLI", "proxy/client/cli"),
        ("OpenAI / Anthropic SDKs", "coding agents, VS Code ext"),
        ("MCP clients", "Claude Desktop, Cursor"),
        ("Python app", "pip install litellm"),
    ]
    ys = [44, 122, 200, 278, 350]
    cb = [box(s, 20, y, 190, 56, t, st, "neutral") for y, (t, st) in zip(ys, clients)]

    group(s, 250, 30, 570, 420, "Python package litellm   ·   2,837 .py files", "muted")
    ui = box(s, 272, 52, 130, 74, "Admin UI export", ["proxy/_experimental/out"], "ui", sub_size=9.5)
    proxy = box(s, 430, 52, 260, 88, "AI Gateway (proxy)", ["litellm/proxy · ~395k lines", "~970 routes", "+ enterprise routes and hooks (try-imported)"], "proxy")
    sdk = box(s, 430, 336, 260, 88, "SDK", ["litellm/ outside proxy · ~540k lines", "~150 providers", "Router, caching, callbacks, cost"], "sdk")
    rb = box(s, 706, 350, 100, 60, "rust_bridge", ["rollout catalog"], "sdk", sub_size=10)

    group(s, 850, 290, 190, 180, "litellm-rust   ·   66 crates", "rust")
    native = box(s, 864, 320, 162, 56, "_native extension", ["python-bridge (PyO3)"], "rust")
    rgw = box(s, 864, 396, 162, 56, "Rust gateway", ["crates/gateway (axum)"], "rust")

    pg = cylinder(s, 720, 44, 120, 64, "Postgres", ["keys, spend, config"])
    rd = cylinder(s, 870, 44, 110, 64, "Redis", ["caches, pub/sub"])
    extras = box(s, 720, 150, 150, 46, "litellm-proxy-extras", ["schema + 220 migrations"], "store", title_size=11.5, sub_size=10)
    prov = box(s, 1070, 160, 145, 300, "Outside services", ["LLM providers", "MCP servers", "guardrail vendors", "observability vendors"], "ext")

    connect(s, cb[0], "r", ui, "l", ta=0.5, tb=0.36)
    path(s, [(210, cb[1].cy), (416, cb[1].cy), (416, 120), (430, 120)])
    connect(s, ui, "r", proxy, "l", ta=0.4, tb=0.26)
    path(s, [(210, cb[2].cy), (480, cb[2].cy), (480, 140)])
    path(s, [(210, cb[3].cy), (510, cb[3].cy), (510, 140)])
    connect(s, cb[4], "r", sdk, "l", tb=0.47)
    connect(s, proxy, "b", sdk, "t", ta=0.7, tb=0.7, label="calls the SDK in-process", label_dx=6, label_anchor="start")
    connect(s, sdk, "r", rb, "l", ta=0.5)
    connect(s, rb, "r", native, "l", ta=0.4, tb=0.6)
    connect(s, proxy, "r", pg, "l", ta=0.3, tb=0.5)
    path(s, [(proxy.x + proxy.w * 0.85, proxy.y), (proxy.x + proxy.w * 0.85, 22), (925, 22), (925, rd.y)])
    connect(s, extras, "t", pg, "b", dashed=True, label="migrates", label_dx=30)
    path(s, [(sdk.x + sdk.w * 0.3, sdk.y + sdk.h), (sdk.x + sdk.w * 0.3, 490), (1142, 490), (1142, prov.y + prov.h)],
         label="provider HTTP", label_seg=1, label_dx=150)
    connect(s, native, "r", prov, "l", tb=0.55)
    connect(s, rgw, "r", prov, "l", tb=0.8)
    s.text(945, 282, "Rust gateway is meant to replace the proxy", 10.5, SUB, 400, italic=True)
    legend(s, 20, 520, [("proxy", "gateway"), ("sdk", "SDK"), ("ui", "UI"), ("rust", "Rust"), ("store", "data"), ("ext", "outside")])
    s.save(out("01-system-overview"))


def layer_stack_intended():
    s = Svg(760, 440, "Intended Python layer stack")
    layers = [
        ("L5  Proxy / AI Gateway", "litellm/proxy", "proxy"),
        ("L4  Router", "router.py · router_strategy · router_utils", "sdk"),
        ("L3  SDK entry points", "main.py · utils.py · responses/ · images/ · */dispatch.py · rust_bridge", "sdk"),
        ("L2  Providers and integrations", "llms/PROVIDER/ENDPOINT/transformation.py · integrations/*", "sdk"),
        ("L1  Core utilities", "litellm_core_utils · caching · secret_managers", "sdk"),
        ("L0  Types and data", "types · models · constants · _logging · _uuid", "store"),
    ]
    for i, (t, sub, k) in enumerate(layers):
        y = 30 + i * 62
        box(s, 40, y, 520, 48, t, [sub], k, code=True, sub_size=10.5)
        if i < len(layers) - 1:
            pass
    path(s, [(600, 30 + 5 * 62 + 24), (600, 54)], color="#30343b", marker="ahk", width=1.6)
    label_at(s, 608, 200, "may import\nanything below", anchor="start", bg=False)
    rules = [
        "import litellm loads no proxy module (test_lazy_imports, test_proxy_types_import)",
        "five public verbs go through */dispatch.py, never main.py directly (TID251)",
    ]
    s.text(40, 412, "Extra rules on top", 10.5, SUB, 700, "start")
    s.text(150, 412, rules[0], 10.5, SUB, 400, "start")
    s.text(150, 426, rules[1], 10.5, SUB, 400, "start")
    s.save(out("02-layer-stack-intended"))


def layer_edges_actual():
    s = Svg(1060, 600, "Upward module-level imports between layers")
    X0, X1 = 230, 730
    bands = {}
    order = ["proxy", "router", "main", "l2", "core", "types"]
    titles = {
        "proxy": ("L5 proxy", "litellm/proxy", "proxy"),
        "router": ("L4 router", "router.py, router_strategy, router_utils", "sdk"),
        "main": ("L3 main / utils", "main.py, utils.py (10,627 lines)", "sdk"),
        "core": ("L1 core_utils", "litellm_core_utils", "sdk"),
        "types": ("L0 types", "types, models, constants", "store"),
    }
    for i, k in enumerate(order):
        y = 50 + i * 80
        if k == "l2":
            bands["integrations"] = box(s, X0, y, 230, 54, "L2 integrations", ["integrations/*"], "sdk", code=True, sub_size=10.5)
            bands["llms"] = box(s, X0 + 270, y, 230, 54, "L2 llms", ["llms/* providers"], "sdk", code=True, sub_size=10.5)
        else:
            t, sub, kind = titles[k]
            bands[k] = box(s, X0, y, X1 - X0, 54, t, [sub], kind, code=True, sub_size=10.5)
    for a, b in [("types", "core"), ("core", "main"), ("main", "router"), ("router", "proxy")]:
        pass

    def w(n):
        return 1.3 + 1.7 * math.log10(n)

    right = [
        ("types", "core", 10), ("core", "llms", 14), ("llms", "main", 116), ("core", "main", 4), ("llms", "router", 4),
        ("core", "router", 2), ("llms", "proxy", 2), ("core", "proxy", 2), ("types", "proxy", 1),
    ]
    left = [("core", "integrations", 94), ("integrations", "main", 3), ("integrations", "router", 1)]
    slots_r = {}
    slots_l = {}

    def slot(table, band, role):
        table.setdefault(band, []).append(role)
        return len(table[band]) - 1

    plan_r = [(src, dst, n, slot(slots_r, src, "out"), slot(slots_r, dst, "in")) for src, dst, n in right]
    plan_l = [(src, dst, n, slot(slots_l, src, "out"), slot(slots_l, dst, "in")) for src, dst, n in left]

    def ypos(table, band, i):
        b = bands[band]
        k = len(table[band])
        return b.y + b.h * (i + 1) / (k + 1)

    for li, (src, dst, n, so, di) in enumerate(plan_r):
        lx = X1 + 28 + li * 30
        ys = ypos(slots_r, src, so)
        yd = ypos(slots_r, dst, di)
        xs = bands[src].x + bands[src].w
        xd = bands[dst].x + bands[dst].w
        path(s, [(xs, ys), (lx, ys), (lx, yd), (xd + 2, yd)], color=RED, dashed=n < 10, marker="ahr", width=w(n))
        label_at(s, lx + 4, yd - 6 if n != 116 else yd - 7, str(n), size=12 if n >= 10 else 11, color=RED, weight=700, anchor="start")
    for li, (src, dst, n, so, di) in enumerate(plan_l):
        lx = X0 - 30 - li * 30
        ys = ypos(slots_l, src, so)
        yd = ypos(slots_l, dst, di)
        path(s, [(X0, ys), (lx, ys), (lx, yd), (X0 - 2, yd)], color=RED, dashed=n < 10, marker="ahr", width=w(n))
        label_at(s, lx - 4, yd - 6, str(n), size=12 if n >= 10 else 11, color=RED, weight=700, anchor="end")

    notes = [
        (20, 382, "94 = litellm_logging.py 48", "+ custom_logger_registry.py 40", "+ 6 others"),
        (1010 - 260, 600 - 70, "", "", ""),
    ]
    s.text(20, 456, "core_utils -> integrations", 10.5, SUB, 600, "start")
    s.text(20, 470, "litellm_logging.py 48", 10.5, SUB, 400, "start")
    s.text(20, 484, "custom_logger_registry.py 40", 10.5, SUB, 400, "start")
    s.text(20, 506, "llms -> main / utils", 10.5, SUB, 600, "start")
    s.text(20, 520, "116 provider modules import litellm.utils", 10.5, SUB, 400, "start")
    s.text(20, 30, "Module-level imports that point UP the stack, counted on main 13118948. Arrow width scales with count.", 11.5, SUB, 500, "start")
    legend(s, 230, 575, [("redline", "upward import (dashed when fewer than 10)")])
    s.text(1040, 580, "Not shown: 395 bare `import litellm` and 1,029 proxy_server global reads", 10.5, SUB, 400, "end")
    s.save(out("02-layer-edges-actual"))



def pipeline_stages():
    s = Svg(1150, 330, "The seven data-plane stages")
    stages = [
        ("1 Endpoint", ["proxy_server.py or", "*_endpoints/*.py", "parse body, stamp ids"], "proxy"),
        ("2 Auth", ["user_api_key_auth", "key, team, user checks", "budget reservation"], "proxy"),
        ("3 Pre-call", ["add_litellm_data_to_request", "function_setup -> Logging", "pre_call_hook"], "proxy"),
        ("4 Route", ["route_request", "Router: deployment,", "retries, fallbacks"], "sdk"),
        ("5 SDK + provider", ["*/dispatch.py -> main.py", "provider Config", "HTTP call"], "sdk"),
        ("6 Log + cost", ["@client wrapper", "response_cost_calculator", "async_success_handler"], "sdk"),
        ("7 Post-call", ["post_call_success_hook", "output guardrails", "x-litellm-* headers"], "proxy"),
    ]
    W, G, X, Y = 142, 20, 20, 92
    bx = []
    for i, (t, sub, k) in enumerate(stages):
        bx.append(box(s, X + i * (W + G), Y, W, 96, t, sub, k, sub_size=10, title_size=12.5))
    for a, b in zip(bx, bx[1:]):
        connect(s, a, "r", b, "l")
    bands = [(0, 2, "proxy owns identity and policy", "proxy"), (3, 3, "Router owns availability", "sdk"),
             (4, 4, "SDK owns translation", "sdk"), (5, 5, "shared side channel", "sdk"), (6, 6, "proxy", "proxy")]
    for a, b, t, k in bands:
        x1 = bx[a].x
        x2 = bx[b].x + W
        col = "#3b6fd8" if k == "proxy" else "#2e8b57"
        s.add(f"<line x1='{x1}' y1='64' x2='{x2}' y2='64' stroke='{col}' stroke-width='3' stroke-linecap='round'/>")
        s.text((x1 + x2) / 2, 54, t, 11, col, 600)
    s.text(20, 26, "Every LLM call through the proxy is meant to pass these stages in order. Direct SDK use is stages 5 and 6 only.", 11.5, SUB, 500, "start")
    path(s, [(bx[2].cx, Y + 96), (bx[2].cx, 232), (bx[6].cx, 232), (bx[6].cx, Y + 96)], label="during_call_hook runs in parallel with stages 4 and 5",
         dashed=True, label_seg=1, label_dy=14)
    pg = cylinder(s, 1000, 262, 130, 54, "Postgres", [])
    path(s, [(bx[5].cx + 20, Y + 96), (bx[5].cx + 20, 289), (pg.x, 289)], dashed=True)
    label_at(s, bx[5].cx + 10, 294, "_ProxyDBLogger -> in-memory queues -> update_spend job (10s)", anchor="end")
    s.save(out("03-pipeline-stages"))


def surface_bridges():
    s = Svg(1060, 700, "How API surfaces reach stage 5")
    group(s, 226, 44, 340, 330, "", "muted")
    s.text(396, 64, "stages 1-4 shared by all three", 11, SUB, 700)
    s.text(396, 78, "endpoint, auth, pre-call hooks, Router", 10.5, SUB, 400)
    cc = box(s, 20, 96, 180, 56, "/v1/chat/completions", ["proxy_server.py:12305"], "neutral", code=True, sub_size=10)
    rs = box(s, 20, 206, 180, 56, "/v1/responses", ["response_api_endpoints"], "neutral", code=True, sub_size=10)
    ms = box(s, 20, 316, 180, 56, "/v1/messages", ["anthropic_endpoints (lazy)"], "neutral", code=True, sub_size=10)
    oc = box(s, 20, 426, 180, 50, "/v1/ocr", ["ocr_endpoints"], "neutral", code=True, sub_size=10)
    chat = box(s, 600, 92, 250, 64, "main.completion", ["~60 _complete_PROVIDER helpers", "47 use BaseLLMHTTPHandler"], "sdk", sub_size=10)
    resp = box(s, 600, 202, 250, 64, "responses/main.py", ["native BaseResponsesAPIConfig", "openai, azure, ..."], "sdk", sub_size=10)
    msg = box(s, 600, 312, 250, 64, "anthropic_messages_handler", ["native Messages config", "Anthropic, Bedrock, Vertex Claude, ..."], "sdk", sub_size=10)
    rust = box(s, 600, 422, 250, 58, "rust_bridge _native", ["OCR always, Bedrock STT always", "Messages when LITELLM_RUST is on"], "rust", sub_size=10)
    prov = box(s, 930, 92, 110, 388, "Provider HTTP", [], "ext")
    for a, b in ((cc, chat), (rs, resp), (ms, msg), (oc, rust)):
        connect(s, a, "r", b, "l")
    for b in (chat, resp, msg, rust):
        connect(s, b, "r", prov, "l", tb=(b.cy - prov.y) / prov.h)
    path(s, [(resp.x + 190, resp.y), (resp.x + 190, chat.y + chat.h)])
    badge(s, resp.x + 190, (resp.y + chat.y + chat.h) / 2, 1)
    path(s, [(chat.x + 60, chat.y + chat.h), (chat.x + 60, resp.y)], dashed=True)
    badge(s, chat.x + 60, (resp.y + chat.y + chat.h) / 2, 2)
    path(s, [(msg.x + 190, msg.y), (msg.x + 190, resp.y + resp.h)])
    badge(s, msg.x + 190, (msg.y + resp.y + resp.h) / 2, 3)
    path(s, [(msg.x + msg.w, msg.y + 12), (888, msg.y + 12), (888, chat.y + chat.h - 12), (chat.x + chat.w, chat.y + chat.h - 12)])
    badge(s, 888, (msg.y + chat.y + chat.h) / 2, 4)
    path(s, [(ms.cx, ms.y + ms.h), (ms.cx, 401), (570, 401), (570, rust.y + 14), (rust.x, rust.y + 14)], dashed=True)
    badge(s, 400, 401, 5)
    key(s, 226, 512, [
        "Responses -> chat bridge when the provider has no native Responses API (responses/litellm_completion_transformation)",
        "Chat -> Responses when the model's mode is responses (responses_api_bridge)",
        "Messages -> Responses when the provider is openai (pass_through/responses_adapters)",
        "Messages -> chat for every other non-native provider (pass_through/adapters)",
        "Rust Messages, opt-in for anthropic, deepseek and vertex_ai only",
    ])
    group(s, 20, 616, 1020, 76, "", "bad", dashed=True)
    s.text(36, 636, "Own pipelines", 11.5, "#8f1f1f", 700, "start")
    s.text(36, 652, "no Router, own Logging", 10.5, "#8f1f1f", 400, "start")
    s.text(36, 666, "object and cost logic", 10.5, "#8f1f1f", 400, "start")
    pt = box(s, 190, 626, 190, 56, "/anthropic/*, /vertex_ai/*", ["29 prefixes (lazy)"], "neutral", code=True, sub_size=10, title_size=11.5)
    ptr = box(s, 410, 626, 250, 56, "pass_through_request", ["raw httpx, re-parses response for cost"], "proxy", sub_size=10)
    mc = box(s, 690, 626, 100, 56, "/mcp", ["ASGI (lazy)"], "neutral", code=True, sub_size=10)
    mco = box(s, 820, 626, 210, 56, "GatewayOperations", ["own permissions, MCPClient"], "proxy", sub_size=10)
    connect(s, pt, "r", ptr, "l")
    connect(s, mc, "r", mco, "l")
    s.save(out("03-surface-bridges"))


def config_sources():
    s = Svg(1000, 410, "Where proxy configuration comes from")
    srcs = [
        box(s, 20, 40, 230, 62, "Environment", ["LITELLM_MASTER_KEY, DATABASE_URL,", "REDIS_*, LITELLM_SALT_KEY"], "neutral", sub_size=10),
        box(s, 20, 122, 230, 62, "config.yaml", ["file or bucket, with includes"], "neutral", sub_size=10),
        box(s, 20, 204, 230, 62, "Secret managers", ["os.environ/ and oidc/ references", "secret_managers/main.py:196"], "neutral", sub_size=10),
    ]
    pg = cylinder(s, 50, 300, 170, 80, "Postgres", ["ProxyModelTable, Config,", "guardrails, MCP ..."])
    lc = box(s, 360, 140, 210, 110, "ProxyConfig.load_config", ["proxy_server.py:6471", "parses YAML twice", "(CLI, then lifespan)"], "proxy", sub_size=10)
    outs = [
        ("litellm.* module globals", "setattr(litellm, key, value), no schema"),
        ("proxy_server globals", "general_settings (SettingsStore), master_key ..."),
        ("Router", "model_list + router_settings, :7246"),
        ("litellm.callbacks", "loggers, guardrails, PROXY_HOOKS"),
        ("litellm.cache, redis_usage_cache", "init_cache, :6206"),
    ]
    ob = [box(s, 660, 30 + i * 76, 320, 60, t, [st], "sdk" if i in (0, 2, 3, 4) else "proxy", sub_size=10) for i, (t, st) in enumerate(outs)]
    for b in srcs:
        connect(s, b, "r", lc, "l", via="hvh", tb=0.2 + 0.3 * srcs.index(b))
    path(s, [(pg.x + pg.w, pg.cy), (465, pg.cy), (465, lc.y + lc.h)], color=RED, dashed=True, marker="ahr",
         label="only via add_deployment (30s poll),\nnever at boot", label_seg=0, label_dy=-14, label_color=RED)
    for b in ob:
        connect(s, lc, "r", b, "l", via="hvh", ta=0.5)
    ob0 = ob[0]
    s.add(f"<rect x='{ob0.x + ob0.w - 64}' y='{ob0.y + 6}' width='58' height='16' rx='8' fill='#fde8e8' stroke='{RED}'/>")
    s.text(ob0.x + ob0.w - 35, ob0.y + 18, "gap G-C1", 9.5, RED, 700)
    s.save(out("04-config-sources"))


def state_in_pod():
    s = Svg(1060, 470, "Where state lives in one proxy worker")
    group(s, 20, 30, 770, 420, "One proxy worker process", "muted")
    group(s, 40, 62, 330, 300, "litellm module   ·   419 names", "sdk")
    rows_l = [("callbacks, success_callback,", "_async_success_callback, ..."), ("cache: Cache", "LLM response cache"),
              ("drop_params, request_timeout,", "model_cost, any litellm_settings key")]
    lb = [box(s, 56, 92 + i * 86, 298, 70, t, [st], "neutral", title_size=11.5, sub_size=10) for i, (t, st) in enumerate(rows_l)]
    group(s, 400, 62, 370, 300, "proxy_server module globals   ·   1,029 imports from 230 files", "proxy")
    rows_r = ["llm_router: Router (+ Router.cache)", "prisma_client", "general_settings: SettingsStore",
              "master_key, premium_user", "user_api_key_cache: UserApiKeyCache", "spend_counter_cache: DualCache",
              "proxy_logging_obj: ProxyLogging", "redis_usage_cache: RedisCache"]
    rb = [box(s, 416, 90 + i * 33, 338, 27, t, [], "neutral", title_size=11, radius=5) for i, t in enumerate(rows_r)]
    cl = box(s, 40, 380, 730, 52, "Class-level state", ["PrismaClient.spend_log_transactions (proxy/utils.py:4664) is shared by every instance"], "bad", title_size=11.5, sub_size=10)
    pg = cylinder(s, 860, 96, 150, 80, "Postgres", ["source of truth"])
    rd = cylinder(s, 860, 290, 150, 80, "Redis", ["shared across pods"])
    path(s, [(rb[1].x + rb[1].w, rb[1].cy), (pg.x, rb[1].cy)])
    path(s, [(rb[4].x + rb[4].w, rb[4].cy), (820, rb[4].cy), (820, 306), (860, 306)], dashed=True)
    path(s, [(rb[5].x + rb[5].w, rb[5].cy), (838, rb[5].cy), (838, 320), (860, 320)])
    path(s, [(rb[7].x + rb[7].w, rb[7].cy), (860, rb[7].cy)])
    path(s, [(lb[1].x + lb[1].w, lb[1].cy), (385, lb[1].cy), (385, 46), (1040, 46), (1040, 340), (1012, 340)], dashed=True)
    s.text(935, 392, "dashed = only when configured:", 10, SUB, 600)
    s.text(935, 406, "enable_redis_auth_cache for the auth cache,", 10, SUB)
    s.text(935, 420, "a Redis backend for litellm.cache", 10, SUB)
    s.save(out("04-state-in-pod"))


def change_propagation():
    s = Svg(1080, 400, "How a control-plane change reaches other pods")
    ui = box(s, 20, 60, 150, 60, "Admin UI", ["POST /model/new"], "ui", sub_size=10)
    a = box(s, 230, 40, 220, 100, "pod A", ["encrypt litellm_params", "ModelRepository.table.create", "add_deployment now"], "proxy", sub_size=10)
    pg = cylinder(s, 530, 36, 200, 70, "Postgres", ["LiteLLM_ProxyModelTable"])
    rd = cylinder(s, 530, 156, 200, 70, "Redis pub/sub", ["litellm_proxy.config_change"])
    b = box(s, 850, 50, 215, 170, "pod B", ["ConfigSyncSubscriber (debounced)", "or add_deployment_job poll (30s)", "then add_deployment", "then Router upsert / delete"], "proxy", sub_size=10)
    connect(s, ui, "r", a, "l", tb=0.4)
    connect(s, a, "r", pg, "l", ta=0.3, tb=0.43, label="write")
    connect(s, a, "r", rd, "l", via="hvh", ta=0.75, label="publish", label_seg=2, label_dy=-6)
    connect(s, rd, "r", b, "l", tb=(rd.cy - b.y) / b.h, label="event")
    connect(s, pg, "r", b, "l", tb=(pg.cy - b.y) / b.h, dashed=True, label="poll every 30s,\nstore_model_in_db only", label_dy=16)
    k = box(s, 20, 280, 190, 60, "key update / delete", ["key_management_endpoints"], "ui", sub_size=10)
    inv = box(s, 270, 280, 230, 60, "delete_cache_key_object", ["auth_checks.py:3315"], "proxy", sub_size=10, code=True)
    rd2 = cylinder(s, 560, 275, 190, 70, "Redis pub/sub", ["auth cache invalidation"])
    b2 = box(s, 850, 280, 215, 60, "pod B drops cached key", [], "proxy")
    connect(s, k, "r", inv, "l")
    connect(s, inv, "r", rd2, "l")
    connect(s, rd2, "r", b2, "l")
    s.text(540, 382, "Without Redis both channels are gone. Model changes fall back to the 30s poll, key revocations wait for the cache TTL.", 11, RED, 500)
    s.save(out("04-change-propagation"))


def callback_sources():
    s = Svg(1090, 460, "How plugins get into litellm.callbacks")
    srcs = [
        ("litellm_settings.callbacks", "[langfuse, datadog, ...]"),
        ("dotted path / s3:// / gcs://", "get_instance_fn"),
        ("guardrails: (YAML v2)", "init_guardrails_v2"),
        ("DB guardrails", "via add_deployment"),
        ("litellm_settings.success_callback", "legacy string or function"),
        ("PROXY_HOOKS", "proxy/hooks/__init__.py"),
        ("enterprise hooks", "try-imported"),
        ("cost_tracking()", "_ProxyDBLogger, only with a DB"),
    ]
    sb = [box(s, 20, 30 + i * 52, 250, 42, t, [st], "neutral", code=True, title_size=11, sub_size=9.5, radius=6) for i, (t, st) in enumerate(srcs)]
    icp = box(s, 340, 24, 240, 62, "initialize_callbacks_on_proxy", ["callback_utils.py:134 · names stay strings"], "proxy", sub_size=9.5, title_size=12)
    bld = box(s, 340, 104, 240, 62, "build_custom_logger_compatible_class", ["litellm_logging.py:4580 · if/elif on name"], "bad", sub_size=9.5, title_size=11)
    gr = box(s, 340, 178, 240, 62, "guardrail registry", ["guardrail_registry.py:727 · auto-discovers", "guardrail_hooks/VENDOR initializers"], "proxy", sub_size=9.5, title_size=12)
    cbl = box(s, 660, 24, 170, 412, "litellm.callbacks", ["one list of", "CustomLogger and", "CustomGuardrail", "instances"], "sdk", sub_size=10.5)
    fan = box(s, 900, 130, 175, 200, "fan-out lists", ["success_callback", "failure_callback", "_async_success_callback", "_async_failure_callback", "input_callback"], "sdk", sub_size=10, code=True)
    connect(s, sb[0], "r", icp, "l", tb=(sb[0].cy - icp.y) / icp.h)
    connect(s, sb[1], "r", icp, "l", via="hvh", tb=0.8)
    connect(s, icp, "b", bld, "t", label="strings", label_dx=30, label_dy=6)
    connect(s, bld, "r", cbl, "l", tb=(bld.cy - cbl.y) / cbl.h)
    connect(s, sb[2], "r", gr, "l", via="hvh", tb=0.3)
    connect(s, sb[3], "r", gr, "l", via="hvh", tb=0.75)
    connect(s, gr, "r", cbl, "l", tb=(gr.cy - cbl.y) / cbl.h)
    for i in (4, 5, 6, 7):
        path(s, [(sb[i].x + sb[i].w, sb[i].cy), (cbl.x, sb[i].cy)], dashed=(i == 4))
    s.text(465, 270, "added directly, no registry", 10.5, SUB, 400, italic=True)
    connect(s, cbl, "r", fan, "l", label="ProxyLogging\nstartup_event", label_dy=-12)
    s.save(out("05-callback-sources"))


def two_hosts():
    s = Svg(1060, 480, "Two Rust hosts on one core")
    group(s, 20, 40, 430, 420, "Python package litellm", "muted")
    proxy = box(s, 40, 72, 390, 46, "Python proxy", ["litellm/proxy"], "proxy", sub_size=10)
    api = box(s, 40, 146, 390, 50, "Public SDK calls", ["litellm.completion, amessages, ocr, ..."], "sdk", sub_size=10)
    disp = box(s, 40, 222, 390, 50, "Dispatch modules", ["chat_completions, messages, responses, ocr, embeddings"], "sdk", sub_size=10)
    cat = box(s, 40, 300, 200, 60, "rust_bridge/catalog.py", ["rollout rules"], "sdk", sub_size=10)
    py = box(s, 40, 388, 200, 56, "Python implementation", ["main.py, llms/*"], "sdk", sub_size=10)
    connect(s, proxy, "b", api, "t", label="calls the SDK", label_dx=50, label_dy=4)
    connect(s, api, "b", disp, "t")
    connect(s, disp, "b", cat, "t", ta=0.18, tb=0.35)
    connect(s, cat, "b", py, "t", ta=0.35, tb=0.35, label="PYTHON_ONLY or fallback", label_dx=8, label_anchor="start", label_dy=4)
    group(s, 480, 40, 560, 420, "litellm-rust workspace", "rust")
    br = box(s, 500, 72, 230, 56, "python-bridge", ["cdylib _native (PyO3)"], "rust", sub_size=10)
    gw = box(s, 790, 72, 230, 56, "gateway binary", ["axum server, own Dockerfile"], "rust", sub_size=10)
    gwi = box(s, 790, 160, 230, 52, "gateway-inference", ["routes, parsing, error envelopes"], "rust", sub_size=10)
    leg = box(s, 500, 160, 230, 52, "callbacks-legacy-python", ["borrowed Python Logging"], "rust", sub_size=10)
    inf = box(s, 560, 244, 400, 56, "inference + inference-chat / messages / responses / ocr / transcription", ["no pricing, budgets or rate limits here"], "rust", title_size=11, sub_size=10)
    llms = box(s, 560, 330, 190, 52, "llms", ["provider transformations"], "rust", sub_size=10)
    host = box(s, 770, 330, 190, 52, "host, host-native/http/python", ["resumable call machine"], "rust", sub_size=10, title_size=11)
    path(s, [(cat.x + cat.w, cat.cy), (462, cat.cy), (462, br.cy), (br.x, br.cy)], label="RUST_REQUIRED or opted in", label_seg=0, label_dx=10)
    connect(s, br, "b", leg, "t")
    connect(s, br, "b", inf, "t", ta=0.8, tb=0.2, via="vhv")
    connect(s, gw, "b", gwi, "t")
    connect(s, gwi, "b", inf, "t", tb=0.8, via="vhv")
    connect(s, inf, "b", llms, "t", tb=0.5, ta=0.24)
    connect(s, inf, "b", host, "t", ta=0.76)
    path(s, [(gw.cx, gw.y), (gw.cx, 24), (235, 24), (235, proxy.y)], color=RED, dashed=True, marker="ahr", label="intended to replace", label_seg=1, label_color=RED)
    s.text(760, 434, "Python owns rollout and fallback. Rust returns errors and never falls back by itself.", 10.5, SUB, 400, italic=True)
    s.save(out("06-two-hosts"))


def crate_layers():
    s = Svg(1020, 540, "litellm-rust crate layers")
    layers = [
        ("L8", "gateway binary", "rust"),
        ("L7 boundaries", "gateway-inference · gateway-auth · gateway-mcp · gateway-ui · python-bridge · callbacks-legacy-python", "rust"),
        ("L6", "router (60 lines: model_name -> Deployment map)", "rust"),
        ("L5 orchestration", "inference · inference-chat · inference-messages · inference-responses · inference-ocr · inference-transcription", "rust"),
        ("L4 providers", "llms (~22.6k lines)", "rust"),
        ("L3 host runtime", "host · host-native · host-http · host-python", "rust"),
        ("L2 adapters", "auth-* · secrets-* · cache-* · cache-response · storage-clickhouse · spend-clickhouse · token-counter", "rust"),
        ("L1 utilities", "core-utils · http · config · model-catalog", "rust"),
        ("L0 leaves", "auth-types · llms-types · router-types · secrets-types · python-compat · coroutine · cache · tracing · framer · db · cost", "store"),
    ]
    for i, (t, sub, k) in enumerate(layers):
        box(s, 20, 30 + i * 56, 700, 44, t, [sub], k, code=True, sub_size=9.5, title_size=11.5)
    notes = [
        ("Stated rule", "dependencies only point down:", "gateway-inference -> inference-FMT -> inference"),
        ("Bends", "llms depends on host and python-compat", "router depends on inference + inference-messages"),
        ("Not built yet", "gateway-management exists, not mounted", "db is a 13-line error enum, no queries"),
    ]
    for i, (h, a, b) in enumerate(notes):
        y = 40 + i * 120
        kind = "neutral" if i == 0 else "bad"
        box(s, 740, y, 265, 96, h, [a, b], kind, sub_size=9.6, title_size=12)
    s.save(out("06-crate-layers"))


def rollout_decision():
    s = Svg(1000, 320, "How a public call picks Python or Rust")
    call = box(s, 20, 120, 170, 60, "litellm.amessages(...)", ["public call"], "neutral", code=True, sub_size=10)
    d = box(s, 230, 120, 150, 60, "messages/dispatch.py", ["PublicDispatch"], "sdk", code=True, sub_size=10)
    r = box(s, 410, 110, 160, 80, "catalog.rollout", ["(route, provider)", "first matching rule wins"], "sdk", sub_size=10)
    n = box(s, 820, 20, 160, 56, "_native", ["no Python fallback"], "rust", sub_size=10)
    sw = box(s, 660, 120, 130, 60, "LITELLM_RUST on?", ["or litellm.rust(True)"], "neutral", sub_size=10, title_size=12)
    n2 = box(s, 820, 120, 160, 60, "_native", ["error after start is terminal"], "rust", sub_size=9.5)
    p = box(s, 820, 236, 160, 60, "Python handler", ["main.py / llms/*"], "sdk", sub_size=10)
    connect(s, call, "r", d, "l")
    connect(s, d, "r", r, "l")
    path(s, [(r.cx, r.y), (r.cx, 48), (n.x, 48)], label="RUST_REQUIRED", label_seg=1)
    connect(s, r, "r", sw, "l", label="RUST_OPT_IN")
    connect(s, sw, "r", n2, "l", label="yes")
    path(s, [(sw.cx, sw.y + sw.h), (sw.cx, 252), (p.x, 252)], label="no", label_seg=0, label_dx=12)
    path(s, [(r.cx, r.y + r.h), (r.cx, 282), (p.x, 282)], label="PYTHON_ONLY (and anything unmatched)", label_seg=1, label_dy=14)
    connect(s, n2, "b", p, "t", dashed=True, label="RustBridgeDeclined\nbefore execution", label_dx=-10, label_anchor="end", label_dy=-4)
    s.save(out("06-rollout-decision"))


def componentized():
    s = Svg(1080, 470, "Componentized deploy (helm/litellm)")
    cl = box(s, 20, 70, 160, 56, "Clients, SDKs,", ["coding agents"], "neutral")
    br = box(s, 20, 200, 160, 56, "Browser", [], "neutral")
    group(s, 220, 30, 840, 420, "helm/litellm  or  terraform/litellm", "muted")
    ing = box(s, 240, 110, 150, 120, "Ingress", ["ingress.yaml:90-108", "path routing"], "neutral", sub_size=10)
    gw = box(s, 470, 50, 280, 86, "litellm-gateway :4000", ["gateway/main.py", "full app, trimmed to gateway allowlist"], "proxy", sub_size=10)
    be = box(s, 470, 160, 280, 86, "litellm-backend :4001", ["backend/main.py", "full app, trimmed to backend allowlist"], "proxy", sub_size=10)
    ui = box(s, 470, 270, 280, 60, "litellm-ui :3000", ["nginx + Next.js export"], "ui", sub_size=10)
    mig = box(s, 470, 356, 280, 60, "litellm-migrations job", ["migrations/run.py"], "store", sub_size=10)
    pg = cylinder(s, 860, 168, 150, 70, "Postgres", [])
    rd = cylinder(s, 860, 58, 150, 70, "Redis", [])
    connect(s, cl, "r", ing, "l", tb=0.3)
    connect(s, br, "r", ing, "l", tb=0.8)
    connect(s, ing, "r", gw, "l", via="hvh", ta=0.25, label="data-plane prefixes", label_seg=2, label_dx=-10)
    connect(s, ing, "r", be, "l", via="hvh", ta=0.55, label="everything else", label_seg=2, label_dx=-10)
    connect(s, ing, "r", ui, "l", via="hvh", ta=0.85, label="/ui", label_seg=2)
    connect(s, ui, "t", be, "b", label="management API", label_dx=50)
    connect(s, gw, "r", rd, "l", ta=(rd.cy - gw.y) / gw.h)
    connect(s, be, "r", pg, "l", ta=(pg.cy - be.y) / be.h)
    s.text(935, 262, "both pods open both stores", 10.5, SUB, 400, italic=True)
    connect(s, mig, "r", pg, "b", via="hv")
    s.add(f"<rect x='472' y='138' width='276' height='20' rx='4' fill='#fde8e8'/>")
    s.text(610, 152, "lazy routes register after the trim, on both pods", 10.5, RED, 700)
    s.save(out("07-componentized"))
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    system_overview()
    layer_stack_intended()
    layer_edges_actual()
    pipeline_stages()
    surface_bridges()
    config_sources()
    state_in_pod()
    change_propagation()
    callback_sources()
    two_hosts()
    crate_layers()
    rollout_decision()
    componentized()
