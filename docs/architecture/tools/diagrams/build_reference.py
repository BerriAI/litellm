import os
import re

from svglib import INK, PALETTE, RED, SUB, Svg, box, label_at, legend, path, text_width

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "..", "diagrams")
MATRIX_MD = os.path.join(HERE, "..", "..", "reference", "flow-subsystem-matrix.md")


def out(name):
    return os.path.join(OUT, name + ".svg")


SURFACES = [
    ("Python SDK", "litellm.completion, ...", "sdk", "litellm/__init__.py + */dispatch.py", "main.py, responses/main.py, */main.py", False),
    ("SDK Router", "litellm.Router", "sdk", "router.py", "fallbacks, retries -> main.py", False),
    ("VS Code extension", "calls OpenAI + Management API", "neutral", None, None, False),
    ("OpenAI-compatible API", "/v1/chat, /v1/embeddings, ...", "proxy", "proxy_server.py + proxy/*_endpoints/*", "83 inline routes, 64 include_router -> Router", False),
    ("Responses API", "/v1/responses", "proxy", "proxy/response_api_endpoints", "-> Router -> responses/main.py", False),
    ("Anthropic API", "/v1/messages", "proxy", "proxy/anthropic_endpoints", "-> Router -> messages dispatch", True),
    ("Provider pass-through", "/anthropic/*, /vertex_ai/*, ...", "proxy", "proxy/pass_through_endpoints", "raw forward, own logging and cost", True),
    ("MCP gateway", "/mcp, /{server}/mcp, /mcp-rest", "proxy", "proxy/_experimental/mcp_server", "own auth call, MCP client", True),
    ("Agents and A2A", "/v1/agents, /a2a/*", "proxy", "proxy/agent_endpoints, a2a", "", True),
    ("Management API", "/key, /team, /user, /model, /spend", "proxy", "proxy/management_endpoints", "spend_tracking, ui_crud_endpoints, proxy_server.py", False),
    ("Admin UI", "/ui, calls Management API", "ui", "ui/litellm-dashboard", "exported to proxy/_experimental/out", False),
    ("lite CLI", "calls Management + OpenAI API", "proxy", "proxy/client/cli", "", False),
    ("litellm CLI", "litellm --config ...", "proxy", "proxy/proxy_cli.py", "-> uvicorn proxy_server:app", False),
    ("Componentized deploy", "helm/litellm, terraform", "store", "gateway/main.py, backend/main.py", "full app trimmed to an allowlist", False),
    ("Rust gateway", "standalone binary", "rust", "litellm-rust/crates/gateway", "axum, own Dockerfile", False),
]

WHO = [
    ("Python developer", (0, 1, 2)),
    ("App / OpenAI SDK client", (3, 4)),
    ("Claude Code / Anthropic SDK", (5, 6)),
    ("MCP client / agent", (7, 8)),
    ("Proxy admin", (9, 10, 11)),
    ("Operator / platform team", (12, 13, 14)),
]


def surface_map():
    top, row_h, bh = 64, 50, 40
    height = top + len(SURFACES) * row_h + 56
    s = Svg(1170, height, "Product surfaces mapped to entry code")
    s.text(20, 36, "Who", 12, SUB, 700, "start")
    s.text(300, 36, "Product surface", 12, SUB, 700, "start")
    s.text(700, 36, "Entry code (litellm/ unless noted)", 12, SUB, 700, "start")

    def row_y(i):
        return top + i * row_h

    surf = [
        box(s, 300, row_y(i), 240, bh, title, [sub], kind, title_size=12.5, sub_size=10)
        for i, (title, sub, kind, *_rest) in enumerate(SURFACES)
    ]
    for i, (_t, _s, kind, code, note, lazy) in enumerate(SURFACES):
        if code is None:
            continue
        code_kind = "ui" if kind == "ui" else ("rust" if kind == "rust" else ("sdk" if kind == "sdk" else "neutral"))
        b = box(s, 700, row_y(i), 450, bh, "", [code] + ([note] if note else []), code_kind, code=True, sub_size=10.5)
        if lazy:
            path(s, [(540, row_y(i) + bh / 2), (b.x, row_y(i) + bh / 2)], color=RED, dashed=True, marker="ahr")
            label_at(s, 620, row_y(i) + bh / 2 + 4, "lazy mount", 10, RED, weight=600)
        else:
            path(s, [(540, row_y(i) + bh / 2), (b.x, row_y(i) + bh / 2)])

    for title, rows in WHO:
        y0, y1 = row_y(rows[0]), row_y(rows[-1]) + bh
        cy = (y0 + y1) / 2
        w = box(s, 20, cy - 22, 200, 44, title, (), "neutral", title_size=12)
        for r in rows:
            path(s, [(w.x + w.w, cy), (260, cy), (260, row_y(r) + bh / 2), (300, row_y(r) + bh / 2)])

    legend(s, 20, height - 18, [("sdk", "SDK"), ("proxy", "AI Gateway"), ("ui", "Admin UI"), ("rust", "Rust"), ("store", "deploy"),
                                ("redline", "route registers on first request, after the gateway/backend trim")])
    s.save(out("ref-surface-map"))


def parse_matrix():
    lines = [ln for ln in open(MATRIX_MD).read().splitlines() if ln.startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = []
    for ln in lines[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) != len(header) or not re.match(r"^[A-L] ", cells[0]):
            break
        rows.append(cells)
    return header[1:], rows


def wrap(text, size, maxw):
    words, rows, cur = text.split(), [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if text_width(cand, size) <= maxw or not cur:
            cur = cand
        else:
            rows.append(cur)
            cur = w
    return rows + ([cur] if cur else [])


def cell_style(v):
    if not v:
        return None, ""
    if v.startswith("Y"):
        return "proxy_full", re.sub(r"^Y\s*\(?|\)$", "", v)
    if v.startswith("opt"):
        return "opt", re.sub(r"^opt\s*\(?|\)$", "", v)
    return "partial", v


def matrix():
    cols, rows = parse_matrix()
    lw, cw, ch, hh = 190, 86, 38, 64
    width = lw + len(cols) * cw + 30
    height = 30 + hh + len(rows) * ch + 60
    s = Svg(width, height, "Flow by subsystem matrix")
    x0, y0 = lw + 10, 30 + hh
    for j, c in enumerate(cols):
        for k, r in enumerate(wrap(c, 10.5, cw - 8)[::-1]):
            s.text(x0 + j * cw + cw / 2, y0 - 10 - k * 13, r, 10.5, INK, 600)
    blue_fill, blue_stroke, _ = PALETTE["proxy"]
    amber_fill, amber_stroke, amber_ink = PALETTE["ext"]
    for i, r in enumerate(rows):
        y = y0 + i * ch
        if i == 6:
            s.add(f"<line x1='20' y1='{y}' x2='{width-20}' y2='{y}' stroke='#9aa1ab' stroke-width='1.4'/>")
        s.text(20, y + ch / 2 + 4, r[0], 11.5, INK, 600, "start")
        for j, v in enumerate(r[1:]):
            x = x0 + j * cw
            style, note = cell_style(v)
            s.add(f"<rect x='{x+2}' y='{y+2}' width='{cw-4}' height='{ch-4}' rx='4' fill='#fafbfc' stroke='#e3e6ea'/>")
            if style is None:
                continue
            if style == "proxy_full":
                s.add(f"<rect x='{x+2}' y='{y+2}' width='{cw-4}' height='{ch-4}' rx='4' fill='{blue_stroke}'/>")
                ink = "#ffffff"
            elif style == "opt":
                s.add(f"<rect x='{x+2}' y='{y+2}' width='{cw-4}' height='{ch-4}' rx='4' fill='{blue_fill}' stroke='{blue_stroke}' stroke-dasharray='4 3'/>")
                ink = "#1d3f8f"
                note = "opt" + (f" ({note})" if note else "")
            else:
                s.add(f"<rect x='{x+2}' y='{y+2}' width='{cw-4}' height='{ch-4}' rx='4' fill='{amber_fill}' stroke='{amber_stroke}'/>")
                ink = amber_ink
            lines = wrap(note, 9, cw - 10)[:3]
            for k, ln in enumerate(lines):
                s.text(x + cw / 2, y + ch / 2 + 3 - (len(lines) - 1) * 5.5 + k * 11, ln, 9, ink, 500)
    ly = height - 22
    s.add(f"<rect x='20' y='{ly-11}' width='14' height='14' rx='3' fill='{blue_stroke}'/>")
    s.text(40, ly, "used", 11, SUB, 400, "start")
    s.add(f"<rect x='90' y='{ly-11}' width='14' height='14' rx='3' fill='{blue_fill}' stroke='{blue_stroke}' stroke-dasharray='4 3'/>")
    s.text(110, ly, "optional, depends on config", 11, SUB, 400, "start")
    s.add(f"<rect x='290' y='{ly-11}' width='14' height='14' rx='3' fill='{amber_fill}' stroke='{amber_stroke}'/>")
    s.text(310, ly, "partial, own variant or setup-only role (the cell says which)", 11, SUB, 400, "start")
    s.text(width - 20, ly, "Rows A-F are data-plane flows, G-L control-plane flows", 11, SUB, 400, "end", italic=True)
    s.save(out("ref-flow-matrix"))


if __name__ == "__main__":
    surface_map()
    matrix()
