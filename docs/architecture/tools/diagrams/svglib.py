"""Tiny SVG drawing helpers for the architecture docs.

Flowcharts are laid out by hand (explicit coordinates). Sequence diagrams are laid out
automatically from a list of steps.
"""
from html import escape

FONT = "Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', 'SFMono-Regular', Menlo, Consolas, monospace"

PALETTE = {
    "proxy": ("#e8f0fe", "#3b6fd8", "#1d3f8f"),
    "sdk": ("#e6f4ea", "#2e8b57", "#1b5e3a"),
    "rust": ("#fdeee4", "#d0692b", "#8a3d12"),
    "ui": ("#f3e8fd", "#8e4fd1", "#582a8c"),
    "store": ("#eef0f3", "#6b7280", "#30343b"),
    "ext": ("#fff6dc", "#c99a1a", "#7a5a00"),
    "bad": ("#fde8e8", "#d33c3c", "#8f1f1f"),
    "neutral": ("#ffffff", "#9aa1ab", "#2b2f36"),
    "muted": ("#f7f8fa", "#c4c9d1", "#5b616b"),
}
RED = "#d33c3c"
INK = "#2b2f36"
SUB = "#5b616b"
LINE = "#7b828c"


def text_width(s, size=12, mono=False):
    factor = 0.61 if mono else 0.56
    return len(s) * size * factor


class Svg:
    def __init__(self, width, height, title=""):
        self.w = width
        self.h = height
        self.title = title
        self.parts = []

    def add(self, s):
        self.parts.append(s)

    def render(self):
        defs = (
            "<defs>"
            f"<marker id='ah' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='9' markerHeight='9' markerUnits='userSpaceOnUse' orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='{LINE}'/></marker>"
            f"<marker id='ahr' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='9' markerHeight='9' markerUnits='userSpaceOnUse' orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='{RED}'/></marker>"
            "<marker id='ahk' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='9' markerHeight='9' markerUnits='userSpaceOnUse' orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='#30343b'/></marker>"
            "</defs>"
        )
        head = (
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{self.w}' height='{self.h}' "
            f"viewBox='0 0 {self.w} {self.h}' font-family=\"{FONT}\">"
        )
        title = f"<title>{escape(self.title)}</title>" if self.title else ""
        bg = f"<rect width='{self.w}' height='{self.h}' fill='#ffffff'/>"
        return head + title + defs + bg + "".join(self.parts) + "</svg>\n"

    def save(self, path):
        with open(path, "w") as f:
            f.write(self.render())

    def text(self, x, y, s, size=12, color=INK, weight=400, anchor="middle", mono=False, italic=False):
        fam = f" font-family=\"{MONO}\"" if mono else ""
        style = " font-style='italic'" if italic else ""
        self.add(
            f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{color}' font-weight='{weight}' "
            f"text-anchor='{anchor}'{fam}{style}>{escape(s)}</text>"
        )

    def lines(self, x, y, rows, size=12, color=INK, weight=400, anchor="middle", gap=None, mono=False):
        gap = gap or size * 1.3
        for i, r in enumerate(rows):
            self.text(x, y + i * gap, r, size, color, weight, anchor, mono)


class Box:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2

    def side(self, s, t=0.5):
        if s == "t":
            return (self.x + self.w * t, self.y)
        if s == "b":
            return (self.x + self.w * t, self.y + self.h)
        if s == "l":
            return (self.x, self.y + self.h * t)
        if s == "r":
            return (self.x + self.w, self.y + self.h * t)
        raise ValueError(s)


def box(svg, x, y, w, h, title, sub=(), kind="neutral", code=False, title_size=13, sub_size=11, radius=8, dashed=False):
    fill, stroke, ink = PALETTE[kind]
    dash = " stroke-dasharray='5 4'" if dashed else ""
    svg.add(
        f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{radius}' fill='{fill}' stroke='{stroke}' stroke-width='1.4'{dash}/>"
    )
    sub = [sub] if isinstance(sub, str) else list(sub)
    titles = title.split("\n") if title else []
    tg = title_size * 1.25
    sg = sub_size * 1.35
    total = len(titles) * tg + len(sub) * sg
    top = y + h / 2 - total / 2 + title_size * 0.85
    for i, t in enumerate(titles):
        svg.text(x + w / 2, top + i * tg, t, title_size, ink, 600)
    st = top + len(titles) * tg + (2 if titles else 0)
    for i, s in enumerate(sub):
        svg.text(x + w / 2, st + i * sg, s, sub_size, SUB, 400, mono=code)
    return Box(x, y, w, h)


def cylinder(svg, x, y, w, h, title, sub=(), kind="store"):
    fill, stroke, ink = PALETTE[kind]
    e = 7
    svg.add(
        f"<path d='M{x},{y+e} a{w/2},{e} 0 0 0 {w},0 a{w/2},{e} 0 0 0 {-w},0 v{h-2*e} a{w/2},{e} 0 0 0 {w},0 v{-(h-2*e)}' "
        f"fill='{fill}' stroke='{stroke}' stroke-width='1.4'/>"
    )
    sub = [sub] if isinstance(sub, str) else list(sub)
    top = y + h / 2 + 4 - (len(sub) * 7)
    svg.text(x + w / 2, top + 2, title, 13, ink, 600)
    for i, s in enumerate(sub):
        svg.text(x + w / 2, top + 17 + i * 14, s, 11, SUB)
    return Box(x, y, w, h)


def group(svg, x, y, w, h, label, kind="muted", label_pos="tl", dashed=False):
    fill, stroke, ink = PALETTE[kind]
    dash = " stroke-dasharray='6 4'" if dashed else ""
    svg.add(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='12' fill='{fill}' stroke='{stroke}' stroke-width='1.2'{dash}/>")
    if label:
        if label_pos == "tl":
            svg.text(x + 12, y + 18, label, 11.5, ink, 700, "start")
        else:
            svg.text(x + w / 2, y + 18, label, 11.5, ink, 700)
    return Box(x, y, w, h)


def label_at(svg, x, y, s, size=11, color=SUB, bg=True, anchor="middle", weight=400):
    rows = s.split("\n")
    w = max(text_width(r, size) for r in rows) + 8
    h = len(rows) * size * 1.3 + 4
    if bg:
        bx = x - w / 2 if anchor == "middle" else (x - 4 if anchor == "start" else x - w + 4)
        svg.add(f"<rect x='{bx:.1f}' y='{y - size - 1:.1f}' width='{w:.1f}' height='{h:.1f}' rx='3' fill='#ffffff' opacity='0.92'/>")
    for i, r in enumerate(rows):
        svg.text(x, y + i * size * 1.3, r, size, color, weight, anchor)


def path(svg, pts, label=None, color=LINE, dashed=False, marker="ah", width=1.4, label_seg=None, label_dx=0, label_dy=-4,
         label_color=None, start_marker=False, label_anchor="middle"):
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    dash = " stroke-dasharray='5 4'" if dashed else ""
    m = f" marker-end='url(#{marker})'" if marker else ""
    sm = f" marker-start='url(#{marker})'" if start_marker else ""
    svg.add(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='{width}'{dash}{m}{sm}/>")
    if label:
        i = label_seg if label_seg is not None else (len(pts) - 1) // 2
        (x1, y1), (x2, y2) = pts[i], pts[i + 1]
        label_at(svg, (x1 + x2) / 2 + label_dx, (y1 + y2) / 2 + label_dy, label, color=label_color or SUB, anchor=label_anchor)


def connect(svg, a, sa, b, sb, label=None, via=None, ta=0.5, tb=0.5, **kw):
    p1 = a.side(sa, ta)
    p2 = b.side(sb, tb)
    pts = [p1]
    if via == "hv":
        pts.append((p2[0], p1[1]))
    elif via == "vh":
        pts.append((p1[0], p2[1]))
    elif via == "vhv":
        my = (p1[1] + p2[1]) / 2
        pts += [(p1[0], my), (p2[0], my)]
    elif via == "hvh":
        mx = (p1[0] + p2[0]) / 2
        pts += [(mx, p1[1]), (mx, p2[1])]
    elif isinstance(via, list):
        pts += via
    pts.append(p2)
    path(svg, pts, label, **kw)


def legend(svg, x, y, items, size=11):
    cx = x
    for kind, name in items:
        fill, stroke, _ = PALETTE[kind] if kind in PALETTE else ("#fff", kind, "")
        if kind == "redline":
            svg.add(f"<line x1='{cx}' y1='{y-4}' x2='{cx+22}' y2='{y-4}' stroke='{RED}' stroke-width='2' stroke-dasharray='5 4'/>")
            svg.text(cx + 28, y, name, size, SUB, 400, "start")
            cx += 34 + text_width(name, size) + 18
            continue
        svg.add(f"<rect x='{cx}' y='{y-11}' width='14' height='14' rx='3' fill='{fill}' stroke='{stroke}'/>")
        svg.text(cx + 20, y, name, size, SUB, 400, "start")
        cx += 26 + text_width(name, size) + 18


# ---------------------------------------------------------------- sequence diagrams

class Seq:
    """Automatic sequence diagram.

    participants: list of (key, title, sub, kind)
    steps: ("m", a, b, text) solid message, ("r", a, b, text) dashed return,
           ("n", a, b, text) note spanning a..b, ("alt"/"par"/"opt", [(label, steps), ...]),
           ("div", text) section divider
    """

    def __init__(self, participants, steps, title="", col_min=130, font=11.5):
        self.ps = participants
        self.steps = steps
        self.title = title
        self.col_min = col_min
        self.font = font

    def _layout_cols(self):
        widths = [max(text_width(t, 12) for t in ([p[1]] + ([p[2]] if p[2] else []))) + 24 for p in self.ps]
        widths = [max(w, 96) for w in widths]
        xs = []
        x = 16
        for i, w in enumerate(widths):
            if i == 0:
                xs.append(x + w / 2)
            else:
                gap = max(self.col_min, (widths[i - 1] + w) / 2 + 14)
                xs.append(xs[-1] + gap)
        self.need = {}
        self._scan(self.steps, xs)
        for (i, j), need in sorted(self.need.items()):
            cur = xs[j] - xs[i]
            if need > cur:
                delta = need - cur
                for k in range(i + 1, len(xs)):
                    xs[k] += delta * min(k - i, j - i) / (j - i)
        self.widths = widths
        return xs

    def _scan(self, steps, xs):
        idx = {p[0]: i for i, p in enumerate(self.ps)}
        for s in steps:
            if s[0] in ("m", "r"):
                i, j = sorted((idx[s[1]], idx[s[2]]))
                if i == j:
                    continue
                need = min(text_width(s[3], self.font) + 24, 340 * (j - i))
                key = (i, j)
                if j - i == 1:
                    self.need[key] = max(self.need.get(key, 0), need)
            elif s[0] in ("alt", "par", "opt", "loop"):
                for _, sub in s[1]:
                    self._scan(sub, xs)

    def render(self, path_out):
        xs = self._layout_cols()
        idx = {p[0]: i for i, p in enumerate(self.ps)}
        head_h = 46
        top = 14
        ops = []
        y = top + head_h + 20

        def wrap(s, maxw):
            if text_width(s, self.font) <= maxw:
                return [s]
            words = s.split(" ")
            rows, cur = [], ""
            for w in words:
                t = (cur + " " + w).strip()
                if text_width(t, self.font) > maxw and cur:
                    rows.append(cur)
                    cur = w
                else:
                    cur = t
            rows.append(cur)
            return rows

        def walk(steps, depth):
            nonlocal y
            for s in steps:
                k = s[0]
                if k in ("m", "r"):
                    a, b = idx[s[1]], idx[s[2]]
                    if a == b:
                        rows = wrap(s[3], 300)
                        ops.append(("self", xs[a], y, rows, k))
                        self_right.append(xs[a] + 40 + max(len(r) for r in rows) * self.font * 0.56)
                        y += 14 * len(rows) + 27
                    else:
                        span = abs(xs[b] - xs[a]) - 16
                        rows = wrap(s[3], max(span, 120))
                        ops.append(("msg", xs[a], xs[b], y + 13 * (len(rows) - 1), rows, k))
                        y += 13 * len(rows) + 15
                elif k == "n":
                    a, b = sorted((idx[s[1]], idx[s[2]]))
                    x1, x2 = xs[a] - 50, xs[b] + 50
                    rows = wrap(s[3], x2 - x1 - 16)
                    ops.append(("note", x1, x2, y - 4, rows))
                    y += 14 * len(rows) + 16
                elif k == "div":
                    ops.append(("div", y, s[1]))
                    y += 26
                elif k in ("alt", "par", "opt", "loop"):
                    y0 = y - 6
                    y += 18
                    seps = []
                    for bi, (label, sub) in enumerate(s[1]):
                        if bi > 0:
                            seps.append((y - 4, label))
                            y += 20
                        else:
                            first_label = label
                        walk(sub, depth + 1)
                    ops.append(("frame", depth, y0, y + 2, k, first_label, seps))
                    y += 22

        self_right = []
        walk(self.steps, 0)
        bottom = y + 8
        width = int(max([xs[-1] + self.widths[-1] / 2 + 20, *self_right]))
        svg = Svg(width, int(bottom + head_h + 18), self.title)
        for i, p in enumerate(self.ps):
            x = xs[i]
            svg.add(f"<line x1='{x}' y1='{top+head_h}' x2='{x}' y2='{bottom}' stroke='#c4c9d1' stroke-width='1.2' stroke-dasharray='4 4'/>")
        for op in ops:
            if op[0] == "frame":
                _, depth, y0, y1, kind, label, seps = op
                x0 = xs[0] - self.widths[0] / 2 + 4 + depth * 8
                x1 = xs[-1] + self.widths[-1] / 2 - 4 - depth * 8
                svg.add(f"<rect x='{x0}' y='{y0}' width='{x1-x0}' height='{y1-y0}' rx='6' fill='#f4f6fa' fill-opacity='0.55' stroke='#9aa1ab' stroke-width='1'/>")
                tag = kind.upper()
                tw = text_width(tag, 10) + 12
                svg.add(f"<path d='M{x0},{y0+16} h{tw} l6,-6 v-10 h{-tw-6} z' fill='#e3e7ee' stroke='#9aa1ab' stroke-width='1'/>")
                svg.text(x0 + 6, y0 + 12, tag, 10, INK, 700, "start")
                svg.text(x0 + tw + 14, y0 + 12, f"[{label}]", 11, SUB, 500, "start", italic=False)
                for sy, sl in seps:
                    svg.add(f"<line x1='{x0}' y1='{sy}' x2='{x1}' y2='{sy}' stroke='#9aa1ab' stroke-dasharray='5 4'/>")
                    svg.text(x0 + 8, sy + 13, f"[{sl}]", 11, SUB, 500, "start")
        for op in ops:
            if op[0] == "msg":
                _, xa, xb, yy, rows, k = op
                dash = " stroke-dasharray='5 4'" if k == "r" else ""
                pad = 3 if xb > xa else -3
                svg.add(f"<line x1='{xa+pad}' y1='{yy}' x2='{xb-pad}' y2='{yy}' stroke='#30343b' stroke-width='1.2'{dash} marker-end='url(#ahk)'/>")
                mx = (xa + xb) / 2
                for i, r in enumerate(rows):
                    svg.text(mx, yy - 5 - (len(rows) - 1 - i) * 13, r, self.font, INK)
            elif op[0] == "self":
                _, x, yy, rows, k = op
                h = 14 * len(rows) + 6
                svg.add(f"<path d='M{x+3},{yy} h26 v{h} h-23' fill='none' stroke='#30343b' stroke-width='1.2' marker-end='url(#ahk)'/>")
                for i, r in enumerate(rows):
                    svg.text(x + 36, yy + 10 + i * 14, r, self.font, INK, 400, "start")
            elif op[0] == "note":
                _, x1, x2, yy, rows = op
                h = 14 * len(rows) + 8
                svg.add(f"<rect x='{x1}' y='{yy-12}' width='{x2-x1}' height='{h}' rx='4' fill='#fff6dc' stroke='#c99a1a'/>")
                for i, r in enumerate(rows):
                    svg.text((x1 + x2) / 2, yy + 2 + i * 14, r, 11, '#5a4300')
            elif op[0] == "div":
                _, yy, t = op
                svg.add(f"<line x1='10' y1='{yy}' x2='{width-10}' y2='{yy}' stroke='#c4c9d1'/>")
                tw = text_width(t, 11) + 16
                svg.add(f"<rect x='{width/2-tw/2}' y='{yy-10}' width='{tw}' height='19' rx='9' fill='#eef0f3' stroke='#c4c9d1'/>")
                svg.text(width / 2, yy + 4, t, 11, SUB, 600)
        for yy in (top, bottom + 6):
            for i, p in enumerate(self.ps):
                key, t, sub, kind = p
                w = self.widths[i]
                fill, stroke, ink = PALETTE[kind]
                svg.add(f"<rect x='{xs[i]-w/2}' y='{yy}' width='{w}' height='{head_h}' rx='7' fill='{fill}' stroke='{stroke}' stroke-width='1.3'/>")
                if sub:
                    svg.text(xs[i], yy + 19, t, 12, ink, 600)
                    svg.text(xs[i], yy + 35, sub, 10.5, SUB, 400, mono=True)
                else:
                    svg.text(xs[i], yy + 27, t, 12, ink, 600)
        svg.save(path_out)


def badge(svg, x, y, n, color="#30343b"):
    svg.add(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='9' fill='#ffffff' stroke='{color}' stroke-width='1.4'/>")
    svg.text(x, y + 4, str(n), 11, color, 700)


def key(svg, x, y, items, size=11, gap=19, color="#30343b"):
    for i, t in enumerate(items):
        badge(svg, x + 9, y + i * gap - 4, i + 1, color)
        svg.text(x + 26, y + i * gap, t, size, SUB, 400, "start")
