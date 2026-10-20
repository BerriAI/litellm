import json, sys, os, glob, hashlib
W = os.environ.get("W", "<scratch>/run"); A, B = sys.argv[1], sys.argv[2]
h = lambda o: hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()[:12]
def ld(p):
    try: return json.load(open(p))
    except Exception: return None
def prov(path, case):
    for k in ("anthropic", "openai"):
        d = ld(f"{W}/tap/{path}.{case}.{k}.json")
        if d: return k, d
    return None, None
def rows(body):
    if body is None: return []
    return body.get("messages") or body.get("input") or []
def ans(path, case):
    r = ld(f"{W}/res/{path}.{case}.client_resp.json") or {}
    b = r.get("body"); st = r.get("status")
    if isinstance(b, dict):
        if "choices" in b: t = b["choices"][0]["message"].get("content")
        elif "content" in b: t = "".join(c.get("text", "") for c in b["content"] if c.get("type") == "text")
        elif "output" in b: t = "".join(c.get("text", "") for i in b["output"] if i.get("type") == "message" for c in i["content"])
        else: t = str(b)[:200]
        return st, (t or "")[:160], b.get("usage")
    return st, str(b)[:300], None
cases = sorted({os.path.basename(f).split(".")[1] for f in glob.glob(f"{W}/res/{A}.*.client_resp.json")} |
               {os.path.basename(f).split(".")[1] for f in glob.glob(f"{W}/res/{B}.*.client_resp.json")})
for c in cases:
    print(f"== {c}")
    pr = {}
    for p in (A, B):
        st, t, u = ans(p, c); hr = ld(f"{W}/tap/{p}.{c}.hr.json"); k, pv = prov(p, c)
        hrs = [(m.get("role"), len(json.dumps(m.get("content")))) for m in hr["req"]["messages"]] if hr else None
        hro = hr["resp"] if hr and hr.get("resp") else {}
        stats = {x: hro.get(x) for x in ("tokens_before", "tokens_after", "tokens_saved", "transforms_applied") if x in hro}
        pr[p] = rows(pv["req"]) if pv else None
        lens = [len(json.dumps(m)) for m in pr[p]] if pr[p] is not None else None
        print(f" {p}: status={st} answer={t!r}\n   usage={u}\n   hr_sent={hrs} hr_stats={stats}\n   provider={k} rowlens={lens} total={sum(lens) if lens else None}")
    if pr[A] is not None and pr[B] is not None:
        diff = [i for i in range(max(len(pr[A]), len(pr[B]))) if i >= len(pr[A]) or i >= len(pr[B]) or pr[A][i] != pr[B][i]]
        print(f"  provider rows differing {A} vs {B}: {diff} (n={len(pr[A])}/{len(pr[B])})")
        for i in diff[:4]:
            if i < len(pr[A]) and i < len(pr[B]):
                a, b = pr[A][i], pr[B][i]
                ka = {k: v for k, v in (a.items() if isinstance(a, dict) else [])}
                print(f"   row{i} keysA={sorted(ka)} keysB={sorted(b) if isinstance(b, dict) else b!r:.80}")
