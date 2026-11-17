# agentic_router.py — Learning-based agentic model-tier router for LiteLLM.
#
# Routes requests across fast/mid/strong model tiers using four experts
# (rule / UCB / LinUCB / k-NN) under an EXP3 meta-selector, with a durable
# quality-reward flywheel (latency / cost / outcome) so routing improves
# with usage.
#
# Usage (SDK):
#     import litellm
#     from litellm.integrations.agentic_router import AgenticRouterHook
#
#     hook = AgenticRouterHook(
#         mode="shadow",                      # log-only; "active" rewrites the model
#         tier_models={"fast": "gpt-4o-mini",
#                      "mid": "gpt-4o",
#                      "strong": "o3"},
#     )
#     litellm.callbacks = [hook]
#
# Proxy (config.yaml):
#     litellm_settings:
#       callbacks: custom_callbacks.agentic_router_handler
#       modify_params: true                  # some versions need this for hook rewrites
#
# Algorithm spec (cross-language source of truth, with JS/Go ports and a
# simulated-replay convergence harness):
#     https://github.com/lhwwxy/agentic-router-gateway  (spec/ROUTING_SPEC.md)

import json
import math
import os
import random
import re
import threading
import time
from pathlib import Path

from litellm.integrations.custom_logger import CustomLogger

__all__ = [
    "classify", "features_of", "Router", "fresh_state", "empty_tiers",
    "recommend_rule", "recommend_ucb", "recommend_lin", "recommend_knn",
    "resolve_tier", "tier_of", "reward_of", "apply_reward", "correct_reward",
    "AgenticRouterHook", "Store", "LIN_GRADUATION", "KNN_GRADUATION",
    "EXPLORATION_EPS", "GAMMA",
]

# ─── 1. Classifier: task type + complexity (deterministic, <1ms) ────────────

_QA_IMPERATIVE = re.compile(r"(请|麻烦|帮我).{0,6}(回答|告诉|说)")
_CODE_RE = re.compile(r"def |function |class |import |require\(|```|\.py|\.ts|\.js|\.go|\.rs\b|bug|error|exception|traceback|报错|代码|函数|接口|脚本")
_WRITE_RE = re.compile(r"(写|创作|起草|润色|翻译|translate|write|blog|article|文案|邮件|报告)")
_SUMMARIZE_RE = re.compile(r"(总结|概括|summar|摘要|要点)")
_RESEARCH_RE = re.compile(r"(查|搜索|检索|research|调研|对比|最新|资料|文档)")
_AGENTIC_RE = re.compile(r"(帮我|请|做|搭建|构建|部署|实现|写一个|生成|build|setup|deploy|implement|create)")
_QA_RE = re.compile(r"[?？]|什么|为什么|怎么|如何|how|what|why|能否|是否")
_COMPLEX_RE = re.compile(r"(重构|优化|设计|分析|架构|debug|调试|排查|review|方案|复杂|多步|逐步|implement|design|architecture)")
_SIMPLE_RE = re.compile(r"(一句|简单|快速|quickly|short|hello|你好|谢谢)")

_PRIOR = {"code": 0.3, "agentic": 0.3, "research": 0.2, "write": 0.15,
          "summarize": 0.05, "qa": 0.0, "other": 0.0}
_FEATURE_TYPES = ["code", "write", "summarize", "research", "agentic", "qa"]


def _round2(x: float) -> float:
    """JS Math.round(x*100)/100 semantics (half away from -inf, not banker's)."""
    return math.floor(x * 100 + 0.5) / 100


def classify(text):
    """Return {"type": str, "complexity": float in [0,1] at 2 decimals}."""
    t = text or ""
    lower = t.lower()
    type_ = "other"
    if _QA_IMPERATIVE.search(lower):
        type_ = "qa"
    elif _CODE_RE.search(lower):
        type_ = "code"
    elif _WRITE_RE.search(lower):
        type_ = "write"
    elif _SUMMARIZE_RE.search(lower):
        type_ = "summarize"
    elif _RESEARCH_RE.search(lower):
        type_ = "research"
    elif _AGENTIC_RE.search(lower):
        type_ = "agentic"
    elif _QA_RE.search(lower):
        type_ = "qa"

    complex_kw = bool(_COMPLEX_RE.search(lower))
    simple_kw = bool(_SIMPLE_RE.search(lower))
    score = _PRIOR.get(type_, 0)
    score += min(len(t) / 2000, 1) * 0.2
    score += 0.25 if "```" in t else 0
    if complex_kw:
        score += 0.25
    if simple_kw and not complex_kw:  # "simple" + "complex" together: no penalty
        score -= 0.25
    return {"type": type_, "complexity": _round2(max(0.0, min(1.0, score)))}


def features_of(cls, text):
    """Shared feature vector for LinUCB / k-NN: [1, complexity, len-norm, one-hot×6]."""
    t = text or ""
    f = [1.0, cls["complexity"], min(len(t), 4000) / 4000]
    for ty in _FEATURE_TYPES:
        f.append(1.0 if cls["type"] == ty else 0.0)
    return f


def _rebuild_features(decision):
    f = [1.0, decision["complexity"], min(decision.get("len") or 0, 4000) / 4000]
    for ty in _FEATURE_TYPES:
        f.append(1.0 if decision["type"] == ty else 0.0)
    return f


# ─── 2. Tier helpers ────────────────────────────────────────────────────────

_FAST_RE = re.compile(r"flash|lite|mini|small|fast|air|nano|tiny|light")
_STRONG_RE = re.compile(r"pro|max|ultra|plus|large|premium|reason|thinking")

DEFAULT_PRICES = {"*": {"fast": {"in": 1, "out": 4},
                        "mid": {"in": 2, "out": 8},
                        "strong": {"in": 3, "out": 16}}}
_CONSERVATIVE = {"in": 3, "out": 16}


def tier_of(model):
    if not model:
        return "unknown"
    if _FAST_RE.search(model):
        return "fast"
    if _STRONG_RE.search(model):
        return "strong"
    return "mid"


def _provider_id(p):
    return (p.get("provider") or p.get("id") or p.get("name")) if isinstance(p, dict) else p


def _tier_target(tier, provider, model_tiers):
    for s in (model_tiers or {}).get("tiers", {}).get(tier) or []:
        if provider and s.startswith(provider + "/"):
            return s[len(provider) + 1:]
    return None


def resolve_tier(tier, provider, model_tiers, opts=None):
    """Resolve a tier to a concrete model; same-provider first, cross-provider
    fallback, then mid→strong/fast fallback. Returns {target, provider, used, note}."""
    opts = opts or {}
    if not model_tiers:
        return {"target": None, "provider": None, "used": tier, "note": "tiers-not-ready"}
    cross = opts.get("crossProvider", True)

    def same(t):
        m = _tier_target(t, provider, model_tiers)
        return {"provider": provider, "model": m} if m else None

    def cross_pick(t):
        if not cross:
            return None
        for p in model_tiers.get("providers") or []:
            pid = _provider_id(p)
            if not pid or pid == provider:
                continue
            m = _tier_target(t, pid, model_tiers)
            if m:
                return {"provider": pid, "model": m}
        return None

    def is_cross(r):
        return bool(r and provider and r["provider"] != provider)

    direct = same(tier) or cross_pick(tier)
    if direct:
        return {"target": direct["model"], "provider": direct["provider"], "used": tier,
                "note": "cross-provider" if is_cross(direct) else ""}
    if tier == "mid":
        for fb in ("strong", "fast"):
            r = same(fb) or cross_pick(fb)
            if r:
                note = "mid->%s fallback" % fb + (" [cross-provider]" if is_cross(r) else "")
                return {"target": r["model"], "provider": r["provider"], "used": fb, "note": note}
    return {"target": None, "provider": None, "used": tier, "note": "no-candidate"}


# ─── 3. Experts ─────────────────────────────────────────────────────────────

def recommend_rule(cls):
    want = "strong" if cls["complexity"] >= 0.7 else ("fast" if cls["complexity"] <= 0.3 else "mid")
    return {"tier": want, "reason": "rule threshold complexity=" + str(cls["complexity"]), "conf": 0.5}


def recommend_ucb(cls, ucb_state, available_tiers=None):
    tiers = available_tiers or ["fast", "mid", "strong"]
    arms, scores = ucb_state["arms"], []
    for tier in tiers:
        arm = arms.get(cls["type"] + ":" + tier) or {"n": 0, "mean": 0}
        u = arm["mean"] + 1.2 * math.sqrt(math.log(ucb_state["total"] + 1) / arm["n"]) if arm["n"] > 0 else math.inf
        scores.append({"tier": tier, "ucb": u, "arm": arm})
    best = scores[0]
    for s in scores:
        if s["ucb"] > best["ucb"]:
            best = s
    infs = [s for s in scores if s["ucb"] == math.inf]
    if len(infs) > 1:  # rotate among unexplored arms
        best = infs[ucb_state["total"] % len(infs)]
    bound = "inf" if math.isinf(best["ucb"]) else _round2(best["ucb"])
    return {"tier": best["tier"],
            "reason": "ucb %s n=%s mean=%s bound=%s" % (cls["type"], best["arm"]["n"],
                                                        _round2(best["arm"]["mean"]), bound),
            "conf": 0.6 if math.isfinite(best["ucb"]) else 0.2}


def recommend_lin(cls, f, lin_state, available_tiers=None):
    tiers = available_tiers or ["fast", "mid", "strong"]
    tn, tier_w, best = lin_state["tierN"], lin_state["tierW"], None
    for tier in tiers:
        w = tier_w.get(tier) or []
        dot = sum((w[i] if i < len(w) else 0) * f[i] for i in range(len(f)))
        bonus = 0.8 * math.sqrt(math.log(lin_state["total"] + 1) / tn[tier]) if tn[tier] > 0 else 0.3
        score = dot + bonus
        if best is None or score > best["score"]:
            best = {"tier": tier, "score": score, "dot": dot}
    return {"tier": best["tier"],
            "reason": "linucb score=%s (dot=%s)" % (_round2(best["score"]), _round2(best["dot"])),
            "conf": 0.7 if lin_state["total"] >= 50 else 0.1}


def recommend_knn(cls, f, knn_state, available_tiers=None):
    tiers = available_tiers or ["fast", "mid", "strong"]
    pool = knn_state["pool"]
    if len(pool) < 5:
        return {"tier": None, "reason": "knn cold (%d/5 seed)" % len(pool), "conf": 0}
    scored = []
    for e in pool:
        d = sum((e["f"][i] - f[i]) ** 2 for i in range(len(f)))
        scored.append({"d": d, "e": e})
    scored.sort(key=lambda x: x["d"])
    k = min(5, len(scored))
    agg = {"fast": {"n": 0, "s": 0.0}, "mid": {"n": 0, "s": 0.0}, "strong": {"n": 0, "s": 0.0}}
    for i in range(k):
        t = scored[i]["e"]["tier"]
        if t in agg:
            agg[t]["n"] += 1
            agg[t]["s"] += scored[i]["e"]["r"]
    best = None
    for tier in tiers:
        a = agg[tier]
        if best is None or (a["n"] > 0 and a["s"] / a["n"] > best["score"]):
            best = {"tier": tier, "score": a["s"] / a["n"] if a["n"] > 0 else -1}
    return {"tier": best["tier"],
            "reason": "knn k=%d neighbors score=%s" % (k, _round2(best["score"])),
            "conf": min(0.8, len(pool) / 100)}


# ─── 4. EXP3 meta-selector ──────────────────────────────────────────────────

DEFAULT_WEIGHTS = {"ucb": 1.0, "linucb": 1.0, "knn": 1.0, "rule": 2.5}
GAMMA = 0.05  # every expert keeps a >= Gamma/K sampling probability


def meta_pick(recs, exp3, rng=None):
    """Softmax-weighted sampling with gamma mixing over the expert pool."""
    rng = rng or random
    weights = exp3["weights"]
    pool = ["ucb"]
    by_name = {"ucb": recs["ucb"], "rule": recs["rule"]}
    if exp3.get("linGraduated") and recs.get("lin") and recs["lin"].get("tier"):
        pool.append("linucb")
        by_name["linucb"] = recs["lin"]
    if exp3.get("knnGraduated") and recs.get("knn") and recs["knn"].get("tier"):
        pool.append("knn")
        by_name["knn"] = recs["knn"]
    pool.append("rule")

    total = sum(weights.get(n, 0) for n in pool)
    k = len(pool)
    gamma = exp3.get("gamma", GAMMA)
    ps = {n: (1 - gamma) * ((weights.get(n, 0) / total) if total > 0 else 1 / k) + gamma / k
          for n in pool}
    r, chosen = rng.random(), "rule"
    for n in pool:
        if r < ps[n]:
            chosen = n
            break
        r -= ps[n]
    return {"chosen": chosen, "ps": ps, "pool": pool, "byName": by_name}


# ─── 5. Reward + flywheel updates ───────────────────────────────────────────

def reward_of(td, tag, dt_ms=None):
    """Proxy quality reward: 1.0 for a clean close, minus penalties for tool
    failures (0.2 ea, cap 0.6), retries (0.2 ea, cap 0.4), cost, latency."""
    if tag != "clean-close":
        return 0, {"base": 0, "toolErrors": td.get("toolErrors", 0), "retries": td.get("retries", 0),
                   "dtMs": dt_ms, "costYuan": td.get("costYuan", 0), "costPenalty": 0}
    reward = 1.0
    tool_penalty = min(0.6, 0.2 * td.get("toolErrors", 0))
    retry_penalty = min(0.4, 0.2 * td.get("retries", 0))
    cost_penalty = min(td.get("costCap", 0.5), td.get("costWeight", 20) * td.get("costYuan", 0))
    reward -= tool_penalty + retry_penalty + cost_penalty
    if dt_ms is not None:
        if dt_ms > 90000:
            reward -= 0.3
        elif dt_ms > 30000:
            reward -= 0.1
    bd = {"base": 1, "toolErrors": td.get("toolErrors", 0), "toolPenalty": tool_penalty,
          "retries": td.get("retries", 0), "retryPenalty": retry_penalty,
          "costYuan": td.get("costYuan", 0), "costPenalty": cost_penalty, "dtMs": dt_ms}
    return max(0, _round2(reward)), bd


def apply_reward(state, rec):
    """Apply one reward record to the flywheel (UCB / LinUCB / k-NN / EXP3)."""
    tier_n = state["lin"]["tierN"]
    rec["linLr"] = 1 / (1 + tier_n.get(rec["linTier"], 0))
    arm = state["ucb"]["arms"].setdefault(rec["clsType"] + ":" + rec["ucbTier"], {"n": 0, "mean": 0})
    arm["n"] += 1
    arm["mean"] += (rec["reward"] - arm["mean"]) / arm["n"]
    state["ucb"]["total"] += 1

    lw = state["lin"]["tierW"].setdefault(rec["linTier"], [0.0] * 9)
    lr = 1 / (1 + tier_n.get(rec["linTier"], 0))
    dot = sum((lw[i] if i < len(lw) else 0) * rec["f"][i] for i in range(len(rec["f"])))
    for i in range(len(rec["f"])):
        lw[i] = (lw[i] if i < len(lw) else 0) + lr * (rec["reward"] - dot) * rec["f"][i]
    tier_n[rec["linTier"]] = tier_n.get(rec["linTier"], 0) + 1
    state["lin"]["total"] += 1

    state["knn"]["pool"].append({"f": list(rec["f"]), "tier": rec["ucbTier"], "r": rec["reward"]})
    if len(state["knn"]["pool"]) > 400:
        state["knn"]["pool"].pop(0)

    pool_len = len(rec.get("pool") or [1])
    if rec.get("pChosen", 0) > 0 and pool_len > 0:
        w = state["exp3"]["weights"]
        w[rec["chosen"]] = w.get(rec["chosen"], 0) * math.exp(0.3 * (rec["reward"] / rec["pChosen"]) / pool_len)
    state["exp3"]["total"] += 1

    # Weight explosion guard: divide by max, with a floor. Without the floor,
    # repeated normalization underflows small weights to exactly 0 (0*x=0,
    # unrecoverable). Ratio capped at 1e6 so a fallen-behind expert can
    # recover via importance weighting in hundreds of rounds, not thousands.
    ws = state["exp3"]["weights"]
    if ws:
        m = max(ws.values())
        if m > 1e50:
            for k in ws:
                ws[k] = max(ws[k] / m, 1e-6)


def correct_reward(state, rec, new_reward):
    """Human-feedback correction (thumbs up/down): rewrites an applied reward."""
    delta = new_reward - rec["reward"]
    if not delta:
        return {"delta": 0}
    arm = state["ucb"]["arms"].get(rec["clsType"] + ":" + rec["ucbTier"])
    if arm and arm["n"] > 0:
        arm["mean"] += delta / arm["n"]
    lw = state["lin"]["tierW"].get(rec["linTier"])
    if lw:
        lr = rec.get("linLr") or 1 / (1 + state["lin"]["tierN"].get(rec["linTier"], 0))
        for i in range(len(rec["f"])):
            lw[i] = (lw[i] if i < len(lw) else 0) + lr * delta * rec["f"][i]
    if state.get("knn"):
        state["knn"]["pool"].append({"f": list(rec["f"]), "tier": rec["ucbTier"], "r": new_reward})
    if rec.get("pChosen", 0) > 0 and rec.get("pool"):
        w = state["exp3"]["weights"]
        w[rec["chosen"]] = w.get(rec["chosen"], 0) * math.exp(0.3 * delta / (rec["pChosen"] * len(rec["pool"])))
    rec["reward"] = new_reward
    return {"delta": delta}


# ─── 6. Store + Router orchestration ────────────────────────────────────────

LIN_GRADUATION = 50     # LinUCB joins the EXP3 pool at 50 samples
KNN_GRADUATION = 30     # k-NN joins the EXP3 pool at 30 experiences
DECISION_WINDOW = 200
EXPLORATION_EPS = 0.08  # tier-level epsilon exploration (0 disables)
MAX_LOG_MB = 10         # JSONL log rotation threshold (AGENTIC_ROUTER_MAX_LOG_MB)


def default_home():
    return os.environ.get("AGENTIC_ROUTER_HOME") or os.path.join(
        os.path.expanduser("~"), ".agentic-router")


def empty_tiers():
    return {"providers": [], "tiers": {"fast": [], "mid": [], "strong": []}}


def fresh_state():
    return {
        "mode": "shadow",  # shadow: log only; active: rewrite the model
        "seq": 0,
        "decisions": [],
        "ucb": {"total": 0, "arms": {}},
        "lin": {"total": 0, "tierN": {"fast": 0, "mid": 0, "strong": 0},
                "tierW": {"fast": [0.0] * 9, "mid": [0.0] * 9, "strong": [0.0] * 9}},
        "knn": {"pool": []},
        "exp3": {"total": 0, "linGraduated": False, "knnGraduated": False,
                 "weights": dict(DEFAULT_WEIGHTS)},
    }


class Store:
    """Local, plain-text, auditable persistence (state.json + JSONL logs).

    JSONL logs (decisions/rewards) are size-rotated: past MAX_LOG_MB the
    current file is moved to <name>.1 (one previous generation is kept) so
    sustained traffic cannot exhaust the filesystem. DECISION_WINDOW only
    bounds the in-memory state snapshot, not these files."""

    def __init__(self, home=None):
        self.home = Path(home or default_home())
        try:
            mb = float(os.environ.get("AGENTIC_ROUTER_MAX_LOG_MB", "") or MAX_LOG_MB)
        except ValueError:
            mb = MAX_LOG_MB
        self.max_log_bytes = max(mb, 1.0) * 1024 * 1024
        self.home.mkdir(parents=True, exist_ok=True)

    def _atomic_write(self, path, content):
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(str(tmp), str(path))

    def load_state(self):
        try:
            p = self.home / "state.json"
            if not p.exists():
                return None
            s = json.loads(p.read_text(encoding="utf-8"))
            return s if isinstance(s, dict) and "ucb" in s else None
        except (OSError, ValueError):
            return None

    def save_state(self, state):
        self._atomic_write(self.home / "state.json", json.dumps(state, ensure_ascii=False))

    def append_jsonl(self, name, obj):
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        p = self.home / name
        try:
            if p.exists() and p.stat().st_size + len(line) > self.max_log_bytes:
                # 尺寸轮转：当前文件挪到 .1（覆盖更旧的），新行写进新文件。
                # 轮转失败不影响请求路径（只影响磁盘占用）。
                old = self.home / (name + ".1")
                if old.exists():
                    old.unlink()
                os.replace(str(p), str(old))
        except OSError:
            pass
        with open(p, "a", encoding="utf-8") as f:
            f.write(line)

    def load_jsonl(self, name):
        try:
            p = self.home / name
            if not p.exists():
                return []
            return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        except (OSError, ValueError):
            return []


class Router:
    """Framework-free orchestration: decide() → settle() → correct()."""

    def __init__(self, store=None, home=None, mode=None, model_tiers=None,
                 cross_provider=True, prices=None, rng=None, exploration=None):
        self.store = store or Store(home)
        self.cross_provider = cross_provider
        self.exploration = EXPLORATION_EPS if exploration is None else exploration
        self.model_tiers = model_tiers or empty_tiers()
        self.prices = prices
        self._rng = rng
        self.state = self.store.load_state() or fresh_state()
        if mode in ("active", "shadow"):
            self.state["mode"] = mode

    @property
    def mode(self):
        return self.state["mode"]

    def set_mode(self, mode):
        if mode not in ("active", "shadow"):
            raise ValueError('mode must be "active" or "shadow"')
        self.state["mode"] = mode
        self._persist()

    def decide(self, text, provider=None, available_tiers=None):
        state = self.state
        cls = classify(text)
        f = features_of(cls, text)
        tiers = list(available_tiers) if available_tiers else ["fast", "mid", "strong"]

        recs = {"rule": recommend_rule(cls),
                "ucb": recommend_ucb(cls, state["ucb"], tiers),
                "lin": recommend_lin(cls, f, state["lin"], tiers),
                "knn": recommend_knn(cls, f, state["knn"], tiers)}

        state["exp3"]["linGraduated"] = state["lin"]["total"] >= LIN_GRADUATION
        state["exp3"]["knnGraduated"] = len(state["knn"]["pool"]) >= KNN_GRADUATION
        meta = meta_pick(recs, state["exp3"], self._rng)
        chosen_rec = meta["byName"].get(meta["chosen"]) or recs["rule"]
        tier = chosen_rec["tier"] or recs["rule"]["tier"]  # k-NN cold start falls back to rule
        explored = False
        # Tier-level epsilon exploration: EXP3 judges experts by realized reward,
        # which punishes the explorer (UCB). Once greedy experts dominate, nobody
        # samples non-first-choice tiers and the correct tier is never learned.
        if self.exploration > 0 and len(tiers) > 1 and random.random() < self.exploration:
            tier = tiers[random.randrange(len(tiers))]
            explored = True
        resolved = resolve_tier(tier, provider, self.model_tiers,
                                {"crossProvider": self.cross_provider})

        state["seq"] += 1
        decision = {"id": state["seq"], "ts": int(time.time() * 1000),
                    "type": cls["type"], "complexity": cls["complexity"],
                    "len": len(text or ""), "mode": state["mode"],
                    "chosen": meta["chosen"], "pChosen": meta["ps"].get(meta["chosen"], 0),
                    "pool": meta["pool"], "tier": tier, "used": resolved["used"],
                    "provider": resolved["provider"], "model": resolved["target"],
                    "note": resolved["note"] + (" [explore]" if explored else ""),
                    "reason": (chosen_rec or {}).get("reason", ""), "settled": False}
        state["decisions"].append(decision)
        if len(state["decisions"]) > DECISION_WINDOW:
            state["decisions"].pop(0)
        self._persist()
        self.store.append_jsonl("decisions.jsonl", decision)
        return decision

    def settle(self, decision_id, success=True, dt_ms=None, cost_yuan=0.0,
               tool_errors=0, retries=0):
        state = self.state
        d = next((x for x in state["decisions"] if x["id"] == decision_id and not x["settled"]), None)
        if not d:
            return None
        d["settled"] = True
        tag = "clean-close" if success is not False else "failed"
        reward, breakdown = reward_of({"toolErrors": tool_errors, "retries": retries,
                                       "costYuan": cost_yuan}, tag, dt_ms)
        rec = {"clsType": d["type"], "ucbTier": d["used"], "linTier": d["used"],
               "f": _rebuild_features(d), "reward": reward, "chosen": d["chosen"],
               "pChosen": d["pChosen"], "pool": d["pool"]}
        apply_reward(state, rec)
        d["reward"] = reward
        d["linLr"] = rec["linLr"]
        self._persist()
        self.store.append_jsonl("rewards.jsonl",
                                {"decisionId": d["id"], "ts": int(time.time() * 1000),
                                 "reward": reward, "breakdown": breakdown})
        return {"reward": reward, "breakdown": breakdown, "tier": d["used"]}

    def correct(self, decision_id, new_reward):
        state = self.state
        d = next((x for x in state["decisions"] if x["id"] == decision_id and x["settled"]), None)
        if not d:
            return None
        rec = {"clsType": d["type"], "ucbTier": d["used"], "linTier": d["used"],
               "f": _rebuild_features(d), "reward": d.get("reward", 0), "chosen": d["chosen"],
               "pChosen": d["pChosen"], "pool": d["pool"], "linLr": d.get("linLr")}
        result = correct_reward(state, rec, new_reward)
        d["reward"] = new_reward
        self._persist()
        return result

    def _persist(self):
        self.store.save_state(self.state)


# ─── 7. LiteLLM integration (CustomLogger hook) ─────────────────────────────

def _normalize_tiers(tier_models):
    """Accept {"fast": "model", ...} or model-name lists; build the tier catalog."""
    tiers = {}
    for tier, models in (tier_models or {}).items():
        if isinstance(models, str):
            models = [models]
        tiers[tier] = [m if "/" in m else "default/" + m for m in (models or [])]
    return {"providers": [{"provider": "default"}], "tiers": tiers}


def _last_user_text(messages):
    for m in reversed(messages or []):
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str) and c.strip():
            return c
        if isinstance(c, list):
            parts = [p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"]
            joined = "\n".join(x for x in parts if x)
            if joined.strip():
                return joined
    return ""


def _destination_budget_exhausted(user_api_key_dict, model):
    """Key-level per-model budgets (model_max_budget / model_spend) are enforced
    by the proxy against the *requested* model before this hook runs. A rewrite
    must not land on a destination model whose budget is already exhausted —
    the proxy would have rejected a direct request to it. Unknown/missing spend
    data never blocks (only a known, exhausted bucket does)."""
    if not user_api_key_dict or not model:
        return False
    max_budget = getattr(user_api_key_dict, "model_max_budget", None)
    if not isinstance(max_budget, (int, float)) or max_budget <= 0:
        return False  # no per-model budget configured for this key
    spend = getattr(user_api_key_dict, "model_spend", None)
    if isinstance(spend, dict) and isinstance(spend.get(model), (int, float)):
        return spend[model] >= max_budget
    return False


def _key_allowed_models(user_api_key_dict):
    """Collect the model allowlist the proxy attached to this key (key models,
    team models, team aliases). Empty set = unrestricted (or unknowable here)."""
    allowed = set()
    for attr in ("models", "allowed_team_models"):
        v = getattr(user_api_key_dict, attr, None)
        if isinstance(v, (list, tuple)):
            allowed.update(m for m in v if isinstance(m, str) and m)
    aliases = getattr(user_api_key_dict, "team_model_aliases", None)
    if isinstance(aliases, dict):
        allowed.update(m for m in aliases.values() if isinstance(m, str) and m)
    return allowed


_ID_KEYS = ("litellm_call_id", "request_id")
_NESTED_SOURCES = ("data", "request_data", "request_obj", "original_request",
                   "kwargs", "body", "litellm_params")


def _find_call_id(*sources):
    """Locate the call id across hook argument shapes: directly on the kwargs
    dict, or nested one level under data / request_data / request_obj / ..."""
    for src in sources:
        if not isinstance(src, dict):
            continue
        for k in _ID_KEYS:
            v = src.get(k)
            if v:
                return v
    for src in sources:
        if not isinstance(src, dict):
            continue
        for k in _NESTED_SOURCES:
            nested = src.get(k)
            if isinstance(nested, dict):
                for kk in _ID_KEYS:
                    v = nested.get(kk)
                    if v:
                        return v
    return None


class AgenticRouterHook(CustomLogger):
    """Learning model-tier router as a LiteLLM callback.

    - async_pre_call_hook: classify + decide; in active mode rewrites data["model"]
      (never to a model the key/team is not allowed to use, and never into a
      model listed in guarded_models — model-level guardrails are wired by the
      proxy for the requested model and a rewrite would skip their input checks)
    - async_post_call_success_hook: latency/cost → reward → flywheel
    - async_post_call_failure_hook: reward 0
    """

    MAX_PENDING = 500    # completed-but-untracked requests can't grow unbounded
    PENDING_TTL = 900.0  # seconds; stale entries are evicted on insert

    def __init__(self, router=None, mode="shadow", tier_models=None,
                 cross_provider=True, home=None, cost_multiplier=1.0,
                 guarded_models=None, **router_kwargs):
        if router is not None:
            self.router = router
        else:
            self.router = Router(home=home, mode=mode,
                                 model_tiers=_normalize_tiers(tier_models),
                                 cross_provider=cross_provider, **router_kwargs)
        self.cost_multiplier = cost_multiplier
        # 模型级 guardrail（PII/拦截）在 proxy 选定模型后才挂接：把请求改写到
        # 受保护模型会跳过它的输入检查。把这些模型列出来（构造参数或
        # AGENTIC_ROUTER_GUARDED_MODELS 环境变量，逗号分隔），active 模式拒绝
        # 路由进入它们——即使 key 的白名单允许。
        self.guarded_models = {m for m in (guarded_models or []) if isinstance(m, str) and m}
        self.guarded_models.update(
            m.strip() for m in os.environ.get("AGENTIC_ROUTER_GUARDED_MODELS", "").split(",")
            if m.strip())
        self._lock = threading.RLock()
        self._pending = {}  # call id → {id, ts, mode, train}

    def _track(self, request_id, decision_id, mode, train):
        with self._lock:
            now = time.time()
            if len(self._pending) > self.MAX_PENDING // 2:
                for k in [k for k, v in self._pending.items()
                          if now - v["ts"] > self.PENDING_TTL]:
                    del self._pending[k]
            self._pending[request_id] = {"id": decision_id, "ts": now,
                                         "mode": mode, "train": train}
            while len(self._pending) > self.MAX_PENDING:
                self._pending.pop(next(iter(self._pending)))  # drop oldest

    def _take(self, *sources):
        request_id = _find_call_id(*sources)
        if not request_id:
            return None
        with self._lock:
            entry = self._pending.pop(request_id, None)
        return entry

    async def async_pre_call_hook(self, user_api_key_dict=None, cache=None,
                                  data=None, call_type=None, **kwargs):
        if not isinstance(data, dict):
            return data
        text = _last_user_text(data.get("messages") or [])
        if not text:
            return data
        request_id = _find_call_id(data, kwargs) or id(data)
        with self._lock:
            decision = self.router.decide(text, "default")
        train = self.router.mode == "active"
        if self.router.mode == "active" and decision.get("model"):
            allowed = _key_allowed_models(user_api_key_dict)
            blocked = None
            if allowed and decision["model"] not in allowed:
                blocked = "key-model-restriction"
            elif decision["model"] in self.guarded_models:
                blocked = "guardrail-model-restriction"
            elif _destination_budget_exhausted(user_api_key_dict, decision["model"]):
                blocked = "budget-model-restriction"
            if blocked is None:
                data["model"] = decision["model"]
                data.setdefault("metadata", {})["agentic_router"] = {
                    "decision_id": decision["id"], "tier": decision["tier"],
                    "expert": decision["chosen"], "reason": decision["reason"]}
            else:
                # Key/team restricted, or the target model is guardrail-protected:
                # keep the caller's requested model, record the block for audits,
                # and mark no-train — the request is served by the original model,
                # so its outcome cannot be attributed to the rejected tier.
                train = False
                data.setdefault("metadata", {})["agentic_router"] = {
                    "decision_id": decision["id"], "tier": decision["tier"],
                    "model": decision["model"], "blocked": blocked,
                    "reason": decision["reason"]}
        self._track(request_id, decision["id"], self.router.mode, train)
        return data

    async def async_post_call_success_hook(self, user_api_key_dict=None, kwargs=None,
                                           response_obj=None, start_time=None,
                                           end_time=None, **kw):
        entry = self._take(kwargs, kw)
        if entry is None:
            return
        if not entry["train"]:
            return  # shadow or rejected rewrite: outcome belongs to the original model
        decision_id = entry["id"]
        dt_ms = None
        if start_time is not None and end_time is not None:
            dt_ms = (end_time - start_time).total_seconds() * 1000
        cost = (kwargs or {}).get("response_cost")
        with self._lock:
            self.router.settle(decision_id, success=True, dt_ms=dt_ms,
                               cost_yuan=(float(cost) if isinstance(cost, (int, float)) else 0.0)
                               * self.cost_multiplier)

    async def async_post_call_failure_hook(self, user_api_key_dict=None, request_obj=None,
                                           original_exception=None, user_model=None,
                                           start_time=None, end_time=None, **kwargs):
        entry = self._take(request_obj, kwargs)
        if entry is None:
            return
        if not entry["train"]:
            return  # shadow or rejected rewrite: outcome belongs to the original model
        decision_id = entry["id"]
        with self._lock:
            self.router.settle(decision_id, success=False)

    # -- ops helpers --
    def stats(self):
        with self._lock:
            s = self.router.state
            return {"mode": s["mode"], "decisions": s["seq"],
                    "ucbArms": len(s["ucb"]["arms"]), "linSamples": s["lin"]["total"],
                    "knnPool": len(s["knn"]["pool"]), "exp3Weights": dict(s["exp3"]["weights"]),
                    "pending": len(self._pending)}

    def activate(self):
        with self._lock:
            self.router.set_mode("active")

    def shadow(self):
        with self._lock:
            self.router.set_mode("shadow")
