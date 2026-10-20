import json, random
def lst(root, seed, canary=None):
    rnd = random.Random(seed)
    rows = [{"path": f"{root}/pkg{i//40}/mod_{i}.{['py','ts','md','json','yaml','lock'][i%6]}", "type": "file", "size": rnd.randint(100, 90000),
             "mtime": f"2026-10-{1+i%18:02d}T12:{i%60:02d}:00Z", "owner": "dev", "mode": "0644"} for i in range(400)]
    if canary: rows[263] = {**rows[263], "owner": canary, "size": 77777}
    return json.dumps(rows)
T = {"type": "function", "function": {"name": "list_dir", "description": "List files as JSON", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
RT = {"type": "function", **T["function"]}
QK = "Which file in /srv/app is owned by zeta-canary-7731? Reply with its exact path and size in bytes. Use the tool."
LK = lst("/srv/app", 11, "zeta-canary-7731")
INSTR = "TASK: Find the single ERROR line in the log below and reply with exactly 'ANSWER: <request id>' and nothing else."
TAIL = "FINAL CONSTRAINT: write the request id in UPPERCASE."
LOG = "\n".join("2026-10-19T12:06:57Z ERROR svc-payments upstream timeout id=req00417 latency_ms=30000" if i == 417 else
    f"2026-10-19T12:{i//60%60:02d}:{i%60:02d}Z INFO svc-{i%7} request ok id=req{i:05d} latency_ms={40+i%90}" for i in range(600))
