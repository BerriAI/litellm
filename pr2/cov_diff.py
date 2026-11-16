"""Changed executable-line coverage and branch coverage of touched functions, from coverage json + git diff."""
import ast, json, re, subprocess, sys
BASE, COV = sys.argv[1], json.load(open(sys.argv[2]))
files = subprocess.check_output(["git", "diff", "--name-only", "--diff-filter=AM", BASE, "HEAD", "--", "litellm/*.py"], text=True).split()
tot = dict(el=0, ec=0, br=0, bc=0); miss_all = []
print("| File | Changed executable lines covered | Touched functions | Branches covered in touched functions | Uncovered changed lines |\n|---|---|---|---|---|")
for f in files:
    diff = subprocess.check_output(["git", "diff", "-U0", BASE, "HEAD", "--", f], text=True)
    changed = set()
    for m in re.finditer(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", diff, re.M):
        s, n = int(m.group(1)), int(m.group(2) or 1)
        changed.update(range(s, s + n))
    c = COV["files"].get(f)
    if c is None:
        print(f"| {f} | not measured | | | |"); continue
    execd, missing = set(c["executed_lines"]), set(c["missing_lines"])
    el = changed & (execd | missing); ec = changed & execd
    tree = ast.parse(open(f).read())
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    touched = [n for n in funcs if changed & set(range(n.lineno, n.end_lineno + 1)) & (execd | missing | {n.lineno})]
    span = set().union(*(range(n.lineno, n.end_lineno + 1) for n in touched)) if touched else set()
    eb = [b for b in c.get("executed_branches", []) if b[0] in span]; mb = [b for b in c.get("missing_branches", []) if b[0] in span]
    tot["el"] += len(el); tot["ec"] += len(ec); tot["br"] += len(eb) + len(mb); tot["bc"] += len(eb)
    ml = sorted(el - ec); miss_all += [(f, ml, sorted(map(tuple, mb)))]
    pct = lambda a, b: f"{a}/{b} ({100*a/b:.1f}%)" if b else "0/0"
    print(f"| {f} | {pct(len(ec), len(el))} | {len(touched)} | {pct(len(eb), len(eb)+len(mb))} | {','.join(map(str, ml)) or 'none'} |")
print(f"| **Total** | {tot['ec']}/{tot['el']} ({100*tot['ec']/max(tot['el'],1):.1f}%) | | {tot['bc']}/{tot['br']} ({100*tot['bc']/max(tot['br'],1):.1f}%) | |")
print("\nMissing branches in touched functions (from_line -> to_line):")
for f, ml, mb in miss_all:
    if mb: print(f, mb)
