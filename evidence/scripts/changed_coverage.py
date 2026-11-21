# Reports line and branch coverage of the executable lines a diff adds or changes.
# usage: changed_coverage.py <source checkout> <base sha> <coverage.json>
import json, re, subprocess, sys
src, base, cov = sys.argv[1:4]
data = json.load(open(cov))["files"]
diff = subprocess.run(["git", "-C", src, "diff", "-U0", base, "HEAD", "--", "litellm"], capture_output=True, text=True).stdout
changed, cur = {}, None
for line in diff.splitlines():
    if line.startswith("+++ b/"):
        cur = line[6:]; changed[cur] = set()
    m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
    if m and cur:
        start, n = int(m.group(1)), int(m.group(2) or 1)
        changed[cur].update(range(start, start + n))
tl = tc = tb = tbc = 0
for f, lines in changed.items():
    d = data.get(f)
    if d is None:
        print(f, "NOT MEASURED"); continue
    ex, miss = set(d["executed_lines"]), set(d["missing_lines"])
    stmts = sorted(lines & (ex | miss)); hit = [l for l in stmts if l in ex]
    br = [b for b in d.get("executed_branches", []) + d.get("missing_branches", []) if b[0] in lines]
    brm = [b for b in d.get("missing_branches", []) if b[0] in lines]
    tl += len(stmts); tc += len(hit); tb += len(br); tbc += len(br) - len(brm)
    print("%s\n  changed executable lines %d, covered %d, uncovered %s\n  branches from changed lines %d, covered %d, uncovered %s" % (
        f, len(stmts), len(hit), [l for l in stmts if l in miss], len(br), len(br) - len(brm), brm))
print("TOTAL lines %d/%d, branches %d/%d" % (tc, tl, tbc, tb))
