#!/usr/bin/env python3
import json, sys, os

def color(s, c):
    codes = {"red":"\033[31m","green":"\033[32m","yellow":"\033[33m","cyan":"\033[36m","bold":"\033[1m","reset":"\033[0m"}
    return f"{codes.get(c,'')}{s}{codes['reset']}"

def main():
    path = os.path.join("local","artifacts","mvp","mvp_report.json")
    try:
        with open(path,"r",encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(color(f"No artifact found at {path}: {e}","red"))
        return 1
    checks = data.get("checks",[])
    if not checks:
        print(color("No checks recorded.","yellow"))
        return 1
    failed = 0
    print(color("\n== Readiness Summary ==","bold"))
    for c in checks:
        name = c.get("name")
        ok = c.get("ok")
        skipped = c.get("skipped")
        if skipped:
            mark = color("⏭ SKIP","yellow")
        elif ok:
            mark = color("✅ PASS","green")
        else:
            mark = color("❌ FAIL","red")
            failed += 1
        print(f" {mark}  {name}")
    print()
    if failed:
        print(color(f"Overall: {failed} failing check(s)","red"))
    else:
        print(color("Overall: all checks PASS","green"))
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())

