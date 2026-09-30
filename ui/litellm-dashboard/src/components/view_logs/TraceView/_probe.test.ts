import { writeFileSync } from "fs";
import { it } from "vitest";
const out: string[] = [];
const log = (...a: unknown[]) => out.push(a.join(" "));

import big from "./__fixtures__/_big_probe.json";
import research from "./__fixtures__/research_trace.json";
import swarm from "./__fixtures__/swarm_trace.json";
import type { Trace } from "./traceTypes";
import { buildOutline, fmtCost, fmtMs, initialOutlineSelection, outlineRows } from "./traceUtils";

const show = (t: Trace) => {
  const all = buildOutline(t);
  const sel = initialOutlineSelection(all);
  const rows = outlineRows(t, sel.ui);
  log(t.summary.name, all.length, rows.length, "sel", sel.selectedId);
  for (const r of rows.slice(0, 60))
    log(
      "  ".repeat(r.depth) +
        `[${r.glyph}] ${r.label}${r.error ? " !" : ""} | ${r.detail ?? ""} | ${r.meta ?? ""} ${r.metaError ?? ""} | ${r.durationMs != null ? fmtMs(r.durationMs) : ""}`,
    );
};
it("probe", () => {
  show(research as Trace);
  show(swarm as Trace);
  show(big as unknown as Trace);
  log(fmtCost(0.128), fmtCost(0.0029), fmtCost(1.142));
  writeFileSync("/tmp/agent-tracing/probe.txt", out.join("\n"));
});
