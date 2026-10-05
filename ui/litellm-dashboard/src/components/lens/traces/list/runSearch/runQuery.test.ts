import { describe, expect, it } from "vitest";

import { fieldValues } from "@/components/shared/search/evaluate";
import { run, runs } from "./__fixtures__/runs";
import { filterRuns, RUN_INDEX } from "./runQuery";

const ids = (query: string) => filterRuns(runs, query).map((r) => r.trace_id);

describe("filterRuns", () => {
  it("matches free text against input, name and trace id, but not agents or models", () => {
    expect(ids("REFUND")).toEqual(["aaa111"]);
    expect(ids("lead")).toEqual(["bbb222"]);
    expect(ids("ccc3")).toEqual(["ccc333"]);
    expect(ids("triage")).toEqual([]);
    expect(ids("gpt-5")).toEqual([]);
  });

  it("splits runs by status, judged by recorded errors", () => {
    expect(ids("status:error")).toEqual(["bbb222"]);
    expect(ids("-status:error")).toEqual(["aaa111", "ccc333"]);
    expect(ids("status:OK")).toEqual(["aaa111", "ccc333"]);
  });

  it("reads agents from the trace, falling back to the service, and models from the run", () => {
    expect(ids("agent:triage")).toEqual(["aaa111"]);
    expect(ids("agent:cron")).toEqual(["ccc333"]);
    expect(ids("model:gpt-5")).toEqual(["aaa111"]);
    expect(ids("name:support")).toEqual(["aaa111"]);
  });

  it("searches the readable input text", () => {
    expect(ids('input:"*vector stores"')).toEqual(["bbb222"]);
    expect(ids("trace_id:bbb222")).toEqual(["bbb222"]);
  });

  it("combines field clauses with free text", () => {
    expect(ids("agent:*e* -status:error refund")).toEqual(["aaa111"]);
  });
});

describe("RUN_INDEX values", () => {
  it("lists the loaded agents and statuses for autocomplete", () => {
    expect(fieldValues(RUN_INDEX, runs, "agent")).toEqual(["billing-agent", "cron", "researcher", "triage"]);
    expect(fieldValues(RUN_INDEX, runs, "status")).toEqual(["error", "ok"]);
    expect(fieldValues(RUN_INDEX, [run({ models: [] })], "model")).toEqual([]);
  });
});
