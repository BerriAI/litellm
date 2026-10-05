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

  it("filters recorded errors separately from root status", () => {
    expect(ids("has_error:true")).toEqual(["bbb222"]);
    expect(ids("-has_error:true")).toEqual(["aaa111", "ccc333"]);
    expect(ids("has_error:FALSE")).toEqual(["aaa111", "ccc333"]);
    expect(ids("root_status:ok has_error:true")).toEqual(["bbb222"]);
    expect(ids("root_status:error")).toEqual([]);
  });

  it("reads agent labels, service and models independently", () => {
    expect(ids("agent:triage")).toEqual(["aaa111"]);
    expect(ids("agent:cron")).toEqual([]);
    expect(ids("service:cron")).toEqual(["ccc333"]);
    expect(ids("model:gpt-5")).toEqual(["aaa111"]);
    expect(ids("name:support")).toEqual(["aaa111"]);
  });

  it("searches the readable input text", () => {
    expect(ids('input:"*vector stores"')).toEqual(["bbb222"]);
    expect(ids("trace_id:bbb222")).toEqual(["bbb222"]);
  });

  it("combines field clauses with free text", () => {
    expect(ids("agent:*e* -has_error:true refund")).toEqual(["aaa111"]);
  });
});

describe("RUN_INDEX values", () => {
  it("lists the loaded agents and statuses for autocomplete", () => {
    expect(fieldValues(RUN_INDEX, runs, "agent")).toEqual(["billing-agent", "researcher", "triage"]);
    expect(fieldValues(RUN_INDEX, runs, "root_status")).toEqual(["ok"]);
    expect(fieldValues(RUN_INDEX, [run({ models: [] })], "model")).toEqual([]);
  });
});
