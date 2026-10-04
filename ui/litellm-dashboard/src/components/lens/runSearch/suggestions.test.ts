import { describe, expect, it } from "vitest";

import { run, runs } from "./__fixtures__/runs";
import { suggest, type Suggestion, type SuggestionMenu } from "./suggestions";

const atEnd = (text: string, source = runs) => suggest(text, text.length, source);
const items = (menu: SuggestionMenu | null): Suggestion[] => menu?.groups.flatMap((g) => g.items) ?? [];
const labels = (menu: SuggestionMenu | null) => items(menu).map((item) => item.label);
const apply = (text: string, item: Suggestion) => text.slice(0, item.from) + item.insert + text.slice(item.to);

describe("suggest", () => {
  it("offers every field, grouped, in an empty box", () => {
    const menu = atEnd("");
    expect(menu?.groups.map((g) => g.heading)).toEqual(["Run attributes", "Content", "Identity"]);
    expect(labels(menu)).toEqual(["name", "agent", "status", "model", "input", "trace_id"]);
    expect(menu?.showOperators).toBe(false);
  });

  it("narrows fields by the typed prefix and completes the key with a colon", () => {
    const menu = atEnd("refund AG");
    expect(labels(menu)).toEqual(["agent"]);
    expect(apply("refund AG", items(menu)[0])).toBe("refund agent:");
    expect(items(menu)[0].completesClause).toBe(false);
  });

  it("keeps a leading minus when completing a negated key", () => {
    expect(apply("-st", items(atEnd("-st"))[0])).toBe("-status:");
  });

  it("offers nothing for free text that names no field", () => {
    expect(atEnd("refund")).toBeNull();
    expect(atEnd("foo:bar")).toBeNull();
  });

  it("offers the loaded values and the operator help once a key has its colon", () => {
    const menu = atEnd("agent:");
    expect(menu?.groups.map((g) => g.heading)).toEqual(["agent"]);
    expect(labels(menu)).toEqual(["billing-agent", "cron", "researcher", "triage"]);
    expect(menu?.showOperators).toBe(true);
  });

  it("narrows values by substring, ignoring wildcards, and drops operator help once a value is typed", () => {
    const menu = atEnd("-agent:*AR*");
    expect(labels(menu)).toEqual(["researcher"]);
    expect(menu?.showOperators).toBe(false);
    expect(atEnd("agent:zzz")).toBeNull();
  });

  it("replaces only the value, adds a separating space at the end, and finishes the clause", () => {
    const item = items(atEnd("-agent:bil"))[0];
    expect(apply("-agent:bil", item)).toBe("-agent:billing-agent ");
    expect(item.completesClause).toBe(true);
    const text = "agent:tri refund";
    expect(apply(text, items(suggest(text, 9, runs))[0])).toBe("agent:triage refund");
  });

  it("quotes a value that contains spaces", () => {
    const spaced = [run({ name: "Order lookup" })];
    expect(apply("name:", items(atEnd("name:", spaced))[0])).toBe('name:"Order lookup" ');
  });

  it("does not list values for free-form fields but still explains the operators", () => {
    const menu = atEnd("input:");
    expect(menu?.groups).toEqual([]);
    expect(menu?.showOperators).toBe(true);
    expect(atEnd("input:x")).toBeNull();
  });

  it("offers fields in the gap between tokens, but nothing mid-token", () => {
    const text = "refund  status:ok";
    expect(apply(text, items(suggest(text, 7, runs))[1])).toBe("refund agent: status:ok");
    expect(suggest(text, 3, runs)).toBeNull();
    expect(suggest("status:ok", 0, runs)).toBeNull();
  });
});
