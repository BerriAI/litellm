import { describe, expect, it } from "vitest";
import draft from "../../../tests/fixtures/workflows/draft";
import { readWorkflowDefinition, sequentialConnections } from "./definition";

const bundle = draft.bundle;

describe("compiled workflow definitions", () => {
  it("preserves the SDK bundle and connects nodes in compiler order, not ID order", () => {
    const definition = readWorkflowDefinition(bundle);
    expect(definition.kind).toBe("ready");
    if (definition.kind !== "ready") return;
    expect(definition.bundle).toEqual(bundle);
    const [first, middle, last] = definition.bundle.nodes;
    expect(sequentialConnections([last, first, middle])).toEqual([
      { source: last.id, target: first.id },
      { source: first.id, target: middle.id },
    ]);
    expect(sequentialConnections([first])).toEqual([]);
  });

  it.each([
    ["empty", { ...bundle, nodes: [] }],
    ["unsupported", { ...bundle, contract_version: "future" }],
    ["invalid", { ...bundle, nodes: [bundle.nodes[0], bundle.nodes[0]] }],
    ["invalid", { ...bundle, nodes: [{ ...bundle.nodes[0], messages: { from: "$input" } }] }],
  ])("classifies boundary cases as %s", (kind, value) => {
    expect(readWorkflowDefinition(value).kind).toBe(kind);
  });
});
