import { describe, expect, it } from "vitest";
import drafts from "../../../tests/fixtures/workflows/drafts.json";
import { executionStates, readWorkflowDefinition, sequentialConnections } from "./definition";

const bundle = drafts["email-reply"].bundle;

describe("compiled workflow definitions", () => {
  it("preserves the SDK bundle fields needed by the inspector", () => {
    const definition = readWorkflowDefinition(bundle);
    expect(definition.kind).toBe("ready");
    if (definition.kind !== "ready") return;
    expect(definition.bundle).toEqual(bundle);
  });

  it("connects adjacent nodes in compiler order, including when IDs are not sorted", () => {
    const definition = readWorkflowDefinition(bundle);
    if (definition.kind !== "ready") throw new Error("Expected a usable SDK fixture");
    const nodes = [...definition.bundle.nodes].reverse();
    expect(sequentialConnections(nodes)).toEqual(
      nodes.slice(1).map((node, index) => ({ source: nodes[index].id, target: node.id })),
    );
    expect(sequentialConnections(nodes.slice(0, 1))).toEqual([]);
  });

  it.each([null, { ...bundle, nodes: [] }])("handles an empty definition", (value) => {
    expect(readWorkflowDefinition(value).kind).toBe("empty");
  });

  it.each([
    { ...bundle, contract_version: "future" },
    { ...bundle, nodes: [{ id: "branch", type: "if" }] },
    { ...bundle, nodes: [{ id: "parallel", type: "parallel" }] },
  ])("declines unsupported semantics instead of drawing a sequential graph", (value) => {
    expect(readWorkflowDefinition(value).kind).toBe("unsupported");
  });

  it.each([
    {},
    { ...bundle, nodes: [bundle.nodes[0], bundle.nodes[0]] },
    { ...bundle, nodes: [{ ...bundle.nodes[0], messages: { from: "$input" } }] },
  ])("rejects malformed bundles and ambiguous node IDs", (value) => {
    expect(readWorkflowDefinition(value).kind).toBe("invalid");
  });

  it("requires the overlay to match the displayed definition", () => {
    const overlay = { definitionKey: "snapshot-1", states: { node_2: "failed" as const } };
    expect(executionStates("snapshot-1", overlay)).toEqual(overlay.states);
    expect(executionStates("snapshot-2", overlay)).toBeUndefined();
    expect(executionStates("snapshot-1")).toBeUndefined();
  });
});
