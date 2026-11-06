import { useState } from "react";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import drafts from "../../../tests/fixtures/workflows/drafts.json";
import { WorkflowDefinitionView } from "./WorkflowDefinitionView";
import type { WorkflowGraphProps } from "./WorkflowGraph";

const bundle = drafts["email-reply"].bundle;

function DefinitionHarness(props: Partial<WorkflowGraphProps>) {
  const [selectedNodeId, onNodeSelect] = useState<string | null>(null);
  return (
    <WorkflowDefinitionView
      bundle={bundle}
      definitionKey="snapshot-1"
      selectedNodeId={selectedNodeId}
      onNodeSelect={onNodeSelect}
      {...props}
    />
  );
}

describe("workflow definition view", () => {
  it("selects the SDK Python node and inspects its bindings, schemas and retry settings", () => {
    render(<DefinitionHarness />);
    fireEvent.click(screen.getByRole("button", { name: /Step 2: prepare_followup/ }));
    const inspector = within(screen.getByRole("complementary", { name: "Workflow definition inspector" }));
    expect(inspector.getByRole("heading", { name: "prepare_followup" })).toBeInTheDocument();
    expect(inspector.getByText("text", { exact: true })).toBeInTheDocument();
    expect(inspector.getByText("node_1", { exact: true })).toBeInTheDocument();
    expect(inspector.getByText("15 s")).toBeInTheDocument();
    expect(inspector.getByText("2", { exact: true })).toBeInTheDocument();
    expect(inspector.getByText("0.5 s")).toBeInTheDocument();
    fireEvent.click(inspector.getByText("Input schema", { exact: true }));
    expect(inspector.getByText(/"additionalProperties": false/)).toHaveTextContent('"text"');
    fireEvent.click(inspector.getByText("Literal value"));
    expect(inspector.getByText(/Review this draft for clarity and accuracy/)).toBeVisible();
    fireEvent.click(inspector.getByRole("button", { name: "Clear node selection" }));
    expect(inspector.getByText(/Select a step to inspect/)).toBeInTheDocument();
  });

  it("shows structured ChatCompletion output from the compiler", () => {
    render(<DefinitionHarness />);
    fireEvent.click(screen.getByRole("button", { name: /Step 3: review-model/ }));
    const inspector = within(screen.getByRole("complementary"));
    fireEvent.click(inspector.getByText("Output schema", { exact: true }));
    expect(inspector.getByText(/"strict": true/)).toHaveTextContent('"approved"');
    expect(inspector.getByText("node_2", { exact: true })).toBeInTheDocument();
  });

  it("inspects literal messages in a single-step SDK workflow", () => {
    render(<DefinitionHarness bundle={drafts["welcome-message"].bundle} />);
    fireEvent.click(screen.getByRole("button", { name: /Step 1: support-model/ }));
    const inspector = within(screen.getByRole("complementary"));
    fireEvent.click(inspector.getByRole("button", { name: "Literal value" }));
    expect(inspector.getByText(/Write a short welcome message/)).toBeVisible();
    expect(inspector.getByText(/No structured output schema/)).toBeInTheDocument();
  });

  it("updates parent-supplied node states without losing selection and ignores other snapshots", () => {
    const states = { node_2: "running" as const };
    const { rerender } = render(<DefinitionHarness execution={{ definitionKey: "snapshot-1", states }} />);
    const pythonNode = screen.getByRole("button", { name: /Step 2: prepare_followup/ });
    fireEvent.click(pythonNode);
    expect(within(pythonNode).getByText("Running")).toBeInTheDocument();
    rerender(<DefinitionHarness execution={{ definitionKey: "snapshot-1", states: { node_2: "failed" } }} />);
    expect(within(pythonNode).getByText("Failed")).toBeInTheDocument();
    expect(pythonNode).toHaveAttribute("aria-pressed", "true");
    rerender(<DefinitionHarness execution={{ definitionKey: "other-snapshot", states }} />);
    expect(screen.queryByText("Running")).not.toBeInTheDocument();
    expect(screen.queryByText("Failed")).not.toBeInTheDocument();
  });

  it("zooms and fits through the graph controls", () => {
    render(<DefinitionHarness />);
    const initialZoom = screen.getByLabelText("Zoom level").textContent;
    fireEvent.click(screen.getByRole("button", { name: "Zoom in" }));
    expect(screen.getByLabelText("Zoom level")).not.toHaveTextContent(initialZoom!);
    fireEvent.click(screen.getByRole("button", { name: "Fit to view" }));
    expect(screen.getByLabelText("Zoom level")).toHaveTextContent(initialZoom!);
  });

  it.each([
    { loading: true, text: "Loading workflow definition" },
    { bundle: null, text: "No workflow to visualize" },
    { bundle: { ...bundle, nodes: [{ type: "if" }] }, text: "This definition is not supported yet" },
    { bundle: {}, text: "Unable to read this definition" },
    { error: "Definition request failed", text: "Definition request failed" },
  ])("shows $text without a stale graph or inspector", ({ text, ...props }) => {
    render(<DefinitionHarness {...props} />);
    expect(screen.getByText(text)).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Workflow graph" })).not.toBeInTheDocument();
    expect(screen.queryByRole("complementary")).not.toBeInTheDocument();
  });
});
