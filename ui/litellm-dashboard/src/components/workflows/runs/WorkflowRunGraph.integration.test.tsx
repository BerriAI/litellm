import { beforeEach, describe, expect, it } from "vitest";
import { fireEvent, screen, within } from "@testing-library/react";
import { renderWithProviders, testQueryClient } from "../../../../tests/test-utils";
import { createWorkflowRunFixtureSource, fixtureWorkflowId } from "../../../../tests/fixtures/workflow-runs/data";
import { WorkflowRunDebugger } from "./WorkflowRunDebugger";
import { WorkflowRunGraphPanel } from "./WorkflowRunGraphPanel";
import type { WorkflowRunGraphState } from "./contracts";

beforeEach(() => testQueryClient.clear());

function renderGraph(state: WorkflowRunGraphState | null, onSelectStep: (stepId: string | null) => void) {
  return <WorkflowRunGraphPanel state={state} onSelectStep={onSelectStep} />;
}

describe("run trace and shared graph integration", () => {
  it("selects a reported step from the pinned graph and clears details for an unreported node", async () => {
    renderWithProviders(
      <WorkflowRunDebugger
        workflowId={fixtureWorkflowId}
        dataSource={createWorkflowRunFixtureSource()}
        runId="run-fixture-running"
        onSelectRun={() => undefined}
        renderGraph={renderGraph}
      />,
    );
    fireEvent.click(await screen.findByRole("button", { name: "Pinned workflow" }));
    const pythonNode = await screen.findByRole("button", { name: "Step 2: format_summary (Python)" });
    fireEvent.click(pythonNode);
    const details = screen.getByRole("region", { name: "Step details" });
    expect(within(details).getByText("step-running-2")).toBeVisible();
    expect(within(details).getByLabelText("Input JSON")).toHaveTextContent('""');
    expect(within(details).getByText("In progress")).toBeVisible();
    expect(pythonNode).toHaveAttribute("aria-pressed", "true");
    expect(within(pythonNode).getByText("Running")).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Step 3: review-model (ChatCompletion)" }));
    expect(screen.queryByRole("region", { name: "Step details" })).not.toBeInTheDocument();
    expect(screen.getByText("No step has been reported for node_3.")).toBeVisible();
  });
});
