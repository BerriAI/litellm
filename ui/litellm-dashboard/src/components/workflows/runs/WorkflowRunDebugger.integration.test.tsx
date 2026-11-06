import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { renderWithProviders, testQueryClient, chooseSelectOption } from "../../../../tests/test-utils";
import type { RunSteps, WorkflowRunDataSource, WorkflowRunGraphState } from "./contracts";
import {
  createWorkflowRunFixtureSource,
  fixtureDefinitions,
  fixtureTraces,
  fixtureWorkflowId,
} from "../../../../tests/fixtures/workflow-runs/data";
import { WorkflowRunDebugger } from "./WorkflowRunDebugger";

function Harness({
  source,
  initialRun = null,
  graphState,
}: {
  source: WorkflowRunDataSource;
  initialRun?: string | null;
  graphState?: (state: WorkflowRunGraphState | null) => void;
}) {
  const [runId, setRunId] = useState(initialRun);
  return (
    <WorkflowRunDebugger
      workflowId={fixtureWorkflowId}
      dataSource={source}
      runId={runId}
      onSelectRun={setRunId}
      onGraphStateChange={graphState}
      pageSize={2}
    />
  );
}

beforeEach(() => testQueryClient.clear());
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("workflow run debugging", () => {
  it("paginates workflow history and resets the cursor when status changes", async () => {
    const source = createWorkflowRunFixtureSource();
    const listRuns = vi.fn(source.listRuns);
    renderWithProviders(<Harness source={{ ...source, listRuns }} />);
    expect(await screen.findByRole("button", { name: "run-fixture-completed" })).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(await screen.findByRole("button", { name: "run-fixture-failed" })).toBeVisible();
    const user = userEvent.setup();
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Filter runs by status" }), "Failed");
    await waitFor(() =>
      expect(listRuns).toHaveBeenLastCalledWith(
        expect.objectContaining({ workflowId: fixtureWorkflowId, status: "failed", cursor: null }),
      ),
    );
    expect(screen.getByText("Page 1")).toBeVisible();
    expect(screen.getByRole("button", { name: "Previous" })).toBeDisabled();
    expect(screen.queryByRole("button", { name: "run-fixture-completed" })).not.toBeInTheDocument();
  });

  it("shows typed Python values without a spend action and emits the pinned definition", async () => {
    const graphState = vi.fn<(state: WorkflowRunGraphState | null) => void>();
    renderWithProviders(
      <Harness source={createWorkflowRunFixtureSource()} initialRun="run-fixture-completed" graphState={graphState} />,
    );
    fireEvent.click(await screen.findByRole("button", { name: "Inspect step 2 node_2" }));
    const details = screen.getByRole("region", { name: "Step details" });
    expect(within(details).getByLabelText("Input JSON")).toHaveTextContent("false");
    expect(within(details).getByLabelText("Output JSON")).toHaveTextContent("0");
    expect(within(details).getByText("$0.000000")).toBeVisible();
    expect(within(details).getByText("25 ms")).toBeVisible();
    expect(within(details).queryByRole("button", { name: "View spend record" })).not.toBeInTheDocument();
    await waitFor(() =>
      expect(graphState).toHaveBeenLastCalledWith(
        expect.objectContaining({
          definition: fixtureDefinitions["snapshot-fixture-v1"],
          selection: {
            runId: "run-fixture-completed",
            snapshotId: "snapshot-fixture-v1",
            stepId: "step-completed-2",
            nodeId: "node_2",
          },
        }),
      ),
    );
  });

  it("retains independent selection for two executions of one node", async () => {
    const graphState = vi.fn<(state: WorkflowRunGraphState | null) => void>();
    renderWithProviders(
      <Harness
        source={createWorkflowRunFixtureSource()}
        initialRun="run-fixture-repeated-node"
        graphState={graphState}
      />,
    );
    fireEvent.click(await screen.findByRole("button", { name: "Inspect step 2 node_1" }));
    expect(screen.getByLabelText("Output JSON")).toHaveTextContent('"Second execution"');
    expect(graphState).toHaveBeenLastCalledWith(
      expect.objectContaining({
        selection: expect.objectContaining({ stepId: "step-repeat-b", nodeId: "node_1" }),
        executions: [
          { stepId: "step-repeat-a", nodeId: "node_1", state: "completed" },
          { stepId: "step-repeat-b", nodeId: "node_1", state: "completed" },
        ],
      }),
    );
  });

  it("opens only the exact authorized spend-detail request, including its original timestamp", async () => {
    const network = vi
      .fn<typeof fetch>()
      .mockResolvedValue(
        Response.json({ messages: [{ content: "Stored prompt" }], response: { content: "Stored response" } }),
      );
    vi.stubGlobal("fetch", network);
    renderWithProviders(<Harness source={createWorkflowRunFixtureSource()} initialRun="run-fixture-completed" />);
    fireEvent.click(await screen.findByRole("button", { name: "View spend record" }));
    expect(await screen.findByLabelText("Spend record fixture-request-summary JSON")).toHaveTextContent(
      "Stored response",
    );
    expect(network).toHaveBeenCalledTimes(1);
    const [url, init] = network.mock.calls[0];
    const parsed = new URL(String(url), "https://gateway.example.test");
    expect(parsed.pathname).toBe("/spend/logs/ui/fixture-request-summary");
    expect(parsed.searchParams.get("start_date")).toBe(fixtureTraces["run-fixture-completed"].steps[0].started_at);
    expect(new Headers(init?.headers).get("Authorization")).toBe("Bearer 123");
  });

  it("shows the failed step and keeps its trace available when the pinned definition fails", async () => {
    const source = createWorkflowRunFixtureSource();
    const getDefinition = vi.fn(source.getDefinition).mockRejectedValueOnce(new Error("Missing snapshot"));
    renderWithProviders(<Harness source={{ ...source, getDefinition }} initialRun="run-fixture-failed" />);
    expect(await screen.findByText("ValidationError: summary must be a string. Received null.")).toBeVisible();
    fireEvent.click(await screen.findByRole("button", { name: "Retry definition" }));
    await waitFor(() => expect(screen.queryByRole("button", { name: "Retry definition" })).not.toBeInTheDocument());
    expect(getDefinition).toHaveBeenLastCalledWith(
      expect.objectContaining({ snapshotId: "snapshot-fixture-v1", runId: "run-fixture-failed" }),
    );
    expect(screen.getByLabelText("Output JSON")).toHaveTextContent("null");
  });

  it("does not display a late response from the previously selected run", async () => {
    const source = createWorkflowRunFixtureSource();
    const delayed = Promise.withResolvers<RunSteps>();
    const getSteps: WorkflowRunDataSource["getSteps"] = (request) =>
      request.runId === "run-fixture-running" ? delayed.promise : source.getSteps(request);
    const graphState = vi.fn<(state: WorkflowRunGraphState | null) => void>();
    renderWithProviders(<Harness source={{ ...source, getSteps }} graphState={graphState} />);
    fireEvent.click(await screen.findByRole("button", { name: "run-fixture-running" }));
    expect(screen.getByText("Loading step trace…")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "run-fixture-completed" }));
    expect(await screen.findByRole("button", { name: "Inspect step 3 node_3" })).toBeVisible();
    await act(async () => delayed.resolve(fixtureTraces["run-fixture-running"]));
    expect(graphState).toHaveBeenLastCalledWith(expect.objectContaining({ runId: "run-fixture-completed" }));
    expect(screen.getByRole("region", { name: "Run inspector" })).not.toHaveTextContent("run-fixture-running");
  });

  it("polls active runs and stops after the gateway reports completion", async () => {
    const source = createWorkflowRunFixtureSource();
    const running = fixtureTraces["run-fixture-running"];
    const completed: RunSteps = {
      ...running,
      state: "completed",
      steps: running.steps.map((step) => ({ ...step, state: "completed", finished_at: step.deadline })),
    };
    const getSteps = vi.fn(source.getSteps).mockResolvedValueOnce(running).mockResolvedValue(completed);
    vi.useFakeTimers({ toFake: ["setInterval", "clearInterval"] });
    renderWithProviders(<Harness source={{ ...source, getSteps }} initialRun={running.run_id} />);
    expect(await screen.findByText("Updates every 2 seconds")).toBeVisible();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    await waitFor(() => expect(screen.queryByText("Updates every 2 seconds")).not.toBeInTheDocument());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(6000);
    });
    expect(getSteps).toHaveBeenCalledTimes(2);
  });

  it("shows an empty pending trace without inventing completed steps or cost", async () => {
    renderWithProviders(<Harness source={createWorkflowRunFixtureSource()} initialRun="run-fixture-pending" />);
    expect(await screen.findByText("No steps have been reported yet.")).toBeVisible();
    const inspector = screen.getByRole("region", { name: "Run inspector" });
    expect(within(inspector).getByText("Pending")).toBeVisible();
    expect(within(inspector).getByText("Not reported")).toBeVisible();
    expect(within(inspector).queryByRole("button", { name: "View spend record" })).not.toBeInTheDocument();
  });

  it("clears a prior session's trace while the replacement source loads", async () => {
    const source = createWorkflowRunFixtureSource();
    const graphState = vi.fn<(state: WorkflowRunGraphState | null) => void>();
    const rendered = renderWithProviders(
      <Harness source={source} initialRun="run-fixture-completed" graphState={graphState} />,
    );
    expect(await screen.findByRole("button", { name: "Inspect step 3 node_3" })).toBeVisible();
    const pending = Promise.withResolvers<RunSteps>();
    const replacement = { ...source, cacheKey: "another-authenticated-session", getSteps: () => pending.promise };
    rendered.rerender(<Harness source={replacement} initialRun="run-fixture-completed" graphState={graphState} />);
    expect(screen.queryByRole("region", { name: "Run inspector" })).not.toBeInTheDocument();
    expect(graphState).toHaveBeenLastCalledWith(null);
    await act(async () => pending.resolve(fixtureTraces["run-fixture-completed"]));
    expect(await screen.findByRole("region", { name: "Run inspector" })).toBeVisible();
  });
});
