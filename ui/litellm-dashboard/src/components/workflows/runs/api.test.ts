import { describe, expect, it, vi } from "vitest";
import { createApiClient } from "@/lib/http/client";
import { createWorkflowRunApi } from "./api";
import { runStepsSchema } from "./contracts";
import {
  fixtureDefinitions,
  fixtureRunSummaries,
  fixtureTraces,
  fixtureWorkflowId,
} from "../../../../tests/fixtures/workflow-runs/data";
import { formatCost, stepDuration } from "./presentation";

function setup(body: unknown) {
  const transport = vi.fn<typeof fetch>().mockImplementation(async () => Response.json(body));
  const client = createApiClient({ getBaseUrl: () => "https://gateway.example.test", fetchImpl: transport });
  const api = createWorkflowRunApi({ accessToken: "test-token", teamId: "test-team", client });
  const signal = new AbortController().signal;
  return { transport, api, signal };
}

describe("workflow run API boundary", () => {
  it("scopes and paginates the run list without querying spend logs", async () => {
    const run = fixtureRunSummaries[0];
    const { api, transport, signal } = setup({ runs: [{ ...run, status: run.state }], next_cursor: "opaque-next" });
    const request = {
      workflowId: fixtureWorkflowId,
      cursor: "opaque-current",
      status: "completed" as const,
      limit: 10,
      signal,
    };
    const result = await api.listRuns(request);
    expect(result).toEqual({ runs: [run], next_cursor: "opaque-next" });
    const [url, init] = transport.mock.calls[0];
    const parsed = new URL(String(url));
    expect(parsed.pathname).toBe("/v1/workflows/runs");
    const expectedQuery = {
      team_id: "test-team",
      workflow_id: fixtureWorkflowId,
      after: "opaque-current",
      status: "completed",
      limit: "10",
    };
    expect(Object.fromEntries(parsed.searchParams)).toEqual(expectedQuery);
    expect(init?.signal).toBe(signal);
    expect(new Headers(init?.headers).get("Authorization")).toBe("Bearer test-token");
    expect(transport).toHaveBeenCalledTimes(1);
  });

  it("rejects cross-workflow rows", async () => {
    const { api, signal } = setup({
      runs: [{ ...fixtureRunSummaries[0], workflow_id: "other", status: "completed" }],
      next_cursor: null,
    });
    const request = { workflowId: fixtureWorkflowId, cursor: null, status: "all" as const, limit: 20, signal };
    await expect(api.listRuns(request)).rejects.toThrow("different workflow");
  });

  it("retrieves the exact run trace and rejects a different run", async () => {
    const trace = fixtureTraces["run-fixture-completed"];
    const { api, transport, signal } = setup(trace);
    await expect(api.getSteps({ runId: trace.run_id, signal })).resolves.toEqual(trace);
    expect(String(transport.mock.calls[0][0])).toContain(`/runs/${trace.run_id}/steps`);
    await expect(api.getSteps({ runId: "another-run", signal })).rejects.toThrow("different run");
  });

  it("rejects a draft or a different workflow in place of the pinned snapshot", async () => {
    const pinned = fixtureDefinitions["snapshot-fixture-v1"];
    const { api, transport, signal } = setup(pinned);
    const request = {
      runId: "run-fixture-completed",
      workflowId: fixtureWorkflowId,
      snapshotId: pinned.snapshot_id,
      signal,
    };
    await expect(api.getDefinition(request)).resolves.toEqual(pinned);
    expect(String(transport.mock.calls[0][0])).toContain(`/runs/${request.runId}/definition`);
    await expect(api.getDefinition({ ...request, snapshotId: "current-draft" })).rejects.toThrow("pinned snapshot");
    await expect(api.getDefinition({ ...request, workflowId: "other" })).rejects.toThrow("pinned snapshot");
  });

  it("propagates cancellation to the transport", async () => {
    const controller = new AbortController();
    const transport = vi.fn<typeof fetch>().mockImplementation(async (_url, init) => {
      init?.signal?.throwIfAborted();
      return Response.json(fixtureTraces["run-fixture-completed"]);
    });
    const client = createApiClient({ getBaseUrl: () => "https://gateway.example.test", fetchImpl: transport });
    const api = createWorkflowRunApi({ accessToken: "test", teamId: "test", client });
    controller.abort();
    await expect(api.getSteps({ runId: "run-fixture-completed", signal: controller.signal })).rejects.toThrow();
  });
});

describe("trace identity and values", () => {
  it("keeps distinct steps for repeated nodes and rejects duplicate step IDs", () => {
    const trace = fixtureTraces["run-fixture-repeated-node"];
    expect(runStepsSchema.parse(trace).steps.map((step) => step.step_id)).toEqual(["step-repeat-a", "step-repeat-b"]);
    expect(runStepsSchema.safeParse({ ...trace, steps: [trace.steps[0], trace.steps[0]] }).success).toBe(false);
    const foreignStep = { ...trace.steps[0], run_id: "other" };
    expect(runStepsSchema.safeParse({ ...trace, steps: [foreignStep] }).success).toBe(false);
  });

  it("preserves scalar JSON, null request IDs and recorded zero cost", () => {
    const step = runStepsSchema.parse(fixtureTraces["run-fixture-completed"]).steps[1];
    expect(step.input).toBe(false);
    expect(step.output).toBe(0);
    expect(step.request_id).toBeNull();
    expect(formatCost(step.response_cost)).toBe("$0.000000");
    expect(formatCost(null)).toBe("Not reported");
  });

  it("reports real elapsed time and distinguishes active and unreported duration", () => {
    const step = fixtureTraces["run-fixture-completed"].steps[1];
    expect(stepDuration(step)).toBe("25 ms");
    expect(stepDuration({ ...step, state: "running", finished_at: null })).toBe("In progress");
    expect(stepDuration({ ...step, state: "failed", finished_at: null })).toBe("Not reported");
  });
});
