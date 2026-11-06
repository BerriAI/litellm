import { z } from "zod";
import { apiClient } from "@/components/networking";
import type { ApiClient } from "@/lib/http/client";
import { runStateSchema, runStepsSchema, type WorkflowRunDataSource } from "./contracts";

const runSummaryFields = {
  run_id: z.string().min(1),
  workflow_id: z.string().min(1),
  snapshot_id: z.string().min(1),
  status: runStateSchema,
  created_at: z.iso.datetime({ offset: true }),
};
const runListSchema = z.object({
  runs: z.array(z.object(runSummaryFields)),
  next_cursor: z.string().min(1).nullable(),
});

const pinnedDefinitionSchema = z.object({
  workflow_id: z.string().min(1),
  snapshot_id: z.string().min(1),
  bundle: z.record(z.string(), z.json()),
});

export interface WorkflowRunApiOptions {
  readonly accessToken: string;
  readonly teamId: string;
  readonly client?: Pick<ApiClient, "get">;
}

export function createWorkflowRunApi({
  accessToken,
  teamId,
  client = apiClient,
}: WorkflowRunApiOptions): WorkflowRunDataSource {
  return {
    cacheKey: crypto.randomUUID(),
    async listRuns({ workflowId, cursor, status, limit, signal }) {
      const page = runListSchema.parse(
        await client.get<unknown>("/v1/workflows/runs", {
          accessToken,
          signal,
          query: {
            team_id: teamId,
            workflow_id: workflowId,
            after: cursor,
            status: status === "all" ? undefined : status,
            limit,
          },
        }),
      );
      if (page.runs.some((run) => run.workflow_id !== workflowId)) {
        throw new Error("The run list returned a different workflow");
      }
      return {
        runs: page.runs.map(({ status: state, ...run }) => ({ ...run, state })),
        next_cursor: page.next_cursor,
      };
    },
    async getSteps({ runId, signal }) {
      const trace = runStepsSchema.parse(
        await client.get<unknown>(`/v1/workflows/runs/${encodeURIComponent(runId)}/steps`, {
          accessToken,
          signal,
        }),
      );
      if (trace.run_id !== runId) throw new Error("The trace returned a different run");
      return trace;
    },
    async getDefinition({ runId, workflowId, snapshotId, signal }) {
      const definition = pinnedDefinitionSchema.parse(
        await client.get<unknown>(`/v1/workflows/runs/${encodeURIComponent(runId)}/definition`, {
          accessToken,
          signal,
        }),
      );
      if (definition.snapshot_id !== snapshotId || definition.workflow_id !== workflowId) {
        throw new Error("The definition does not match this run's pinned snapshot");
      }
      return definition;
    },
  };
}
