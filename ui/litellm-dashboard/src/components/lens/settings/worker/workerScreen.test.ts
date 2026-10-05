import { describe, expect, it } from "vitest";
import type { Worker, WorkerCreated } from "../../model/types";
import { workerScreen, type WorkerScreenInput } from "./workerScreen";

const worker: Worker = {
  id: "worker",
  name: "Lens worker",
  last_seen: "1970-01-01T00:00:00Z",
  scope: { all_teams: true, api_key_hash: "", team_id: "" },
  revoked: false,
  analysis_key_id: "b".repeat(64),
};
const revoked: Worker = { ...worker, revoked: true };
const other: Worker = { ...worker, id: "other" };

const created: WorkerCreated = { token: "lens-test-token", image: "lens-worker:v1", worker };

const idle: WorkerScreenInput = { workers: [], created: null, editingId: null, connected: false };
const screenOf = (input: Partial<WorkerScreenInput>) => workerScreen({ ...idle, ...input });

describe("workerScreen", () => {
  it("shows the setup form when no active worker exists", () => {
    expect(screenOf({})).toEqual({ kind: "form", editing: null });
    expect(screenOf({ workers: [revoked] })).toEqual({ kind: "form", editing: null });
  });

  it("lists active workers until one is being edited", () => {
    expect(screenOf({ workers: [worker] })).toEqual({ kind: "list" });
    expect(screenOf({ workers: [worker], editingId: worker.id })).toEqual({ kind: "form", editing: worker });
  });

  it("drops a stale edit once that worker is revoked", () => {
    expect(screenOf({ workers: [revoked], editingId: worker.id })).toEqual({ kind: "form", editing: null });
    expect(screenOf({ workers: [revoked, other], editingId: worker.id })).toEqual({ kind: "list" });
  });

  it("keeps the install screen up after registration, whatever the list says", () => {
    expect(screenOf({ workers: [worker], created, editingId: worker.id })).toEqual({
      kind: "install",
      created,
      connected: false,
    });
    expect(screenOf({ created, connected: true })).toEqual({ kind: "install", created, connected: true });
  });
});
