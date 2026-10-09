import type { LensList, Worker, WorkerCreated } from "../../model/types";

export type WorkerScreen =
  | { readonly kind: "list" }
  | { readonly kind: "form"; readonly editing: Worker | null }
  | { readonly kind: "install"; readonly created: WorkerCreated; readonly connected: boolean };

export interface WorkerScreenInput {
  readonly workers: LensList["workers"];
  readonly created: WorkerCreated | null;
  readonly editingId: string | null;
  readonly connected: boolean;
}

export function workerScreen({ workers, created, editingId, connected }: WorkerScreenInput): WorkerScreen {
  if (created) return { kind: "install", created, connected };
  const active = workers.filter((worker) => !worker.revoked);
  const editing = active.find((worker) => worker.id === editingId) ?? null;
  if (editing) return { kind: "form", editing };
  return active.length ? { kind: "list" } : { kind: "form", editing: null };
}
