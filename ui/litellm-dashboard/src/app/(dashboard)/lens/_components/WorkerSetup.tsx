"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { apiClient } from "@/components/networking";
import type { EngineList, WorkerCreated } from "./engineData";

export function WorkerSetup({
  accessToken,
  workers,
  onClose,
  onChanged,
}: {
  accessToken: string;
  workers: EngineList["workers"];
  onClose: () => void;
  onChanged: () => void;
}) {
  const [created, setCreated] = useState<WorkerCreated | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const createWorker = async () => {
    setBusy(true);
    setError("");
    try {
      setCreated(
        await apiClient.post<WorkerCreated>("/engine/workers/register", {
          accessToken,
          body: { name: "Lens worker" },
        }),
      );
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not create credential");
    } finally {
      setBusy(false);
    }
  };
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>Connect a worker</DialogTitle>
          <DialogDescription>
            Run a small Lens worker alongside your existing LiteLLM proxy or on another server. One worker can serve all
            your lenses.
          </DialogDescription>
        </DialogHeader>
        <ol className="list-decimal pl-5 space-y-3 text-sm">
          <li>Create a credential scoped to the activity you can access.</li>
          <li>
            On your server, set <code>LITELLM_URL</code> to this proxy&apos;s address and <code>LENS_WORKER_TOKEN</code>{" "}
            to the credential below.
          </li>
          <li>
            Set <code>LENS_WORKER_IMAGE</code> to the versioned worker image for your LiteLLM release, then run{" "}
            <code>
              docker run -d --restart unless-stopped --read-only --cap-drop ALL --security-opt no-new-privileges
              --env-file lens.env &quot;$LENS_WORKER_IMAGE&quot;
            </code>
            .
          </li>
        </ol>
        <p className="text-xs text-muted-foreground">
          No source checkout or second LiteLLM proxy is needed. Upgrade your existing proxy to a Lens-enabled release
          first. The worker polls LiteLLM for scans started here or due on a schedule. It connects outward to LiteLLM.
          No incoming port or GPU is required. Store the credential in your server&apos;s secret manager or environment
          file.
        </p>
        {created ? (
          <div className="space-y-2">
            <p className="text-sm font-medium">Copy this credential now. It is shown only once.</p>
            <textarea
              readOnly
              aria-label="Worker credential"
              className="w-full rounded-md border p-3 font-mono text-xs"
              value={created.token}
            />
            <Button variant="outline" onClick={() => navigator.clipboard.writeText(created.token)}>
              Copy credential
            </Button>
          </div>
        ) : (
          <Button disabled={busy} onClick={createWorker}>
            {busy ? "Creating…" : "Create worker credential"}
          </Button>
        )}
        {workers
          ?.filter((w) => !w.revoked)
          .map((worker) => (
            <div key={worker.id} className="flex justify-between items-center border-t pt-3 text-sm">
              <span>{worker.name}</span>
              <Button
                variant="ghost"
                size="sm"
                onClick={async () => {
                  try {
                    await apiClient.delete(`/engine/workers/${worker.id}`, { accessToken });
                    onChanged();
                  } catch (e) {
                    setError(e instanceof Error ? e.message : "Could not revoke worker");
                  }
                }}
              >
                Revoke access
              </Button>
            </div>
          ))}
        {error && (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        )}
      </DialogContent>
    </Dialog>
  );
}
