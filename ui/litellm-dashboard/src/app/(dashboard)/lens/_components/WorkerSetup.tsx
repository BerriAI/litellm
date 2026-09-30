"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { serverRootPath } from "@/lib/serverRootPath";
import { apiClient, proxyBaseUrl } from "@/components/networking";
import type { EngineList, WorkerCreated } from "./engineData";

export const LENS_WORKER_IMAGE =
  "ghcr.io/berriai/litellm-lens-worker@sha256:654bdb62df533402cc778d318db0e9c56ed1bce7d866a6c684c3436bedc6a7eb";

function initialProxyAddress(): string {
  const url = new URL(proxyBaseUrl || serverRootPath, window.location.origin);
  if (["localhost", "127.0.0.1", "[::1]"].includes(url.hostname)) url.hostname = "host.docker.internal";
  return url.toString().replace(/\/$/, "");
}

export function workerSetupCommand(address: string, token: string): string {
  const quote = (value: string) => "'" + value.replaceAll("'", "'\\''") + "'";
  return [
    "docker run -d --restart unless-stopped --read-only --cap-drop ALL",
    "  --security-opt no-new-privileges --platform linux/amd64",
    `  -e ${quote("LITELLM_URL=" + address)}`,
    `  -e ${quote("LENS_WORKER_TOKEN=" + token)}`,
    `  ${LENS_WORKER_IMAGE}`,
  ].join(" \\\n");
}

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
  const [address, setAddress] = useState(initialProxyAddress);
  const [copied, setCopied] = useState(false);
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
        <p className="text-sm">
          LiteLLM already handles your lenses and results. This Docker container runs their analysis in the background.
          Generate a command below, then run it on a server with Docker. It connects automatically.
        </p>
        <label className="grid gap-2 text-sm">
          LiteLLM address
          <Input value={address} onChange={(event) => setAddress(event.target.value)} />
        </label>
        <p className="text-xs text-muted-foreground">
          Use an address the container can reach. If you run Docker on another server, enter this proxy’s network
          address.
        </p>
        {created ? (
          <div className="space-y-3">
            <p className="text-sm font-medium">Run this command on your server</p>
            <textarea
              readOnly
              aria-label="Docker setup command"
              rows={7}
              className="w-full rounded-md border p-3 font-mono text-xs"
              value={workerSetupCommand(address, created.token)}
            />
            <Button
              onClick={async () => {
                await navigator.clipboard.writeText(workerSetupCommand(address, created.token));
                setCopied(true);
              }}
            >
              {copied ? "Copied" : "Copy Docker command"}
            </Button>
            <p className="text-xs text-muted-foreground">
              The command includes a private worker token, shown only here. It lets this worker run your lenses; no
              other API key is needed. Keep the command private.
            </p>
            <p className="text-sm" role="status">
              {workers.some(
                (worker) => worker.id === created.worker.id && Date.now() - Date.parse(worker.last_seen) < 120000,
              )
                ? "Worker connected. You can start a scan."
                : "Waiting for your worker to connect…"}
            </p>
          </div>
        ) : (
          <Button disabled={busy || !address.trim()} onClick={createWorker}>
            {busy ? "Generating…" : "Generate setup command"}
          </Button>
        )}
        <p className="text-xs text-muted-foreground">
          A compatible worker image is already selected. One worker can serve all your lenses and keeps running when you
          close this page. You can stop its access below.
        </p>
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
