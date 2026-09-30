"use client";

import { useEffect, useState } from "react";
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
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 15000);
    return () => window.clearInterval(timer);
  }, []);
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
          body: { name: "Lens analyzer" },
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
          <DialogTitle>Set up Lens analysis</DialogTitle>
          <DialogDescription>
            Lens reads your agents’ logs and finds issues in the background. Run its analyzer once with Docker.
          </DialogDescription>
        </DialogHeader>
        <label className="grid gap-2 text-sm">
          Your LiteLLM deployment URL
          <Input value={address} onChange={(event) => setAddress(event.target.value)} />
        </label>
        <p className="text-xs text-muted-foreground">
          The analyzer connects to this deployment to read logs and save findings.
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
              Keep this command private. It includes the analyzer’s access token.
            </p>
            <p className="text-sm" role="status">
              {workers.some((worker) => worker.id === created.worker.id && now - Date.parse(worker.last_seen) < 120000)
                ? "Analyzer connected. You can start a scan."
                : "Waiting for your analyzer to connect…"}
            </p>
          </div>
        ) : (
          <Button disabled={busy || !address.trim()} onClick={createWorker}>
            {busy ? "Generating…" : "Generate setup command"}
          </Button>
        )}
        {workers
          ?.filter((w) => !w.revoked)
          .map((worker) => (
            <div key={worker.id} className="flex justify-between items-center border-t pt-3 text-sm">
              <span>
                {worker.name}
                <span className="block text-xs text-muted-foreground">
                  {now - Date.parse(worker.last_seen) < 120000 ? "Connected · ready to analyze" : "Not connected"}
                </span>
              </span>
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
