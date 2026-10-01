"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { serverRootPath } from "@/lib/serverRootPath";
import { apiClient, proxyBaseUrl } from "@/components/networking";
import { AnalysisKey } from "./AnalysisKey";
import type { EngineList, WorkerCreated } from "./engineData";

export const LENS_WORKER_IMAGE =
  "ghcr.io/berriai/litellm-lens-worker@sha256:40fdb82113dd4474cb6e833cf28552487d87c8baf61693a1c3fc2863b7968c6a";

function initialProxyAddress(): string {
  const url = new URL(proxyBaseUrl || serverRootPath, window.location.origin);
  if (["localhost", "127.0.0.1", "[::1]"].includes(url.hostname)) url.hostname = "host.docker.internal";
  return url.toString().replace(/\/$/, "");
}

export function workerSetupCommand(address: string, token: string): string {
  const quote = (value: string) => "'" + value.replaceAll("'", "'\\''") + "'";
  return [
    "docker run -d --restart unless-stopped --read-only --cap-drop ALL",
    "  --tmpfs /tmp:rw,noexec,nosuid,size=1g",
    "  --security-opt no-new-privileges --platform linux/amd64 --add-host host.docker.internal:host-gateway",
    `  -e ${quote("LITELLM_URL=" + address)}`,
    `  -e ${quote("LENS_WORKER_TOKEN=" + token)}`,
    `  ${LENS_WORKER_IMAGE}`,
  ].join(" \\\n");
}

function workerStatus(worker: EngineList["workers"][number], now: number): string {
  if (!worker.analysis_key_id) return "Billing key required";
  return now - Date.parse(worker.last_seen) < 120000 ? "Connected · ready to analyze" : "Not connected";
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
  const [analysisKey, setAnalysisKey] = useState<string | null>(null);
  const [editingWorker, setEditingWorker] = useState<string | null>(null);
  const [address, setAddress] = useState(initialProxyAddress);
  const [copied, setCopied] = useState(false);
  const [created, setCreated] = useState<WorkerCreated | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const actionLabel = editingWorker ? "Save billing key" : "Generate setup command";
  const editBilling = (worker: EngineList["workers"][number]) => {
    setCreated(null);
    setEditingWorker(worker.id);
    setAnalysisKey(worker.analysis_key_id ?? null);
  };
  const createWorker = async () => {
    setBusy(true);
    setError("");
    try {
      if (editingWorker) {
        await apiClient.put(`/engine/workers/${editingWorker}/billing-key`, {
          accessToken,
          body: { analysis_key_id: analysisKey },
        });
        setEditingWorker(null);
        setAnalysisKey(null);
        onChanged();
        return;
      }
      setCreated(
        await apiClient.post<WorkerCreated>("/engine/workers/register", {
          accessToken,
          body: { name: "Lens analyzer", analysis_key_id: analysisKey },
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
        {!editingWorker && (
          <>
            <label className="grid gap-2 text-sm">
              Your LiteLLM deployment URL
              <Input value={address} onChange={(event) => setAddress(event.target.value)} />
            </label>
            <p className="text-xs text-muted-foreground">
              The analyzer connects to this deployment to read logs and save findings.
            </p>
          </>
        )}
        {editingWorker && (
          <p className="text-sm font-medium">
            Billing for {workers.find((worker) => worker.id === editingWorker)?.name}
          </p>
        )}
        {!created && (
          <AnalysisKey
            key={editingWorker ?? "new"}
            accessToken={accessToken}
            value={analysisKey}
            onChange={setAnalysisKey}
            name="worker"
          />
        )}
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
          <Button disabled={busy || !address.trim() || !analysisKey} onClick={createWorker}>
            {busy ? "Saving…" : actionLabel}
          </Button>
        )}
        {editingWorker && (
          <Button
            variant="ghost"
            onClick={() => {
              setEditingWorker(null);
              setAnalysisKey(null);
            }}
          >
            Cancel
          </Button>
        )}
        {workers
          ?.filter((w) => !w.revoked)
          .map((worker) => (
            <div key={worker.id} className="flex justify-between items-center border-t pt-3 text-sm">
              <span>
                {worker.name}
                <span className="block text-xs text-muted-foreground">{workerStatus(worker, now)}</span>
              </span>
              <Button variant="ghost" size="sm" onClick={() => editBilling(worker)}>
                Billing key
              </Button>
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
