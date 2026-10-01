"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { serverRootPath } from "@/lib/serverRootPath";
import { apiClient, proxyBaseUrl } from "@/components/networking";
import { CheckCircle2, Loader2, Plus } from "lucide-react";
import { workerConnected } from "./lensData";
import { AnalysisKeyDetails } from "./AnalysisKeyDetails";
import { Switch } from "@/components/ui/switch";
import { AnalysisKey, AnalysisAccessFields, createAnalysisKey, type AnalysisAccess } from "./AnalysisKey";
import type { LensList, WorkerCreated } from "./lensData";

export const LENS_WORKER_IMAGE =
  "ghcr.io/berriai/litellm-lens-worker@sha256:67eba741c1b97c749975c5c38e2370a603e1105babc908d613c1b79d7b995393";

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

function workerStatus(worker: LensList["workers"][number], now: number): string {
  if (!worker.analysis_key_id) return "Billing key required";
  return workerConnected(worker, now) ? "Connected" : "Not connected";
}

export function WorkerSetup({
  accessToken,
  workers,
  onClose,
  onChanged,
  onReady,
}: {
  accessToken: string;
  workers: LensList["workers"];
  onClose: () => void;
  onChanged: () => void;
  onReady?: () => void;
}) {
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 2000);
    return () => window.clearInterval(timer);
  }, []);
  const [adding, setAdding] = useState(!workers.some((w) => !w.revoked));
  const [access, setAccess] = useState<AnalysisAccess>({ model: null, budget: "100" });
  const [useExisting, setUseExisting] = useState(false);
  const [preparedKey, setPreparedKey] = useState<string | null>(null);
  const [analysisKey, setAnalysisKey] = useState<string | null>(null);
  const [editingWorker, setEditingWorker] = useState<string | null>(null);
  const [address, setAddress] = useState(initialProxyAddress);
  const [copied, setCopied] = useState(false);
  const [created, setCreated] = useState<WorkerCreated | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const connected = created
    ? workers.some((w) => w.id === created.worker.id && workerConnected(w, now))
    : workers.some((w) => workerConnected(w, now));
  const formVisible = adding || !!editingWorker;
  const createdTitle = connected ? "Analysis is ready" : "Install the analysis service";
  const formTitle = editingWorker ? "Analysis access" : "Enable investigations";
  const validAccess = useExisting
    ? !!analysisKey
    : !!access.model && Number.isFinite(Number(access.budget)) && Number(access.budget) > 0;
  const baseTitle = formVisible ? formTitle : "Analysis worker";
  const dialogTitle = created ? createdTitle : baseTitle;
  const actionLabel = editingWorker ? "Save analysis access" : "Get install command";
  const cancelForm = () => {
    if (!workers.some((w) => !w.revoked)) {
      onClose();
      return;
    }
    setEditingWorker(null);
    setAdding(false);
    setAnalysisKey(null);
  };
  const addWorker = () => {
    setAdding(true);
    setUseExisting(false);
    setPreparedKey(null);
  };
  const editBilling = (worker: LensList["workers"][number]) => {
    setAdding(false);
    setUseExisting(true);
    setCreated(null);
    setEditingWorker(worker.id);
    setAnalysisKey(worker.analysis_key_id ?? null);
  };
  const createWorker = async () => {
    setBusy(true);
    setError("");
    try {
      const parsedAddress = new URL(address);
      if (!["http:", "https:"].includes(parsedAddress.protocol) || parsedAddress.username || parsedAddress.password)
        throw new Error("Enter an HTTP or HTTPS proxy URL without credentials");
      const keyId = useExisting ? analysisKey : preparedKey ?? (await createAnalysisKey(accessToken, access));
      if (!useExisting) setPreparedKey(keyId);
      if (editingWorker) {
        await apiClient.put(`/lens/workers/${editingWorker}/billing-key`, {
          accessToken,
          body: { analysis_key_id: keyId },
        });
        setEditingWorker(null);
        setAnalysisKey(null);
        onChanged();
        return;
      }
      setCreated(
        await apiClient.post<WorkerCreated>("/lens/workers/register", {
          accessToken,
          body: { name: "Lens worker", analysis_key_id: keyId },
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
        if (!open && !busy) onClose();
      }}
    >
      <DialogContent className="sm:max-w-xl max-h-[90dvh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-xl">{dialogTitle}</DialogTitle>
          <DialogDescription>
            {created && connected
              ? "You can now run investigations."
              : formVisible
                ? "Lens needs a small service on your server to run investigations."
                : "Worker status and model access"}
          </DialogDescription>
        </DialogHeader>
        {formVisible && !created && (
          <div className="space-y-5">
            {useExisting ? (
              <AnalysisKey accessToken={accessToken} value={analysisKey} onChange={setAnalysisKey} />
            ) : (
              <AnalysisAccessFields
                accessToken={accessToken}
                value={access}
                onChange={(next) => {
                  setAccess(next);
                  setPreparedKey(null);
                }}
              />
            )}
            <details className="text-sm" open={editingWorker ? true : undefined}>
              <summary className="cursor-pointer text-muted-foreground">Advanced options</summary>
              <div className="mt-4 space-y-5">
                <label className="flex items-center justify-between gap-4">
                  Use an existing virtual key
                  <Switch checked={useExisting} onCheckedChange={setUseExisting} />
                </label>
                {!editingWorker && (
                  <div className="space-y-2">
                    <label htmlFor="worker-proxy-address" className="block text-sm font-medium">
                      LiteLLM proxy URL
                    </label>
                    <Input id="worker-proxy-address" value={address} onChange={(e) => setAddress(e.target.value)} />
                    <p className="text-xs text-muted-foreground">Your server must be able to reach this address.</p>
                  </div>
                )}
              </div>
            </details>
          </div>
        )}
        {created ? (
          <div className="space-y-3">
            {!connected && (
              <>
                <p className="text-sm leading-6">Run this command once on your server. Docker must be installed.</p>
                <Button
                  variant="default"
                  onClick={async () => {
                    try {
                      await navigator.clipboard.writeText(workerSetupCommand(address, created.token));
                      setCopied(true);
                    } catch {
                      setError("Clipboard access failed. Allow clipboard access and try again.");
                    }
                  }}
                >
                  {copied ? "Copied" : "Copy Docker command"}
                </Button>
                <p className="text-xs text-muted-foreground">The command contains a private worker token.</p>
                <details className="text-sm">
                  <summary className="cursor-pointer text-muted-foreground">View command</summary>
                  <pre
                    aria-label="Docker command preview"
                    className="mt-3 max-h-48 overflow-auto rounded-md bg-muted/40 p-3 text-xs leading-5"
                  >
                    {workerSetupCommand(address, created.token)}
                  </pre>
                </details>
              </>
            )}
            <div
              role="status"
              className={`flex items-center gap-2 text-sm ${connected ? "text-emerald-700 dark:text-emerald-400" : "text-muted-foreground"}`}
            >
              {connected ? <CheckCircle2 className="size-5" /> : <Loader2 className="size-4 animate-spin" />}
              {connected ? "Worker connected" : "Waiting for your worker to connect…"}
            </div>
            {!connected && (
              <details className="text-sm">
                <summary className="cursor-pointer">Not connecting?</summary>
                <p className="mt-2 leading-6 text-muted-foreground">
                  Run the command on a machine with Docker. Check that it can reach the proxy URL above, then inspect
                  the container logs for a connection or authentication error. This page updates automatically.
                </p>
              </details>
            )}
            {connected && (
              <Button onClick={onReady ?? onClose}>{onReady ? "Continue to investigation" : "Done"}</Button>
            )}
          </div>
        ) : null}
        {!created && formVisible && (
          <DialogFooter>
            <Button variant="outline" disabled={busy} onClick={cancelForm}>
              Cancel
            </Button>
            <Button disabled={busy || !address.trim() || !validAccess} onClick={createWorker}>
              {busy ? "Preparing…" : actionLabel}
            </Button>
          </DialogFooter>
        )}
        {!created &&
          !formVisible &&
          workers
            .filter((w) => !w.revoked)
            .map((worker) => (
              <section key={worker.id} className="space-y-5 border-b pb-5 text-sm">
                <div className="flex items-center justify-between gap-3">
                  <h3 className="font-medium">{worker.name}</h3>
                  <span
                    className={`flex items-center gap-2 ${workerConnected(worker, now) ? "text-emerald-700 dark:text-emerald-400" : "text-muted-foreground"}`}
                  >
                    {workerConnected(worker, now) && <CheckCircle2 className="size-4" />}
                    {workerStatus(worker, now)}
                  </span>
                </div>
                {worker.analysis_key_id && (
                  <AnalysisKeyDetails accessToken={accessToken} keyId={worker.analysis_key_id} showName />
                )}
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <Button variant="outline" size="sm" onClick={() => editBilling(worker)}>
                    Change virtual key
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={async () => {
                      try {
                        await apiClient.delete(`/lens/workers/${worker.id}`, { accessToken });
                        onChanged();
                      } catch (e) {
                        setError(e instanceof Error ? e.message : "Could not revoke worker");
                      }
                    }}
                  >
                    Revoke access
                  </Button>
                </div>
              </section>
            ))}
        {!formVisible && !created && (
          <Button variant="outline" onClick={addWorker}>
            <Plus className="size-4" /> Add worker
          </Button>
        )}
        {error && (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        )}
      </DialogContent>
    </Dialog>
  );
}
