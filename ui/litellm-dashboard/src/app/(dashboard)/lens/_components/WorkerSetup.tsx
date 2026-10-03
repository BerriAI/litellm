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
import { CheckCircle2, Copy, Loader2 } from "lucide-react";
import { workerConnected } from "./lensData";
import { AnalysisKeyDetails } from "./AnalysisKeyDetails";
import { Switch } from "@/components/ui/switch";
import { AnalysisKey, AnalysisAccessFields, createAnalysisKey, type AnalysisAccess } from "./AnalysisKey";
import type { LensList, WorkerCreated } from "./lensData";

export const LENS_WORKER_IMAGE =
  "ghcr.io/berriai/litellm-lens-worker@sha256:44f0597c7583dcfef999ece9a8bc02cfeb9f0f5167a1221cee3bd10b1b79271b";

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
  const [access, setAccess] = useState<AnalysisAccess>({ model: null, budget: "100" });
  const [useExisting, setUseExisting] = useState(false);
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
  const formVisible = !workers.some((w) => !w.revoked) || !!editingWorker;
  const createdTitle = connected ? "Worker connected" : "Run the worker";
  const formTitle = editingWorker ? "Analysis access" : "Connect a worker";
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
    setAnalysisKey(null);
  };
  const editBilling = (worker: LensList["workers"][number]) => {
    setUseExisting(true);
    setCreated(null);
    setEditingWorker(worker.id);
    setAnalysisKey(worker.analysis_key_id ?? null);
  };
  const createWorker = async () => {
    setBusy(true);
    setError("");
    let newKey: string | null = null;
    try {
      const parsedAddress = new URL(address);
      if (!["http:", "https:"].includes(parsedAddress.protocol) || parsedAddress.username || parsedAddress.password)
        throw new Error("Enter an HTTP or HTTPS proxy URL without credentials");
      const keyId = useExisting ? analysisKey : await createAnalysisKey(accessToken, access);
      if (!useExisting) newKey = keyId;
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
      const message = e instanceof Error ? e.message : "Could not create credential";
      setError(message);
      if (newKey) {
        try {
          const current = await apiClient.get<LensList>("/lens", { accessToken });
          if (!current.workers.some((worker) => !worker.revoked && worker.analysis_key_id === newKey)) {
            await apiClient.post("/key/delete", { accessToken, body: { keys: [newKey] } });
          }
          onChanged();
        } catch {
          setError(
            `${message}. Could not confirm cleanup. Check the Lens analysis key in Virtual Keys before retrying.`,
          );
        }
      }
    } finally {
      setBusy(false);
    }
  };
  const setupDescription = created
    ? "Run this command on a server with Docker."
    : "Deploy the worker on your server to run investigations.";
  const awaitingConnection = !editingWorker && !connected;
  const describeSetup = awaitingConnection && (formVisible || !!created);
  const completed = !!created && connected;
  const installSize = connected ? "sm:max-w-sm p-8" : "sm:max-w-lg";
  const modalSize = created ? installSize : "sm:max-w-xl";
  const description = describeSetup ? setupDescription : "Worker status and model access";
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open && !busy) onClose();
      }}
    >
      <DialogContent className={`max-h-[90dvh] overflow-y-auto ${modalSize}`}>
        <DialogHeader className={completed ? "items-center gap-3 text-center sm:text-center" : undefined}>
          {completed && (
            <div className="flex size-12 items-center justify-center rounded-full bg-emerald-50 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-400">
              <CheckCircle2 className="size-6" />
            </div>
          )}
          <DialogTitle className="text-xl leading-7">{dialogTitle}</DialogTitle>
          <DialogDescription className={describeSetup || completed ? undefined : "sr-only"}>
            {completed ? "Ready to run investigations." : description}
          </DialogDescription>
        </DialogHeader>
        {formVisible && !created && (
          <div className="min-w-0 space-y-5">
            {useExisting ? (
              <AnalysisKey accessToken={accessToken} value={analysisKey} onChange={setAnalysisKey} />
            ) : (
              <AnalysisAccessFields
                accessToken={accessToken}
                value={access}
                onChange={(next) => {
                  setAccess(next);
                }}
              />
            )}
            <details className="text-sm" open={editingWorker ? true : undefined}>
              <summary className="cursor-pointer font-medium">Advanced options</summary>
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
          <div className="min-w-0 space-y-5">
            {!connected && (
              <>
                <Button
                  variant="default"
                  className="w-full gap-2"
                  onClick={async () => {
                    try {
                      await navigator.clipboard.writeText(workerSetupCommand(address, created.token));
                      setCopied(true);
                    } catch {
                      setError("Clipboard access failed. Allow clipboard access and try again.");
                    }
                  }}
                >
                  {copied ? <CheckCircle2 className="size-4" /> : <Copy className="size-4" />}
                  {copied ? "Copied" : "Copy Docker command"}
                </Button>
                <details className="text-sm">
                  <summary className="cursor-pointer text-muted-foreground">View command</summary>
                  <p className="mt-3 text-xs text-muted-foreground">Contains a private worker token.</p>
                  <pre
                    aria-label="Docker command preview"
                    className="mt-3 max-h-48 overflow-auto rounded-md bg-muted/40 p-3 text-xs leading-5"
                  >
                    {workerSetupCommand(address, created.token)}
                  </pre>
                </details>
              </>
            )}
            {!connected && (
              <div className="space-y-3 border-t pt-5">
                <div role="status" className="flex items-center gap-2.5 text-sm">
                  <Loader2 className="size-4 shrink-0 animate-spin text-muted-foreground" /> Waiting for your worker to
                  connect…
                </div>
                <details className="pl-6.5 text-sm text-muted-foreground">
                  <summary className="cursor-pointer">Not connecting?</summary>
                  <p className="mt-2 break-words leading-6">
                    Check that Docker is running and can reach {address}. Inspect the container logs for connection or
                    authentication errors. This page updates automatically.
                  </p>
                </details>
              </div>
            )}
            {connected && (
              <Button className="w-full" onClick={onReady ?? onClose}>
                {onReady ? "New investigation" : "Done"}
              </Button>
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
              <section key={worker.id} className="space-y-5 text-sm">
                <div className="flex items-center justify-between gap-3">
                  {workers.filter((w) => !w.revoked).length > 1 && <h3 className="font-medium">{worker.name}</h3>}
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
                    Settings
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={async () => {
                      try {
                        await apiClient.delete(`/lens/workers/${worker.id}`, { accessToken });
                        setAnalysisKey(null);
                        setUseExisting(false);
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
        {error && (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        )}
      </DialogContent>
    </Dialog>
  );
}
