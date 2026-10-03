"use client";
import { WorkerList } from "./WorkerList";

import { initialProxyAddress } from "./workerCommand";
import { usePrepareWorker, useRevokeWorker } from "../../api/mutations";
import { WorkerInstall } from "./WorkerInstall";
import { WorkerForm } from "./WorkerForm";

import { useNow } from "@/hooks/useNow";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { CheckCircle2 } from "lucide-react";
import { workerConnected } from "../../model/status";
import { analysisAccessSchema, type AnalysisAccess } from "./workerSchema";
import type { LensList, WorkerCreated } from "../../model/types";

export function WorkerDialog({
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
  const revokeWorker = useRevokeWorker(accessToken);
  const now = useNow(2000);
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
  const uninstalledScreen = formVisible ? "form" : "list";
  const screen = created ? "install" : uninstalledScreen;
  const createdTitle = connected ? "Worker connected" : "Run the worker";
  const formTitle = editingWorker ? "Analysis access" : "Connect a worker";
  const validAccess = useExisting ? !!analysisKey : analysisAccessSchema.safeParse(access).success;
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
  const onPrepared = (created: WorkerCreated | null) => {
    if (created) setCreated(created);
    else {
      setEditingWorker(null);
      setAnalysisKey(null);
    }
  };
  const prepareWorker = usePrepareWorker(accessToken, { onChanged, onPrepared });
  const createWorker = async () => {
    setBusy(true);
    setError("");
    try {
      const registration = { address, useExisting, analysisKey, access, editingWorker };
      await prepareWorker.mutateAsync(registration);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not create credential");
    } finally {
      setBusy(false);
    }
  };
  const revoke = async (id: string) => {
    try {
      await revokeWorker.mutateAsync(id);
      setAnalysisKey(null);
      setUseExisting(false);
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not revoke worker");
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
        {screen === "form" && (
          <WorkerForm
            useExisting={useExisting}
            accessToken={accessToken}
            analysisKey={analysisKey}
            setAnalysisKey={setAnalysisKey}
            access={access}
            setAccess={setAccess}
            editingWorker={editingWorker}
            setUseExisting={setUseExisting}
            address={address}
            setAddress={setAddress}
          />
        )}
        {screen === "install" && created ? (
          <WorkerInstall
            connected={connected}
            address={address}
            created={created}
            copied={copied}
            setCopied={setCopied}
            setError={setError}
            onReady={onReady}
            onClose={onClose}
          />
        ) : null}
        {screen === "form" && (
          <DialogFooter>
            <Button variant="outline" disabled={busy} onClick={cancelForm}>
              Cancel
            </Button>
            <Button disabled={busy || !address.trim() || !validAccess} onClick={createWorker}>
              {busy ? "Preparing…" : actionLabel}
            </Button>
          </DialogFooter>
        )}
        {screen === "list" && (
          <WorkerList workers={workers} now={now} accessToken={accessToken} editBilling={editBilling} revoke={revoke} />
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
