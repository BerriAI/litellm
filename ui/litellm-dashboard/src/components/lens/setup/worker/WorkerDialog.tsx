"use client";
import { WorkerList } from "./WorkerList";

import { initialProxyAddress } from "./workerCommand";
import { usePrepareWorker, useRevokeWorker } from "../../api/mutations";
import { WorkerInstall } from "./WorkerInstall";
import { WorkerForm } from "./WorkerForm";

import { useNow } from "@/hooks/useNow";

import { FormProvider } from "react-hook-form";
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
import { workerFormSchema, type WorkerFormInput } from "./workerSchema";
import type { LensList, WorkerCreated } from "../../model/types";
import { useZodForm } from "@/lib/forms/useZodForm";
import { cn } from "@/lib/cva.config";

function defaultWorkerFormValues(): WorkerFormInput {
  return {
    useExisting: false,
    analysisKey: null,
    access: { model: null, budget: "100" },
    address: typeof window === "undefined" ? "" : initialProxyAddress(),
  };
}

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
  const form = useZodForm(workerFormSchema, {
    defaultValues: defaultWorkerFormValues(),
    mode: "onChange",
  });
  const { formState, reset, setValue } = form;
  const [editingWorker, setEditingWorker] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [created, setCreated] = useState<WorkerCreated | null>(null);
  const [error, setError] = useState("");
  const connected = created
    ? workers.some((w) => w.id === created.worker.id && workerConnected(w, now))
    : workers.some((w) => workerConnected(w, now));
  const formVisible = !workers.some((w) => !w.revoked) || !!editingWorker;
  const uninstalledScreen = formVisible ? "form" : "list";
  const screen = created ? "install" : uninstalledScreen;
  const createdTitle = connected ? "Worker connected" : "Run the worker";
  const formTitle = editingWorker ? "Analysis access" : "Connect a worker";
  const baseTitle = formVisible ? formTitle : "Analysis worker";
  const dialogTitle = created ? createdTitle : baseTitle;
  const actionLabel = editingWorker ? "Save analysis access" : "Get install command";
  const cancelForm = () => {
    if (!workers.some((w) => !w.revoked)) {
      onClose();
      return;
    }
    setEditingWorker(null);
    reset(defaultWorkerFormValues());
  };
  const editBilling = (worker: LensList["workers"][number]) => {
    reset({
      ...defaultWorkerFormValues(),
      useExisting: true,
      analysisKey: worker.analysis_key_id ?? null,
    });
    setCreated(null);
    setEditingWorker(worker.id);
  };
  const onPrepared = (created: WorkerCreated | null) => {
    if (created) setCreated(created);
    else {
      setEditingWorker(null);
      setValue("analysisKey", null);
    }
  };
  const prepareWorker = usePrepareWorker(accessToken, { onChanged, onPrepared });
  const createWorker = form.handleSubmit(async (values) => {
    setError("");
    try {
      const registration = {
        address: values.address,
        useExisting: values.useExisting,
        analysisKey: values.analysisKey,
        access: values.access,
        editingWorker,
      };
      await prepareWorker.mutateAsync(registration);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not create credential");
    }
  });
  const revoke = async (id: string) => {
    try {
      await revokeWorker.mutateAsync(id);
      setValue("analysisKey", null);
      setValue("useExisting", false);
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
  const modalSize = cn(
    created && connected && "sm:max-w-sm p-8",
    created && !connected && "sm:max-w-lg",
    !created && "sm:max-w-xl",
  );
  const description = describeSetup ? setupDescription : "Worker status and model access";
  return (
    <FormProvider {...form}>
      <Dialog
        open
        onOpenChange={(open) => {
          if (!open && !formState.isSubmitting) onClose();
        }}
      >
        <DialogContent className={cn("max-h-[90dvh] overflow-y-auto", modalSize)}>
          <DialogHeader className={cn(completed && "items-center gap-3 text-center sm:text-center")}>
            {completed && (
              <div className="flex size-12 items-center justify-center rounded-full bg-emerald-50 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-400">
                <CheckCircle2 className="size-6" />
              </div>
            )}
            <DialogTitle className="text-xl leading-7">{dialogTitle}</DialogTitle>
            <DialogDescription className={cn(!(describeSetup || completed) && "sr-only")}>
              {completed ? "Ready to run investigations." : description}
            </DialogDescription>
          </DialogHeader>
          {screen === "form" && <WorkerForm accessToken={accessToken} editingWorker={editingWorker} />}
          {screen === "install" && created ? (
            <WorkerInstall
              connected={connected}
              address={form.getValues("address")}
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
              <Button variant="outline" disabled={formState.isSubmitting} onClick={cancelForm}>
                Cancel
              </Button>
              <Button disabled={!formState.isValid || formState.isSubmitting} onClick={() => void createWorker()}>
                {formState.isSubmitting ? "Preparing…" : actionLabel}
              </Button>
            </DialogFooter>
          )}
          {screen === "list" && (
            <WorkerList
              workers={workers}
              now={now}
              accessToken={accessToken}
              editBilling={editBilling}
              revoke={revoke}
            />
          )}
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
        </DialogContent>
      </Dialog>
    </FormProvider>
  );
}
