"use client";
import { WorkerList } from "./WorkerList";

import { initialProxyAddress } from "./workerCommand";
import { useRevokeWorker } from "../../api/mutations";
import { usePrepareWorker } from "./usePrepareWorker";
import { WorkerInstall } from "./WorkerInstall";
import { WorkerForm } from "./WorkerForm";

import { FormProvider } from "react-hook-form";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { CheckCircle2 } from "lucide-react";
import { useWorkerConnected } from "../../useWorkerConnected";
import { workerFormSchema, type WorkerFormInput } from "./workerSchema";
import type { LensList } from "../../model/types";
import { useZodForm } from "@/lib/forms/useZodForm";
import { cn } from "@/lib/cva.config";
import { SettingsCard } from "../SettingsSection";

function defaultWorkerFormValues(): WorkerFormInput {
  return {
    useExisting: false,
    analysisKey: null,
    access: { model: null, budget: "100" },
    address: typeof window === "undefined" ? "" : initialProxyAddress(),
  };
}

export function WorkerSettings({ workers, onReady }: { workers: LensList["workers"]; onReady?: () => void }) {
  const revokeWorker = useRevokeWorker();
  const prepareWorker = usePrepareWorker();
  const form = useZodForm(workerFormSchema, {
    defaultValues: defaultWorkerFormValues(),
    mode: "onChange",
  });
  const { formState, reset, setValue } = form;
  const [editingWorker, setEditingWorker] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [clipboardError, setClipboardError] = useState("");
  const created = prepareWorker.data ?? null;
  const error = prepareWorker.error?.message ?? revokeWorker.error?.message ?? clipboardError;
  const hasActiveWorker = workers.some((w) => !w.revoked);
  const connected = useWorkerConnected(created ? workers.filter((w) => w.id === created.worker.id) : workers);
  const formVisible = !hasActiveWorker || !!editingWorker;
  const uninstalledScreen = formVisible ? "form" : "list";
  const screen = created ? "install" : uninstalledScreen;
  const createdTitle = connected ? "Worker connected" : "Run the worker";
  const formTitle = editingWorker ? "Analysis access" : "Connect a worker";
  const title = created ? createdTitle : formTitle;
  const actionLabel = editingWorker ? "Save analysis access" : "Get install command";
  const cancelForm = () => {
    setEditingWorker(null);
    reset(defaultWorkerFormValues());
  };
  const editBilling = (worker: LensList["workers"][number]) => {
    reset({
      ...defaultWorkerFormValues(),
      useExisting: true,
      analysisKey: worker.analysis_key_id ?? null,
    });
    prepareWorker.reset();
    setEditingWorker(worker.id);
  };
  const createWorker = form.handleSubmit((values) => {
    const registration = {
      address: values.address,
      useExisting: values.useExisting,
      analysisKey: values.analysisKey,
      access: values.access,
      editingWorker,
    };
    prepareWorker.mutate(registration, {
      onSuccess: (result) => {
        if (result) return;
        setEditingWorker(null);
        setValue("analysisKey", null);
      },
    });
  });
  const revoke = (id: string) =>
    revokeWorker.mutate(id, {
      onSuccess: () => {
        setValue("analysisKey", null);
        setValue("useExisting", false);
      },
    });
  const busy = prepareWorker.isPending;
  const setupDescription = created
    ? "Run this command on a server with Docker."
    : "Deploy the worker on your server to run investigations.";
  const awaitingConnection = !editingWorker && !connected;
  const describeSetup = awaitingConnection && (formVisible || !!created);
  const completed = !!created && connected;
  const description = describeSetup ? setupDescription : "Choose which key pays for analysis.";
  if (screen === "list")
    return (
      <>
        <WorkerList workers={workers} editBilling={editBilling} revoke={revoke} />
        {error && (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        )}
      </>
    );
  return (
    <FormProvider {...form}>
      <SettingsCard className={cn("flex flex-col gap-5", completed && "items-center text-center")}>
        <header className={cn("space-y-1", completed && "flex flex-col items-center gap-3")}>
          {completed && (
            <div className="flex size-12 items-center justify-center rounded-full bg-emerald-50 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-400">
              <CheckCircle2 className="size-6" />
            </div>
          )}
          <h3 className="text-base font-semibold">{title}</h3>
          <p className="text-sm text-muted-foreground">{completed ? "Ready to run investigations." : description}</p>
        </header>
        {screen === "form" && <WorkerForm editingWorker={editingWorker} />}
        {screen === "install" && created ? (
          <WorkerInstall
            connected={connected}
            address={form.getValues("address")}
            created={created}
            copied={copied}
            setCopied={setCopied}
            setError={setClipboardError}
            onReady={onReady}
            onClose={prepareWorker.reset}
          />
        ) : null}
        {screen === "form" && (
          <div className="flex justify-end gap-2">
            {hasActiveWorker && (
              <Button variant="outline" disabled={busy} onClick={cancelForm}>
                Cancel
              </Button>
            )}
            <Button disabled={!formState.isValid || busy} onClick={() => void createWorker()}>
              {busy ? "Preparing…" : actionLabel}
            </Button>
          </div>
        )}
        {error && (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        )}
      </SettingsCard>
    </FormProvider>
  );
}
