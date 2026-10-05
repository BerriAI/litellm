"use client";

import { useState, type ReactNode } from "react";
import { FormProvider } from "react-hook-form";
import { Button } from "@/components/ui/button";
import { useZodForm } from "@/lib/forms/useZodForm";
import { useRevokeWorker } from "../../data/mutations";
import type { LensList, Worker } from "../../model/types";
import { useWorkerConnected } from "../../hooks/useWorkerConnected";
import { SettingsCard } from "../SettingsSection";
import { usePrepareWorker } from "./usePrepareWorker";
import { initialProxyAddress } from "./workerCommand";
import { WorkerForm } from "./WorkerForm";
import { WorkerInstall } from "./WorkerInstall";
import { WorkerList } from "./WorkerList";
import { workerFormSchema, type WorkerFormInput } from "./workerSchema";
import { workerScreen } from "./workerScreen";

function defaultWorkerFormValues(): WorkerFormInput {
  return {
    useExisting: false,
    analysisKey: null,
    access: { model: null, budget: "100" },
    address: typeof window === "undefined" ? "" : initialProxyAddress(),
  };
}

function ErrorText({ message }: { message: string | undefined }) {
  if (!message) return null;
  return (
    <p role="alert" className="text-sm text-destructive">
      {message}
    </p>
  );
}

function submitLabel(editing: Worker | null, busy: boolean): string {
  if (busy) return "Preparing…";
  return editing ? "Save analysis access" : "Get install command";
}

function WorkerFormCard({
  editing,
  busy,
  valid,
  onCancel,
  onSubmit,
  children,
}: {
  editing: Worker | null;
  busy: boolean;
  valid: boolean;
  onCancel: () => void;
  onSubmit: () => void;
  children: ReactNode;
}) {
  return (
    <SettingsCard className="flex flex-col gap-5">
      <header className="space-y-1">
        <h3 className="text-base font-semibold">{editing ? "Analysis access" : "Connect a worker"}</h3>
        <p className="text-sm text-muted-foreground">
          {editing ? "Choose which key pays for analysis." : "Deploy the worker on your server to run investigations."}
        </p>
      </header>
      <WorkerForm editingWorker={editing?.id ?? null} />
      <div className="flex justify-end gap-2">
        {editing && (
          <Button variant="outline" disabled={busy} onClick={onCancel}>
            Cancel
          </Button>
        )}
        <Button disabled={!valid || busy} onClick={onSubmit}>
          {submitLabel(editing, busy)}
        </Button>
      </div>
      {children}
    </SettingsCard>
  );
}

export function WorkerSettings({
  workers,
  readyAction,
}: {
  workers: LensList["workers"];
  /** Shown once a freshly installed worker connects; defaults to a Done button that returns to the list. */
  readyAction?: ReactNode;
}) {
  const revokeWorker = useRevokeWorker();
  const prepareWorker = usePrepareWorker();
  const form = useZodForm(workerFormSchema, {
    defaultValues: defaultWorkerFormValues(),
    mode: "onChange",
  });
  const { formState, reset, setValue } = form;
  const [editingId, setEditingId] = useState<string | null>(null);
  const created = prepareWorker.data ?? null;
  const connected = useWorkerConnected(created ? workers.filter((w) => w.id === created.worker.id) : null);
  const screenInput = { workers, created, editingId, connected };
  const screen = workerScreen(screenInput);
  const error = prepareWorker.error?.message ?? revokeWorker.error?.message;
  const cancelForm = () => {
    setEditingId(null);
    reset(defaultWorkerFormValues());
  };
  const editBilling = (worker: Worker) => {
    reset({ ...defaultWorkerFormValues(), useExisting: true, analysisKey: worker.analysis_key_id ?? null });
    prepareWorker.reset();
    setEditingId(worker.id);
  };
  const submit = (editing: Worker | null) =>
    form.handleSubmit((values) => {
      const registration = {
        address: values.address,
        useExisting: values.useExisting,
        analysisKey: values.analysisKey,
        access: values.access,
        editingWorker: editing?.id ?? null,
      };
      prepareWorker.mutate(registration, {
        onSuccess: (result) => {
          if (result) return;
          setEditingId(null);
          setValue("analysisKey", null);
        },
      });
    })();
  const revoke = (id: string) =>
    revokeWorker.mutate(id, {
      onSuccess: () => {
        setValue("analysisKey", null);
        setValue("useExisting", false);
      },
    });
  switch (screen.kind) {
    case "list":
      return (
        <>
          <WorkerList workers={workers} onEditBilling={editBilling} onRevoke={revoke} />
          <ErrorText message={error} />
        </>
      );
    case "form":
      return (
        <FormProvider {...form}>
          <WorkerFormCard
            editing={screen.editing}
            busy={prepareWorker.isPending}
            valid={formState.isValid}
            onCancel={cancelForm}
            onSubmit={() => void submit(screen.editing)}
          >
            <ErrorText message={error} />
          </WorkerFormCard>
        </FormProvider>
      );
    case "install":
      return (
        <WorkerInstall address={form.getValues("address")} created={screen.created} connected={screen.connected}>
          {readyAction ?? (
            <Button className="w-full" onClick={() => prepareWorker.reset()}>
              Done
            </Button>
          )}
        </WorkerInstall>
      );
  }
}
