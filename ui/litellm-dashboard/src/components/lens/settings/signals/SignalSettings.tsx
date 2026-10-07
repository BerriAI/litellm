"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Flag, Trash2 } from "lucide-react";
import Link from "next/link";
import { useId, useState } from "react";

import { SearchSelect } from "@/components/shared/SearchSelect";
import { StatusDot } from "@/components/shared/StatusDot";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import { uiHref } from "@/utils/uiHref";

import { useLensApi } from "../../data/LensServices";
import { lensKeys, lensQueries } from "../../data/queries";
import { SIGNAL_LIBRARY, signalsConfigured, systemOneModels } from "../../model/signals";
import { WatchPicker } from "../../setup/WatchPicker";
import type { SignalConfig } from "../../model/types";
import { SettingsCard, SettingsSection } from "../SettingsSection";
import {
  MAX_SIGNALS,
  configFrom,
  draftFrom,
  draftProblems,
  newRow,
  type SignalDraft,
  type SignalRow,
} from "./signalDraft";

const LIBRARY_QUESTIONS: ReadonlySet<string> = new Set(SIGNAL_LIBRARY.map((signal) => signal.question));

export function SignalSettings() {
  return (
    <SettingsSection
      heading="Signals"
      description="Set once for every trace. A System 1 model checks each run and Traces flags matches in red."
    >
      <SignalConfigLoader />
    </SettingsSection>
  );
}

function SignalConfigLoader() {
  const api = useLensApi();
  const config = useQuery(lensQueries.signalConfig(api));
  if (config.error)
    return (
      <p role="alert" className="text-sm text-destructive">
        Could not load signals: {config.error.message}
      </p>
    );
  if (!config.data) return <Skeleton aria-label="Loading signals" className="h-40 w-full" />;
  return <SignalForm saved={config.data} />;
}

const FieldError = ({ children }: { children?: string }) =>
  children ? <p className="mt-1 text-xs text-destructive">{children}</p> : null;

function SetupCallout({ hasModels }: { hasModels: boolean }) {
  return (
    <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/5 p-3 text-sm">
      <Flag aria-hidden="true" className="mt-0.5 size-4 shrink-0 text-destructive" />
      <div className="space-y-1">
        <p className="font-medium">Choose a System 1 model to start flagging traces</p>
        {hasModels ? (
          <p className="text-xs text-muted-foreground">Pick one of the evaluation models on this proxy below.</p>
        ) : (
          <p className="text-xs text-muted-foreground">
            This proxy has no System 1 models yet. Add one with mode evaluation, for example typesafe/jev-latest, on{" "}
            <Link
              href={uiHref("models-and-endpoints")}
              className="font-medium text-foreground underline underline-offset-2"
            >
              Models + Endpoints
            </Link>
            .
          </p>
        )}
      </div>
    </div>
  );
}

function SignalFields({
  row,
  problems,
  onChange,
  onRemove,
}: {
  row: SignalRow;
  problems?: { name?: string; question?: string };
  onChange: (row: SignalRow) => void;
  onRemove: () => void;
}) {
  const label = row.name.trim() || "new signal";
  return (
    <li className="grid gap-2 rounded-md border border-border p-3 sm:grid-cols-[200px_minmax(0,1fr)_auto]">
      <div>
        <Input
          aria-label="Signal name"
          placeholder="User frustration"
          value={row.name}
          aria-invalid={Boolean(problems?.name)}
          onChange={(event) => onChange({ ...row, name: event.target.value })}
        />
        <FieldError>{problems?.name}</FieldError>
      </div>
      <div>
        <Textarea
          aria-label={`Question for ${label}`}
          placeholder="Does the user show frustration with the agent in this run?"
          rows={2}
          value={row.question}
          aria-invalid={Boolean(problems?.question)}
          onChange={(event) => onChange({ ...row, question: event.target.value })}
        />
        <FieldError>{problems?.question}</FieldError>
      </div>
      <Button variant="ghost" size="icon-sm" aria-label={`Remove ${label}`} onClick={onRemove}>
        <Trash2 aria-hidden="true" />
      </Button>
    </li>
  );
}

export function SignalForm({ saved }: { saved: SignalConfig }) {
  const api = useLensApi();
  const queryClient = useQueryClient();
  const modelId = useId();
  const thresholdId = useId();
  const [draft, setDraft] = useState<SignalDraft>(() => draftFrom(saved));
  const [draftBase, setDraftBase] = useState<SignalConfig>(() => saved);
  const [savedElsewhere, setSavedElsewhere] = useState(false);
  const savedKey = JSON.stringify(saved);
  const draftBaseKey = JSON.stringify(draftBase);
  if (savedKey !== draftBaseKey) {
    const draftIsDirty = JSON.stringify(configFrom(draft)) !== JSON.stringify(configFrom(draftFrom(draftBase)));
    setDraftBase(saved);
    if (draftIsDirty) {
      setSavedElsewhere(true);
    } else {
      setDraft(draftFrom(saved));
      setSavedElsewhere(false);
    }
  }
  const details = useQuery(lensQueries.modelDetails(api));
  const models = systemOneModels(details.data?.data ?? []);
  const options = models.map((info) => ({
    value: info.model_group,
    label: info.model_group,
    sublabel: info.providers.join(", "),
  }));
  const problems = draftProblems(draft);
  const next = configFrom(draft);
  const dirty = JSON.stringify(next) !== JSON.stringify(configFrom(draftFrom(draftBase)));
  const save = useMutation({
    mutationFn: (config: SignalConfig) => api.saveSignalConfig(config),
    onSuccess: (config) => {
      queryClient.setQueryData(lensKeys.signalConfig(api.scope), config);
      setDraft(draftFrom(config));
      setDraftBase(config);
      setSavedElsewhere(false);
      void queryClient.invalidateQueries({ queryKey: ["traceSignals"] });
    },
  });
  const setRows = (rows: readonly SignalRow[]) => setDraft((current) => ({ ...current, rows }));
  const addRow = () => setRows([...draft.rows, newRow(crypto.randomUUID())]);
  const active = signalsConfigured(saved);
  const custom = draft.rows.filter((row) => !LIBRARY_QUESTIONS.has(row.question));
  const picked = new Set(
    SIGNAL_LIBRARY.filter((signal) => draft.rows.some((row) => row.question === signal.question)).map(
      (signal) => signal.id,
    ),
  );
  const pick = (next: ReadonlySet<string>) =>
    setRows([
      ...SIGNAL_LIBRARY.filter((signal) => next.has(signal.id)).map(
        (signal) =>
          draft.rows.find((row) => row.question === signal.question) ?? {
            key: crypto.randomUUID(),
            id: draft.rows.some((row) => row.id === signal.id) ? "" : signal.id,
            name: signal.name,
            question: signal.question,
          },
      ),
      ...custom,
    ]);
  const loadLatest = () => {
    setDraft(draftFrom(saved));
    setDraftBase(saved);
    setSavedElsewhere(false);
  };

  return (
    <>
      <SettingsCard className="space-y-4">
        <p role="status" className="inline-flex items-center gap-2 text-sm">
          <StatusDot state={active ? "ok" : "off"} />
          {active ? `Flagging traces with ${saved.model}` : "Signals are off"}
        </p>
        {!saved.model && !details.isPending && <SetupCallout hasModels={models.length > 0} />}
        {savedElsewhere && (
          <div role="alert" className="flex items-center justify-between gap-3 rounded-md border p-3 text-sm">
            <span>Signals were changed elsewhere</span>
            <Button type="button" variant="outline" size="sm" onClick={loadLatest}>
              Load latest
            </Button>
          </div>
        )}
        <div className="grid gap-4 sm:grid-cols-[minmax(0,1fr)_160px]">
          <div className="space-y-1.5">
            <label htmlFor={modelId} className="text-sm font-medium">
              System 1 model
            </label>
            <SearchSelect
              inputId={modelId}
              aria-label="System 1 model"
              options={options}
              value={draft.model}
              onValueChange={(value) => setDraft((current) => ({ ...current, model: value ?? "" }))}
              placeholder={details.isPending ? "Loading models…" : "Choose a System 1 model"}
              disabled={details.isPending}
              emptyText="No System 1 models on this proxy"
              allowClear
            />
            <p className="text-xs text-muted-foreground">
              Decisions API models onboarded with mode evaluation, such as TypeSafe JEV
            </p>
          </div>
          <div className="space-y-1.5">
            <label htmlFor={thresholdId} className="text-sm font-medium">
              Flag at score
            </label>
            <div className="flex items-center gap-1.5">
              <Input
                id={thresholdId}
                type="number"
                inputMode="numeric"
                min={5}
                max={95}
                step={5}
                value={Number.isNaN(draft.thresholdPercent) ? "" : draft.thresholdPercent}
                aria-invalid={Boolean(problems.threshold)}
                onChange={(event) =>
                  setDraft((current) => ({ ...current, thresholdPercent: event.target.valueAsNumber }))
                }
              />
              <span className="text-sm text-muted-foreground">%</span>
            </div>
            <FieldError>{problems.threshold}</FieldError>
          </div>
        </div>
        <div className="space-y-4 border-t border-border pt-4">
          <WatchPicker
            label="Flag runs where"
            options={SIGNAL_LIBRARY}
            selected={picked}
            onChange={pick}
            onAddCustom={addRow}
            addDisabled={draft.rows.length >= MAX_SIGNALS}
          />
          {custom.length > 0 && (
            <ul aria-label="Custom signals" className="space-y-2">
              {custom.map((row) => (
                <SignalFields
                  key={row.key}
                  row={row}
                  problems={problems.rows.get(row.key)}
                  onChange={(changed) => setRows(draft.rows.map((other) => (other.key === row.key ? changed : other)))}
                  onRemove={() => setRows(draft.rows.filter((other) => other.key !== row.key))}
                />
              ))}
            </ul>
          )}
          {draft.rows.length === 0 && (
            <p className="text-xs text-muted-foreground">Pick at least one signal to flag traces</p>
          )}
          <FieldError>{problems.signals}</FieldError>
        </div>
        <div className="flex items-center justify-end gap-3 border-t border-border pt-4">
          {save.error && (
            <p role="alert" className="text-sm text-destructive">
              Could not save signals: {save.error.message}
            </p>
          )}
          {save.isSuccess && !dirty && <p className="text-xs text-muted-foreground">Saved</p>}
          <Button disabled={!dirty || problems.any || save.isPending} onClick={() => save.mutate(next)}>
            {save.isPending ? "Saving…" : "Save signals"}
          </Button>
        </div>
      </SettingsCard>
    </>
  );
}
