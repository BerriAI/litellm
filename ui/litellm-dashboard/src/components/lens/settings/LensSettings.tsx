"use client";

import { useId, type ReactNode } from "react";
import { Activity, ArrowUpRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { StatusDot } from "@/components/shared/StatusDot";
import { useStoredValue } from "@/lib/storage";
import { LENS_INTRO_DISMISSED } from "../storage";
import { WorkerSettings } from "./worker/WorkerSettings";
import { SettingsCard, SettingsSection } from "./SettingsSection";
import type { LensList } from "../model/types";

const TRACING_DOCS = "https://docs.litellm.ai/docs/proxy/lens";

function TracingSection({ enabled, onOpenTraces }: { enabled: boolean; onOpenTraces: () => void }) {
  return (
    <SettingsSection heading="Tracing" description="Where your agents send runs so Lens can read them.">
      <SettingsCard className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3 text-sm">
          <Activity aria-hidden="true" className="size-4 text-muted-foreground" />
          <span role="status" className="inline-flex items-center gap-2">
            <StatusDot state={enabled ? "ok" : "off"} />
            {enabled ? "Tracing enabled" : "Tracing is not enabled"}
          </span>
        </div>
        <div className="flex items-center gap-3">
          <a
            href={TRACING_DOCS}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-xs text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
          >
            Docs
            <ArrowUpRight aria-hidden="true" className="size-3" />
          </a>
          <Button variant="outline" size="sm" onClick={onOpenTraces}>
            {enabled ? "Connect an agent" : "Enable tracing"}
          </Button>
        </div>
      </SettingsCard>
    </SettingsSection>
  );
}

function IntroductionSection() {
  const [dismissed, setDismissed] = useStoredValue(LENS_INTRO_DISMISSED);
  const id = useId();
  return (
    <SettingsSection heading="Introduction" description="The getting started dialog shown when you open Lens.">
      <SettingsCard className="flex items-center justify-between gap-3">
        <Label htmlFor={id} className="text-sm font-normal">
          Show the introduction on each new session
        </Label>
        <Switch id={id} checked={!dismissed} onCheckedChange={(show) => setDismissed(!show)} />
      </SettingsCard>
    </SettingsSection>
  );
}

export function LensSettings({
  list,
  workerReadyAction,
  onOpenTraces,
}: {
  list: LensList;
  /** Replaces the worker install card's Done button once the new worker connects. */
  workerReadyAction?: ReactNode;
  onOpenTraces: () => void;
}) {
  return (
    <div aria-label="Settings" role="region" className="flex w-full flex-col divide-y divide-border">
      <TracingSection enabled={list.tracing_enabled} onOpenTraces={onOpenTraces} />
      <SettingsSection
        heading="Analysis worker"
        description="Runs investigations on your server and bills model usage to an analysis key."
      >
        <WorkerSettings workers={list.workers} readyAction={workerReadyAction} />
      </SettingsSection>
      <IntroductionSection />
    </div>
  );
}
