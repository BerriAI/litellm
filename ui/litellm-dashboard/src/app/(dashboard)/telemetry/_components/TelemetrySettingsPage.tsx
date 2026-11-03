"use client";

import { useState } from "react";
import { Switch } from "@/components/ui/switch";
import { FieldLabel } from "@/components/ui/field";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Activity } from "lucide-react";
import { Page, PageTabs, PageTabsContent, PageTabsList, PageTabsTrigger } from "@/components/shared/Page";
import { PageHeader, PageHeaderControls, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import { useUrlTab } from "@/hooks/useUrlTab";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import {
  type TelemetryDestination,
  type TelemetryGroup,
  type TelemetrySettings,
  useTelemetrySettings,
  useUpdateTelemetrySettings,
} from "@/app/(dashboard)/hooks/telemetry/useTelemetrySettings";
import { GROUP_COPY, PROXY_GROUPS, UI_GROUPS, relativeDepths, toggleGroup, type Requires } from "./telemetryGroups";
import { TelemetryEnterpriseDialog } from "./TelemetryEnterpriseDialog";

const TABS = ["proxy", "ui"] as const;

const destinationText = (settings: TelemetrySettings): string => {
  const every = `every ${settings.flush_interval_seconds} s`;
  switch (settings.destination) {
    case "http":
      return `One JSON report per worker ${every} and on shutdown, by HTTP POST to LITELLM_TELEMETRY_ENDPOINT. A failed send is retried next window`;
    case "local_table":
      return `One report per worker ${every} and on shutdown, stored in this proxy's LiteLLM_TelemetryReport table for ${settings.retention_days} days. Nothing leaves the proxy. Admins can export from GET /telemetry/reports`;
    case null:
      return "No destination is chosen, so nothing is recorded";
  }
};

const DESTINATION_LABELS: Record<TelemetryDestination, string> = {
  local_table: "Local table (LiteLLM_TelemetryReport in this proxy's database)",
  http: "HTTP (POST to LITELLM_TELEMETRY_ENDPOINT)",
};

type SettingsUpdate = ReturnType<typeof useUpdateTelemetrySettings>;

const enabledGroups = (settings: TelemetrySettings): Set<TelemetryGroup> =>
  new Set(settings.groups.filter((info) => info.enabled).map((info) => info.group));

const isOn = (settings: TelemetrySettings): boolean => enabledGroups(settings).size > 0;

const savedDestination = (settings: TelemetrySettings): TelemetryDestination | null =>
  settings.destination_set_by_environment ? null : settings.stored_destination ?? null;

function DestinationPicker({
  settings,
  update,
  requestEnable,
}: {
  settings: TelemetrySettings;
  update: SettingsUpdate;
  requestEnable: () => void;
}) {
  const unchangeable = !settings.editable || settings.destination_set_by_environment;
  const locked = unchangeable || update.isPending || !isOn(settings);
  const missing = settings.destination === null && enabledGroups(settings).size > 0;
  return (
    <section className="space-y-2">
      <h3 className="font-medium">Destination</h3>
      {missing && (
        <p className="text-xs text-destructive">Telemetry is on but no destination is chosen, so nothing is recorded</p>
      )}
      <RadioGroup
        value={settings.destination ?? ""}
        disabled={locked}
        onValueChange={(value: unknown) =>
          settings.destination === null
            ? requestEnable()
            : update.mutate({ groups: [...enabledGroups(settings)], destination: value as TelemetryDestination })
        }
      >
        {(Object.keys(DESTINATION_LABELS) as TelemetryDestination[]).map((destination) => (
          <FieldLabel key={destination} className="text-xs font-normal">
            <RadioGroupItem
              value={destination}
              disabled={!settings.available_destinations.includes(destination)}
              aria-label={DESTINATION_LABELS[destination]}
            />
            {DESTINATION_LABELS[destination]}
          </FieldLabel>
        ))}
      </RadioGroup>
      {settings.destination_set_by_environment && (
        <p className="text-xs text-muted-foreground">Set by LITELLM_TELEMETRY_DESTINATION</p>
      )}
    </section>
  );
}

function GroupList({
  groups,
  settings,
  update,
  requestEnable,
}: {
  groups: readonly TelemetryGroup[];
  settings: TelemetrySettings;
  update: SettingsUpdate;
  requestEnable: () => void;
}) {
  const requires: Requires = new Map(settings.groups.map((info) => [info.group, info.requires ?? null]));
  const enabled = enabledGroups(settings);
  const depths = relativeDepths(groups, requires);
  const onToggle = (group: TelemetryGroup, on: boolean) => {
    if (on) {
      requestEnable();
      return;
    }
    update.mutate({ groups: [...toggleGroup(group, on, enabled, requires)], destination: savedDestination(settings) });
  };
  return (
    <ul className="divide-y rounded-md border">
      {groups.map((group) => {
        const copy = GROUP_COPY[group];
        const locked = !settings.editable || update.isPending || !isOn(settings);
        const depth = depths.get(group) ?? 0;
        return (
          <li
            key={group}
            data-depth={depth}
            className="flex items-start gap-3 py-2 pr-3"
            style={{ paddingLeft: `${0.75 + depth * 1.5}rem` }}
          >
            <Switch
              size="sm"
              aria-label={copy.title}
              checked={enabled.has(group)}
              disabled={locked}
              onCheckedChange={(on) => onToggle(group, on)}
            />
            <div className="space-y-0.5 text-xs text-muted-foreground">
              <div className="text-sm font-medium text-foreground">{copy.title}</div>
              {copy.slices !== undefined && <p>Slices by: {copy.slices}</p>}
              <p>
                {copy.slices !== undefined ? "Counts" : "Sends"}: {copy.counts}
              </p>
              <p>Helps catch: {copy.helps}</p>
            </div>
          </li>
        );
      })}
      {update.error !== null && <li className="p-3 text-xs text-destructive">{update.error.message}</li>}
    </ul>
  );
}

function TelemetrySwitch({
  settings,
  update,
  requestEnable,
}: {
  settings: TelemetrySettings;
  update: SettingsUpdate;
  requestEnable: () => void;
}) {
  const on = isOn(settings);
  return (
    <div className="flex items-center gap-2 text-sm font-medium">
      <Switch
        aria-label="Telemetry"
        checked={on}
        disabled={!settings.editable || update.isPending}
        onCheckedChange={(next) =>
          next ? requestEnable() : update.mutate({ groups: [], destination: savedDestination(settings) })
        }
      />
      <span aria-hidden>{on ? "On" : "Off"}</span>
    </div>
  );
}

function ReportPreview({ settings }: { settings: TelemetrySettings }) {
  return (
    <section className="space-y-1">
      <h3 className="font-medium">
        {settings.report_is_sample ? "Sample report" : "Last report sent from this worker"}
      </h3>
      <p className="text-xs text-muted-foreground">
        {settings.report_is_sample
          ? "Made-up traffic through the real filter, with the groups that are on (all groups while telemetry is off)"
          : "Exactly what this worker last sent or stored"}
      </p>
      <pre className="max-h-96 overflow-auto rounded-md bg-muted p-3 text-xs">
        {JSON.stringify(settings.report, null, 2)}
      </pre>
    </section>
  );
}

export default function TelemetrySettingsPage() {
  const [tab, setTab] = useUrlTab(TABS, "proxy");
  const { data: settings, isLoading, error } = useTelemetrySettings();
  const update = useUpdateTelemetrySettings();
  const { premiumUser } = useAuthorized();
  const [enableDialogOpen, setEnableDialogOpen] = useState(false);
  const requestEnable = () => setEnableDialogOpen(true);
  if (isLoading) return <Page>Loading telemetry settings...</Page>;
  if (settings === undefined)
    return <Page className="text-destructive">Could not load telemetry settings: {error?.message}</Page>;
  return (
    <Page className="text-sm">
      <PageTabs value={tab} onValueChange={(value) => setTab(value as (typeof TABS)[number])}>
        <PageHeader>
          <PageHeaderTitle>
            <Activity />
            Telemetry
          </PageHeaderTitle>
          <PageHeaderDescription>
            Choose what privacy-safe, aggregated usage data this proxy shares
          </PageHeaderDescription>
          <PageHeaderControls>
            <TelemetrySwitch settings={settings} update={update} requestEnable={requestEnable} />
            <PageTabsList>
              <PageTabsTrigger value="proxy">Proxy requests</PageTabsTrigger>
              <PageTabsTrigger value="ui">Admin UI</PageTabsTrigger>
            </PageTabsList>
          </PageHeaderControls>
        </PageHeader>
        <PageTabsContent value="proxy" className="gap-4">
          <GroupList groups={PROXY_GROUPS} settings={settings} update={update} requestEnable={requestEnable} />
        </PageTabsContent>
        <PageTabsContent value="ui" className="gap-4">
          <GroupList groups={UI_GROUPS} settings={settings} update={update} requestEnable={requestEnable} />
          <p className="text-xs text-muted-foreground">
            Sent only while Page navigation is on, to this proxy at POST /telemetry/ui_events, and counted into the same
            report as proxy traffic
          </p>
        </PageTabsContent>
      </PageTabs>
      <DestinationPicker settings={settings} update={update} requestEnable={requestEnable} />
      <section className="space-y-1">
        <h3 className="font-medium">How and when it is sent</h3>
        <p className="text-xs text-muted-foreground">{destinationText(settings)}. Changes apply from the next window</p>
        <p className="text-xs text-muted-foreground">
          Off by default. Each switch adds aggregated counts, never prompts, responses, keys, ids or header values.
          LITELLM_TELEMETRY_DISABLED=true turns everything off
        </p>
      </section>
      <ReportPreview settings={settings} />
      <TelemetryEnterpriseDialog
        open={enableDialogOpen}
        onOpenChange={setEnableDialogOpen}
        premiumUser={premiumUser === true}
      />
    </Page>
  );
}
