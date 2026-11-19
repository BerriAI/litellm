"use client";

import type { ParseKeys, TFunction } from "i18next";
import { useState } from "react";
import { useTranslation } from "react-i18next";
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

const destinationText = (settings: TelemetrySettings, t: TFunction): string => {
  const every = t("telemetry.destinationText.every", { seconds: settings.flush_interval_seconds });
  switch (settings.destination) {
    case "http":
      return t("telemetry.destinationText.http", { every });
    case "local_table":
      return t("telemetry.destinationText.localTable", { every, count: settings.retention_days });
    case null:
      return t("telemetry.destinationText.none");
  }
};

const DESTINATION_LABELS: Record<TelemetryDestination, ParseKeys> = {
  local_table: "telemetry.destination.localTable",
  http: "telemetry.destination.http",
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
  const { t } = useTranslation();
  const unchangeable = !settings.editable || settings.destination_set_by_environment;
  const locked = unchangeable || update.isPending || !isOn(settings);
  const missing = settings.destination === null && enabledGroups(settings).size > 0;
  return (
    <section className="space-y-2">
      <h3 className="font-medium">{t("telemetry.destination.title")}</h3>
      {missing && <p className="text-xs text-destructive">{t("telemetry.destination.missing")}</p>}
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
              aria-label={t(DESTINATION_LABELS[destination])}
            />
            {t(DESTINATION_LABELS[destination])}
          </FieldLabel>
        ))}
      </RadioGroup>
      {settings.destination_set_by_environment && (
        <p className="text-xs text-muted-foreground">{t("telemetry.destination.setByEnvironment")}</p>
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
  const { t } = useTranslation();
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
              aria-label={t(copy.title)}
              checked={enabled.has(group)}
              disabled={locked}
              onCheckedChange={(on) => onToggle(group, on)}
            />
            <div className="space-y-0.5 text-xs text-muted-foreground">
              <div className="text-sm font-medium text-foreground">{t(copy.title)}</div>
              {copy.slices !== undefined && (
                <p>
                  {t("telemetry.groupList.slicesBy")}
                  {t(copy.slices)}
                </p>
              )}
              <p>
                {copy.slices !== undefined ? t("telemetry.groupList.counts") : t("telemetry.groupList.sends")}:{" "}
                {t(copy.counts)}
              </p>
              <p>
                {t("telemetry.groupList.helpsCatch")}
                {t(copy.helps)}
              </p>
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
  const { t } = useTranslation();
  const on = isOn(settings);
  return (
    <div className="flex items-center gap-2 text-sm font-medium">
      <Switch
        aria-label={t("telemetry.switch.label")}
        checked={on}
        disabled={!settings.editable || update.isPending}
        onCheckedChange={(next) =>
          next ? requestEnable() : update.mutate({ groups: [], destination: savedDestination(settings) })
        }
      />
      <span aria-hidden>{on ? t("telemetry.switch.on") : t("telemetry.switch.off")}</span>
    </div>
  );
}

function ReportPreview({ settings }: { settings: TelemetrySettings }) {
  const { t } = useTranslation();
  return (
    <section className="space-y-1">
      <h3 className="font-medium">
        {settings.report_is_sample ? t("telemetry.report.sampleTitle") : t("telemetry.report.lastTitle")}
      </h3>
      <p className="text-xs text-muted-foreground">
        {settings.report_is_sample ? t("telemetry.report.sampleDescription") : t("telemetry.report.lastDescription")}
      </p>
      <pre className="max-h-96 overflow-auto rounded-md bg-muted p-3 text-xs">
        {JSON.stringify(settings.report, null, 2)}
      </pre>
    </section>
  );
}

export default function TelemetrySettingsPage() {
  const { t } = useTranslation();
  const [tab, setTab] = useUrlTab(TABS, "proxy");
  const { data: settings, isLoading, error } = useTelemetrySettings();
  const update = useUpdateTelemetrySettings();
  const { premiumUser } = useAuthorized();
  const [enableDialogOpen, setEnableDialogOpen] = useState(false);
  const requestEnable = () => setEnableDialogOpen(true);
  if (isLoading) return <Page>{t("telemetry.page.loading")}</Page>;
  if (settings === undefined)
    return (
      <Page className="text-destructive">
        {t("telemetry.page.loadFailed")}
        {error?.message}
      </Page>
    );
  return (
    <Page className="text-sm">
      <PageTabs value={tab} onValueChange={(value) => setTab(value as (typeof TABS)[number])}>
        <PageHeader>
          <PageHeaderTitle>
            <Activity />
            {t("telemetry.page.title")}
          </PageHeaderTitle>
          <PageHeaderDescription>{t("telemetry.page.description")}</PageHeaderDescription>
          <PageHeaderControls>
            <TelemetrySwitch settings={settings} update={update} requestEnable={requestEnable} />
            <PageTabsList>
              <PageTabsTrigger value="proxy">{t("telemetry.page.tabs.proxy")}</PageTabsTrigger>
              <PageTabsTrigger value="ui">{t("telemetry.page.tabs.ui")}</PageTabsTrigger>
            </PageTabsList>
          </PageHeaderControls>
        </PageHeader>
        <PageTabsContent value="proxy" className="gap-4">
          <GroupList groups={PROXY_GROUPS} settings={settings} update={update} requestEnable={requestEnable} />
        </PageTabsContent>
        <PageTabsContent value="ui" className="gap-4">
          <GroupList groups={UI_GROUPS} settings={settings} update={update} requestEnable={requestEnable} />
          <p className="text-xs text-muted-foreground">{t("telemetry.page.uiNote")}</p>
        </PageTabsContent>
      </PageTabs>
      <DestinationPicker settings={settings} update={update} requestEnable={requestEnable} />
      <section className="space-y-1">
        <h3 className="font-medium">{t("telemetry.howSent.title")}</h3>
        <p className="text-xs text-muted-foreground">
          {destinationText(settings, t)}. {t("telemetry.howSent.changesApply")}
        </p>
        <p className="text-xs text-muted-foreground">{t("telemetry.howSent.offByDefault")}</p>
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
