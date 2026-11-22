"use client";

import { Settings } from "lucide-react";
import { useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import RouterSettingsSummary from "@/components/common_components/RouterSettingsSummary";
import { hasRouterSettings } from "@/components/common_components/routerSettingsPayload";
import { modelOrAccessGroupHref } from "@/utils/entityLinks";

import { formatCount } from "./formatters";
import { LimitTable } from "./LimitTable";
import { ResourceLink } from "./ResourceLink";
import { BadgeRow, Caption, FactLabel } from "./SectionCard";

export interface OtherSettingsData {
  readonly tags: readonly string[];
  readonly allowedRoutes: readonly string[];
  readonly passThroughRoutes: readonly string[];
  readonly prompts: readonly string[];
  readonly promptCaching: boolean;
  readonly estimatedOutputTokens: number | null;
  readonly estimatedOutputTokensPerModel: readonly (readonly [string, number])[];
  readonly routerSettings: Record<string, unknown> | null | undefined;
  readonly metadataJson: string;
  readonly metadataFieldCount: number;
}

interface OtherSettingsProps {
  data: OtherSettingsData;
  accessGroupNames: ReadonlySet<string> | undefined;
}

type JsonView = "router" | "metadata" | null;

export const hasOtherSettings = (data: OtherSettingsData): boolean =>
  data.tags.length > 0 ||
  data.allowedRoutes.length > 0 ||
  data.passThroughRoutes.length > 0 ||
  data.prompts.length > 0 ||
  data.promptCaching ||
  data.estimatedOutputTokens !== null ||
  data.estimatedOutputTokensPerModel.length > 0 ||
  hasRouterSettings(data.routerSettings) ||
  data.metadataFieldCount > 0;

function Item({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <FactLabel>{label}</FactLabel>
      <div className="min-w-0 [overflow-wrap:anywhere]">{children}</div>
    </div>
  );
}

function Badges({ values }: { values: readonly string[] }) {
  return (
    <BadgeRow>
      {values.map((value) => (
        <Badge key={value} variant="secondary">
          {value}
        </Badge>
      ))}
    </BadgeRow>
  );
}

export function OtherSettings({ data, accessGroupNames }: OtherSettingsProps) {
  const { t } = useTranslation();
  const [jsonView, setJsonView] = useState<JsonView>(null);
  if (!hasOtherSettings(data)) return null;
  const routerFieldCount = data.routerSettings
    ? Object.values(data.routerSettings).filter((value) => value != null).length
    : 0;

  return (
    <section className="pt-2" data-testid="key-other-settings">
      <div className="mb-3 flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h2 className="m-0 flex items-center gap-2 text-[15px] leading-[22px] font-semibold">
          <Settings className="size-4 text-muted-foreground" aria-hidden />
          {t("keyDetail.overview.other.title")}
        </h2>
        <Caption>{t("keyDetail.overview.other.subtitle")}</Caption>
      </div>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(200px,1fr))] gap-x-6 gap-y-4">
        {data.tags.length > 0 ? (
          <Item label={t("keyDetail.overview.other.tags")}>
            <Badges values={data.tags} />
          </Item>
        ) : null}
        {data.allowedRoutes.length > 0 ? (
          <Item label={t("keyDetail.overview.other.allowedRoutes")}>
            <span className="font-mono text-xs">{data.allowedRoutes.join(", ")}</span>
          </Item>
        ) : null}
        {data.passThroughRoutes.length > 0 ? (
          <Item label={t("keyDetail.overview.other.passThroughRoutes")}>
            <span className="font-mono text-xs">{data.passThroughRoutes.join(", ")}</span>
          </Item>
        ) : null}
        {data.prompts.length > 0 ? (
          <Item label={t("keyDetail.overview.other.prompts")}>
            <Badges values={data.prompts} />
          </Item>
        ) : null}
        {data.promptCaching ? (
          <Item label={t("keyDetail.overview.other.promptCaching")}>{t("keyDetail.overview.other.on")}</Item>
        ) : null}
        {data.estimatedOutputTokens !== null ? (
          <Item label={t("keyDetail.overview.other.estimatedOutputTokens")}>
            {t("keyDetail.overview.other.byDefault", { tokens: formatCount(data.estimatedOutputTokens) })}
          </Item>
        ) : null}
        {hasRouterSettings(data.routerSettings) ? (
          <Item label={t("keyDetail.overview.other.routerSettings")}>
            <span className="inline-flex items-center gap-1.5">
              {t("keyDetail.overview.other.fields", { count: routerFieldCount })}
              <Button variant="link" size="sm" className="h-auto p-0" onClick={() => setJsonView("router")}>
                {t("keyDetail.overview.other.view")}
              </Button>
            </span>
          </Item>
        ) : null}
        {data.metadataFieldCount > 0 ? (
          <Item label={t("keyDetail.overview.other.metadata")}>
            <span className="inline-flex items-center gap-1.5">
              {t("keyDetail.overview.other.fields", { count: data.metadataFieldCount })}
              <Button variant="link" size="sm" className="h-auto p-0" onClick={() => setJsonView("metadata")}>
                {t("keyDetail.overview.other.viewJson")}
              </Button>
            </span>
          </Item>
        ) : null}
      </div>
      {data.estimatedOutputTokensPerModel.length > 0 ? (
        <div className="mt-4 max-w-[560px]">
          <LimitTable
            title={t("keyDetail.overview.other.estimatedPerModel")}
            searchLabel={t("keyDetail.overview.rateLimits.searchModels")}
            columns={[t("keyDetail.overview.rateLimits.model"), t("keyDetail.overview.other.tokens")]}
            rows={data.estimatedOutputTokensPerModel.map(([model, tokens]) => ({
              key: model,
              cells: [
                <ResourceLink key="model" href={modelOrAccessGroupHref(model, accessGroupNames)}>
                  {model}
                </ResourceLink>,
                formatCount(tokens),
              ],
            }))}
          />
        </div>
      ) : null}
      <Dialog open={jsonView !== null} onOpenChange={(open) => (open ? undefined : setJsonView(null))}>
        <DialogContent className="sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle>
              {jsonView === "router"
                ? t("keyDetail.overview.other.routerSettings")
                : t("keyDetail.overview.other.metadata")}
            </DialogTitle>
          </DialogHeader>
          {jsonView === "router" ? (
            <RouterSettingsSummary routerSettings={data.routerSettings} />
          ) : (
            <pre className="max-h-[60vh] overflow-auto rounded-md bg-muted p-3 font-mono text-xs">
              {data.metadataJson}
            </pre>
          )}
        </DialogContent>
      </Dialog>
    </section>
  );
}
