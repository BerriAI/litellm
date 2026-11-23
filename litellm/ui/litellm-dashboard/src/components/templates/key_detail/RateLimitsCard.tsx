"use client";

import { Gauge } from "lucide-react";
import { useTranslation } from "react-i18next";

import { modelOrAccessGroupHref } from "@/utils/entityLinks";
import { cn } from "@/lib/cva.config";

import { formatCount } from "./formatters";
import type { KeyRateLimits } from "./keyOverviewSections";
import { LimitTable } from "./LimitTable";
import { ResourceLink } from "./ResourceLink";
import { FactLabel, SectionCard } from "./SectionCard";

interface RateLimitsCardProps {
  limits: KeyRateLimits;
  accessGroupNames: ReadonlySet<string> | undefined;
}

function LimitValue({ value }: { value: number | null }) {
  const { t } = useTranslation();
  return value === null ? (
    <span className="text-muted-foreground">{t("keyDetail.overview.rateLimits.noLimit")}</span>
  ) : (
    <>{formatCount(value)}</>
  );
}

export function RateLimitsCard({ limits, accessGroupNames }: RateLimitsCardProps) {
  const { t } = useTranslation();
  const tiles = [
    { key: "rpm", label: t("keyDetail.overview.rateLimits.rpm"), value: limits.rpm },
    { key: "tpm", label: t("keyDetail.overview.rateLimits.tpm"), value: limits.tpm },
    { key: "tpd", label: t("keyDetail.overview.rateLimits.tpd"), value: limits.tpd },
    { key: "parallel", label: t("keyDetail.overview.rateLimits.parallel"), value: limits.parallel },
  ];
  const hasTables = limits.perModel.length > 0 || limits.perTag.length > 0;

  return (
    <SectionCard icon={Gauge} title={t("keyDetail.overview.rateLimits.title")} testId="key-rate-limits-card">
      <div className="grid grid-cols-2 gap-y-3 md:grid-cols-4 md:gap-y-0">
        {tiles.map((tile, index) => (
          <div
            key={tile.key}
            className={cn(
              "flex min-w-0 flex-col gap-1.5 px-4",
              index % 2 === 0 ? "pl-0 md:pl-4" : "border-l border-border",
              index >= 2 && "border-t border-border pt-3 md:border-t-0 md:pt-0",
              index === 2 && "md:border-l",
              index === 0 && "md:pl-0",
            )}
          >
            <FactLabel>{tile.label}</FactLabel>
            <div className="text-[15px] leading-[22px] font-semibold tabular-nums">
              <LimitValue value={tile.value} />
            </div>
          </div>
        ))}
      </div>
      {hasTables ? (
        <div className="grid grid-cols-1 items-start gap-x-8 gap-y-4 border-t border-border pt-3 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
          {limits.perModel.length > 0 ? (
            <LimitTable
              title={t("keyDetail.overview.rateLimits.perModel")}
              searchLabel={t("keyDetail.overview.rateLimits.searchModels")}
              columns={[
                t("keyDetail.overview.rateLimits.model"),
                t("keyDetail.overview.rateLimits.rpm"),
                t("keyDetail.overview.rateLimits.tpm"),
              ]}
              rows={limits.perModel.map((row) => ({
                key: row.model,
                cells: [
                  <ResourceLink key="model" href={modelOrAccessGroupHref(row.model, accessGroupNames)}>
                    {row.model}
                  </ResourceLink>,
                  <LimitValue key="rpm" value={row.rpm} />,
                  <LimitValue key="tpm" value={row.tpm} />,
                ],
              }))}
            />
          ) : null}
          {limits.perTag.length > 0 ? (
            <LimitTable
              title={t("keyDetail.overview.rateLimits.perTag")}
              searchLabel={t("keyDetail.overview.rateLimits.searchTags")}
              columns={[t("keyDetail.overview.rateLimits.tag"), t("keyDetail.overview.rateLimits.rpm")]}
              rows={limits.perTag.map((row) => ({
                key: row.tag,
                cells: [
                  <span key="tag" className="font-medium">
                    {row.tag}
                  </span>,
                  <LimitValue key="rpm" value={row.rpm} />,
                ],
              }))}
            />
          ) : null}
        </div>
      ) : null}
    </SectionCard>
  );
}
