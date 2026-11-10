import type * as React from "react";
import { useTranslation } from "react-i18next";

import { ProviderLogo } from "@/components/molecules/models/ProviderLogo";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";
import { cn } from "@/lib/cva.config";
import {
  NUMERIC_CELL_CLASS,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

import {
  buildPricingTable,
  type DeploymentPricing,
  describePricingTier,
  formatTokenCount,
  type PriceGroup,
  SERVICE_TIER_LABELS,
  type ServiceTier,
} from "./deploymentPricing";

function PriceGroupRows({ group, tiers }: { group: PriceGroup; tiers: readonly ServiceTier[] }) {
  const { t } = useTranslation();
  const name =
    group.threshold === null
      ? t("deploymentPricing.basePrices")
      : t("deploymentPricing.aboveThreshold", { threshold: formatTokenCount(group.threshold) });
  return (
    <TableBody aria-label={name}>
      {group.threshold !== null && (
        <TableRow className="hover:bg-transparent">
          <TableHead colSpan={tiers.length + 1} scope="colgroup" className="h-8 text-xs text-muted-foreground">
            {name}
          </TableHead>
        </TableRow>
      )}
      {group.rows.map((row) => (
        <TableRow key={row.label}>
          <TableHead scope="row" className="font-normal">
            {row.label}
          </TableHead>
          {tiers.map((tier) => (
            <TableCell key={tier} className={NUMERIC_CELL_CLASS}>
              {row.prices[tier]}
            </TableCell>
          ))}
        </TableRow>
      ))}
    </TableBody>
  );
}

function DeploymentPrices({ deployment }: { deployment: DeploymentPricing }) {
  const { t } = useTranslation();
  const table = buildPricingTable(deployment.costs);
  const tiers = deployment.tiered_pricing;
  const providerName = getProviderLogoAndName(deployment.provider).displayName;
  return (
    <section
      aria-label={t("deploymentPricing.providerPricing", { provider: providerName })}
      className="flex flex-col gap-2 rounded-lg border border-border p-3"
    >
      <h4 className="flex items-center gap-2 text-sm font-medium">
        <ProviderLogo provider={deployment.provider} />
        {providerName}
      </h4>
      {table.groups.length > 0 && (
        <Table>
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              <TableHead>{t("deploymentPricing.chargeColumn")}</TableHead>
              {table.tiers.map((tier) => (
                <TableHead key={tier} className={NUMERIC_CELL_CLASS}>
                  {t(SERVICE_TIER_LABELS[tier])}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          {table.groups.map((group) => (
            <PriceGroupRows key={group.threshold ?? "base"} group={group} tiers={table.tiers} />
          ))}
        </Table>
      )}
      {table.hasBlankCells && <p className="text-xs text-muted-foreground">{t("deploymentPricing.blankCells")}</p>}
      {tiers.length > 0 && (
        <div className="flex flex-col gap-1 text-sm">
          <p className="text-xs font-medium text-muted-foreground">{t("deploymentPricing.tieredPricing")}</p>
          <ul className="flex flex-col gap-0.5 tabular-nums">
            {tiers.map((tier) => (
              <li key={tier.range.join("-")}>{describePricingTier(tier)}</li>
            ))}
          </ul>
        </div>
      )}
      {table.unknown.length > 0 && (
        <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1 text-sm">
          {table.unknown.map(({ key, label, price }) => (
            <div key={key} className="contents">
              <dt className="truncate">{label}</dt>
              <dd className="text-right tabular-nums">{price}</dd>
            </div>
          ))}
        </dl>
      )}
    </section>
  );
}

export function DeploymentPricingBreakdown({
  deployments,
  heading,
  className,
}: {
  deployments: readonly DeploymentPricing[] | null | undefined;
  heading?: React.ReactNode;
  className?: string;
}) {
  const { t } = useTranslation();
  if (!deployments || deployments.length === 0) return null;
  return (
    <div className={cn("flex flex-col gap-3", className)}>
      {heading}
      {deployments.length > 1 && (
        <p className="text-xs text-muted-foreground">
          {t("deploymentPricing.deploymentCount", { count: deployments.length })}
        </p>
      )}
      {deployments.map((deployment, index) => (
        <DeploymentPrices key={`${deployment.provider}-${index}`} deployment={deployment} />
      ))}
      <p className="text-xs text-muted-foreground">{t("deploymentPricing.requestAdjustments")}</p>
    </div>
  );
}
