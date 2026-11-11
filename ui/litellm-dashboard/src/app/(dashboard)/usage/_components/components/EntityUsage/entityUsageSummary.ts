import type { ParseKeys, TFunction } from "i18next";
import i18n from "@/i18n/i18n";
import { formatNumberWithCommas } from "@/utils/dataUtils";

export interface SummaryTile {
  title: string;
  value: string;
  className?: string;
  tooltip?: string;
  expandable?: boolean;
}

interface SpendSummaryMetadata {
  total_spend: number;
  total_flat_cost?: number;
  total_ptu_hours?: number;
  total_api_requests: number;
  total_successful_requests: number;
  total_failed_requests: number;
  total_tokens: number;
}

export const SUMMARY_TILE_LABELS: Readonly<Record<string, ParseKeys>> = {
  "PTU Hours": "usage.entitySummary.titles.ptuHours",
  "Total Cost": "usage.entitySummary.titles.totalCost",
  "Total Spend": "usage.entitySummary.titles.totalSpend",
  "Total Requests": "usage.entitySummary.titles.totalRequests",
  "Successful Requests": "usage.entitySummary.titles.successfulRequests",
  "Failed Requests": "usage.entitySummary.titles.failedRequests",
  "Total Tokens": "usage.entitySummary.titles.totalTokens",
  "Request Cost": "usage.entitySummary.titles.requestCost",
  "Flat Cost": "usage.entitySummary.titles.flatCost",
};

export const TOTAL_COST_TOOLTIP: ParseKeys = "usage.entitySummary.tooltips.totalCost";

export const REQUEST_COST_TOOLTIP: ParseKeys = "usage.entitySummary.tooltips.requestCost";

export const FLAT_COST_TOOLTIP: ParseKeys = "usage.entitySummary.tooltips.flatCost";

export const PTU_HOURS_TOOLTIP: ParseKeys = "usage.entitySummary.tooltips.ptuHours";

export const hasFlatCost = (metadata: SpendSummaryMetadata): boolean => (metadata.total_flat_cost ?? 0) > 0;

export const hasPtuHours = (metadata: SpendSummaryMetadata): boolean => (metadata.total_ptu_hours ?? 0) > 0;

export const buildSummaryTiles = (
  metadata: SpendSummaryMetadata,
  showFlatCost: boolean,
  showPtuHours: boolean = false,
  t: TFunction = i18n.t.bind(i18n),
): SummaryTile[] => {
  const flatCost = metadata.total_flat_cost ?? 0;
  const ptuHoursTile: SummaryTile[] = showPtuHours
    ? [
        {
          title: "PTU Hours",
          value: formatNumberWithCommas(metadata.total_ptu_hours ?? 0, 2),
          tooltip: t(PTU_HOURS_TOOLTIP),
        },
      ]
    : [];
  return [
    showFlatCost
      ? {
          title: "Total Cost",
          value: `$${formatNumberWithCommas(metadata.total_spend + flatCost, 2)}`,
          tooltip: t(TOTAL_COST_TOOLTIP),
          expandable: true,
        }
      : { title: "Total Spend", value: `$${formatNumberWithCommas(metadata.total_spend, 2)}` },
    { title: "Total Requests", value: metadata.total_api_requests.toLocaleString() },
    {
      title: "Successful Requests",
      value: metadata.total_successful_requests.toLocaleString(),
      className: "text-success",
    },
    { title: "Failed Requests", value: metadata.total_failed_requests.toLocaleString(), className: "text-destructive" },
    { title: "Total Tokens", value: metadata.total_tokens.toLocaleString() },
    ...ptuHoursTile,
  ];
};

export const buildCostBreakdownTiles = (
  metadata: SpendSummaryMetadata,
  t: TFunction = i18n.t.bind(i18n),
): SummaryTile[] => [
  {
    title: "Request Cost",
    value: `$${formatNumberWithCommas(metadata.total_spend, 2)}`,
    className: "text-info",
    tooltip: t(REQUEST_COST_TOOLTIP),
  },
  {
    title: "Flat Cost",
    value: `$${formatNumberWithCommas(metadata.total_flat_cost ?? 0, 2)}`,
    className: "text-violet-600",
    tooltip: t(FLAT_COST_TOOLTIP),
  },
];
